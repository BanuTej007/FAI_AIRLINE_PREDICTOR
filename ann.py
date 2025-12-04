from preprocess import *

class ANNModel:
    def __init__(self, learning_rate=0.0005):
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = None
        self.train_columns = None   # store encoded training columns

    # Build ANN model
    def build_model(self, input_dim):
        self.model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.15),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

    # Training
    def train(self, X_train_raw, y_train, validation_split=0.15, epochs=100, batch_size=16):

        # One-hot encode
        X_train = pd.get_dummies(X_train_raw, drop_first=True)

        # Save training columns
        self.train_columns = X_train.columns

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Build ANN if not built
        if self.model is None:
            self.build_model(X_train_scaled.shape[1])

        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            verbose=1
        )

        return history

    # Predict (used for split test)
    def predict(self, X_raw):
        # One-hot encode
        X = pd.get_dummies(X_raw, drop_first=True)

        # Align columns
        X = X.reindex(columns=self.train_columns, fill_value=0)

        # Scale
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled).flatten()

    # Evaluate
    def evaluate(self, y_true, y_pred, title="Dataset"):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        print(f"\n--- {title} Performance ---")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return mae, rmse, r2, mape

    # Plot Actual vs Predicted
    def plot_results(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2)

        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Flight Prices (ANN Model)")
        plt.grid(True)
        plt.show()

    # Predict from NEW FILE & save CSV
    def predict_new_file(self, df_test, output_csv_name="ann_predictions.csv"):

        # Drop Price column if it exists
        X_predict = df_test.drop("Price", axis=1, errors="ignore")

        # One-hot encode
        X_predict_enc = pd.get_dummies(X_predict, drop_first=True)

        # Match columns with training phase
        X_predict_enc = X_predict_enc.reindex(columns=self.train_columns, fill_value=0)

        # Scale
        X_predict_scaled = self.scaler.transform(X_predict_enc)

        # Predict
        predictions = self.model.predict(X_predict_scaled).flatten()

        # Save to CSV
        pd.DataFrame({'Price_Prediction': predictions}).to_csv(output_csv_name, index=False)

        print(f"\nPredictions saved to {output_csv_name}")
        return predictions
