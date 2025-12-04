from preprocess import *

class RandomForestModel:
    def __init__(self, cv=3, scoring='neg_mean_absolute_error'):
        # Define hyperparameter grid inside class
        self.param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        self.cv = cv
        self.scoring = scoring
        self.best_model = None
        self.grid_search = None
        self.train_columns = None   # for encoding alignment

    # Fit model + store training columns (after encoding)
    def train(self, X_train_raw, y_train):
        # One-hot encode training data
        X_train = pd.get_dummies(X_train_raw, drop_first=True)

        # Save training columns so test/new file can match it later
        self.train_columns = X_train.columns

        base_rf = RandomForestRegressor(random_state=42)

        self.grid_search = GridSearchCV(
            base_rf,
            self.param_grid,
            cv=self.cv,
            n_jobs=-1,
            verbose=1,
            scoring=self.scoring,
            error_score="raise"
        )

        self.grid_search.fit(X_train, y_train)
        self.best_model = self.grid_search.best_estimator_

        print("\nBest Random Forest Parameters:", self.grid_search.best_params_)
        return self.best_model

    # Predict (for normal test split)
    def predict(self, X_raw):
        if self.best_model is None:
            raise Exception("Model is not trained yet!")

        X = pd.get_dummies(X_raw, drop_first=True)

        # Align columns to training columns
        X = X.reindex(columns=self.train_columns, fill_value=0)

        return self.best_model.predict(X)

    # Evaluate metrics
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

    # Plot results
    def plot_results(self, y_test, y_pred):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--',
            lw=2
        )
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Flight Prices (Random Forest)")
        plt.grid(True)
        plt.show()

    # Predict new file & save CSV
    def predict_new_file(self, df_test, output_csv_name="rf_predictions.csv"):
        if self.best_model is None:
            raise Exception("Model is not trained yet!")

        # Remove Price column if present
        X_predict = df_test.drop('Price', axis=1, errors='ignore')

        # Encode
        X_predict_enc = pd.get_dummies(X_predict, drop_first=True)

        # Align columns
        X_predict_enc = X_predict_enc.reindex(columns=self.train_columns, fill_value=0)

        # Predict
        predictions = self.best_model.predict(X_predict_enc)

        # Save CSV
        pd.DataFrame({'Price_Prediction': predictions}).to_csv(output_csv_name, index=False)

        print(f"\nPredictions saved to {output_csv_name}")
        return predictions

