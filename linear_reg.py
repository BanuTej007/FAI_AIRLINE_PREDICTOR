from preprocess import *

class FlightPriceModelLR:
    """
    A class to train and evaluate machine learning models for flight price prediction.
    """

    def __init__(self, model=None, test_size=0.2, random_state=42):
        """
        Initializing the model class.
        """
        self.model = model if model is not None else LinearRegression()
        self.test_size = test_size
        self.random_state = random_state

        # Data containers
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_encoded = None

        # Encoding
        self.label_encoders = {}
        self.categorical_cols = []
        self.feature_names = []

        # Predictions
        self.y_train_pred = None
        self.y_val_pred = None

        # Metrics
        self.train_metrics = {}
        self.val_metrics = {}

    def train(self, df_train, target_col='Price', verbose=True):
        """
        Main training pipeline that orchestrates all steps.
        Returns:
        dict
            Dictionary containing training and validation metrics
        """
        if verbose:
            print("-" * 60)
            print("Starting Model Training Pipeline")
            print("-" * 60)

        X, y = self._prepare_data(df_train, target_col, verbose)
        self.X_encoded = self._encode_categorical(X, verbose)
        self._split_data(self.X_encoded, y, verbose)
        self._train_model(verbose)
        self._make_predictions(verbose)
        self._evaluate_model(verbose)

        if verbose:
            self._display_feature_importance()

        if verbose:
            print("\n" + "-" * 60)
            print("Training Complete!")
            print("-" * 60)

        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

    def predict(self, df_test, verbose=True):
        """
        Making predictions on test data.

        Returns:
        np.array
            Array of predictions
        """
        if self.model is None or not hasattr(self.model, 'predict'):
            raise ValueError("Model has not been trained yet. Call train() first.")

        if verbose:
            print("\n" + "-" * 60)
            print("Making Predictions on Test Data")
            print("-" * 60)

        X_test = df_test.drop('Price', axis=1, errors='ignore')
        X_test_encoded = self._encode_test_data(X_test, verbose)

        predictions = self.model.predict(X_test_encoded)

        if verbose:
            print(f"\nGenerated {len(predictions)} predictions")
            print(f"  Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
            print(f"  Mean prediction: {predictions.mean():.2f}")

        return predictions

    def _prepare_data(self, df, target_col, verbose):
        """Separating features and target variable."""
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        if verbose:
            print(f"\nData prepared")
            print(f"  Features shape: {X.shape}")
            print(f"  Target shape: {y.shape}")

        return X, y

    def _encode_categorical(self, X, verbose):
        """Encode categorical variables using Label Encoding."""
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        additional_categorical = ['Dep_Date', 'Dep_Month', 'Arrival_Month', 'Arrival_Day', 'Total_Stops', 'Duration_min']
        self.categorical_cols.extend(additional_categorical)

        if verbose:
            print(f"\nFound {len(self.categorical_cols)} categorical columns")
            print(f"  Columns: {self.categorical_cols}")

        # Create a copy for encoding
        X_encoded = X.copy()

        for col in self.categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.feature_names = X_encoded.columns.tolist()

        if verbose:
            print(f"Encoded categorical variables")

        return X_encoded

    def _encode_test_data(self, X_test, verbose):
        """Encoding test data using fitted encoders from training."""
        X_test_encoded = X_test.copy()

        for col in self.categorical_cols:
            if col in X_test_encoded.columns:
                # Handle unseen categories by assigning -1
                X_test_encoded[col] = X_test_encoded[col].apply(
                    lambda x: self._encode_value(col, x)
                )

        if verbose:
            print(f"Encoded test data using fitted encoders")

        return X_test_encoded

    def _encode_value(self, col, value):
        """Helper method to encode a single value, handling unseen categories."""
        try:
            if str(value) in self.label_encoders[col].classes_:
                return self.label_encoders[col].transform([str(value)])[0]
            else:
                return -1
        except:
            return -1

    def _split_data(self, X, y, verbose):
        """Splitting data into training and validation sets."""
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        if verbose:
            print(f"\nData split into train/validation")
            print(f"  Training set: {self.X_train.shape}")
            print(f"  Validation set: {self.X_val.shape}")

    def _train_model(self, verbose):
        """Train the machine learning model."""
        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print(f"\nModel trained successfully")
            print(f"  Model type: {type(self.model).__name__}")

    def _make_predictions(self, verbose):
        """Make predictions on train and validation sets."""
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_val_pred = self.model.predict(self.X_val)

        if verbose:
            print(f"\nPredictions generated")

    def _evaluate_model(self, verbose):
      """Evaluating model performance on train and validation sets."""
      # Training metrics
      self.train_metrics = {
          'r2': r2_score(self.y_train, self.y_train_pred),
          'rmse': np.sqrt(mean_squared_error(self.y_train, self.y_train_pred)),
          'mae': mean_absolute_error(self.y_train, self.y_train_pred)
      }

      # Validation metrics
      self.val_metrics = {
          'r2': r2_score(self.y_val, self.y_val_pred),
          'rmse': np.sqrt(mean_squared_error(self.y_val, self.y_val_pred)),
          'mae': mean_absolute_error(self.y_val, self.y_val_pred)
      }

      if verbose:
          self._print_metrics()

    def _print_metrics(self):
        """Printing formatted metrics."""
        print("\n" + "-" * 60)
        print("Model Performance")
        print("-" * 60)

        print("\nTraining Set:")
        print(f"  R² Score: {self.train_metrics['r2']:.4f}")
        print(f"  RMSE: {self.train_metrics['rmse']:.2f}")
        print(f"  MAE: {self.train_metrics['mae']:.2f}")

        print("\nValidation Set:")
        print(f"  R² Score: {self.val_metrics['r2']:.4f}")
        print(f"  RMSE: {self.val_metrics['rmse']:.2f}")
        print(f"  MAE: {self.val_metrics['mae']:.2f}")

        # Checking for overfitting
        r2_diff = abs(self.train_metrics['r2'] - self.val_metrics['r2'])
        if r2_diff > 0.1:
            print(f"\nWarning: Possible overfitting detected (R² difference: {r2_diff:.4f})")

    def _display_feature_importance(self, top_n=10):
        """Displaying feature importance (coefficients for linear models)."""
        if hasattr(self.model, 'coef_'):
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)

            print("\n" + "=" * 60)
            print(f"Top {top_n} Most Important Features")
            print("=" * 60)
            print(feature_importance.head(top_n).to_string(index=False))

    def get_metrics(self):
        """Returning training and validation metrics."""
        return {
            'train': self.train_metrics,
            'validation': self.val_metrics
        }

    def get_model(self):
        """Returning the trained model."""
        return self.model

    def save_predictions(self, predictions, output_file='test_predictions.csv'):
        """
        Saving predictions to a CSV file.
        """
        pd.DataFrame({'Price_Prediction': predictions}).to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")

    def plot_results(self, dataset='validation'):
      y_true = self.y_val if dataset == 'validation' else self.y_train
      y_pred = self.y_val_pred if dataset == 'validation' else self.y_train_pred
      residuals = y_true - y_pred

      fig, axes = plt.subplots(1, 2, figsize=(14, 6))

      # Plot 1: Actual vs Predicted
      axes[0].scatter(y_true, y_pred, alpha=0.5, s=30)
      axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
      axes[0].set_xlabel('Actual Price (INR)')
      axes[0].set_ylabel('Predicted Price (INR)')
      axes[0].set_title('Actual vs Predicted Prices (Linear Regression)')
      axes[0].grid(True, alpha=0.3)

      # Plot 2: Residuals vs Predicted
      axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
      axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
      axes[1].set_xlabel('Predicted Price (INR)')
      axes[1].set_ylabel('Residuals (INR)')
      axes[1].set_title('Residual Plot')
      axes[1].grid(True, alpha=0.3)

      plt.tight_layout()
      plt.show()