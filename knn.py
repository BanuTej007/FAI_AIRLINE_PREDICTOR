from preprocess import *
from linear_reg import *

class FlightPriceModelKNN(FlightPriceModelLR):
    '''Class to implement KNN model for flight prediction model'''

    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski', p=2, test_size=0.2, random_state=42):

        super().__init__(test_size=test_size, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p

        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)

        self.scaler = StandardScaler()

        self.X_train_scaled = None
        self.X_val_scaled = None

    def train(self, df_train, target_column = 'Price', verbose=True):
        if verbose:
            print("=" * 60)
            print("Starting KNN Model Training Pipeline")
            print("=" * 60)
            print(f"\nKNN Parameters:")
            print(f"  n_neighbors: {self.n_neighbors}")
            print(f"  weights: {self.weights}")
            print(f"  metric: {self.metric}")
            print(f"  p: {self.p}")

        X, y = self._prepare_data(df_train, target_column, verbose)
        self.X_encoded = self._encode_categorical(X, verbose)
        self._split_data(self.X_encoded, y, verbose)
        self.scaleFeatures(verbose)
        self.trainModel(verbose)
        self.makePredictions(verbose)
        self._evaluate_model(verbose)

        if verbose:
            self.display_feature_importance()

            print("\n" + "=" * 60)
            print("Training Complete")
            print("=" * 60)

        return {'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}

    def predict(self, df_test, verbose = True):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        if verbose:
            print("\n" + "=" * 60)
            print("Making Predictions on Test Data")
            print("=" * 60)

        X_test = df_test.drop('Price', axis=1, errors='ignore')
        X_test_encoded = self._encode_test_data(X_test, verbose)
        X_test_scaled = self.scaler.transform(X_test_encoded)
        predictions = self.model.predict(X_test_scaled)

        if verbose:
            print(f"\nGenerated {len(predictions)} predictions")
            print(f"  Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
            print(f"  Mean prediction: {predictions.mean():.2f}")

        return predictions
    def scaleFeatures(self, verbose):
        """Scale features using StandardScaler (critical for KNN)."""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)

        if verbose:
            print(f"\nFeatures scaled using StandardScaler")

    def trainModel(self, verbose):
        """Train KNN model using scaled data."""
        self.model.fit(self.X_train_scaled, self.y_train)

        if verbose:
            print(f"\nKNN Model trained successfully")
            print(f"  Model type: {type(self.model).__name__}")

    def makePredictions(self, verbose):
        """Make predictions using scaled data."""
        self.y_train_pred = self.model.predict(self.X_train_scaled)
        self.y_val_pred = self.model.predict(self.X_val_scaled)

        if verbose:
            print(f"\nPredictions generated")

    def display_feature_importance(self, top_n=10):
        """Display feature importance using permutation method."""
        print("\n" + "=" * 60)
        print(f"Top {top_n} Most Important Features (Permutation Importance)")
        print("=" * 60)

        baseline_score = r2_score(self.y_val, self.y_val_pred)
        importances = []

        for i, _ in enumerate(self.feature_names):
            X_val_permuted = self.X_val_scaled.copy()
            np.random.seed(self.random_state)
            X_val_permuted[:, i] = np.random.permutation(X_val_permuted[:, i])

            permuted_pred = self.model.predict(X_val_permuted)
            permuted_score = r2_score(self.y_val, permuted_pred)
            importances.append(baseline_score - permuted_score)

        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(feature_importance.head(top_n).to_string(index=False))

    def find_optimal_k(self, df_train, target_col='Price', k_range=range(3, 31), verbose=True):
        """Find optimal k using cross-validation."""
        # if verbose:
        print("\n" + "=" * 60)
        print("Finding Optimal K")
        print("=" * 60)

        X, y = self._prepare_data(df_train, target_col, verbose=False)
        X_encoded = self._encode_categorical(X, verbose=False)
        X_scaled = self.scaler.fit_transform(X_encoded)

        cv_scores = []
        for k in k_range:
            knn = KNeighborsRegressor(n_neighbors=k, weights=self.weights)
            scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            cv_scores.append(-scores.mean())

        plt.figure(figsize=(10, 6))
        plt.plot(k_range, cv_scores, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('K Value')
        plt.ylabel('Mean Squared Error (CV)')
        plt.title('KNN: Finding Optimal K')
        plt.grid(True, alpha=0.3)
        plt.show()

        optimal_k = list(k_range)[np.argmin(cv_scores)]

        # if verbose:
        print(f"\nOptimal K: {optimal_k}")
        print(f"Best MSE: {min(cv_scores):.2f}")

        return optimal_k

    def grid_search_tune(self, df_train, target_col='Price', verbose=True):
        """Perform GridSearchCV for hyperparameter tuning."""
        # if verbose:
        print("\n" + "=" * 60)
        print("Grid Search Hyperparameter Tuning")
        print("=" * 60)

        X, y = self._prepare_data(df_train, target_col, verbose=False)
        X_encoded = self._encode_categorical(X, verbose=False)
        X_scaled = self.scaler.fit_transform(X_encoded)

        optimal_k = self.find_optimal_k(df_train, target_col, verbose=True)

        param_grid = {
            'n_neighbors': [max(1, optimal_k - 2), optimal_k, optimal_k + 2],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }

        grid_search = GridSearchCV(
            KNeighborsRegressor(),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            verbose=1 if verbose else 0,
            n_jobs=-1
        )

        grid_search.fit(X_scaled, y)

        # if verbose:
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: {-grid_search.best_score_:.2f}")

        self.model = grid_search.best_estimator_
        self.n_neighbors = grid_search.best_params_['n_neighbors']
        self.weights = grid_search.best_params_['weights']
        self.metric = grid_search.best_params_['metric']
        self.p = grid_search.best_params_['p']

        return grid_search.best_params_, -grid_search.best_score_


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
        axes[0].set_title('Actual vs Predicted Prices (KNN)')
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