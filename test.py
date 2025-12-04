from preprocess import *
from linear_reg import *
from knn import *
from random_forest import *
from ann import *

print("Linear Regression")
lr_model = FlightPriceModelLR(model=LinearRegression())
lr_metrics = lr_model.train(df_train, verbose=True)

# Make predictions on test data
test_predictions = lr_model.predict(df_test, verbose=True)
print(test_predictions)
lr_model.save_predictions(test_predictions, 'linear_regression_predictions.csv')

if 'Price' in df_test.columns:

    y_test = df_test['Price']

    # Calculate metrics
    test_r2 = r2_score(y_test, test_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100

    # Print metrics
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION - TEST SET PERFORMANCE")
    print("=" * 60)
    print(f"R² Score:  {test_r2:.4f}")
    print(f"MAE:       {test_mae:.2f}")
    print(f"RMSE:      {test_rmse:.2f}")
    print(f"MAPE:      {test_mape:.2f}%")
    print("=" * 60)

print("\nK Nearest Neighbor\n")
model_knn = FlightPriceModelKNN(n_neighbors=5, weights='distance')
optimal_k = model_knn.find_optimal_k(df_train)
knn_metrics = model_knn.train(df_train, target_column='Price', verbose=True)
knn_predictions = model_knn.predict(df_test)

# Plots
model_knn.plot_results('validation')
lr_model.plot_results('validation')

X = df_train.drop("Price", axis=1)
y = df_train["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestModel()

# Train
rf_model.train(X_train, y_train)

# Predict
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_model.evaluate(y_train, y_train_pred_rf, "Train Set")
rf_model.evaluate(y_test, y_test_pred_rf, "Test Set")

# Predict for new file and save CSV
rf_model.predict_new_file(df_test, "rf_predictions.csv")

rf_model.plot_results(y_test, y_test_pred_rf)

ann_model = ANNModel()

ann_model.train(X_train, y_train)

# Predict on split test
y_train_pred_ann = ann_model.predict(X_train)
y_test_pred_ann = ann_model.predict(X_test)

print("\nArtificial Neural Networks")
ann_model.evaluate(y_train, y_train_pred_ann, "Train Set")
ann_model.evaluate(y_test, y_test_pred_ann, "Test Set")

# Predict new file + save CSV
ann_model.predict_new_file(df_test, "ann_predictions.csv")

ann_model.plot_results(y_test, y_test_pred_ann)