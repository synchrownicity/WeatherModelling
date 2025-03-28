import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model_path, X_test, y_test, feature_names=None):
    rf = joblib.load(model_path)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.5f}, R²: {r2:.5f}")

    ### Predicted vs Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # ideal prediction line
    plt.xlabel("Actual Relative Humidity")
    plt.ylabel("Predicted Relative Humidity")
    plt.title("Actual vs. Predicted Relative Humidity")
    plt.grid(True)
    plt.show()

    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    residuals = y_test - y_pred

    # Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Relative Humidity")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.grid(True)
    plt.show()

    # Feature importance
    importances = rf.feature_importances_
    # Already in your evaluate_model
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]


    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.grid(True)
    plt.show()
