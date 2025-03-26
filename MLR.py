import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ensure data is sorted by time (assuming already sorted in dataset)

def time_based_split(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Splits data into train, validation, and test sets while preserving time order.
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    N = X.shape[0]
    train_end = int(N * train_ratio)
    val_end = train_end + int(N * val_ratio)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(final_X, final_y)

# Train multiple linear regression model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Predictions
y_train_pred = mlr.predict(X_train)
y_val_pred = mlr.predict(X_val)
y_test_pred = mlr.predict(X_test)

# Model evaluation
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
print(f"Validation MSE: {val_mse:.4f}, R2: {val_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

# Plot predictions vs actual

def MLR_plot(actual_y, pred_y):
    plt.figure(figsize=(10,5))
    plt.plot(actual_y, label='Actual', linestyle='dashed')
    plt.plot(pred_y, label='Predicted')
    plt.legend()
    plt.title("MLR Predictions vs Actual on Test Set")
    plt.show()

# MLR_plot(y_test, y_test_pred)
