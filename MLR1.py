import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import data_preprocessing as dp

final_X = dp.final_X
final_y = dp.final_y

def create_lagged_features(X, N):
    """Creates lagged features using the past N time steps."""
    X_lagged = []
    for i in range(N, len(X)):
        X_lagged.append(X[i-N:i])
    return np.array(X_lagged)

def time_series_split(X, y):
    """Splits time series data into train (70%), validation (10%), and test (20%)."""
    split_1 = int(len(X) * 0.7)
    split_2 = int(len(X) * 0.8)
    
    X_train, y_train = X[:split_1], y[:split_1]
    X_val, y_val = X[split_1:split_2], y[split_1:split_2]
    X_test, y_test = X[split_2:], y[split_2:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(X_train, y_train):
    """Trains a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val, X_test, y_test, forecast_horizon):
    """Evaluates the model using MSE and MAE."""
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"{forecast_horizon}-hour Forecast:")
    print(f"Validation MSE: {val_mse}, MAE: {val_mae}")
    print(f"Test MSE: {test_mse}, MAE: {test_mae}\n")
    
    return val_mse, val_mae, test_mse, test_mae

# Example usage:
N = 100  # Lag value
forecast_horizons = [1]  # Forecasting time steps

# Assume final_X and final_y contain the time series data
X_lagged = create_lagged_features(final_X, N)
y_lagged = final_y[N:]  # Align target variable

X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(X_lagged, y_lagged)

for k in forecast_horizons:
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val, X_test, y_test, k)
