import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import data_preprocessing as dp
from sklearn.model_selection import TimeSeriesSplit

# Ensure data is sorted by time (assuming already sorted in dataset)

# Function to create lag variables
def create_lagged_features(X, target_idx, N, forecast_horizon=1):
    """Create lagged features using past N time points before the target index, for forecasting a target after forecast_horizon time points."""
    lagged_features = []
    target_idx_shifted = target_idx + forecast_horizon  # Shift target index by forecast horizon

    # Ensure that we don't exceed available time points
    if target_idx_shifted >= len(X):
        raise IndexError(f"Target index {target_idx_shifted} exceeds the available data range.")
    
    # Create lag features only if they are within bounds
    for i in range(N):
        lag_idx = target_idx_shifted - (i + 1)
        if lag_idx < 0:
            raise IndexError(f"Lag index {lag_idx} is out of bounds for the available data.")
        lagged_features.append(X[lag_idx])  # Use past N values up to shifted target

    return np.hstack(lagged_features)  # Stack them as features

# Function to find the optimal N using validation set
def find_optimal_N(X_train, y_train, X_val, y_val, max_N=500):
    """Find the optimal number of past time points (N) for best prediction."""
    best_N = 1
    best_mse = float('inf')
    for N in range(1, max_N + 1):
        X_train_lagged = np.array([create_lagged_features(X_train, i, N) for i in range(N, len(X_train))])
        y_train_lagged = y_train[N:]
        X_val_lagged = np.array([create_lagged_features(X_val, i, N) for i in range(N, len(X_val))])
        y_val_lagged = y_val[N:]

        if X_train_lagged.shape[0] == 0 or X_val_lagged.shape[0] == 0:
            continue

        model = LinearRegression()
        model.fit(X_train_lagged, y_train_lagged)
        y_pred = model.predict(X_val_lagged)
        mse = mean_squared_error(y_val_lagged, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_N = N
    
    return best_N

# Function to train and evaluate the final model
def train_and_evaluate(X_train, y_train, X_test, y_test, best_N, forecast_horizon=1):
    """Train MLR model using best N and evaluate it on test set."""
    X_train_lagged = []
    y_train_lagged = []
    for i in range(best_N, len(X_train)):  # Start from best_N to ensure we have enough lag data
        try:
            X_train_lagged.append(create_lagged_features(X_train, i, best_N, forecast_horizon))
            y_train_lagged.append(y_train[i])
        except IndexError:
            continue

    X_train_lagged = np.array(X_train_lagged)
    y_train_lagged = np.array(y_train_lagged)

    X_test_lagged = []
    y_test_lagged = []
    for i in range(best_N, len(X_test)):
        try:
            X_test_lagged.append(create_lagged_features(X_test, i, best_N, forecast_horizon))
            y_test_lagged.append(y_test[i])
        except IndexError:
            continue

    X_test_lagged = np.array(X_test_lagged)
    y_test_lagged = np.array(y_test_lagged)

    model = LinearRegression()
    model.fit(X_train_lagged, y_train_lagged)
    y_pred = model.predict(X_test_lagged)
    test_mse = mean_squared_error(y_test_lagged, y_pred)

    print(f"Test MSE with N={best_N} for {forecast_horizon}-hour forecast: {test_mse}")
    return model, y_pred

# Assuming X and y are already preprocessed
time_split_1 = int(len(dp.final_X) * 0.7)  # 70% train
time_split_2 = int(len(dp.final_X) * 0.8)  # Next 10% validation, last 20% test
X_train, y_train = dp.final_X[:time_split_1], dp.final_y[:time_split_1]
X_val, y_val = dp.final_X[time_split_1:time_split_2], dp.final_y[time_split_1:time_split_2]
X_test, y_test = dp.final_X[time_split_2:], dp.final_y[time_split_2:]


forecast_horizon_1hr = 1
forecast_horizon_6hr = 6
forecast_horizon_24hr = 24


print(create_lagged_features(X_train, 50000, 1000))
# Finding optimal N for each configuration 
# best_N_1hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=500)
# best_N_6hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=50)
# best_N_24hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=50)

# Train final models using best N values
# test_model_1hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_1hr, forecast_horizon=forecast_horizon_1hr)
# test_model_6hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_6hr, forecast_horizon=forecast_horizon_6hr)
# test_model_24hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_24hr, forecast_horizon=forecast_horizon_24hr)


# Plot predictions vs actual

def MLR_plot(actual_y, pred_y):
    plt.figure(figsize=(10,5))
    plt.plot(actual_y, label='Actual', linestyle='dashed')
    plt.plot(pred_y, label='Predicted')
    plt.legend()
    plt.title("MLR Predictions vs Actual on Test Set")
    plt.show()

# MLR_plot(y_test, y_pred)
