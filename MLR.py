import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import data_preprocessing as dp
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score


import data_preprocessing as dp
final_X = dp.final_X
final_y = dp.final_y

# Create lagg variables
def create_lagged_features(X, target_idx, N, k):
    """Create lagged features using past N time points before the target index, for forecasting a target after forecast_horizon time points."""
    lagged_features = []
    target_idx_shifted = target_idx + k  # Shift target index by forecast horizon, k

    # Ensure that we don't exceed available time points
    if target_idx_shifted >= len(X):
        return None  # Skip if out of bounds
    
    # Create lag features only if they are within bounds
    for i in range(N):
        lag_idx = target_idx_shifted - (i + 1)
        if lag_idx < 0:
            return None  # Skip if out of bounds
        lagged_features.append(X[lag_idx])  # Use past N values up to shifted target

    return np.array(lagged_features).reshape(1, -1)  # Ensure 2D shape

# Function to find the optimal N using validation set
## can adjust the range of N values 
def find_optimal_N(X_train, y_train, X_val, y_val, min_N=100, max_N=101, k=1):
    """Find the optimal number of past time points (N) for best prediction using a forecast horizon."""
    
    best_N = min_N
    best_mse = float('inf')

    # Precompute lagged features ONCE for all N values
    train_features = {N: [] for N in range(min_N, max_N + 1)}
    val_features = {N: [] for N in range(min_N, max_N + 1)}

    # Store y_lagged values separately to avoid recomputation
    y_train_lagged_dict = {N: [] for N in range(min_N, max_N + 1)}
    y_val_lagged_dict = {N: [] for N in range(min_N, max_N + 1)}

    # Precompute train set lagged features
    for i in range(max_N, len(X_train) - k):
        for N in range(min_N, max_N + 1):
            features = create_lagged_features(X_train, i, N, k)
            if features is not None:
                train_features[N].append(features.flatten())
                y_train_lagged_dict[N].append(y_train[i + k])

    # Precompute validation set lagged features
    for i in range(max_N, len(X_val) - k):
        for N in range(min_N, max_N + 1):
            features = create_lagged_features(X_val, i, N, k)
            if features is not None:
                val_features[N].append(features.flatten())
                y_val_lagged_dict[N].append(y_val[i + k])

    # Iterate over N and fit models efficiently
    for N in range(min_N, max_N + 1):
        if len(train_features[N]) == 0 or len(val_features[N]) == 0:
            continue  # Skip if there aren't enough valid features

        X_train_lagged, y_train_lagged = np.array(train_features[N]), np.array(y_train_lagged_dict[N])
        X_val_lagged, y_val_lagged = np.array(val_features[N]), np.array(y_val_lagged_dict[N])

        model = LinearRegression()
        model.fit(X_train_lagged, y_train_lagged)
        y_pred = model.predict(X_val_lagged)
        mse = mean_squared_error(y_val_lagged, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_N = N

    return best_N



# Function to train the model
def train_model(X_train, y_train, best_N, k):
    """Train MLR model using best N."""
    X_train_lagged, y_train_lagged = [], []
    for i in range(best_N, len(X_train) - k):  # Prevent out-of-bounds
        features = create_lagged_features(X_train, i, best_N, k)
        if features is not None:
            X_train_lagged.append(features.flatten())
            y_train_lagged.append(y_train[i + k])  # predicts y val at point i+k

    X_train_lagged, y_train_lagged = np.array(X_train_lagged), np.array(y_train_lagged)
    model = LinearRegression()
    model.fit(X_train_lagged, y_train_lagged)
    return model

# Function to evaluate the model
def evaluate_model(model, X, y, best_N, k, dataset_name="Test"):
    """Evaluate the trained model on a given dataset."""
    X_lagged, y_lagged = [], []
    for i in range(best_N, len(X) - k):  # Prevent out-of-bounds
        features = create_lagged_features(X, i, best_N, k)
        if features is not None:
            X_lagged.append(features.flatten())
            y_lagged.append(y[i + k])  # Predicting k steps ahead
    
    X_lagged, y_lagged = np.array(X_lagged), np.array(y_lagged)
    y_pred = model.predict(X_lagged)
    mse = mean_squared_error(y_lagged, y_pred)
    print(f"{dataset_name} MSE with N={best_N} for {k}-hour forecast: {mse}")
    return mse, y_pred, y_lagged

<<<<<<< HEAD
# Assuming X and y are already preprocessed
time_split_1 = int(len(dp.final_X) * 0.7)  # 70% train
time_split_2 = int(len(dp.final_X) * 0.8)  # Next 10% validation, last 20% test
X_train, y_train = dp.final_X[:time_split_1], dp.final_y[:time_split_1]
X_val, y_val = dp.final_X[time_split_1:time_split_2], dp.final_y[time_split_1:time_split_2]
X_test, y_test = dp.final_X[time_split_2:], dp.final_y[time_split_2:]
=======

def eval_r2(y_true, y_pred, dataset_name="Test"):
    """Evaluate and print the R² score for a dataset."""
    r2 = r2_score(y_true, y_pred)
    print(f"{dataset_name} R² score: {r2}")
    return r2

# train-validation-test split
time_split_1 = int(len(final_X) * 0.7)  # 70% train
time_split_2 = int(len(final_X) * 0.8)  # Next 10% validation, last 20% test
X_train, y_train = final_X[:time_split_1], final_y[:time_split_1]
X_val, y_val = final_X[time_split_1:time_split_2], final_y[time_split_1:time_split_2]
X_test, y_test = final_X[time_split_2:], final_y[time_split_2:]
>>>>>>> e3768a8b01dd3f3eb596328d567060e680c8b807

# k-values
k1 = 1
k6 = 6
k24 = 24



print(create_lagged_features(X_train, 50000, 1000))
# Finding optimal N for each configuration 
<<<<<<< HEAD
# best_N_1hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=500)
# best_N_6hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=50)
# best_N_24hr = find_optimal_N(X_train, y_train, X_val, y_val, max_N=50)

# Train final models using best N values
# test_model_1hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_1hr, forecast_horizon=forecast_horizon_1hr)
# test_model_6hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_6hr, forecast_horizon=forecast_horizon_6hr)
# test_model_24hr = train_and_evaluate(X_train, y_train, X_test, y_test, best_N_24hr, forecast_horizon=forecast_horizon_24hr)
=======
best_N_1hr = find_optimal_N(X_train, y_train, X_val, y_val, k = k1)
#best_N_6hr = find_optimal_N(X_train, y_train, X_val, y_val, k = k6)
#best_N_24hr = find_optimal_N(X_train, y_train, X_val, y_val, k = k24)

>>>>>>> e3768a8b01dd3f3eb596328d567060e680c8b807


# Train models
model_1hr = train_model(X_train, y_train, best_N_1hr, k1)
#model_6hr = train_model(X_train, y_train, best_N_6hr, k6)
#model_24hr = train_model(X_train, y_train, best_N_24hr, k24)




# MSE
## 1 hour forecast
_, train_pred_1h, train_actual_y1 = evaluate_model(model_1hr, X_train, y_train, best_N_1hr, k1, "Train")
_, val_pred_1h, val_actual_y1 = evaluate_model(model_1hr, X_val, y_val, best_N_1hr, k1, "Validation")
_, test_pred_1h, test_actual_y1 = evaluate_model(model_1hr, X_test, y_test, best_N_1hr, k1, "Test")

## 6 hour forecast
#_, train_pred_6h = evaluate_model(model_6hr, X_train, y_train, best_N_6hr, k6, "Train")
#_, val_pred_6h = evaluate_model(model_6hr, X_val, y_val, best_N_6hr, k6, "Validation")
#_, test_pred_6h = evaluate_model(model_6hr, X_test, y_test, best_N_6hr, k6, "Test")

## 24 hour forecast
#_, train_pred_24h = evaluate_model(model_24hr, X_train, y_train, best_N_24hr, k24, "Train")
#_, val_pred_24h = evaluate_model(model_24hr, X_val, y_val, best_N_24hr, k24, "Validation")
#_, test_pred_24h = evaluate_model(model_24hr, X_test, y_test, best_N_24hr, k24, "Test")


# R2
## 1 hour forecast
eval_r2(train_actual_y1, train_pred_1h, dataset_name="Train")
eval_r2(val_actual_y1, val_pred_1h, dataset_name="Val")
eval_r2(test_actual_y1, test_pred_1h, dataset_name="Test")



""" Current results - 1h forecast
Train MSE with N=50 for 1-hour forecast: 0.0048800727280796635
Validation MSE with N=50 for 1-hour forecast: 0.006093609529610243
Test MSE with N=50 for 1-hour forecast: 0.005394733389488955
Train R² score: 0.7960293090425572
Val R² score: 0.7748829135392314
Test R² score: 0.7715625233141626 """

# Plot predictions vs actual

def MLR_plot(actual_y, pred_y):
    plt.figure(figsize=(10,5))
    plt.plot(actual_y, label='Actual', linestyle='dashed')
    plt.plot(pred_y, label='Predicted')
    plt.legend()
    plt.title("MLR Predictions vs Actual on Test Set")
    plt.show()

# MLR_plot(y_train, train_pred_1h)
# MLR_plot(y_val, val_pred_1h)
# MLR_plot(y_test, test_pred_1h)
# MLR_plot(y_test, test_pred_6h)
# MLR_plot(y_test, test_pred_24h)
