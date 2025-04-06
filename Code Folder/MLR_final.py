import data_preprocessing as dp
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json 
import joblib


# Number of data points in train-validation and test sets
data_N = len(dp.final_y); print(f"\nTotal no. of datapoints: {data_N}")
train_val_N = int(data_N*0.8); print(f"80% datapoints in (train + val) sets: {train_val_N}")
val_N = int(data_N*0.1); print(f"10% datapoints in validation set: {val_N}")
test_N = int(data_N*0.2); print(f"20% datapoints in test set: {test_N}\n")

# Train-Validation, test sets
train_val_X = dp.final_X[:train_val_N, :]
train_val_y = dp.final_y[:train_val_N]
test_X = dp.final_X[train_val_N:, :]
test_y = dp.final_y[train_val_N:]

# Hyperparameters to be tuned
N = [24, 168, 720]  # half-daily, daily, weekly, monthly, yearly
delays = [1, 6, 24]

# Get the max split (i.e., minimum Train size in starting fold) given train_val size (F), gap (G), and size of validation set (V)
def get_TSS_max_split(G, F=train_val_N, V=val_N):
    max_split = (F - 2*G - V) // V  # Floor division to round down to nearest int
    return max_split



# Updated model function for Linear Regression
# Updated model function for Linear Regression

import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def train_model(N, delays, train_val_X, train_val_y, test_X, test_y):
    for delay in delays:
        for n in N:
            gap = n + delay
            max_split = get_TSS_max_split(G=gap)
            tss = TimeSeriesSplit(n_splits=max_split, test_size=9774, gap=gap)


            for idx, (train_index, val_index) in enumerate(tss.split(train_val_X)):
                # In each fold
                X_train, X_val = train_val_X[train_index, :], train_val_X[val_index, :]
                y_train, y_val = train_val_y[train_index], train_val_y[val_index]


                # Training
                train_size = len(X_train)
                X_train_window = X_train[:train_size - gap, :]
                y_train_window = y_train[gap:]

                # Validation
                val_size = len(X_val)
                X_val_window = X_val[:val_size - gap, :]
                y_val_window = y_val[gap:]

                # Test
                test_size = len(test_X)
                X_test_window = test_X[:test_size - gap, :]
                y_test_window = test_y[gap:]

                # Initialize the Linear Regression model from scikit-learn
                model = LinearRegression()

                # Fit training data
                model.fit(X_train_window, y_train_window)

                # Validate on validation set
                val_predictions = model.predict(X_val_window)
                val_rmse = np.sqrt(mean_squared_error(y_val_window, val_predictions))
                val_mae = mean_absolute_error(y_val_window, val_predictions)

                # Test on test set
                test_predictions = model.predict(X_test_window)
                test_rmse = np.sqrt(mean_squared_error(y_test_window, test_predictions))
                test_mae = mean_absolute_error(y_test_window, test_predictions)

                # Save the test metrics to the history dictionary
                history = {
                    'val_root_mean_squared_error': val_rmse,
                    'val_mean_absolute_error': val_mae,
                    'test_root_mean_squared_error': test_rmse,
                    'test_mean_absolute_error': test_mae
                }

                # Save the training, validation, and test history for each fold (RMSE, MAE, etc.)
                filename = f"Training_history_N_{n}_delay_{delay}_{idx}th_fold.json"
                joblib.dump(model, 'mlr_model.joblib')  # Saves the whole model including architecture and weights

                with open(filename, "w") as f:
                    json.dump(history, f)

train_model(N, delays, train_val_X, train_val_y, test_X, test_y)



import json
import matplotlib.pyplot as plt
import numpy as np

def averager(paths):
    main_train_rmse = []
    main_val_rmse = []
    main_train_mae = []
    main_val_mae = []

    for N in paths:
        files = paths[N]
        train_rmse_lst = []
        train_mae_lst = []
        val_rmse_lst = []
        val_mae_lst = []
        for file in files:
            with open(file, "r") as f:
                history = json.load(f)
                # Accessing root mean squared error and mean absolute error
                train_rmse = history["test_root_mean_squared_error"]  # Assuming these keys exist
                val_rmse = history['val_root_mean_squared_error']
                train_mae = history["test_mean_absolute_error"]
                val_mae = history["val_mean_absolute_error"]

                # Append the metrics for this fold
                train_rmse_lst.append(train_rmse)
                train_mae_lst.append(train_mae)
                val_rmse_lst.append(val_rmse)
                val_mae_lst.append(val_mae)
            
        # Averaging metrics across folds for this N value
        main_train_rmse.append(np.mean(train_rmse_lst))
        main_train_mae.append(np.mean(train_mae_lst))
        main_val_rmse.append(np.mean(val_rmse_lst))
        main_val_mae.append(np.mean(val_mae_lst))
    
    return main_train_rmse, main_train_mae, main_val_rmse, main_val_mae


# Update the paths dictionary to use your new file naming convention
paths = {
    'N24': [
        'Training_history_N_24_delay_1_0th_fold.json',
        'Training_history_N_24_delay_1_1th_fold.json',
        'Training_history_N_24_delay_1_2th_fold.json',
        'Training_history_N_24_delay_1_3th_fold.json'
    ],
    'N168': [
        'Training_history_N_168_delay_1_0th_fold.json',
        'Training_history_N_168_delay_1_1th_fold.json',
        'Training_history_N_168_delay_1_2th_fold.json',
        'Training_history_N_168_delay_1_3th_fold.json'
    ],
    'N720': [
        'Training_history_N_720_delay_1_0th_fold.json',
        'Training_history_N_720_delay_1_1th_fold.json',
        'Training_history_N_720_delay_1_2th_fold.json',
        'Training_history_N_720_delay_1_3th_fold.json'
    ]
}

# Get the averaged results
train_rmse, train_mae, val_rmse, val_mae = averager(paths)

# Plot the results
N = [24, 168, 720]
plt.plot(N, train_rmse, label='Train RMSE')
plt.plot(N, val_rmse, label='Val RMSE')
plt.plot(N, train_mae, label='Train MAE')
plt.plot(N, val_mae, label='Val MAE')

# Add labels and title
plt.xlabel('N')
plt.ylabel('MAE / RMSE')
plt.title('Plot of MAE and RMSE vs N')

# Show the plot
plt.legend()
plt.show()


mlr_model = joblib.load('mlr_model.joblib')
predictions = mlr_model.predict(test_X)

rmse = np.sqrt(mean_squared_error(test_y, predictions))
print("Root Mean Squared Error:", rmse)

mae = mean_absolute_error(test_y, predictions)
print("Mean Absolute Error:", mae)

# Plot actual vs predicted humidity values
plt.figure(figsize=(10, 6))
plt.plot(test_y, label='Actual Values', color='blue')
plt.plot(predictions, label='Predicted Values', color='orange')
plt.xlabel('Time Points')
plt.ylabel('Relative Humidity')
plt.title('Actual vs Predicted Relative Humidity Values')
plt.legend()
plt.show()




