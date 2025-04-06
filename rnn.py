import data_preprocessing as dp
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, SimpleRNN, Input
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json

print("TensorFlow version:", tf.__version__)
print("Available devices:", tf.config.list_physical_devices())

# Optional: Force GPU usage if available
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

if physical_devices:
    print("Using device:", physical_devices[0])
else:
    print("No GPU detected.")

# Number of data points in train-validation and test sets
data_N = len(dp.final_y); print(f"\nTotal no. of datapoints: {data_N}")
train_val_N = int(data_N*0.8); print(f"80% datapoints in (train + val) sets: {train_val_N}")
val_N = int(data_N*0.1); print(f"10% datapoints in validatate set: {val_N}")
test_N = int(data_N*0.2); print(f"20% datapoints in test set: {test_N}\n")

# Train-Validation, test sets
train_val_X = dp.final_X[:train_val_N, :]
train_val_y = dp.final_y[:train_val_N]
test_X = dp.final_X[train_val_N:, :]
test_y = dp.final_y[train_val_N:]

# Check ending indices of Train-Validation; starting indices of Test set
def check_idx(row_to_check, reference_arr):
    arr_match = np.all(reference_arr == row_to_check, axis=1)
    idx = np.where(arr_match)[0][0]
    return idx
last_row_in_train_val_X = train_val_X[-1, :]; print(f"Last row index in (train + val) set: {check_idx(last_row_in_train_val_X, dp.final_X)}")
first_row_in_test_X = test_X[0, :]; print(f"First row index in test set: {check_idx(first_row_in_test_X, dp.final_X)}\n")

# Hyperparameters to be tuned
N = [12, 24, 168, 720] # half-daily, daily, weekly, monthly, yearly
delays = [1, 6, 24]
base_lr = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
seq_len = [] # depends on chosen N as it has to be <= N

# Get the max split (ie minimum Train size in starting fold) given train_val size (F), gap (G) and size of validation set (V)
def get_TSS_max_split(G, F=train_val_N, V=val_N):
    max_split = (F-2*G-V)//V # Floor division to round down to nearest int
    return max_split
# 12_0_model_weights
#regex
def train_model(N, delays, train_val_X=train_val_X, train_val_y=train_val_y, test_X=test_X, test_y=test_y):
    for delay in delays:
        # In each delay
        for n in N:
            # For each N parameters
            gap = n + delay
            # Split dataset into train and val
            max_split = get_TSS_max_split(G=gap)
            tss = TimeSeriesSplit(n_splits=max_split, test_size=9774, gap=gap)

            for idx, (train_index, val_index) in enumerate(tss.split(train_val_X)):
                # In each fold
                X_train, X_val = train_val_X[train_index, :], train_val_X[val_index,:]
                y_train, y_val = train_val_y[train_index], train_val_y[val_index]

                #training
                # create rolling window for prediction
                train_size = len(X_train)
                X_train_window = X_train[:train_size-delay, :]
                y_train_window = y_train[delay:]
                train_generator = TimeseriesGenerator(X_train_window, y_train_window, length=n, batch_size=24)

                #validate
                # create rolling window for prediction
                val_size = len(X_val)
                X_val_window = X_val[:val_size-delay, :]
                y_val_window = y_val[delay:]
                val_generator = TimeseriesGenerator(X_val_window, y_val_window, length=n, batch_size=24)

                #test
                # create rolling window for prediction
                test_size = len(test_X)
                X_test_window = test_X[:test_size-delay, :]
                y_test_window = test_y[delay:]
                test_generator = TimeseriesGenerator(X_test_window, y_test_window, length=n, batch_size=24)

                # fit training and validation data into model
                model = Sequential([
                    Input(shape=(n, 6)),
                    SimpleRNN(32),
                    Dense(1)
                    ])
                model.compile(optimizer='adam',loss='mse', 
                                    metrics=[tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
                history = model.fit(train_generator, validation_data=val_generator, epochs=1)

                # test model on test set
                pred = model.predict(test_generator)
                y_true = y_test_window[n:]
                test_rmse = np.sqrt(mean_squared_error(y_true, pred))
                test_mae = mean_absolute_error(y_true, pred)

                # Add the test metrics to the history dictionary
                history.history['test_root_mean_squared_error'] = test_rmse
                history.history['test_mean_absolute_error'] = test_mae

                # # 1. Saving the model weights
                # model_train.save_weights(f"model_weights{idx}.h5")  # Save weights only

                # # 2. Saving the full model (architecture + weights)
                model.save(f"Full model_N{n}_delay{delay}_{idx}th fold.h5")  # Saves the whole model including architecture and weights

                # 3. Saving the training, validation, test history for each fold (RMSE, MAE, etc.)
                with open(f"Training history_N{n}_delay{delay}_{idx}th fold.json", "w") as f:
                    json.dump(history.history, f)


train_model(N=N, delays=delays)

# def model_best_N(X, y, N, val_N = val_N):
#     train_rmse = []
#     val_rmse = []

#     for n in N:
#         X_seq, y_seq = create_sequences(X, y, n)
#         # X = np.expand_dims(X, axis=1)  # Add channel dim for RNN (samples, timesteps, features)
#         max_fold = int((len(y)-val_N)/val_N)

#         # TimeSeriesSplit with gap = N
#         tscv = TimeSeriesSplit(n_splits=max_fold, gap=n+1)
        
#         fold_train_rmse = []
#         fold_val_rmse = []
        
#         for train_idx, val_idx in tscv.split(X):
#             # Split data
#             X_train, X_val = X_seq[train_idx,:], X_seq[val_idx,:]
#             y_train, y_val = y_seq[train_idx], y_seq[val_idx]
#             print(np.shape(X_train))
            
#             # Vanilla RNN model, start with 1 hidden layer
#             # model = tf.keras.Sequential([
#             #     tf.keras.layers.SimpleRNN(32, input_shape=(n, 6)),
#             #     tf.keras.layers.Dense(1)
#             # ])
            
#             # model.compile(optimizer='adam', loss='mse')
#             # # start train
#             # model.fit(X_train, y_train, epochs=10, verbose=0)

#             # # Predictions
#             # train_pred = model.predict(X_train).flatten()
#             # val_pred = model.predict(X_val).flatten()
            
#             # # Calculate RMSE
#             # fold_train_rmse.append(np.sqrt(mean_squared_error(y_train, train_pred)))
#             # fold_val_rmse.append(np.sqrt(mean_squared_error(y_val, val_pred)))
        
#         # Average RMSE across folds
#         train_rmse.append(np.mean(fold_train_rmse))
#         val_rmse.append(np.mean(fold_val_rmse))
#     return train_rmse, val_rmse

# model_best_N()
# rmse_train_N, rmse_val_N = model_best_N(train_val_X, train_val_y, N)