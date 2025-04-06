# LSTM training with hyperparameter sweep (past_steps and learning_rate)
import numpy as np
import pickle
import os
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import data_preprocessing as dp

# Load and scale data
X_raw = dp.final_X
y_raw = dp.final_y.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

# Parameters to sweep
past_steps_list = [24, 168, 720]         # Daily, weekly, monthly
learning_rates = [1e-3, 1e-4]            # Reasonable range
forecast_horizon = 1

# Group-based train/test split
group_size = 100
num_groups = len(X_raw) // group_size
indices = np.arange(num_groups)
np.random.seed(42)
np.random.shuffle(indices)
train_groups = indices[:int(0.8 * num_groups)]
test_groups = indices[int(0.8 * num_groups):]

train_idx = np.concatenate([np.arange(g * group_size, (g + 1) * group_size) for g in train_groups])
test_idx = np.concatenate([np.arange(g * group_size, (g + 1) * group_size) for g in test_groups])

print("Line 37 done")

# Sequence creation
def create_sequences(X, y, past_steps, forecast_horizon, valid_indices):
    X_seq, y_seq = [], []
    for i in valid_indices:
        if i - past_steps - forecast_horizon + 1 < 0 or i >= len(X):
            continue
        X_seq.append(X[i - forecast_horizon - past_steps + 1 : i - forecast_horizon + 1])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

best_score = float("inf")
best_model = None

print("Line 57 done, starting hyperparameter sweep")

# Sweep through hyperparameters
for past_steps, lr in product(past_steps_list, learning_rates):
    print(f"Training with past_steps={past_steps}, learning_rate={lr}")

    X_train, y_train = create_sequences(X_raw, y_scaled, past_steps, forecast_horizon, train_idx)
    X_test, y_test = create_sequences(X_raw, y_scaled, past_steps, forecast_horizon, test_idx)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    print("Model creation done, starting fitting...")

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )

    print("Fitting done, starting prediction...")

    y_pred = model.predict(X_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}\n")

    if rmse < best_score:
        best_score = rmse
        best_model = model
        best_hparams = (past_steps, lr)
        best_y_pred_inv = y_pred_inv
        best_y_test_inv = y_test_inv

# Save best model and results
os.makedirs("output", exist_ok=True)
np.save("output/y_pred.npy", best_y_pred_inv)
np.save("output/y_test.npy", best_y_test_inv)
np.save("output/residuals.npy", best_y_test_inv.flatten() - best_y_pred_inv.flatten())

with open("output/scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y, f)

best_model.save("output/lstm_model.keras")

with open("output/best_hyperparams.txt", "w") as f:
    f.write(f"Best Past Steps: {best_hparams[0]}\n")
    f.write(f"Best Learning Rate: {best_hparams[1]}\n")
    f.write(f"Best RMSE: {best_score:.4f}\n")

print("Hyperparameter sweep complete. Best model and outputs saved.")
