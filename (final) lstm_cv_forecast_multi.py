# 3-fold cross-validation for 1h, 6h, 24h forecasts using TimeSeriesSplit
import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import data_preprocessing as dp

X_raw = dp.final_X
y_raw = dp.final_y.reshape(-1, 1)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

past_steps = 24  # Use best from tuning
learning_rate = 1e-3
horizons = [1, 6, 24]  # Forecast 1h, 6h, 24h

results = {}

for forecast_horizon in horizons:
    print(f"\nRunning CV for {forecast_horizon}h forecast")
    X_seq, y_seq = [], []
    for i in range(past_steps + forecast_horizon - 1, len(X_raw)):
        X_seq.append(X_raw[i - past_steps - forecast_horizon + 1: i - forecast_horizon + 1])
        y_seq.append(y_scaled[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    tscv = TimeSeriesSplit(n_splits=6)  # change fold number
    fold_rmses, fold_maes = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
        X_train, y_train = X_seq[train_idx], y_seq[train_idx]
        X_val, y_val = X_seq[val_idx], y_seq[val_idx]

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        y_val_pred = model.predict(X_val)
        y_val_inv = scaler_y.inverse_transform(y_val)
        y_val_pred_inv = scaler_y.inverse_transform(y_val_pred)

        rmse = np.sqrt(mean_squared_error(y_val_inv, y_val_pred_inv))
        mae = mean_absolute_error(y_val_inv, y_val_pred_inv)

        fold_rmses.append(rmse)
        fold_maes.append(mae)
        print(f"Fold {fold+1}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    results[f"{forecast_horizon}h"] = {
        "avg_rmse": np.mean(fold_rmses),
        "avg_mae": np.mean(fold_maes),
        "last_pred": y_val_pred_inv,
        "last_true": y_val_inv,
        "last_residuals": y_val_inv.flatten() - y_val_pred_inv.flatten()
    }

    # Save outputs for latest fold only
    np.save(f"output/y_pred_{forecast_horizon}h.npy", y_val_pred_inv)
    np.save(f"output/y_test_{forecast_horizon}h.npy", y_val_inv)
    np.save(f"output/residuals_{forecast_horizon}h.npy", y_val_inv.flatten() - y_val_pred_inv.flatten())

    print(f"→ Avg RMSE: {np.mean(fold_rmses):.4f}, Avg MAE: {np.mean(fold_maes):.4f}")

# Save summary
with open("output/cv_summary.txt", "w") as f:
    for key, val in results.items():
        f.write(f"{key}: RMSE={val['avg_rmse']:.4f}, MAE={val['avg_mae']:.4f}\n")
