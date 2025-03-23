import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data_preprocessing as dp

# Assuming final_df is already created
X = dp.final_X
Y = dp.final_y.reshape(-1, 1)  # from your earlier preprocessing

# Normalize features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Normalize target
scaler_Y = MinMaxScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# Create sequences for LSTM input
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  # past 24 hours
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, time_steps)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1)

# Predict and inverse transform
Y_pred = model.predict(X_test)
Y_pred_inv = scaler_Y.inverse_transform(Y_pred)
Y_test_inv = scaler_Y.inverse_transform(Y_test)

# Plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(Y_test_inv, label="Actual")
plt.plot(Y_pred_inv, label="Predicted")
plt.title("LSTM Forecast - Relative Humidity")
plt.xlabel("Time steps")
plt.ylabel("Relative Humidity")
plt.legend()
plt.show()