# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 19:32:46 2025

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load Excel file
file_path = 'sample data.xlsx'
data = pd.read_excel(file_path, sheet_name=0,engine='openpyxl')

# Assign features and target
features = ['SS_in', 'COD_in', 'NH3-N_in', 'TP_in', 'TN_in', 'PH_in', 
      'NH3-N', 'TP', 'TN', 'PH', 'Outflow', 'Inflow_2', 'Outflow_1', 'Outflow_2']

           
target ='COD'


# Time Series Data Generation Function (Including Target Historical Values)
def create_time_series_data(data, time_steps, target_delay, num_features):
    X, y = [], []
    for i in range(len(data) - time_steps - target_delay):
        X.append(data[i:i+time_steps, :])  
        y.append(data[i+time_steps+target_delay, num_features])  
    return np.array(X), np.array(y).reshape(-1, 1)

# main (prediction horizon=target_delay+1)
def train_final_model():
    time_steps = 12
    target_delay = 1

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_end = int(len(data) * train_ratio)
    
    val_end = train_end + int(len(data) * test_ratio)
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
   

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_x.fit(train_data[features])
    scaler_y.fit(train_data[target].values.reshape(-1, 1))

    train_features = scaler_x.transform(train_data[features])
    train_target = scaler_y.transform(train_data[target].values.reshape(-1, 1))
    test_features = scaler_x.transform(test_data[features])
    test_target = scaler_y.transform(test_data[target].values.reshape(-1, 1))
    val_features = scaler_x.transform(val_data[features])
    val_target = scaler_y.transform(val_data[target].values.reshape(-1, 1))

    # historical target into features
    train_combined = np.hstack((train_features, train_target))
    test_combined = np.hstack((test_features, test_target))
    val_combined = np.hstack((val_features, val_target))

    num_features = len(features) + 1

    X_train, y_train = create_time_series_data(train_combined, time_steps, target_delay, len(features))
    X_test, y_test = create_time_series_data(test_combined, time_steps, target_delay, len(features))
    X_val, y_val = create_time_series_data(val_combined, time_steps, target_delay, len(features))


    X_train = X_train.reshape(-1, time_steps, num_features)
    X_test = X_test.reshape(-1, time_steps, num_features)
    X_val = X_val.reshape(-1, time_steps, num_features)

    model = Sequential([
        LSTM(66, return_sequences=False, input_shape=(time_steps, num_features)),
        Dropout(0.3589),
        Dense(1, activation='linear')
    ])

    optimizer = Adam(learning_rate=0.002441)
    model.compile(optimizer=optimizer, loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30, batch_size=32,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    y_val_pred = model.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    print(f"validation R²: {r2_val:.4f}")

    # tarin/validation Loss 
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # prediction and evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_pred_original = scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    def calculate_mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def calculate_rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    mape_test = calculate_mape(y_test_original, y_test_pred_original)
    rmse_test = calculate_rmse(y_test_original, y_test_pred_original)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual', marker='.')
    plt.plot(y_test_pred_original, label='Predicted', linestyle='--', marker='.')
    plt.title(f'Actual vs Predicted COD Values (R²={r2_test:.2f}, MAPE={mape_test:.2f}%, RMSE={rmse_test:.2f})')
    plt.xlabel('Test Sample Index')
    plt.ylabel('COD Value (mg/l)')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"train R²: {r2_train:.4f}")
    print(f"test R²: {r2_test:.4f}, MAPE: {mape_test:.2f}%")
    print(f"test RMSE: {rmse_test:.4f}")

# Execution training
train_final_model()
