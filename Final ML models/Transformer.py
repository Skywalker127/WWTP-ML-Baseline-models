# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 19:54:50 2025

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add,
    GlobalAveragePooling1D
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# Load Excel file
file_path = 'sample data.xlsx'
data = pd.read_excel(file_path, sheet_name=0,engine='openpyxl')

# Assign features and target
features = ['SS_in', 'COD_in', 'NH3-N_in', 'TP_in', 'TN_in', 'PH_in', 
      'NH3-N', 'TP', 'TN', 'PH', 'Outflow', 'Inflow_2', 'Outflow_1', 'Outflow_2']

           
target ='COD'


data.fillna(method='ffill', inplace=True)
if data.isnull().sum().sum() > 0:
    data.fillna(method='bfill', inplace=True)
if data.isnull().sum().sum() > 0:
    data.fillna(0, inplace=True)

data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

# ------------------------------
# Add the sequence constructor to the target history
# ------------------------------
def create_sequences(data, features, target, time_steps, target_delay):
    X_data = data[features].values
    y_data = data[target].values.reshape(-1, 1)

    total_samples = len(data) - time_steps - target_delay
    train_size = int(total_samples * 0.8)
    val_size = int(total_samples * 0.1)
    test_size = total_samples - train_size - val_size


    scaler_x = MinMaxScaler().fit(X_data[:train_size])
    scaler_y = MinMaxScaler().fit(y_data[:train_size])

    X_scaled = scaler_x.transform(X_data)
    y_scaled = scaler_y.transform(y_data)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - time_steps - target_delay):
        feature_seq = X_scaled[i:i+time_steps]
        target_history = y_scaled[i:i+time_steps]
        combined_seq = np.concatenate([feature_seq, target_history], axis=1)

        X_seq.append(combined_seq)
        y_seq.append(y_scaled[i + time_steps + target_delay])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y

# ------------------------------
# Location Coding Layer
# ------------------------------
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len=1000, d_model=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_enc = layers.Embedding(input_dim=max_len, output_dim=d_model)
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_enc(positions)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model
        })
        return config

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x_ff = Dense(ff_dim, activation="gelu")(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ------------------------------
# Transformer Model Construction
# ------------------------------
def build_optimized_model(time_steps, num_features):
    inputs = Input(shape=(time_steps, num_features))
    x = PositionalEncoding(max_len=time_steps, d_model=num_features)(inputs)
    for _ in range(3):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

# ------------------------------
# 模型训练与评估
# ------------------------------

    
def run_delay_experiment():
    time_steps = 2
    results = []

    for target_delay in range(0,1):  # Delay from n to n
        print(f"\n====== Target Delay: {target_delay} ======")

        X_train, y_train, X_val, y_val, X_test, y_test, scaler_y = create_sequences(
            data, features, target, time_steps, target_delay
        )

        model = build_optimized_model(time_steps, len(features) + 1)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=0
        )

        y_pred = model.predict(X_test)
        y_test_orig = scaler_y.inverse_transform(y_test)
        y_pred_orig = scaler_y.inverse_transform(y_pred)

        def calculate_mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100

        r2 = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mape = calculate_mape(y_test_orig, y_pred_orig)

        print(f"R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")

        results.append({
            "delay": target_delay,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

    
    df = pd.DataFrame(results)
    print("\n=== Delay Performance Summary ===")
    print(df)

   
    plt.figure(figsize=(10, 6))
    plt.plot(df["delay"], df["R2"], label='R²', marker='o')
    plt.plot(df["delay"], df["RMSE"], label='RMSE', marker='x')
    plt.xlabel("Target Delay")
    plt.ylabel("Score")
    plt.title("Model Performance vs Target Delay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ------------------------------
# Execution training
# ------------------------------
if __name__ == "__main__":
    run_delay_experiment()