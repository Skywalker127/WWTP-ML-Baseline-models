# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:45:25 2025

@author: Administrator
"""
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Load Excel file
file_path = 'sample data.xlsx'
data = pd.read_excel(file_path, sheet_name=0,engine='openpyxl')

# Assign features and target
features = ['SS_in', 'COD_in', 'NH3-N_in', 'TP_in', 'TN_in', 'PH_in', 
      'NH3-N', 'TP', 'TN', 'PH', 'Outflow', 'Inflow_2', 'Outflow_1', 'Outflow_2']

           
target ='COD'

def create_time_series_data(data, time_steps, target_delay, num_features):
    X, y = [], []
    for i in range(len(data) - time_steps - target_delay):
        X.append(data[i:i+time_steps, :num_features].flatten())
        y.append(data[i+time_steps+target_delay, num_features])
    return np.array(X), np.array(y).reshape(-1, 1)

def train_xgboost_model():
    time_steps = 5
    target_delay = 0 #prediction horizon=target delay+1

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:train_size+val_size]
    test_data = data.iloc[train_size+val_size:]

    # Normalisation
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

    train_combined = np.hstack((train_features, train_target))
    test_combined = np.hstack((test_features, test_target))
    val_combined = np.hstack((val_features, val_target))

    num_features = len(features)
    X_train, y_train = create_time_series_data(train_combined, time_steps, target_delay, num_features)
    X_test, y_test = create_time_series_data(test_combined, time_steps, target_delay, num_features)
    X_val, y_val = create_time_series_data(val_combined, time_steps, target_delay, num_features)

    # train
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha = 0.1,  # L1 regularisation coefficient
        reg_lambda = 0.5 # L2 regularisation coefficient
    )
    model.set_params(eval_metric='rmse')

    model.fit(
        X_train, y_train.ravel(),
        eval_set=[(X_train, y_train.ravel()), (X_val, y_val.ravel())],
        verbose=False
    )

    #  evals_result
    results = model.evals_result()
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']

    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('XGBoost Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    # R²
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(f"train R²: {r2_train:.4f}")
    print(f"validate R²: {r2_val:.4f}")
    print(f"test R²: {r2_test:.4f}")

    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    rmse_original = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual', marker='.')
    plt.plot(y_pred_original, label='Predicted', linestyle='--', marker='.')
    plt.title(f'Actual vs Predicted TN Values (R²={r2_test:.2f}, MAPE={mape:.2f}%, RMSE={rmse_original:.2f})')
    plt.xlabel('Test Sample Index')
    plt.ylabel('TN Value (mg/l)')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"final model R²: {r2_test:.4f}, MAPE: {mape:.2f}%, RMSE: {rmse_original:.4f}")

train_xgboost_model()
