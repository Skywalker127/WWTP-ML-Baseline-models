# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:28:56 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm  
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)  

# Font settings
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  

# Load Excel file
file_path = 'sample data.xlsx'
data = pd.read_excel(file_path, sheet_name=0,engine='openpyxl')


# ensure id and time series exist
if 'id' not in data.columns or 'COD' not in data.columns:
    raise ValueError("Please check the time series")

data = data.sort_values(by='id')

target_col = 'COD' 

target_delay = 1  # Prediction horizon

if target_delay > 0:
    data[target_col] = data[target_col].shift(-target_delay)

data = data.dropna(subset=[target_col])

# time series extraction
time_series = data[target_col].dropna()
print(f"Time series length: {len(time_series)}")

# ADF test
def adf_test(series):
    result = adfuller(series)
    print("ADF statistical measure: {:.4f}".format(result[0]))
    print("p value: {:.4f}".format(result[1]))
    print("Key value:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    return result[1]  

# Verify whether a differential is required
d = 0
p_value = adf_test(time_series)
if p_value > 0.05:
    d = 1  
    time_series = time_series.diff().dropna()
    print("The data is non-stationary; perform first-order differencing...")
    adf_test(time_series)
else:
    print("The data has stabilised; no differential is required.")

train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]

# Plot ACF and PACF graphs to aid in determining p and q.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(train, ax=axes[0])
axes[0].set_title("ACF")
plot_pacf(train, ax=axes[1])
axes[1].set_title("PACF")
plt.show()

# Selecting appropriate p and q values (based on ACF/PACF)
p = 1  # Selecting the PACF cutoff point
q = 0  # Selecting the ACF cutoff point
order = (p, d, q)

# Rolling window prediction on the test set
predictions = []
history = list(train)
update_interval = 1  

print("Begin scrolling forecast...")
predictions = []
history = list(train)  
update_interval = 1  

# **Ensure the first model training**
model = ARIMA(history, order=order)
model_fit = model.fit()

for t in tqdm(range(len(test)), desc="ARIMA Forecast progress", unit="step"):
    # **If the update interval is reached, retrain the model.**
    if t % update_interval == 0:  
        model = ARIMA(history, order=order)
        model_fit = model.fit()

    # **Make predictions using the latest model**
    forecast = model_fit.forecast(steps=1)[0]
    predictions.append(forecast)

    # **Update historical data**
    # **Use the actual value from the time step before `target_delay`, rather than the current value at time `t`.**
    if t >= target_delay:
        history.append(test.iloc[t - target_delay])  # Update history with past actual values
    else:
        history.append(forecast)  # In the initial stages, when there is insufficient historical data, use predicted values to fill in the gaps.


mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test, predictions)
mape = np.mean(np.abs((test - predictions) / test)) * 100

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')
print(f'MAPE: {mape:.2f}%')

# Visualisation results
plt.figure(figsize=(10,5))
plt.plot(train.index, train, label='training data', color='blue')
plt.plot(test.index, test, label='Actual value', color='black')
plt.plot(test.index, predictions, label='ARIMA Predicted value', color='red', linestyle='dashed')
plt.legend()
plt.xlabel('Time step')
plt.ylabel(target_col)
plt.title(f'ARIMA prediction {target_delay}h {target_col}')
plt.show()

