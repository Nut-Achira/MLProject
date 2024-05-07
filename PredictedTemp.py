#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:29:09 2024

@author: nutachirasawad
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split

# Load the data from an Excel file
data_path = 'tempdata.xlsx'
data = pd.read_excel(data_path)

# Assuming data is monthly starting from 1st October 1752
start_date = '1752-10-01'
dates = pd.date_range(start=start_date, periods=len(data), freq='M')
data['Date'] = dates

# Display initial data information
print(data.head())
print(data.info())

# Handling missing values by filling them with the mean of the column if necessary
data['Average_Temp'].fillna(data['Average_Temp'].mean(), inplace=True)

# Check for and remove any duplicates
data = data.drop_duplicates()

# Resetting the index after cleaning
data.reset_index(drop=True, inplace=True)

# Plotting temperature over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Average_Temp'], label='Temperature')
plt.title('Temperature Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Seasonal decomposition to understand trends, seasonality, and noise
decomposition = seasonal_decompose(data.set_index('Date')['Average_Temp'], model='additive')
fig = decomposition.plot()
plt.show()

# Split the data into training and the last 12 months for testing
train_data = data[:-12]
test_data = data[-12:]

# Define the SARIMA model parameters
order = (2, 0, 2)
seasonal_order = (0, 1, 2, 12)  

# Initialize and fit the SARIMA model
sarima_model = SARIMAX(train_data['Average_Temp'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit()

# Show model summary
print(sarima_results.summary())

# Forecast for the next 12 months
forecast_steps = 12
predicted_values = sarima_results.get_forecast(steps=forecast_steps)
predicted_mean = predicted_values.predicted_mean
predicted_ci = predicted_values.conf_int()

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Average_Temp'], label='Historical Data')
plt.plot(test_data['Date'], test_data['Average_Temp'], label='Actual Temperature', color='red')
plt.plot(predicted_values.predicted_mean.index, predicted_values.predicted_mean, label='Forecasted Temperature', color='green')
plt.fill_between(predicted_values.predicted_mean.index, predicted_ci.iloc[:, 0], predicted_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Temperature Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Calculate and print performance metrics
mse = mean_squared_error(test_data['Average_Temp'], predicted_mean)
print('Mean Squared Error:', mse)
forecast_data = pd.DataFrame({
    'Forecasted Temp': predicted_mean,
    'Lower CI': predicted_ci.iloc[:, 0],
    'Upper CI': predicted_ci.iloc[:, 1]
})

# Display the forecast table
print("Forecast for the Next Year:")
print(forecast_data)

# Additionally, save the forecast table to a CSV file if needed
forecast_data.to_csv('forecast_next_year.csv', index=True)

# Plotting remains as previously defined
plt.figure(figsize=(14, 7))
plt.plot(data['Date'], data['Average_Temp'], label='Historical Data')
plt.plot(test_data['Date'], test_data['Average_Temp'], label='Actual Temperature', color='red')
plt.plot(predicted_values.predicted_mean.index, predicted_values.predicted_mean, label='Forecasted Temperature', color='green')
plt.fill_between(predicted_values.predicted_mean.index, predicted_ci.iloc[:, 0], predicted_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Temperature Forecast vs Actuals')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()

# Performance metrics
mse = mean_squared_error(test_data['Average_Temp'], predicted_mean)
print('Mean Squared Error:', mse)