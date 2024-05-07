# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#define data in excel file

import pandas as pd
import numpy as np
from pandas import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from itertools import product

# Load the data
data_path = 'tempdata.xlsx'
data = pd.read_excel(data_path)

# Create a date range and assign it to the DataFrame
start_date = '1752-10-01'
dates = pd.date_range(start=start_date, periods=len(data), freq='M')
data['Date'] = dates
data.set_index('Date', inplace=True)

# Fill missing values and remove duplicates
data.fillna(data.mean(), inplace=True)
data.drop_duplicates(inplace=True)

# Function to find the best SARIMA model
def auto_arima(data, seasonal=True, m=12):
    """
    Performs a grid search across combinations of SARIMA parameters and seasonal adjustments.
    """
    p = d = q = range(0, 3)  # Consider 0, 1, 2 for ARIMA parameters
    pdq = list(product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], m) for x in list(product(p, d, q))]

    best_aic = np.inf
    best_params = None
    best_seasonal_params = None
    best_model = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = param
                    best_seasonal_params = param_seasonal
                    best_model = results
            except:
                continue

    print('Best SARIMA{}x{}12 - AIC:{}'.format(best_params, best_seasonal_params, best_aic))
    return best_model

# Run the auto_arima function to find the best model
best_sarima_model = auto_arima(data['Average_Temp'], seasonal=True, m=12)

# Print the summary of the best SARIMA model
print(best_sarima_model.summary())

# Forecasting and plotting the results
data['forecast'] = best_sarima_model.predict(start=pd.to_datetime('2020-01-01'), 
                                             end=pd.to_datetime('2021-01-01'), 
                                             dynamic=True)
plt.figure(figsize=(12, 8))
plt.plot(data['Average_Temp'], label='Actual')
plt.plot(data['forecast'], label='Forecast')
plt.title('Best SARIMA Model Forecast')
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.legend()
plt.show()
