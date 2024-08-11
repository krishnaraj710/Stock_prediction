

import sys
print(sys.version)

# prompt: import keras and tensorflow

!pip install tensorflow
!pip install keras
import tensorflow as tf
import keras



import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(7)

# Load the dataset
data = pd.read_csv('reliance.csv')

# Extract the 'Close' price column
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create the training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
train_x, train_y = create_dataset(train_data, time_step)
test_x, test_y = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

# Define the LSTM model without stateful=True
def get_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(60, input_shape=(input_shape[0], input_shape[1])))
    model.add(Dense(1))
    return model

# Build and compile the model
model = get_lstm((train_x.shape[1], train_x.shape[2]))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mape'])

# Train the model
history = model.fit(train_x, train_y, verbose=2, epochs=50, batch_size=32, validation_data=(test_x, test_y), shuffle=False)

# Make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Inverse transform predictions and actual values
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
train_y = scaler.inverse_transform([train_y])
test_y = scaler.inverse_transform([test_y])

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plot the results
plt.figure(figsize=(14,5))

# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(scaled_data), label="True Price")
plt.plot(train_predict_plot, label="Train Prediction")
plt.plot(test_predict_plot, label="Test Prediction")
plt.legend()
plt.show()