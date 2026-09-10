import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop("Time",axis=1,inplace=True)
# Split the data into input features (X) and target variable (y)
X = data.drop('AQI', axis=1)
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the input data for CNN
X_train = np.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(Flatten())
# Add LSTM layer
model.add(LSTM(64, activation='relu',return_sequences=True))

# Add output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# # Print the model summary
# print(model.summary())
# model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))
# # Generate predictions on the test set
# predictions = model.predict(X_test_reshaped)

# # Perform any necessary inverse scaling or post-processing on the predictions
# mse = mean_squared_error(y_test, predictions)
# print('Mean Squared Error:', mse)
# print(r2_score(y_test,predictions))
