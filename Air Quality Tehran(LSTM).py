from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import calendar
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error,mean_squared_error
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
# data['Time'] = pd.to_datetime(data['Time'].astype(
#     str))
data.drop('Time',inplace=True,axis=1)
X = data.drop('AQI', axis=1)
y = data['AQI']

# Scale the input features between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the input data to be 3D (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# scaler = MinMaxScaler()
# AQI = data.AQI.values.reshape(-1, 1)
# scaled_AQI = scaler.fit_transform(AQI)

# seq_len = 60

# def split_into_sequences(data, seq_len):
#     n_seq = len(data) - seq_len + 1
#     return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

# def get_train_test_sets(data, seq_len, train_frac):
#     sequences = split_into_sequences(data, seq_len)
#     n_train = int(sequences.shape[0] * train_frac)
#     X_train = sequences[:n_train, :-1, :]
#     y_train = sequences[:n_train, -1, :]
#     X_test = sequences[n_train:, :-1, :]
#     y_test = sequences[n_train:, -1, :]
#     return X_train, y_train, X_test, y_test

# X_train, y_train, X_test, y_test = get_train_test_sets(scaled_AQI, seq_len, train_frac=0.8)
# window_size = seq_len - 1
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_shape=(1, X_train.shape[2])))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)
y_pred=model.predict(X_test)
# Visualize the results
# plt.plot(timestamps[train_size+sequence_length:], y_test[:, 0], label='Actual AQI')
# plt.plot(timestamps[train_size+sequence_length:], predictions[:, 0], label='Predicted AQI')
# plt.xlabel('Timestamp')
# plt.ylabel('AQI')
# plt.title('Air Quality Index Prediction')
# plt.legend()
# plt.show()
y_test = np.array(y_test).reshape(-1, 1)
MSE = np.square(np.subtract(y_test, y_pred)).mean()
RMSE = math.sqrt(MSE)
print(MSE)
print(RMSE)
print(r2_score(y_test,y_pred))
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(y_pred, color='orange', label='Pred')
plt.legend()
plt.show()