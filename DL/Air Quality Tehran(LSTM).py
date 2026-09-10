from tensorflow import keras
from keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from keras.layers import CuDNNLSTM
from keras.models import Sequential
import calendar
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
import sklearn.model_selection as ms

data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data['Time'] = pd.to_datetime(data['Time'].astype(
    str))

scaler = MinMaxScaler()
AQI = data.AQI.values.reshape(-1, 1)
scaled_AQI = scaler.fit_transform(AQI)

seq_len = 60

def split_into_sequences(data, seq_len):
    n_seq = len(data) - seq_len + 1
    return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

def get_train_test_sets(data, seq_len, train_frac):
    sequences = split_into_sequences(data, seq_len)
    n_train = int(sequences.shape[0] * train_frac)
    X_train = sequences[:n_train, :-1, :]
    y_train = sequences[:n_train, -1, :]
    X_test = sequences[n_train:, :-1, :]
    y_test = sequences[n_train:, -1, :]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = get_train_test_sets(scaled_AQI, seq_len, train_frac=0.8)

# timestamps = pd.to_datetime(data['Time'])
# values = data[['temp', 'humidity', 'windspeed', 'sealevelpressure',
# 'visibility', 'solarradiation', 'uvindex', 'CO', 'O3', 'NO2', 'SO2', 'PM10', 'PM2.5']]  # Assuming you want to use 'PM2.5', 'Temperature', and 'Humidity' as features
# values = np.array(values)
dropout = 0.2
window_size = seq_len - 1

# build a 3-layer LSTM RNN
model = keras.Sequential()

model.add(
    LSTM(window_size, return_sequences=True, 
         input_shape=(window_size, X_train.shape[-1]))
)

model.add(Dropout(rate=dropout))
# Bidirectional allows for training of sequence data forwards and backwards
model.add(
    Bidirectional(LSTM((window_size * 2), return_sequences=True)
)) 

model.add(Dropout(rate=dropout))
model.add(
    Bidirectional(LSTM(window_size, return_sequences=False))
) 

model.add(Dense(units=1))
# linear activation function: activation is proportional to the input
model.add(Activation('linear'))
batch_size = 16

model.compile(
    loss='mean_squared_error',
    optimizer='adam'
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=batch_size,
    shuffle=False,
    validation_split=0.2
)
y_pred = model.predict(X_test)

# invert the scaler to get the absolute price data
y_test_orig = scaler.inverse_transform(y_test)
y_pred_orig = scaler.inverse_transform(y_pred)


# Visualize the results
# plt.plot(timestamps[train_size+sequence_length:], y_test[:, 0], label='Actual AQI')
# plt.plot(timestamps[train_size+sequence_length:], predictions[:, 0], label='Predicted AQI')
# plt.xlabel('Timestamp')
# plt.ylabel('AQI')
# plt.title('Air Quality Index Prediction')
# plt.legend()
# plt.show()
import math
MSE = np.square(np.subtract(y_test, y_pred)).mean()
RMSE = math.sqrt(MSE)
print(MSE)
print(RMSE)
print(r2_score(y_test,y_pred))
