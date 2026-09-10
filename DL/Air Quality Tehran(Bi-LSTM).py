from keras.models import Sequential
from keras.layers import LSTM, Dense,Bidirectional
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
data.drop("Time",axis=1,inplace=True)
# Split the data into input features (X) and target variable (y)
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

# Create a bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(64, activation='relu'), input_shape=(1, X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Rescale the predictions to their original scale
# y_pred = scaler.inverse_transform(y_pred)

# Reshape the target variable for evaluation
y_test = np.array(y_test).reshape(-1, 1)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print(r2_score(y_test,y_pred))
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(y_pred, color='orange', label='Pred')
plt.legend()
plt.show()