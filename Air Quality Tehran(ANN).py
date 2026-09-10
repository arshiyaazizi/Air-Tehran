import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

data = pd.read_excel(r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop("Time",inplace=True,axis=1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the feature values
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
# loss = model.evaluate(X_test, y_test)
# print('Test Loss:', loss)
# Assuming you have a new sample X_new for prediction
# X_new = scaler.transform()
prediction = model.predict(X_test)
# print('Predicted air quality:', prediction)
loss = model.evaluate(X_test, y_test)
print("Test Loss:", loss)

# ann = tf.keras.models.Sequential()

# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# y_pred = ann.predict(X_test)
# y_pred = (y_pred > 0.5)
import math
MSE = np.square(np.subtract(y_test, prediction)).mean()
RMSE = math.sqrt(MSE)
print(MSE)
print(RMSE)
print(r2_score(y_test,prediction))
