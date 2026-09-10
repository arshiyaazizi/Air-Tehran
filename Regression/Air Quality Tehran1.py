from sklearn.preprocessing import StandardScaler
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import seaborn as sns
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error
import array
import statsmodels.formula.api as sm

# Load and preprocess the data
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop('Time', inplace=True, axis=1)

# Normalize the values to a range between 0 and 1
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

# Split the data into training and test sets
x = data[:, :-1]
y = data[:, -1]
splitlimit = int(len(x) * 0.8)
x_train, x_test = x[:splitlimit], x[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]


# Build the ExtraTreesRegressor model
model=ExtraTreesRegressor(n_estimators=2000,random_state=0)
reg=model.fit(x_train, y_train)

# Make predictions on the test set
prediction= reg.predict(x_test)

# Evaluate the model
MSE = np.square(np.subtract(y_test, prediction)).mean()
RMSE = math.sqrt(MSE)
print(mean_absolute_error(y_test,prediction))
print(MSE)
print(RMSE)
print(r2_score(y_test,prediction))
