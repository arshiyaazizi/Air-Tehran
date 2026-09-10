import sklearn.model_selection as ms
import math
from numpy import asarray
import pandas as pd
import pandas_datareader as web
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import matplotlib as mpl
from arch import arch_model
import xgboost as xgb
from itertools import filterfalse
import seaborn as sns
from datetime import time, date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
# number = np.arange(1, 155, dtype=int)
# z = pd.DataFrame(number)
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop('Time',inplace=True,axis=1)
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
x=data[:,:-1]
y=data[:,-1]
x_train, x_test, y_train, y_test = ms.train_test_split(
    x, y, test_size=0.3)

# data['Time'] = pd.to_datetime(data['Time'])
# data = data.set_index('Time')
# pd.to_datetime(data.index)
# train = data.loc[data.index < '2022-12-01']
# test = data.loc[data.index >= '2022-12-01']
# X = data.iloc[:, 1:-1].values
# y = data.iloc[:, -1].values
# x_train, x_test, y_train, y_test = ms.train_test_split(
    # X, y, test_size=0.3)


# def Create_Features(data):
#     data = data.copy()
#     data['hour'] = data.index.hour
#     data['dayofweek'] = data.index.dayofweek
#     data['quarter'] = data.index.quarter
#     data['month'] = data.index.month
#     data['year'] = data.index.year
#     data['dayofyear'] = data.index.dayofyear
#     return data


# data = Create_Features(data)
# train = Create_Features(train)
# test = Create_Features(test)
# Features = ['temp', 'humidity', 'windgust', 'windspeed', 'sealevelpressure',
#             'visibility', 'solarradiation', 'uvindex', 'CO', 'O3', 'NO2', 'SO2', 'PM10', 'PM2.5']
# target = 'AQI'

# x_train = train[Features]
# y_train = train[target]
# x_test = test[Features]
# y_test = test[target]
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(x_train, y_train)
prediction= reg.predict(x_test)
# data = data.merge(test[['prediction']], how='left',
                #   right_index=True, left_index=True)
# test['eror'] = np.abs(test[target]-test['prediction'])
# test['date'] = test.index.date
# True_predict = test.groupby(['date'])['eror'].mean(
# ).sort_values(ascending=True).head(5)
# False_predict = test.groupby(['date'])['eror'].mean(
# ).sort_values(ascending=False).head(5)
# score = np.sqrt(mean_squared_error(test['AQI'], test['prediction']))
# print(f'RMSE score on test set : {score:0.2f}')
# print(r2_score(y_test,test['prediction']))
# MSE = np.square(np.subtract(y_test, test['prediction'])).mean()
# RMSE = math.sqrt(MSE)
# print(MSE)
# print(mean_absolute_error(y_test,test['prediction']))
# fi=pd.DataFrame(data=reg.feature_importances_,index=reg.feature_names_in_,columns=['Importance'])
# fi.sort_values('Importance').plot(kind='barh',title='Features Importance')
MSE = np.square(np.subtract(y_test, prediction)).mean()
RMSE = math.sqrt(MSE)
print(mean_absolute_error(y_test,prediction))
print(MSE)
print(RMSE)
print(r2_score(y_test,prediction))
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(prediction, color='orange', label='Pred')
plt.legend()
plt.show()
