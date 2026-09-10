import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn import linear_model as lm
from sklearn.metrics import r2_score,mean_absolute_error
import seaborn as sns
import sklearn.model_selection as ms
import array
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as sm
number = np.arange(1, 155, dtype=int)
z = pd.DataFrame(number)
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
# data['Time'] = pd.to_datetime(data['Time'])
# data = data.set_index('Time')
# pd.to_datetime(data.index)
# train = data.loc[data.index < '2022-12-01']
# test = data.loc[data.index >= '2022-12-01']
# X = data.iloc[:, 1:-1].values
# y = data.iloc[:, -1].values
data.drop('Time',inplace=True,axis=1)
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)
x=data[:,:-1]
y=data[:,-1]
x_train, x_test, y_train, y_test = ms.train_test_split(
    x, y, test_size=0.3)


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
# Features = ['temp', 'dew', 'humidity', 'windgust', 'windspeed', 'winddir',
#        'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
#        'uvindex', 'severerisk', 'CO', 'O3', 'NO2', 'SO2', 'PM10', 'PM2.5']
# target = 'AQI'
# # print(data.columns)
# x_train = train[Features]
# y_train = train[target]
# x_test = test[Features]
# y_test = test[target]
lm = LinearRegression()
# rfe = RFECV(estimator=lm, step=1, cv=5)

# Fit the model with the dataset
# rfe.fit(data, data["AQI"])

# Print the selected features
# print("Selected Features:", data[rfe.support_])
lmfit = lm.fit(x_train, y_train)
# coefficients = pd.concat(
#     [pd.DataFrame(x_train.columns), pd.DataFrame(np.transpose(lm.coef_))], axis=1)
# print(coefficients)
prediction= lmfit.predict(x_test)

# data = data.merge(test[['prediction']], how='left',
                #   right_index=True, left_index=True)
# print(y_predicate)
# y_eror = Tex-y_predicate
# print(y_test)
# print(r2_score(y_test, test['prediction']))
# print(lm.coef_)
# print(lm.intercept_)
# MSE = np.square(np.subtract(y_test, test['prediction'])).mean()
# RMSE = math.sqrt(MSE)
# print(MSE)
# print(mean_absolute_error(y_test,test['prediction']))
# fi=pd.DataFrame(data=lmfit.feature_importances_,index=lmfit.feature_names_in_,columns=['Importance'])
# fi.sort_values('Importance').plot(kind='barh',title='Features Importance')

MSE = np.square(np.subtract(y_test, prediction)).mean()
RMSE = math.sqrt(MSE)
print(mean_absolute_error(y_test,prediction))
print(MSE)
print(RMSE)
print(r2_score(y_test,prediction))
# plt.show()
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(prediction, color='orange', label='Pred')
plt.legend()
plt.show()