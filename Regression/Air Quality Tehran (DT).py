from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import statsmodels.formula.api as sm
import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score, r2_score,mean_absolute_error
number = np.arange(1, 155, dtype=int)
z = pd.DataFrame(number)
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
dtreeR = DecisionTreeRegressor()
dtreeR.fit(x_train, y_train)
prediction= dtreeR.predict(x_test)
# data = data.merge(test[['prediction']], how='left',
                #   right_index=True, left_index=True)
# print(y_predicate)
# y_eror = Tex-y_predicate
# print(y_test)
# print(r2_score(y_test, test['prediction']))
# MSE = np.square(np.subtract(y_test, test['prediction'])).mean()
# RMSE = math.sqrt(MSE)
# print(MSE)
# print(mean_absolute_error(y_test,test['prediction']))
# fi=pd.DataFrame(data=dtreeR.feature_importances_,index=dtreeR.feature_names_in_,columns=['Importance'])
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
# plt.show()
