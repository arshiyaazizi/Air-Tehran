from lazypredict.Supervised import LazyRegressor
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
from sklearn import linear_model as lm
from sklearn.metrics import r2_score, accuracy_score
import seaborn as sns
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error
import array
import statsmodels.formula.api as sm
number = np.arange(1, 155, dtype=int)
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.3)
# data['Time'] = pd.to_datetime(data['Time'])
# # data = data.dropna()
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
# # fig,ax=plt.subplots(figsize=(15,5))
# # train.plot(ax=ax,label='Training set')
# # test.plot(ax=ax,label='Testing set')
# # ax.axvline('01-01-2005',ls='--',color='Black')
# # plt.show()
# train = Create_Features(train)
# test = Create_Features(test)
# Features = ['temp', 'humidity', 'windgust', 'windspeed', 'sealevelpressure',
#             'visibility', 'solarradiation', 'precip', 'uvindex', 'CO', 'O3', 'NO2', 'SO2', 'PM10', 'PM2.5']
# target = 'AQI'

# x_train = train[Features]
# y_train = train[target]
# x_test = test[Features]
# y_test = test[target]

# print(data)
# score = np.sqrt(mean_squared_error(['AQI'], test['prediction']))

# ax = data.loc[(data.index > '2023-01-01') & (data.index < '2023-01-30')
#               ]['AQI'].plot(figsize=(15, 5), title='Month Of Data')
# data.loc[(data.index > '2023-01-01') & (data.index < '2023-01-30')
#          ]['prediction'].plot(figsize=(15, 5), title='Month Of Data')
# plt.show()
# print(y_predicate)
# y_eror = y_test-test['prediction']
# print(y_eror)
reg = LazyRegressor(verbose=0, ignore_warnings=False,
                    custom_metric=None, predictions=False, random_state=13)
model, predicate = reg.fit(x_train, x_test, y_train, y_test)
model_dictionary = reg.provide_models(x_train, x_test, y_train, y_test)
print(model)
# print(clf.coef_)
# print(clf.intercept_)
# reg = ExtraTreesRegressor(
#     n_estimators=100, random_state=0).fit(x_train, y_train)
# score = reg.score(x_test, y_test)
# print(score)
