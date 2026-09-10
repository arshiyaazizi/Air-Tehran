from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import calendar
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
from sklearn import linear_model as lm
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import seaborn as sns
import sklearn.model_selection as ms
from sklearn.metrics import mean_squared_error
import array
import statsmodels.formula.api as sm
number = np.arange(1, 155, dtype=int)
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
# data['Time'] = pd.to_datetime(data['Time'].astype(
#     str))
data.drop('Time', inplace=True, axis=1)
ds=data.describe()
print(ds)
# scaler = MinMaxScaler()
# scaler.fit(data)
# data = scaler.transform(data)
# x=data[:,:-1]
# print(x)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# plt.plot(data)
# X, y = data(random_state=170, n_samples=600, centers = 5)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# cluster the data into five clusters
# dbscan = DBSCAN(eps=0.5, min_samples=5)
# dbscan.fit(x)
# labels = dbscan.labels_
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# print(f"Number of clusters: {n_clusters}")
# print("Cluster labels:")
# print(labels)
# plot the cluster assignments
# data = data.set_index('Time')
# pd.to_datetime(data.index)



# y=data[:,-1]
# X = data.iloc[:, :-1]
# y = data.iloc[:, -1]
# sc = MinMaxScaler(feature_range=(0, 1))
# data_set_scaled = sc.fit_transform(data)
# print(data_set_scaled)
# X = []
# backcandles = 30
# print(data_set_scaled.shape[0])

# for j in range(8):
    # X.append([])
#     for i in range(backcandles, data_set_scaled.shape[0]):
#         X[j].append(data_set_scaled[i - backcandles:i, j])

# X = np.moveaxis(X, [0],[2])
# X, yi = np.array(X), np.array(data_set_scaled[:, -1])
# y = np.reshape(yi, (len(yi), 1))

# splitlimit = int(len(X) * 0.8)
# x_train, x_test = X[:splitlimit], X[splitlimit:]
# y_train, y_test = y[:splitlimit], y[splitlimit:]

x_train, x_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.3)
# number=np.arange(1,330,dtype=int)
# z=pd.DataFrame(number)
# print(np.where(data == 'NaN'))
# print(data)

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
# data['month'] = data['month'].apply(lambda x: calendar.month_abbr[x])
# fig,ax=plt.subplots(figsize=(15,5))
# train.plot(ax=ax,label='Training set')
# test.plot(ax=ax,label='Testing set')
# ax.axvline('01-01-2005',ls='--',color='Black')
# plt.show()
# train = Create_Features(x_train)
# test = Create_Features(x_test)
# Features = ['temp', 'humidity', 'windgust', 'windspeed', 'sealevelpressure',
# 'visibility', 'solarradiation', 'uvindex', 'CO', 'O3', 'NO2', 'SO2', 'PM10', 'PM2.5']
# target = 'AQI'

# param_names={0:'temp', 1:'humidity',2: 'windgust',3: 'windspeed',4: 'sealevelpressure',
# 5:'visibility',6: 'solarradiation',7: 'uvindex',8: 'CO',9: 'O3',10: 'NO2',11: 'SO2',12: 'PM10', 13:'PM2.5'}
# x_train = x_train[Features]
# y_train = y_train[target]
# x_test = x_test[Features]
# y_test = y_test[target]
# nsamples, nx, ny = x_train.shape
# d2_train_dataset = y_train.reshape((nsamples,nx*ny))
reg = ExtraTreesRegressor(
    n_estimators=1000, random_state=0).fit(x_train, y_train)
prediction= reg.predict(x_test)
# data.loc[:,'prediction'] = pd.Series(prediction)
# prediction += [np.nan] * (len(data) - len(prediction))
# data['prediction']=prediction
# data['prediction'] = data.apply(prediction)
# model=ExtraTreesRegressor()
# model.fit(x_train, y_train)
# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
# indices = np.argsort(importances)[::-1]

# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), [param_names[i] for i in indices], rotation=90)
# plt.xlim([-1, X.shape[1]])
# plt.show()
# data = data.merge(prediction']], how='left',
#                   right_index=True, left_index=True)
# print(data.tail())
# score = np.sqrt(mean_squared_error(test['AQI'], test['prediction']))
# print(score)
# ax = data.loc[(data.index > '2021-03-21') & (data.index < '2023-03-20')
#               ]['AQI'].plot(figsize=(15, 5), ylabel='AQI')
# data.loc[(data.index > '2021-03-21') & (data.index < '2023-03-20')
#          ]['prediction'].plot(figsize=(15, 5), ylabel='AQI')
# plt.plot(z, prediction,color="blue",label='Pred')
# plt.plot(z, y_test,color="red",label='Test')
# plt.show()
# print(y_predicate)
# y_eror = y_test-test['prediction']
# print(y_eror)
# MSE = np.square(np.subtract(y_test, prediction)).mean()
# RMSE = math.sqrt(MSE)
# print(mean_absolute_error(y_test,prediction))
# print(MSE)
# print(RMSE)
print(r2_score(y_test,prediction))
# print(data.corr)
# print(clf.coef_)
# print(clf.intercept_)
# print(data.isna().sum())
# score = reg.score(x_test, y_test)
# print(score)
# print(data['month'])
# fig, ax = plt.subplots(figsize=(15, 5))
# sns.boxplot(data=data, x='month', y='AQI',ax=ax)
# fi = pd.DataFrame(data=reg.feature_importances_,
#                   index=reg.feature_names_in_, columns=['Importance'])
# fi.sort_values('Importance').plot(kind='barh', title='Features Importance')
# plt.show()
# train['AQI'].plot(ax=ax,label='Training set',ylabel='AQI')
# test['AQI'].plot(ax=ax,label='Testing set')
# ax.axvline('2022-12-01',ls='--',color='Black')
# print(data)
# months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# ax1 = data[data["Outcome"]=="0"].plot(kind="scatter",x='DiabetesPedigreeFunction',y='Pregnancies',color='blue')
# ax = data.loc[(data.index > '2022-03-21') & (data.index < '2023-03-20')
#   ]['AQI'].plot(figsize=(15, 5), ylabel='AQI')
# plt.legend(labels=['Outcome'])
# plt.title('Relationship between DiabetesPedigreeFunction and Pregnancies', size=20)
# plt.xlabel('month', size=18)
# plt.ylabel('AQI', size=18)
# plt.show()


# plt.figure(figsize=(16,8))
# plt.plot(y_test, color='blue', label='Test')
# plt.plot(prediction, color='orange', label='Pred')
# plt.legend()
# plt.show()
