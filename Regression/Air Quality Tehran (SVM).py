import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.model_selection as ms
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn
from sklearn import svm
from sklearn.metrics import r2_score,mean_absolute_error
import seaborn as sns
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
SVM_MODEL = svm.SVR(kernel="linear", C=1, gamma=1)
SVM_MODEL.fit(x_train, y_train)
SVM_MODEL.score(x_train, y_train)
prediction= SVM_MODEL.predict(x_test)
# data = data.merge(test[['prediction']], how='left',
                #   right_index=True, left_index=True)
# print(test['prediction'])
# y_eror = x_test-test['prediction']
# print(y_test)
# print(r2_score(y_test, test['prediction']))
# fi=pd.DataFrame(data=SVM_MODEL.feature_importances_,index=SVM_MODEL.feature_names_in_,columns=['Importance'])
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