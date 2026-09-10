from sklearn.preprocessing import StandardScaler
import calendar
import math
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error,mean_squared_error
import seaborn as sns
import sklearn.model_selection as ms
from sklearn.model_selection import cross_val_score
import statsmodels.formula.api as sm

number = np.arange(1, 155, dtype=int)
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
# data['Time'] = pd.to_datetime(data['Time'].astype(
#     str))
data.drop('Time', inplace=True, axis=1)
data.drop('solarradiation', inplace=True, axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
data=scaler.fit_transform(data)
X = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = ms.train_test_split(
    X, y, test_size=0.3)
# number=np.arange(1,330,dtype=int)
# z=pd.DataFrame(number)
# print(np.where(data == 'NaN'))
# print(data)

model = ExtraTreesRegressor(
    n_estimators=1000, random_state=0).fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=10)
prediction= model.predict(x_test)
print(scores.mean)

# data.loc[:,'prediction'] = pd.Series(prediction)
# prediction += [np.nan] * (len(data) - len(prediction))
# data['prediction']=prediction
# data['prediction'] = data.apply(prediction)


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
# print(y_predicate)
# y_eror = y_test-test['prediction']
# print(y_eror)
MSE = np.square(np.subtract(y_test, prediction)).mean()
RMSE = math.sqrt(MSE)
print(mean_absolute_error(y_test,prediction))
print(MSE)
print(RMSE)
print(r2_score(y_test,prediction))
# print(data.corr)
# print(clf.coef_)
# print(clf.intercept_)
# print(data.isna().sum())
# score = reg.score(x_test, y_test)






# fi = pd.DataFrame(data=model.feature_importances_,
#                   index=model.feature_names_in_, columns=['Importance'])
# fi.sort_values('Importance').plot(kind='barh', title='Features Importance')
# plt.show()
# train['AQI'].plot(ax=ax,label='Training set',ylabel='AQI')
# test['AQI'].plot(ax=ax,label='Testing set')
# ax.axvline('2022-12-01',ls='--',color='Black')



# plt.figure(figsize=(16,8))
# plt.plot(y_test, color='blue', label='Test')
# plt.plot(prediction, color='orange', label='Pred')
# plt.legend()
# plt.show()
