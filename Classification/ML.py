from sklearn.preprocessing import MinMaxScaler
import time
from pandas import value_counts
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_predict,cross_val_score,train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree           
from IPython.display import Image, display              
import pydotplus    
data = pd.read_excel(
    r"/home/arshia/arshia/Msc/Air Tehran/Orginal Air Quality Tehran.xlsx")
data.drop('Time',axis=1,inplace=True)
def probThreshold(inpProb):
    if inpProb  <  51:
        return(0)
    if 50<inpProb < 101:
        return(1)
    if 100<inpProb<151:
        return(2)
    if 150<inpProb<201:
        return(3)
    if 200<inpProb<301:
        return(4)
    else:
        return(5)
          
data['Outcomes']=data['AQI'].apply(probThreshold)
# print(count(data,'Outcomes','1'))        
# Features=['temp','humidity','precip','windspeed', 'sealevelpressure',
#        'visibility', 'solarradiation', 'uvindex', 'CO', 'O3', 'NO2', 'SO2',
#        'PM10', 'PM2.5']
# Target=['Outcomes']
# X=data[Features].values
# y=data[Target].values
X=data.iloc[:,:-2].values
y=data.iloc[:,-1].values
X_temp, X_test, y_temp, y_test = train_test_split(X, y,test_size=.15, random_state =12)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=.176, random_state=12) #0.176 * 0.85 = 0.15

unique, counts = np.unique(y_train, return_counts=True)
print("TrainSet (class: count) = ", dict(zip(unique, counts)), "TrainSize : ", y_train.shape[0])

unique, counts = np.unique(y_test, return_counts=True)
print("TestSet (class: count) =", dict(zip(unique, counts)), "TestSize : ", y_test.shape[0])

unique, counts = np.unique(y_val, return_counts=True)
print("ValSet (class: count) = ", dict(zip(unique, counts)), "ValSize : ", y_val.shape[0])

print(y.shape[0], " = ", (y_train.shape[0] + y_test.shape[0] + y_val.shape[0]))
# print(X1)
# y=data.iloc[:,-1]
# data.dropna()
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123) 
# class_to_count= 1
# countwith=data['Outcomes'].value_counts().get(class_to_count)
# print(countwith)
# count=data.groupby('Outcomes').count().loc[class_to_count]
# print(count)
# plt.scatter(data['AQI'], data['Outcomes'])
# plt.xlabel('AQI', size=18)
# plt.ylabel('Outcomes', size=18)
# plt.show()
# clf = MLPClassifier(random_state=1, max_iter=300)
# md = 6
# clf = DecisionTreeClassifier()
clf=ExtraTreeClassifier(random_state=0)
# clf=BaggingClassifier(extra,random_state=0)
# clf=KNeighborsClassifier()
# clf=AdaBoostClassifier()
# clf=RandomForestClassifier()
# clf=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,    max_depth=1, random_state=0)
start = time.time()
models=clf.fit(X_train,y_train)
stop = time.time()
print(f"Training time: {stop - start}s")
prediction=models.predict(X_test)
# data1['Outcome1']=prediction
print(accuracy_score(y_test,prediction))
print(balanced_accuracy_score(y_test,prediction))
print(f1_score(y_test,prediction,average='macro'))
print(classification_report(y_test, prediction))        
# dot_data = tree.export_graphviz(models, out_file=None,filled=True, rounded=True)            
# graph = pydotplus.graph_from_dot_data(dot_data)            
# display(Image(data=graph.create_png()))
fig, ax = plt.subplots(figsize=(8, 5))   
cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, prediction),display_labels=["class_1", "class_2", "class_3","class_4","class_5","class_6"],)
cmp.plot(ax=ax)            
plt.show()
# conf_matrix = confusion_matrix(y_test, prediction) 
# sns.heatmap(conf_matrix, annot = True, cmap= 'Blues') 
# plt.ylabel('True') 
# plt.xlabel('False') 
# plt.title('Confusion Matrix') 
# plt.show()
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(prediction, color='orange', label='Pred')
plt.legend()
plt.show()
fi = pd.DataFrame(data=clf.feature_importances_,
                  index=clf.feature_names_in_, columns=['Importance'])
fi.sort_values('Importance').plot(kind='barh', title='Features Importance')
plt.show()
# feature_importances = clf.feature_importances_
# sorted_indices = feature_importances.argsort()[::-1]
# sorted_feature_names = data.feature_names[sorted_indices]
# sorted_importances = feature_importances[sorted_indices]

# Create a bar plot of the feature importances
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.barplot(sorted_importances, sorted_feature_names)