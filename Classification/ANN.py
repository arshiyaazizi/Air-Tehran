from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import time
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
# Features=['temp','humidity','precip','windspeed', 'sealevelpressure',
#        'visibility', 'solarradiation', 'uvindex', 'CO', 'O3', 'NO2', 'SO2',
#        'PM10', 'PM2.5']
# Target=['Outcomes']
# X=data[Features].values
# y=data[Target].values
X=data.iloc[:,:-2].values
y=data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
classifier=tf.keras.Sequential()
classifier.add(tf.keras.layers.Dense(units=10,input_dim=14,kernel_initializer='uniform',activation='relu'))
classifier.add(tf.keras.layers.Dense(units=5,kernel_initializer='uniform',activation='relu'))
classifier.add(tf.keras.layers.Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model=classifier.fit(X_train,y_train,batch_size=5,epochs=100)
# print(model)
Predictions=classifier.predict(X_test)
print(accuracy_score(y_test,Predictions))
print(balanced_accuracy_score(y_test,Predictions))
print(f1_score(y_test,Predictions,average='macro'))
print(classification_report(y_test, Predictions))  
# fig, ax = plt.subplots(figsize=(8, 5))   
# cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, Predictions),display_labels=["class_1", "class_2", "class_3","class_4","class_5","class_6"],)
# cmp.plot(ax=ax)            
# plt.show()
# conf_matrix = confusion_matrix(y_test, Predictions) 
# sns.heatmap(conf_matrix, annot = True, cmap= 'Blues') 
# plt.ylabel('True') 
# plt.xlabel('False') 
# plt.title('Confusion Matrix') 
# plt.show()
# plt.figure(figsize=(16,8))
# plt.plot(y_test, color='blue', label='Test')
# plt.plot(Predictions, color='orange', label='Pred')
# plt.legend()
# plt.show()




# def make_classification_ann(Optimizer_Trial, Neurons_Trial,Neurons_Trial1):
    # from keras.models import Sequential
    # from keras.layers import Dense
    
    # # Creating the classifier ANN model
    # classifier = Sequential()
    # classifier.add(Dense(units=Neurons_Trial, input_dim=14, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=Neurons_Trial1, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # classifier.compile(optimizer=Optimizer_Trial, loss='binary_crossentropy', metrics=['accuracy'])
            
    # return classifier


# Parameter_Trials={'batch_size':[5,10,20,30],
                #       'epochs':[5, 10, 50 ,100],
                #     'Optimizer_Trial':['adam', 'rmsprop'],
                #   'Neurons_Trial': [5,10,20,50],
                #   'Neurons_Trial1': [5,10,20]
                #  }

# # Creating the classifier ANN
# classifierModel=KerasClassifier(make_classification_ann, verbose=0)
# grid_search=GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring='f1', cv=5)

# ########################################

# # Measuring how much time it took to find the best params
# import time
# StartTime=time.time()

# # Running Grid Search for different paramenters
# grid_search.fit(X_train,y_train, verbose=1)

# EndTime=time.time()
# print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

# printing the best parameters
# print('\n#### Best hyperparamters ####')
# print(grid_search.best_params_)
# Printing the best parameter
# print(ResultsData.sort_values(by='Accuracy', ascending=False).head(1))
 
# Visualizing the results
# ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line', rot=20)
# plt.show()
# Predictions=classifier.predict(X_test)
 
# Scaling the test data back to original scale
# Test_Data=predictorScalerfit.inverse_transform(X_test)
# Generating a data frame for analyzing the test data
# TestingData=pd.DataFrame(data=Test_Data, columns=Features)
# TestingData['Outcomes']=y_test
# TestingData['PredictedSurvivalProb']=Predictions
# print(metrics.classification_report(y_test, Predictions))
# print(metrics.confusion_matrix(y_test, Predictions))
# print(metrics.precision_score(y_test, Predictions))
# print(metrics.roc_auc_score(y_test, Predictions))