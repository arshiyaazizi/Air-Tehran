import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,roc_auc_score,accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn import metrics
from sklearn.metrics import make_scorer,accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,LSTM,Embedding
from keras.preprocessing import sequence
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
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
X=data.iloc[:,:-2].values
y=data.iloc[:,-1].values
scaler=StandardScaler()
X=scaler.fit_transform(X)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
X=X.reshape((X.shape[0],1,X.shape[1]))
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
model = Sequential()
model.add(LSTM(50,input_shape=(X_train.shape[1],X_train.shape[2]),activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32,validation_split=0.2)
Predictions=model.predict(X_test).argmax(axis=1)
Accuracy=model.evaluate(X_test,y_test)[1]
print(f'Test Accuracy:{Accuracy*100:.2f}%')
# print(metrics.classification_report(y_test, Predictions))
# print(metrics.confusion_matrix(y_test, Predictions))
# print(metrics.precision_score(y_test, Predictions))
print(accuracy_score(y_test,Predictions))
print(balanced_accuracy_score(y_test,Predictions))
print(f1_score(y_test,Predictions,average='macro'))
print(classification_report(y_test, Predictions))  
fig, ax = plt.subplots(figsize=(8, 5))   
cmp = ConfusionMatrixDisplay(confusion_matrix(y_test, Predictions),display_labels=["class_1", "class_2", "class_3","class_4","class_5","class_6"],)
cmp.plot(ax=ax)            
plt.show()
conf_matrix = confusion_matrix(y_test, Predictions) 
sns.heatmap(conf_matrix, annot = True, cmap= 'Blues') 
plt.ylabel('True') 
plt.xlabel('False') 
plt.title('Confusion Matrix') 
plt.show()
plt.figure(figsize=(16,8))
plt.plot(y_test, color='blue', label='Test')
plt.plot(Predictions, color='orange', label='Pred')
plt.legend()
plt.show()







# def make_classification_rnn(Optimizer_Trial,Neurons_Trial,activation):
    
#             model = Sequential()
#             model.add(LSTM(Neurons_Trial,input_shape=(X_train.shape[1],X_train.shape[2]),activation=activation))
#             model.add(Dense(len(np.unique(y)), activation='softmax'))
#             model.compile(loss='sparse_categorical_crossentropy', optimizer=Optimizer_Trial, metrics=['accuracy'])
            
#             return model


# Parameter_Trials={'batch_size':[5,10,20,30],
#                       'epochs':[5, 10, 50 ,100],
#                     'Optimizer_Trial':['adam', 'rmsprop'],
#                   'Neurons_Trial': [5,10,20,50],
#                   'activation':['relu','tanh']
#                  }

# # # Creating the classifier RNN
# classifierModel=KerasClassifier(make_classification_rnn, verbose=0)
# grid_search=GridSearchCV(estimator=classifierModel, param_grid=Parameter_Trials, scoring=make_scorer(accuracy_score), cv=5)

# # ########################################

# # # Measuring how much time it took to find the best params
# import time
# StartTime=time.time()

# # # Running Grid Search for different paramenters
# grid_search.fit(X_train,y_train)

# EndTime=time.time()
# print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

# # printing the best parameters
# print('\n#### Best hyperparamters ####')
# print(grid_search.best_params_)
# print("best Accuracy:{:,.2f}%".format(grid_search.best_score_*100))