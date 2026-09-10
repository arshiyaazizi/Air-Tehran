from keras import callbacks
import pandas as pd
import numpy as np
import os
import shutil
import pickle as pk
import pickle
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model,Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
#########################################################################################################

data = pd.read_excel(
    r"/home/arshia/arshia/Data Science/Air Tehran/Orginal Air Quality Tehran.xlsx")
############################################################################################################

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

X=data.iloc[:,:-2]
y=data.iloc[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
##################################################################################################

#########ENCODER####################
# dummy_y_train = to_categorical(y_train)
# dummy_y_test = to_categorical(y_test)
#########################################################################################

###########ENCODER ANOTHER##########
encoder = OneHotEncoder()
encoded_Y = encoder.fit(y.values.reshape(-1,1))
encoded_Y = encoded_Y.transform(y.values.reshape(-1,1)).toarray()
print(encoded_Y)
##############################################################################################
# import sklearn.model_selection as ms
# Trx, Tex, Try, Tey = ms.train_test_split(X, encoded_Y, train_size=0.7,random_state=123)
# Vax, Tex, Vay, Tey = ms.train_test_split(Tex, Tey, train_size=0.5)
# print(Trx.shape,Try.shape,Tex.shape,Tey.shape,Vax.shape,Vay.shape)
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

# Generate TrainX and TrainY
trainX, testX, trainY, testY = train_test_split(X, encoded_Y, test_size= 1 - train_ratio)
# Genearate ValX, TestX, ValY and TestY
valX, testX, valY, testY = train_test_split(testX, testY, test_size=test_ratio/(test_ratio + validation_ratio),random_state=123)
##################################################################################################################################################

#######  CHECK ALL     #########
y_part = [trainY, valY, testY]

for y_part in y_part:
    re_transformed_array = encoder.inverse_transform(y_part)
    
    unique_elements, counts_elements = np.unique(re_transformed_array, return_counts=True)
    unique_elements_and_counts = pd.DataFrame(np.asarray((unique_elements, counts_elements)).T)
    unique_elements_and_counts.columns = ['unique_elements', 'count']
#     print('---------------')
#     print(unique_elements_and_counts)
    
list_trainY = unique_elements_and_counts['unique_elements'].to_list()
list_valY = unique_elements_and_counts['unique_elements'].to_list()
list_testY = unique_elements_and_counts['unique_elements'].to_list()
# print(list_trainY)
# print(list_valY)
# print(list_testY)
check_test =  all(item in list_testY for item in list_trainY)
 
# if check_test is True:
#     print('OK !')
#     print("The list_testY contains all elements of the list_trainY.")    
# else :
#     print()
#     print('No !')
#     print("List_testY doesn't have all elements of the list_trainY.")
########################################################################################################


############MODEL DETAILS#####################

checkpoint_no = 'ANN'
model_name = 'AQC_ANN'
input_shape = trainX.shape[1]

n_batch_size = 30

n_steps_per_epoch = int(trainX.shape[0] / n_batch_size)
n_validation_steps = int(valX.shape[0] / n_batch_size)
n_test_steps = int(testX.shape[0] / n_batch_size)

n_epochs = 100

num_classes = trainY.shape[1]

# print('Input Shape: ' + str(input_shape))
# print('Batch Size: ' + str(n_batch_size))
# print()
# print('Steps per Epoch: ' + str(n_steps_per_epoch))
# print()
# print('Validation Steps: ' + str(n_validation_steps))
# print('Test Steps: ' + str(n_test_steps))
# print()
# print('Number of Epochs: ' + str(n_epochs))
# print()
# print('Number of Classes: ' + str(num_classes))    

# ##############################################################################################
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
# print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Prepare a directory to store all the checkpoints.
checkpoint_dir = './'+ checkpoint_no
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
keras_callbacks = [ModelCheckpoint(filepath = checkpoint_dir + '/' + model_name, 
                                   monitor='loss', save_best_only=True, mode='auto')]
# history = model.fit(trainX,
#                     trainY,
#                     steps_per_epoch=n_steps_per_epoch,
#                     epochs=n_epochs,
#                     batch_size=n_batch_size,
#                     validation_data=(valX, valY),
#                     validation_steps=n_validation_steps,
#                     callbacks=[keras_callbacks])
es = callbacks.EarlyStopping(monitor='loss', #val_accuracy #val_loss
                                   mode='auto',
                                   patience=20,
                                   restore_best_weights=True) # important - otherwise you just return the last weigths...

#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200, verbose=1, epsilon=1e-4, mode='auto')

# #model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
history = model.fit(trainX,
                    trainY,
                    callbacks=[es, keras_callbacks], #callbacks=[es],
                    epochs=n_epochs, 
                    validation_data=(valX,valY),
                    validation_steps=n_validation_steps,
                    batch_size=n_batch_size , #1 , 3 , 487 , 1461
                    shuffle=False,
                   # validation_split=0.2,
                    verbose=1)
hist_df = pd.DataFrame(history.history)
hist_df['epoch'] = hist_df.index + 1
cols = list(hist_df.columns)
cols = [cols[-1]] + cols[:-1]
hist_df = hist_df[cols]
hist_df.to_csv(checkpoint_no + '/' + 'history_df_' + model_name + '.csv')
# print(hist_df.head())
########################################################################################################

############################ Obtaining class assignments ####################################################################

class_assignment = dict(zip(y, encoded_Y))

df_temp = pd.DataFrame([class_assignment], columns=class_assignment.keys())
df_temp = df_temp.stack()
df_temp = pd.DataFrame(df_temp).reset_index().drop(['level_0'], axis=1)
df_temp.columns = ['Category', 'Allocated Number']

df_temp.to_csv(checkpoint_no + '/' + 'class_assignment_df_' + model_name + '.csv')

# print('Class assignment:')
# print(class_assignment)
##############################################################################################################################

################################# Validation #########################################################

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

########################################################################################################

###################################### Load best model ###################################################################

# #Loading the automatically saved model
model_reloaded = load_model(checkpoint_no + '/' + model_name)

#Saving the best model in the correct path and format
root_directory = os.getcwd()
checkpoint_dir = os.path.join(root_directory, checkpoint_no)
model_name_temp = os.path.join(checkpoint_dir, model_name + '.h5')
model_reloaded.save(model_name_temp)
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
# Deletion of the automatically created folder under Model Checkpoint File.
folder_name_temp = os.path.join(checkpoint_dir, model_name)
shutil.rmtree(folder_name_temp, ignore_errors=True)
best_model = load_model(model_name_temp)
# print(best_model)

#############################################################################################################################

################################## Model Testing #############################################################################

test_loss, test_acc = best_model.evaluate(testX,
                                          testY,
                                          steps=n_test_steps)
# print()
# print('Test Accuracy:', test_acc)

################################################################################################################################

########################################## Prediction #################################################################

y_pred = model.predict(testX)
# print(y_pred[:5])
# encoder_reload = pk.load(open(checkpoint_dir + '\\' + 'encoder.pkl','rb'))
re_transformed_y_pred = encoder.inverse_transform(y_pred)
re_tranformed_testY=encoder.inverse_transform(testY)
# # print(re_transformed_y_pred[:5])
# testX['re_transformed_y_pred'] = re_transformed_y_pred
# print(testX)
print(classification_report(testY.argmax(axis=1), y_pred.argmax(axis=1)))








# #### Best hyperparamters ####
# #{'batch_size': 30, 'epochs': 100, 'model__neurons_1': 20, 'model__neurons_2': 50, 'optimizer': 'rmsprop'}
# # del model
# # del Sequential
# from keras.models import Sequential
# model = Sequential()
# model.add(Dense(units=20, input_shape=(X_test.shape[1],), activation='relu')) # input shape is (features,)
# model.add(Dense(units=50, activation='relu')) # input shape is (features,)
# model.add(Dense(units=6, activation='softmax'))
# model.summary()

# # compile the model
# #https://keras.io/api/models/model_training_apis/
# model.compile(optimizer='RMSprop',
#               loss='categorical_crossentropy', # this is different instead of binary_crossentropy (for regular classification)
#               metrics=['accuracy'])

# import keras
# from keras.callbacks import EarlyStopping

# # early stopping callback
# # This callback will stop the training when there is no improvement in
# #https://keras.io/api/callbacks/early_stopping/
# es = keras.callbacks.EarlyStopping(monitor='loss', #val_accuracy #val_loss
#                                    mode='auto',
#                                    patience=20,
#                                    restore_best_weights=True) # important - otherwise you just return the last weigths...

# from keras.callbacks import ModelCheckpoint
# mcp_save = ModelCheckpoint('/content/drive/My Drive/best_model.keras', save_best_only=True, monitor='loss', mode='auto') #.mdl_wts.hdf5
# #reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=200, verbose=1, epsilon=1e-4, mode='auto')

# #model.fit(Xtr_more, Ytr_more, batch_size=batch_size, epochs=50, verbose=0, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)
# # now we just update our model fit call
# #https://keras.io/api/models/model_training_apis/
# history = model.fit(X_train,
#                     dummy_y_train,
#                     callbacks=[es, mcp_save], #callbacks=[es],
#                     epochs=1000, # you can set this to a big number!
#                     batch_size=30 , #1 , 3 , 487 , 1461
#                     shuffle=False,
#                    # validation_split=0.2,
#                     verbose=1)

# import matplotlib.pyplot as plt
# history_dict = history.history

# # learning curve
# # accuracy
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']

# # loss
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']

# # range of X (no. of epochs)
# epochs = range(1, len(acc) + 1)

# # plot
# # "r" is for "solid red line"
# plt.plot(epochs, acc, 'r', label='Training accuracy')
# # b is for "solid blue line"
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()

# history_dict = history.history
# #print(history_dict['accuracy'])
# print(max(history_dict['accuracy']))
# print(sum(history_dict['accuracy'])/len(history_dict['accuracy']))
# #print(max(history_dict['val_accuracy']))
# #print(sum(history_dict['val_accuracy'])/len(history_dict['val_accuracy']))

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# # load a saved model
# #from keras.models import load_model
# #saved_model = load_model('/content/drive/My Drive/best_model.keras')
# #model = saved_model

# preds = model.predict(X_test) # see how the model did!
# #print(preds)
# print(preds[0]) # i'm spreading that prediction across three nodes and they sum to 1
# print(np.sum(preds[0])) # sum it up! Should be 1

# matrix = confusion_matrix(dummy_y_test.argmax(axis=1), preds.argmax(axis=1))
# print(matrix)

# # more detail on how well things were predicted
# print(classification_report(dummy_y_test.argmax(axis=1), preds.argmax(axis=1)))

# from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
# print(accuracy_score(dummy_y_test.argmax(axis=1),preds.argmax(axis=1)))
# print(balanced_accuracy_score(dummy_y_test.argmax(axis=1),preds.argmax(axis=1)))
# print(f1_score(dummy_y_test.argmax(axis=1),preds.argmax(axis=1),average='macro'))
# print(classification_report(dummy_y_test.argmax(axis=1), preds.argmax(axis=1)))

# from sklearn.metrics import ConfusionMatrixDisplay
# fig, ax = plt.subplots(figsize=(8, 5))
# cmp = ConfusionMatrixDisplay(confusion_matrix(dummy_y_test.argmax(axis=1), preds.argmax(axis=1)),display_labels=["class_1", "class_2", "class_3","class_4","class_5","class_6"])
# cmp.plot(ax=ax)
# plt.show()

# #https://michael-fuchs-python.netlify.app/2021/02/23/nn-artificial-neural-network-for-multi-class-classfication/
# #https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# #https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

# from sklearn.model_selection import GridSearchCV


# #def create_model(neurons_1,neurons_2,activation,optimizer,dropout):
# def make_classification_ann(optimizer='rmsprop', neurons_1=50, neurons_2=20):
#     from keras.models import Sequential
#     from keras.layers import Dense

#    # Creating the classifier ANN model
#     classifier = Sequential()
#     classifier.add(Dense(units=neurons_1, input_dim=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=neurons_2, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='softmax'))
#     classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return classifier


# batch_size = [20,30,487]
# epochs = [100]
# Optimizer_Trial = ['Adam', 'rmsprop']
# Neurons_Trial = [20, 32, 100]
# Neurons_Trial1 = [10, 16, 50]

# # # Creating the classifier ANN
# param_grid = dict(optimizer=Optimizer_Trial, epochs=epochs, batch_size=batch_size, model__neurons_1=Neurons_Trial, model__neurons_2=Neurons_Trial1)
# classifierModel=KerasClassifier(make_classification_ann, verbose=0)
# #from scikeras.wrappers import KerasClassifier
# #from sklearn.metrics import make_scorer, f1_score
# #scorer = make_scorer(f1_score, average = 'weighted')
# grid_search=GridSearchCV(estimator=classifierModel, param_grid=param_grid, scoring='accuracy', cv=6)

# # ########################################

# # # Measuring how much time it took to find the best params
# import time
# StartTime=time.time()

# # # Running Grid Search for different paramenters
# ResultsData = grid_search.fit(X_train, dummy_y_train, verbose=1)

# EndTime=time.time()
# print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

# # printing the best parameters
# print('\n#### Best hyperparamters ####')
# print(grid_search.best_params_)
# ############### Total Time Taken:  98 Minutes #############
# #{'batch_size': 30, 'epochs': 100, 'model__neurons_1': 20, 'model__neurons_2': 50, 'optimizer': 'rmsprop'}
# # Printing the best parameter
# print(ResultsData.sort_values(by='Accuracy', ascending=False).head(1))

# from sklearn.model_selection import GridSearchCV
# # from scikeras.wrappers import KerasClassifier

# # Function to create model, required for KerasClassifier
# def create_model(optimizer='rmsprop', init='uniform'):
#     # create model
#     model = Sequential()
#     model.add(Dense(50, input_dim=8, kernel_initializer=init, activation='relu'))
#     model.add(Dense(20, kernel_initializer=init, activation='relu'))
#     model.add(Dense(6, kernel_initializer=init, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

# model = KerasClassifier(model=create_model, verbose=0)
# print(model.get_params().keys())
# # grid search epochs, batch size and optimizer
# optimizers = ['rmsprop', 'adam']
# init = ['uniform']
# epochs = [100]
# batches = [20,30, 487]
# param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, model__init=init)
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
# grid_result = grid.fit(X_train, dummy_y_test)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))