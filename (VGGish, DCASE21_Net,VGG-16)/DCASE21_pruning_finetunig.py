#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%
import tensorflow as tf
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
# from tensorflow.keras.layers.core import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix

from sklearn.metrics import log_loss
import os


#%% build a model

input_shape=(40,500,1)
##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(16, kernel_size=(7, 7),padding='same',input_shape=input_shape))

model.add(BatchNormalization(axis=-1)) #layer2
convout1= Activation('relu')
model.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

model.add(Conv2D(16, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Dropout(0.30))



#''''''''''''''''''''''''''''''''''''''''''''''

model.add(Conv2D(32, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(4, 100)))

model.add(Dropout(0.30))


model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(10, activation='softmax'))


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

#%%


# Z=model.get_weights()

W_dcas =  np.load('~/unpruned_model_weights.npy', allow_pickle=True)   #Folder: /DCASE21_Net/unpruned_model_DCASE21_Net_48.58/

model.set_weights(W_dcas)

#%%  DATA load (Folder: ~/DCASE21_Net/dataset)


x_train=np.load('~/X_train.npy')
x_test=np.load('~/X_test.npy')
labels_test=np.load('~/Y_test.npy')
labels_train=np.load('~/Y_train.npy')



C = tf.constant(10, name = "C")

one_hot_matrix_test = tf.one_hot(labels_test, C, on_value = 1.0, off_value = 0.0, axis =-1)

one_hot_matrix_train = tf.one_hot(labels_train, C, on_value = 1.0, off_value = 0.0, axis =-1)
sess = tf.Session()

y_test = sess.run(one_hot_matrix_test)
y_train = sess.run(one_hot_matrix_train)
sess.close()



pred_label=model.predict(x_test)


pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')


logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label, normalize=True)

print(logloss_overall)



#%% important index load at a given layer-wise pruning ration (p1,p2 and p3) 

p1 = 0.5
p2 = 0.5
p3 = 0.5
L1=np.arange(0,np.ceil((1-p1)*16))#
L2=np.arange(0,np.ceil((1-p2)*16))#
L3=np.arange(0,np.ceil((1-p3)*32))

L1=sorted(np.load('~/important_index/PCS_Pruning/sim_index1.npy')[16-len(L1):16]) #indexes computed using Proposed_pruning.py script and saved in "importance_scores" folder.

L2= sorted(np.load('~/important_index/PCS_Pruning/sim_index2.npy')[16-len(L2):16])

L3 =sorted(np.load('~/important_index/PCS_Pruning/sim_index3.npy')[32-len(L3):32])


D3=L3  #list of indexes to be removed from the dense layer

w_f=[]
for i in range(len(L3)):
        w_f.append(list(range(D3[i]*2,D3[i]*2+2)))

w_f=np.hstack(w_f)


	

Total_filter=len(L1)+len(L2)+len(L3)#+len(L4)+len(L5)+len(L6)+len(L7)+len(L8)+len(L9)+len(L10)+len(L11)+len(L12)+len(L13)


W=W_dcas



W_pruned=[W[0][:,:,:,L1],W[1][L1],W[2][L1],W[3][L1],W[4][L1],W[5][L1],W[6][:,:,L1,:][:,:,:,L2],W[7][L2],W[8][L2],W[9][L2],W[10][L2],W[11][L2],	W[12][:,:,L2,:][:,:,:,L3],W[13][L3],W[14][L3],W[15][L3],W[16][L3],W[17][L3],W[18][w_f,:],W[19],W[20],W[21]]
	
#%% Obtaining pruned model

input_shape=(40,500,1)
##model building
model1 = Sequential()
#convolutional layer with rectified linear unit activation
layer_C1= Conv2D(len(L1), kernel_size=(7, 7),padding='same',input_shape=input_shape)
layer_C1.trainable =True
model1.add(layer_C1)
layer_BN1 = BatchNormalization(axis=-1)
layer_BN1.trainable = True
model1.add(layer_BN1) #layer2


convout1= Activation('relu')
# convout1.trainable= True
model1.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

layer_C2=Conv2D(len(L2), kernel_size=(7, 7),padding='same')
layer_C2.trainable= True
model1.add(layer_C2)

layer_BN2= BatchNormalization()#layer2
layer_BN2.trainable=  True
model1.add(layer_BN2)


convout2= Activation('relu')
model1.add(convout2) #laye

model1.add(MaxPooling2D(pool_size=(5, 5)))

model1.add(Dropout(0.30))



layer_C3=Conv2D(len(L3), kernel_size=(7, 7),padding='same')
layer_C3.trainable= True
model1.add(layer_C3)

layer_BN3 =  BatchNormalization() #layer2
layer_BN3.trainable= True
model1.add(layer_BN3)


convout3= Activation('relu')
model1.add(convout3) #laye

model1.add(MaxPooling2D(pool_size=(4, 100)))

model1.add(Dropout(0.30))

# model1.set_weights(W_dcas[0:18])

model1.add(Flatten())

model1.add(Dense(100,activation='relu'))
model1.add(Dropout(0.30))

model1.add(Dense(10, activation='softmax'))


model1.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


model1.set_weights(W_pruned)


model1.summary()



pred_label=model1.predict(x_test)



# model1.save('L1_pruning_Pruned_L123_43.94.h5')

pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'pruned_accuracy')
# model1.summary()

#%% finetuning of the pruned model.....
checkpointer = ModelCheckpoint(filepath='~/best_weights_dcase2021.h5py',monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)
hist=model1.fit(x_train, labels_train,batch_size=32,epochs=100,verbose=1,validation_data=(x_test, y_test),callbacks=[checkpointer])


model1.load_weights('~/best_weights_dcase2021.h5py')

model1.save('~/best_weights_dcase2021.h5')

np.save('~/history1.npy',hist.history)
pred_label=model1.predict(x_test)
pred=np.argmax(pred_label,1)
asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
model1.summary()




