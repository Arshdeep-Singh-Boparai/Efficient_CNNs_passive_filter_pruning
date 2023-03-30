#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
#%%

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
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
#import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix

import os
#%% weight load

model_unpruned = load_model('~/VGG16_cifar10/unpruned_model_vgg16_9358/unpruned_VGG16_cifar10_9358.h5')


#%% important index load
# specify pruning ratio p = {25,50,75,90}% for each convolutional layer

p = 0  

p1 = 0     # C1 layer 
p2 = 0
p3 = 0
p4 = 0
p5 = 0
p6 = 0
p7 = 0
p8 = 0
p9 =  p
p10 = p
p11 = p
p12 = p
p13 = p     # C13 layer 



L1=np.arange(0,int((1-p1)*64))#np.load('sim_index1.npy')#[32:64]
L2=np.arange(0,int((1-p2)*64))#np.load('sim_index2.npy')#np.arange(0,64)#
L3=np.arange(0,int((1-p3)*128))#np.load('sim_index3.npy')
L4=np.arange(0,int((1-p4)*128))#np.load('sim_index4.npy')
L5=np.arange(0,int((1-p5)*256))#np.load('sim_index5.npy')#np.arange(0,256)#
L6=np.arange(0,int((1-p6)*256))#np.load('sim_index6.npy')#np.arange(0,256)#
L7=np.arange(0,int((1-p7)*256))#
L8=np.arange(0,int((1-p8)*512))#
L9=np.arange(0,int((1-p9)*512))
L10=np.arange(0,int((1-p10)*512))
L11=np.arange(0,int((1-p11)*512))
L12=np.arange(0,int((1-p12)*512))
L13=np.arange(0,int((1-p13)*512))


L1=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index1.npy')[64-len(L1):64])
L2=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index2.npy')[64-len(L2):64])#np.arange(0,64)#
L3=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index3.npy')[128-len(L3):128])
L4=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index4.npy')[128-len(L4):128])
L5=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index5.npy')[256-len(L5):256])
L6=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index6.npy')[256-len(L6):256])
L7=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index7.npy')[256-len(L7):256])#[64:]
L8=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index8.npy')[512-len(L8):512])
L9=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index9.npy')[512-len(L9):512])#[256:]
L10=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index10.npy')[512-len(L10):512])#[256:]
L11=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index11.npy')[512-len(L11):512])#[256:]
L12=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index12.npy')[512-len(L12):512])#[256:]
L13=sorted(np.load('~/codes_after_rebuttal/importance_scores/VGG16/Proposed_Pruning/sim_index13.npy')[512-len(L13):512])#[256:]


Total_filter=len(L1)+len(L2)+len(L3)+len(L4)+len(L5)+len(L6)+len(L7)+len(L8)+len(L9)+len(L10)+len(L11)+len(L12)+len(L13)
print(Total_filter)
#%%

W=model_unpruned.get_weights()

W_pruned=[W[0][:,:,:,L1],W[1][L1],W[2][L1],W[3][L1],W[4][L1],W[5][L1],W[6][:,:,L1,:][:,:,:,L2],W[7][L2],W[8][L2],W[9][L2],W[10][L2],W[11][L2],	W[12][:,:,L2,:][:,:,:,L3],W[13][L3],W[14][L3],W[15][L3],W[16][L3],W[17][L3],W[18][:,:,L3,:][:,:,:,L4],W[19][L4],W[20][L4],W[21][L4],W[22][L4],W[23][L4],W[24][:,:,L4,:][:,:,:,L5],W[25][L5],W[26][L5],W[27][L5],W[28][L5],W[29][L5],W[30][:,:,L5,:][:,:,:,L6],W[31][L6],W[32][L6],W[33][L6],W[34][L6],W[35][L6],W[36][:,:,L6,:][:,:,:,L7],W[37][L7],W[38][L7],W[39][L7],W[40][L7],W[41][L7],W[42][:,:,L7,:][:,:,:,L8],W[43][L8],W[44][L8],W[45][L8],W[46][L8],W[47][L8],W[48][:,:,L8,:][:,:,:,L9],W[49][L9],W[50][L9],W[51][L9],W[52][L9],W[53][L9],W[54][:,:,L9,:][:,:,:,L10],W[55][L10],W[56][L10],W[57][L10],W[58][L10],W[59][L10],W[60][:,:,L10,:][:,:,:,L11],W[61][L11],W[62][L11],W[63][L11],W[64][L11],W[65][L11],W[66][:,:,L11,:][:,:,:,L12],W[67][L12],W[68][L12],W[69][L12],W[70][L12],W[71][L12],W[72][:,:,L12,:][:,:,:,L13],W[73][L13],W[74][L13],W[75][L13],W[76][L13],W[77][L13],W[78][L13,:],W[79],W[80],W[81],W[82],W[83],W[84],W[85]]
								



#%%

model = Sequential()
weight_decay = 0.0005

model.add(Conv2D(len(L1), (3, 3), padding='same',input_shape=[32,32,3],kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())       
model.add(Dropout(0.3))

model.add(Conv2D(len(L2), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(len(L3), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L4), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(len(L5), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L6), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L7), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(len(L8), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L9), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L10), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(len(L11), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(len(L12), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(len(L13), (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.set_weights(W_pruned)  # load pruned weights
model.summary()

#%%data load
# x_train=np.load('train_data.npy')
sgd = tf.keras.optimizers.SGD(lr=.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
#odel.compile(loss=tf.keras.losses.categorical_crossentropy,
 #            optimizer=tf.keras.optimizers.Adam(),
  #           metrics=['accuracy'])



print('model loaded')
model.summary()

#%%data load
(x_train, labels_train), (x_test, labels_test) = cifar10.load_data()

C = tf.constant(10, name = "C")

one_hot_matrix_test = tf.one_hot(labels_test, C, on_value = 1.0, off_value = 0.0, axis =-1)

one_hot_matrix_train = tf.one_hot(labels_train, C, on_value = 1.0, off_value = 0.0, axis =-1)
sess = tf.Session()

y_test = sess.run(one_hot_matrix_test)
y_train = sess.run(one_hot_matrix_train)
sess.close()

#%% 

mean = np.mean(x_train,axis=(0,1,2,3))
std = np.std(x_train, axis=(0, 1, 2, 3))
X_test = (x_test-mean)/(std+1e-7)
X_train = (x_train-mean)/(std+1e-7)


#%% pre-trained model testing

predicted_x = model.predict(X_test)
y_true = y_test#np.argmax(y_test,1)
y_pred = np.argmax(predicted_x,1)
C=confusion_matrix(labels_test, y_pred)

#%%
acc=np.trace(C)*100/np.shape(y_pred)[0]

print(acc, 'pruned',  np.shape(y_true),np.shape(y_pred),np.shape(X_train),np.shape(X_test))



#%%  training the model


checkpointer = ModelCheckpoint(filepath='~/best_weights.h5py',monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)
hist=model.fit(X_train,y_train,batch_size=128,epochs=100,verbose=1,validation_data=(X_test,y_test),callbacks=[checkpointer])

model.load_weights('~/best_weights.h5py')



model.save('~/pruned_VGG16_cifar10.h5')

np.save('~/history1.npy',hist.history)


#%% predictions

predicted_x = model.predict(X_test)
y_true = y_test#np.argmax(y_test,1)
y_pred = np.argmax(predicted_x,1)
C=confusion_matrix(labels_test, y_pred)

#%%
acc=np.trace(C)*100/np.shape(y_pred)[0]

model.summary()
print(acc,'  after finetuning')

