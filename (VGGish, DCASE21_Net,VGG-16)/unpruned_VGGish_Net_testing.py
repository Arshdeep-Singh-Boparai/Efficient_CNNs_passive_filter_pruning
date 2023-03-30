#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py    
import numpy as np     

import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (BatchNormalization, SeparableConv2D, Conv2D,MaxPooling2D,  ZeroPadding2D,AveragePooling2D, Activation, Flatten, Dropout, Dense)
from tensorflow.keras import backend as K


from tensorflow.keras.callbacks import ModelCheckpoint
import os
from numpy import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras.models import load_model
from sklearn.metrics import log_loss



#%% data load (from given link)

#%%
x_test=np.load('~/test_data.npy')
labels_test=np.load('~/labels_test.npy')


C = tf.constant(10, name = "C")
      
one_hot_matrix_test = tf.one_hot(labels_test, C, on_value = 1.0, off_value = 0.0, axis =-1)
      
# one_hot_matrix_train = tf.one_hot(labels_train, C, on_value = 1.0, off_value = 0.0, axis =-1)  
# sess = tf.Session()
  
# y_test = sess.run(one_hot_matrix_test)
# y_train = sess.run(one_hot_matrix_train) 
# sess.close()

y_test = one_hot_matrix_test.numpy()

# print(np.shape(x_train),np.shape(x_test))

x_test = np.reshape(x_test, [2518,19,96,64,1])


#%%
# nessi.get_keras_size(model)


model_all = load_model('~/VGGish_Net/unpruned_model_VGGish_64.69/unpruned_VGGish.h5') # [load pruned/unpruned model from the given links]

#%% Baseline testing

X_test = np.reshape(x_test, [2518*19,96,64,1])


score=model_all.predict(X_test)
#%%
num_segment=19;
pred_prob=[]
t=0
num=int(np.size(labels_test)/num_segment);
for i in range(num):
    pred=np.sum(score[t:t+num_segment,:],0)
    pred_prob.append(pred)
    t=t+num_segment


pred_prob=np.argmax((np.asarray(pred_prob)),1)

k=0
true_labels=[]
for i in range(num):
	true_labels.append(labels_test[k])
	k=k+num_segment


true_label=np.asarray(true_labels)
asd=confusion_matrix(true_label,pred_prob);
accu=(np.trace(asd)/np.size(true_label))*100;
print(accu,'pruned model accuracy')

