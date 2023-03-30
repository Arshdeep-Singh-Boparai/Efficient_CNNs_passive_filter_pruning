#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:58:48 2022

"""

import numpy as np
import tensorflow.keras as K
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model

#%%
def preprocess_data(X,Y):
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

#%% dataset load

(x_train, y_train) , (x_test, y_test) = K.datasets.cifar10.load_data()
print((x_train.shape, y_train.shape))
x_train, y_train = preprocess_data(x_train, y_train)
x_test , y_test  = preprocess_data(x_test, y_test)
print((x_train.shape, y_train.shape))


#%% pre-trained model load (unpruned or pruned model)


model = load_model('~/ResNet50_cifar10/pruned_ResNet50_cifar10_8308/pruned_ResNet50_cifar10.h5')

#%% evaluate

predicted_x = model.predict(x_test)
y_true = np.argmax(y_test,1)
y_pred = np.argmax(predicted_x,1)
C=confusion_matrix(y_true, y_pred)


acc=np.trace(C)*100/np.shape(y_pred)[0]

model.summary()
print(acc,'resnet50_CIFAR10_unpruned_300_epochs_accuracy')
