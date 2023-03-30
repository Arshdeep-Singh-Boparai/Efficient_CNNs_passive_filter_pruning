
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

import os


#%%
'''
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



W_dcas =  np.load('~/unpruned_model_weights.npy', allow_pickle=True)
model.set_weights(W_dcas)
'''

model = load_model(~/DCASE21_Net/unpruned_model_DCASE21_Net_48.58/unpruned_model.h5') # for pruned model, please load pruned_model.h5 [from the given link]

#%%  DATA load (from the given link)


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

