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
#import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import img_to_array, array_to_img
from sklearn.metrics import confusion_matrix

#%%
(xtrain,labels_train),(xtest, labels_test)= tf.keras.datasets.mnist.load_data()
xtrain=np.dstack([xtrain]*3)
xtest=np.dstack([xtest]*3)
xtrain = xtrain.reshape(-1, 28,28,3)
xtest= xtest.reshape (-1,28,28,3)
xtrain = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in xtrain])
xtest = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in xtest])

C = tf.constant(10, name = "C")

one_hot_matrix_test = tf.one_hot(labels_test, C, on_value = 1.0, off_value = 0.0, axis =-1)

one_hot_matrix_train = tf.one_hot(labels_train, C, on_value = 1.0, off_value = 0.0, axis =-1)
# sess = tf.Session()

y_test = one_hot_matrix_test.numpy()#sess.run(one_hot_matrix_test)
y_train = one_hot_matrix_train.numpy() #sess.run(one_hot_matrix_train)
# sess.close()

#%% 

mean = np.mean(xtrain,axis=(0,1,2,3))
std = np.std(xtrain, axis=(0, 1, 2, 3))
X_test = (xtest-mean)/(std+1e-7)
X_train = (xtrain-mean)/(std+1e-7)

model =load_model(~/VGG16_MNIST/unpruned_model_99.49/unpruned_VGG16_MNIST.h5') ## load the unpruned model from given link 
#%% pre-trained model testing

predicted_x = model.predict(X_test)
y_true = y_test#np.argmax(y_test,1)
y_pred = np.argmax(predicted_x,1)
C=confusion_matrix(labels_test, y_pred)

#%%
acc=np.trace(C)*100/np.shape(y_pred)[0]

print(acc, 'unpruned')
