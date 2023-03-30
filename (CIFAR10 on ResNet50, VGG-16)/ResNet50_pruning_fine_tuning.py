#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 18:29:20 2022
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import layer_utils
#from keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow
#%matplotlib inline

#import tensoflow.keras.backend as K
#K.set_image_data_format('channels_last')
#K.set_learning_phase(1)
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from sklearn.metrics import confusion_matrix

import os
from tensorflow.keras.models import load_model

#%%

def preprocess_data(X,Y):
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    
    return X


#%%

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)



    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    # X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
    #                     kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    
    return X


#%%




def ResNet50(input_shape=(32, 32, 3), classes=10):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    #X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    #X = Flatten()(X)
    #X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


#%%


def ResNet50_pruned(P, input_shape=(32, 32, 3),  classes=10):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    p0, p12, p36, p54, p72, p96, p114, p132, p150, p174, p192, p210, p228, p246, p264, p288, p306 = P
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(int(64*(1-p0)), (7, 7), strides=(2, 2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, int(64*(1-p12)), 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, int(64*(1-p36)), 256], stage=2, block='b')
    X = identity_block(X, 3, [64, int(64*(1-p54)), 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, int(128*(1-p72)), 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, int(128*(1-p96)), 512], stage=3, block='b')
    X = identity_block(X, 3, [128, int(128*(1-p114)), 512], stage=3, block='c')
    X = identity_block(X, 3, [128, int(128*(1-p132)), 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, int(256*(1-p150)), 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, int(256*(1-p174)), 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, int(256*(1-p192)), 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, int(256*(1-p210)), 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, int(256*(1-p228)), 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, int(256*(1-p246)), 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, int(512*(1-p264)), 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, int(512*(1-p288)), 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, int(512*(1-p306)), 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###
    # X = K.layers.GlobalAveragePooling2D()(X)
    # # X = K.layers.Flatten()(X)
    # X = K.layers.Dense(256, activation='relu')(X)
    # X = K.layers.Dense(10, activation='softmax')(X) 
    # output layer
    # X = Flatten()(X)
    # X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50_pruned')

    return model

#%%

(x_train, y_train) , (x_test, y_test) = K.datasets.cifar10.load_data()
print((x_train.shape, y_train.shape))
x_train, y_train = preprocess_data(x_train, y_train)
x_test , y_test  = preprocess_data(x_test, y_test)
print((x_train.shape, y_train.shape))


#%%  specify which convolutional layers (3 x 3) to prune

p = 0.50  # specify pruning ratio p = {25,50,75,90}%

p0 = p                    #stage 1
p12 = p                   #stage 2 (Conv block branch b 3 x 3) 
p36 = p                   #stage 2 (ID block 1 branch b 3 x 3) 
p54 = p                   #stage 2 (ID block 2 branch b 3 x 3) 

p72 = p                   #stage 3 (Conv block branch b 3 x 3) 
p96 = p                   #stage 3 (ID block 1 branch b 3 x 3) 
p114 = p                  #stage 2 (ID block 2 branch b 3 x 3) 
p132 = p                  #stage 2 (ID block 3 branch b 3 x 3) 

p150 = p                 #stage4 (Conv block branch b 3 x 3) 
p174 = p                 #stage 4 (ID block 1 branch b 3 x 3) 
p192= p                  #stage 4 (ID block 2 branch b 3 x 3) 
p210 = p                 #stage 4 (ID block 3 branch b 3 x 3) 
p228 = p                 #stage 4 (ID block 4 branch b 3 x 3) 
p246 = p                 #stage 4 (ID block 5 branch b 3 x 3) 

p264 = p                  #stage 5 (Conv block branch b 3 x 3) 
p288 = p                  #stage 5 (ID block 1 branch b 3 x 3) 
p306 = p                  #stage 5 (ID block 2 branch b 3 x 3) 


#stage 1

L_0 = sorted(np.load('/home/arshdeep/Pruning/SPL/ICLR_anonymize_code_test/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index0.npy')[int(np.ceil(64*p0)):64])


# stage 2
L_12 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index12.npy')[int(np.ceil(64*p12)):64])
L_36 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index36.npy')[int(np.ceil(64*p36)):64])
L_54 = sorted(np.load('~t/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index54.npy')[int(np.ceil(64*p54)):64])

# stage 3
L_72 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index72.npy')[int(np.ceil(128*p72)):128])
L_96 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index96.npy')[int(np.ceil(128*p96)):128])
L_114 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index114.npy')[int(np.ceil(128*p114)):128])
L_132 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index132.npy')[int(np.ceil(128*p132)):128])

# stage 4
L_150 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index150.npy')[int(np.ceil(256*p150)):256])
L_174 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index174.npy')[int(np.ceil(256*p174)):256])
L_192 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index192.npy')[int(np.ceil(256*p192)):256])
L_210 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index210.npy')[int(np.ceil(256*p210)):256])
L_228 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index228.npy')[int(np.ceil(256*p228)):256])
L_246 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index246.npy')[int(np.ceil(256*p246)):256])

# stage 5


L_264 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index264.npy')[int(np.ceil(512*p264)):512])
L_288 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index288.npy')[int(np.ceil(512*p288)):512])
L_306 = sorted(np.load('~/codes_after_rebuttal/importance_scores/ResNet50/Proposed_Pruning/sim_index306.npy')[int(np.ceil(512*p306)):512])


P = [p0, p12, p36, p54, p72, p96, p114, p132, p150, p174, p192, p210, p228, p246, p264, p288, p306]
#%%


#%% load pre-trained weights and eliminate redundant weights

model = load_model('~/ResNet50_cifar10/unpruned_ResNet50_cifar10_8337/unpruned_ResNet50_cifar10.h5') # follow the given link to download model
Z = model.get_weights()


W = [ ]

W.append(Z[0][:,:,:,L_0])  # middle conv layer
for i in range(1,6):
    W.append(Z[i][L_0])

W.append(Z[6][:,:,L_0,:])
W.extend(Z[7:12]) 


# Stage 2 middle layer pruning.....................................................................
# conv block
W.append(Z[12][:,:,:,L_12])  # middle con layer
for i in range(13,18):
    W.append(Z[i][L_12])     
    
    

W.append(Z[18][:,:,L_0,:])   # branch 1 as it is since stage 1 is pruned 

W.append(Z[19]) 


W.append(Z[20][:,:,L_12,:])  #' next conv channels' branch 2c
W.extend(Z[21:36])



#----ID block 1 stage 2------------------------------------------------------------------

W.append(Z[36][:,:,:,L_36])
for i in range(37,42):
    W.append(Z[i][L_36])  


# W.extend(Z[180:182]) 
W.append(Z[42][:,:,L_36,:])
W.extend(Z[43:54])

#--ID bloack 2 stage 2.............................

W.append(Z[54][:,:,:,L_54])
for i in range(55,60):
    W.append(Z[i][L_54]) 


W.append(Z[60][:,:,L_54,:])
W.extend(Z[61:72])


# ---------Stage 3 middle layer pruning..............----------------------
# conv block

W.append(Z[72][:,:,:,L_72])  # middle con layer
for i in range(73,78):
    W.append(Z[i][L_72])     
    
    

W.extend(Z[78:80])   # branch 1 as it is 
W.append(Z[80][:,:,L_72,:])  #' next conv channels' branch 2c
W.extend(Z[81:96])


# ID block 1 stage 3------------------------------


W.append(Z[96][:,:,:,L_96])
for i in range(97,102):
    W.append(Z[i][L_96])  


# W.extend(Z[180:182]) 
W.append(Z[102][:,:,L_96,:])
W.extend(Z[103:114])

#---ID 3 stage 3...........

W.append(Z[114][:,:,:,L_114])
for i in range(115,120):
    W.append(Z[i][L_114])  


# W.extend(Z[180:182]) 
W.append(Z[120][:,:,L_114,:])
W.extend(Z[121:132])

# ID 4 stage 4..........................


W.append(Z[132][:,:,:,L_132])
for i in range(133,138):
    W.append(Z[i][L_132])  


# W.extend(Z[180:182]) 
W.append(Z[138][:,:,L_132,:])
W.extend(Z[139:150])


# Stage 4 middel layer pruning in all blocks---------------------------------------------------------------------------
W.append(Z[150][:,:,:,L_150])  # middle con layer
for i in range(151,156):
    W.append(Z[i][L_150])     
    
    

W.extend(Z[156:158])   # branch 1 as it is 
W.append(Z[158][:,:,L_150,:])  #' next conv channels' branch 2c
W.extend(Z[159:174])


#-------------------Stage 4 ID block 1 (only 3 x 3)......................................
W.append(Z[174][:,:,:,L_174])
for i in range(175,180):
    W.append(Z[i][L_174])  


# W.extend(Z[180:182]) 
W.append(Z[180][:,:,L_174,:])
W.extend(Z[181:192])

#---------------------------------Stage 4 ID block 2------------------------------------------------------------------------------------


W.append(Z[192][:,:,:,L_192])
for i in range(193,198):
    W.append(Z[i][L_192]) 


W.append(Z[198][:,:,L_192,:])
W.extend(Z[199:210])



#------------------------------Stage 4 ID block 3.................................................
W.append(Z[210][:,:,:,L_210])
for i in range(211,216):
    W.append(Z[i][L_210]) 


W.append(Z[216][:,:,L_210,:])
W.extend(Z[217:228])


# ------------------------------Stage 4 ID block 4--------------------------------------------
W.append(Z[228][:,:,:,L_228])
for i in range(229,234):
    W.append(Z[i][L_228]) 


W.append(Z[234][:,:,L_228,:])
W.extend(Z[235:246])

#-------------------------------Stage 4 ID block 5-------------------------------------------
W.append(Z[246][:,:,:,L_246])
for i in range(247,252):
    W.append(Z[i][L_246]) 


W.append(Z[252][:,:,L_246,:])

W.extend(Z[253:264])

# stage 5--------------------------------------conv block ------------------------------------------------------------------------
W.append(Z[264][:,:,:,L_264])
for i in range(265,270):
    W.append(Z[i][L_264]) 
    

W.extend(Z[270:272]) 
W.append(Z[272][:,:,L_264,:])  
W.extend(Z[273:288])


#----------------------------------Stage 5 ID block 1
W.append(Z[288][:,:,:,L_288])
for i in range(289,294):
    W.append(Z[i][L_288]) 

W.append(Z[294][:,:,L_288,:])
W.extend(Z[295:306])


#---------------------------------------Stage 5 ID block 2
W.append(Z[306][:,:,:,L_306])
for i in range(307,312):
    W.append(Z[i][L_306]) 

    
W.append(Z[312][:,:,L_306,:])
W.extend(Z[313:])

#%%........................classifier.....


resnet_pruned = ResNet50_pruned(P, input_shape = (32, 32, 3), classes = 10)

model_pruned = K.models.Sequential()

model_pruned.add(resnet_pruned)
model_pruned.add(K.layers.GlobalAveragePooling2D())
#model_pruned.add(K.layers.Flatten())
#model.add(K.layers.BatchNormalization())
model_pruned.add(K.layers.Dense(256, activation='relu'))

model_pruned.add(K.layers.Dense(10, activation='softmax'))

model_pruned.set_weights(W) # load weights after eliminating pruned weights


#%% pruned model finetuning


check_point = K.callbacks.ModelCheckpoint(filepath="~/best_weights_SGD_100.h5py",
                                              monitor="val_acc", save_weights_only=True,
                                              save_best_only=True,
                                              )


lr_schedule = K.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)

model_pruned.compile(loss='categorical_crossentropy', optimizer=K.optimizers.SGD(learning_rate=lr_schedule), metrics=['accuracy'])


hist = model_pruned.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_data=(x_test, y_test),callbacks=[check_point])



#%% save model, weights, history
np.save("~/history_SGD_100.npy",hist.history)

model_pruned.load_weights("~/best_weights_SGD_100.h5py")

Z=model.get_weights()
np.save("~/best_weights_SGD_100.npy",Z)



model_pruned.save("~/pruned_ResNet50_cifar10.h5")




#%% predictions.....................................
predicted_x = model_pruned.predict(x_test)
y_true = np.argmax(y_test,1)
y_pred = np.argmax(predicted_x,1)
C=confusion_matrix(y_true, y_pred)
acc=np.trace(C)*100/np.shape(y_pred)[0]
model_pruned.summary()
print(acc,'resnet_fine_tuning_accuracy')


