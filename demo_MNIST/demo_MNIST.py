#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import tkinter as tk
from scipy.spatial import distance
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from PIL import  Image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tkinter import *
from PIL import Image, ImageTk
import tkinter as tk
import requests

#%%
HEIGHT = 300
WIDTH = 400

def test_function(entry):
	print("This is the entry:", entry)




def load_models():
    a= grab_and_assign_CNN_model()
    if a== "Unpruned":
        auto_path='/~/unpruned_VGG16_MNIST.h5'   #set path after downloading models from VGG16_MNIST Model
    elif a=="Pruned":
        auto_path='~/pruned_VGG16_MNIST.h5'

    CNN_model=load_model(auto_path)
        
    return CNN_model,a



    
    

def classification_result(xtest):
    mean = 34.52383
    std =  71.70004
    xtest = np.reshape(xtest,[1,32,32,3])
    xtest = (xtest-mean)/(std+1e-7)
    # xtest = np.reshape(xtest,[1,32,32,3])
    CNN_model,model_name =load_models()
#    label['text']="Model loaded"
    
    test_prob=CNN_model.predict(xtest)
    stringlist = [] 
    CNN_model.summary(print_fn=lambda x: stringlist.append(x))
    N_parm = stringlist[124]
#    label['text']="Predicting autencoder output"
    out_pred = np.argmax(test_prob)
    return out_pred, test_prob[0][out_pred], np.sort(test_prob), np.argsort(test_prob),model_name, N_parm


    


       
    
    
    

def load_audio(audio_path, sr=None):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=8000, mono=True,duration=10.0)
    sound_sample *= 256	

    label['text']="audio loaded"
    label['text']=classification_result(sound_sample)

    return sound_sample, sr




def display_input_image(img):
    image1 = Image.fromarray(img.astype(np.uint8))
    image2 = image1.resize((256, 256), Image.ANTIALIAS)
    test = ImageTk.PhotoImage(image2)
    label3 = tk.Label(image=test)
    label3.image = test
    label3.place(x=400, y=300)
    
    return


def display_model_summary():
    CNN_model = load_models()
    # label3['text']= CNN_model.summary()
    return 
    
    
    


def load_image_and_predict(image_path):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    img = np.loadtxt(image_path)
    img = np.reshape(img, [1,28,28])
    xtest = np.dstack([img]*3)
    xtest= xtest.reshape (-1,28,28,3)
    xtest = np.asarray([img_to_array(array_to_img(im, scale=False).resize((32,32))) for im in xtest]) 
    label['text']= "image loaded"
    display_input_image(img[0,:,:])
    # display_model_summary()
    category, confidence, sorted_prob, sorted_arg, model_name, N_parm = classification_result(xtest)
    label['text'] = 'Predicted digit is: ' + str(category) #+  'with' + str(confidence) 
    if model_name == 'Unpruned':
        label['text'] = 'Predicted digit with Unpruned is: ' + str(category)
        label2['text'] =  'Unpruned Top-3 predictions: ' + str(sorted_arg[0][6:10][::-1]) + ' with prob. ' + str(sorted_prob[0][6:10][::-1])
        label5['text'] = 'Unruned N/k: ' + N_parm
        
    else:
        label['text'] = 'Predicted digit with Pruned is: ' + str(category)
        label3['text'] =   'Pruned Top-3 predictions: ' + str(sorted_arg[0][6:10][::-1]) + ' with prob. ' + str(sorted_prob[0][6:10][::-1])
        label6['text'] = 'Pruned N/k: ' + N_parm

    return img

def grab_and_assign_CNN_model():
    chosen_option = var.get()
#    label_chosen_variable= Label(root, text=chosen_option)
#    label_chosen_variable.grid(row=100, column=80)
    return chosen_option

def grab_and_assign_Threshold():
    return var2.get()



def grab_and_assign_scene():
    chosen_option = var1.get()
#    label_chosen_variable= Label(root, text=chosen_option)
#    label_chosen_variable.grid(row=3, column=2)
    return chosen_option


def refresh():
    label2['text'] = [ ]
    label3['text'] = [ ]
    return

#%%
import os
os.chdir('~/MNIST_test_data_with_groundtruth')
root = tk.Tk()

root.title("|| Demonstration Handwritten digit classification || Anonymous submission || ")

root.configure(background='grey')
var = StringVar(root)
var.set("Select CNN Model")


drop_menu = OptionMenu(root, var,  "Unpruned", "Pruned", command=grab_and_assign_CNN_model)
drop_menu.grid(row=300, column=250)

label_left=Label(root, text="CNN model--->")
label_left.grid(row=300, column=200)


root.geometry('600x600')


frame = tk.Frame(root, bg='green', bd=15)
# frame.place(relx=0.5, rely=0.75, relwidth=0.8, relheight=0.05, anchor='n')
frame.place(relx=0.5, rely=0.1, relwidth=0.85, relheight=0.05, anchor='n')

entry = tk.Entry(frame, font=15)
entry.place(relwidth=3, relheight=2)


button = tk.Button(frame, text="Upload  & prediction ", font=7, command=lambda: load_image_and_predict(entry.get()))
button.place(relx=0.67, relheight=2, relwidth=0.34)



lower_frame = tk.Frame(root, bg='red', bd=10)
lower_frame.place(relx=0.5, rely=0.8, relwidth=0.9, relheight=0.1, anchor='n')
#/home/arshdeep/Desktop/intel_logo.png
label = tk.Label(lower_frame)

label.config(font=("Courier", 24))
label.place(relwidth=1, relheight=1)


lower_frame2 = tk.Frame(root, bg='yellow', bd=10)
lower_frame2.place(relx=0.5, rely=0.9, relwidth=1, relheight=0.1, anchor='n')

label2 = tk.Label(lower_frame2)
label2.config(font=("Courier", 24))
label2.place(x=1, y = 1)

lower_frame3 = tk.Frame(root, bg='green', bd=10)
lower_frame3.place(relx=0.5, rely=0.95, relwidth=1, relheight=0.1, anchor='n')

label3 = tk.Label(lower_frame3)
label3.config(font=("Courier", 24))
label3.place(x=1, y = 1)


button = tk.Button(lower_frame, text="Clear predictions", font=7, command=refresh)
button.place(relx=0.7, relheight=1, relwidth=0.3)



lower_frame5 = tk.Frame(root, bg='grey', bd=10)
lower_frame5.place(relx=0.75, rely=0.3, relwidth=0.4, relheight=0.051, anchor='n')

label5 = tk.Label(lower_frame5)
label5.config(font=("Courier", 24))
label5.place(x=1, y = 1)

lower_frame6 = tk.Frame(root, bg='grey', bd=10)
lower_frame6.place(relx=0.75, rely=0.35, relwidth=0.3, relheight=0.051, anchor='n')

label6 = tk.Label(lower_frame6)
label6.config(font=("Courier", 24))
label6.place(x=1, y = 1)
    
    
root.mainloop()
