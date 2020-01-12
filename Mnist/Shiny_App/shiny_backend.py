# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:02:22 2020

@author: Josh
"""

#used to predict for R shiny

import numpy as np
import pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow import keras

def predict_pixels(pixels):
    
    pixels = pixels.reshape(1, 28, 28, 1)
    
    neural_net = keras.models.load_model("conv_nn.h5")
    neural_net.compile(optimizer = 'adam',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    
    predictions = neural_net.predict_classes(pixels)
    
    
    return predictions

def pixel_probs(pixels):
    
    pixels = pixels.reshape(1, 28, 28, 1)
    
    neural_net = keras.models.load_model("conv_nn.h5")
    neural_net.compile(optimizer = 'adam',
                       loss = 'categorical_crossentropy',
                       metrics = ['accuracy'])
    
    probs = neural_net.predict(pixels)
    
    return probs

def make_fig(X, y):
    
    fig, ax = plt.subplots()
    
    ax.plot(X, y, 'k', linewidth = "40")
    plt.axis('off')
    
    return fig

def get_pixels(fig):
    
    fig.tight_layout(pad = 0)
    fig.canvas.draw()
    
    
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype = np.uint8)
    
    data = buf.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    
    data_gray = (data @ [.2989, .5870, .1140])
    
    data_resized = resize(data_gray, (28, 28, 1)) #resizes image to 28x28x1
    
    data_scaled = (255 - data_resized) / 255.0 #flips to black-white and scales data
    
    plt.close()
    
    return data_scaled