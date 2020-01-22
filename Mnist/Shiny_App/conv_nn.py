# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:51:47 2020

@author: Josh
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

(Xtrain, ytrain), (Xtest, ytest) = keras.datasets.mnist.load_data()

Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2], 1).astype(np.float64)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1).astype(np.float64)


Xtrain = Xtrain / 255.0
Xtest = Xtest / 255.0

ytrain = keras.utils.to_categorical(ytrain, 10)
ytest = keras.utils.to_categorical(ytest, 10)

model = keras.Sequential([
    
    keras.layers.Conv2D(128, (3, 3), input_shape = (Xtrain.shape[1], Xtrain.shape[2], 1), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(196, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(pool_size = (2, 2)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Flatten(),
    keras.layers.Dense(380, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(380, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation = 'softmax')
    ])

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(Xtrain, ytrain, validation_data = (Xtest, ytest), epochs = 25, batch_size = 16)

keras.models.save_model(model, "F:/School/MachineLearning_AndrewNg/Shiny_App/conv_nn.h5")
