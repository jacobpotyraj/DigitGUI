# -*- coding: utf-8 -*-
"""
Created on Fri July 24 14:52:47 2020

@author: potyraj
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
import numpy as np
import cv2


class NeuralNet(object):

    
     def runCNN(self):
      # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # reshape to be [samples][width][height][channels]

        # feature scaling and normalization
        self.training_images = X_train.reshape((60000, 28 , 28, 1)).astype('float32') / 255
        self.training_targets = to_categorical(y_train)

        self.test_images = X_test.reshape((10000, 28 , 28, 1)).astype('float32') / 255
        self.test_targets = to_categorical(y_test)

        self.input_shape = (self.training_images.shape[1],)
        
       	# create model
       	self.model = Sequential()
       	self.model.add(Conv2D(20, (9, 9), activation='relu'))
       	self.model.add(MaxPooling2D(pool_size=(2,2)))

       	self.model.add(Dropout(0.3))
       	self.model.add(Flatten())
       	self.model.add(Dense(100, activation='relu'))
       	
       	self.model.add(Dense(10, activation='softmax'))
       	# Compile model
        opt = SGD(lr=0.01, momentum=0.9)
       	self.model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])

        # build the model
      
        # Fit the model
        self.model.fit(self.training_images, self.training_targets,  epochs=10, batch_size=100, verbose=0)
   

     def predict(self, image):
        input = cv2.resize(image, (28 , 28)).reshape((28, 28, 1)).astype('float32') / 255

        return self.model.predict_classes(np.array([input]))

