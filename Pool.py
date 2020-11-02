# -*- coding: utf-8 -*-
"""
Created on Wed July 22 17:08:57 2020

@author: potyraj
"""
import numpy as np
#import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
#import skimage.measure 
import scipy.signal as sp
#2x2 mean pooling


def Pool(x):
    
    xrow, xcol, numFilters = x.shape[0:3]
   
    yrow = xrow//2
    ycol = xcol//2
    y = np.zeros((int(yrow), int(ycol), numFilters))
  

    manualPoolOutput = np.zeros((int(yrow), int(ycol), numFilters))
  
    for k in np.arange(numFilters): 
        z = 2 #size of filter 2x2 ZxZ
        kernel = np.ones([z,z])/(z**2)

        for r in range(y.shape[0]):
           
            for c in range(y.shape[1]):
                manualPoolOutput[r,c,k] = ((kernel*x[r*2:r*2+2,c*2:c*2+2,k]).sum())
         
        plt.imshow(manualPoolOutput[:,:,k], cmap='Greys')           
  
    return manualPoolOutput