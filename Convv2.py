# -*- coding: utf-8 -*-
"""
Created on Wed July 22 17:07:55 2020

@author: potyraj
"""

import numpy as np
from scipy import signal as sp
from timeit import default_timer as timer

def Convv2(x, W):
    Wrow, Wcol, numFilters = W.shape
    x = x.reshape(x.shape[0:2])
    xrow, xcol = x.shape
    yrow = xrow - Wrow + 1
    ycol = xcol - Wcol + 1
    manualConvOutput = np.zeros((yrow, ycol, numFilters))

    for k in range(0,numFilters):
       
        weights = W[:,:,k]

        for j in range(0,yrow):
           
            for i in range(0,ycol):
                
                    jj = j+Wrow 
                    ii = i+Wcol
                
                    manualConvOutput[j,i,k] = (x[j:jj,i:ii]*weights).sum()
            
    return manualConvOutput
