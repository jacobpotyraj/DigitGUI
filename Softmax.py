# -*- coding: utf-8 -*-
"""
Created on Wed July 22 17:09:15 2020

@author: potyraj
"""

import numpy as np


def Softmax(x):
    x  = np.subtract(x, np.max(x))        # prevent overflow
    ex = np.exp(x)
    
    return ex / np.sum(ex)