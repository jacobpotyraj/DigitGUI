# -*- coding: utf-8 -*-
"""
Created on Wed July 22 17:07:45 2020

@author: potyraj
"""

import numpy as np
from scipy import signal as sp
from Softmax import *
from ReLU import *
from Convv2 import *
from Pool import *


def CNN(W1, W5, Wo, x_train, ans_train):

   #This will cause the numberOfBatches to update weights after every batch
    batchSize = 100
    #batchSize = 50
    totalToTest = x_train.shape[0]
    numberOfBatches = int(totalToTest/batchSize)
    learningRate = .01 #alpha
    beta = .90 #momentum
    
    batchList2 = np.arange(0, totalToTest, batchSize)
    

    mm1 = np.zeros_like(W1)
    mm5 = np.zeros_like(W5)
    mm0 = np.zeros_like(Wo)
    
    print('Number of Batches: ', numberOfBatches)
    
    for batch in range(len(batchList2)):
            dW1 = np.zeros_like(W1)
            dW5 = np.zeros_like(W5)
            dWo = np.zeros_like(Wo)
            
            begin = batchList2[batch]
            print('Batch number: ', batch)
            for k in range(begin, begin+batchSize):
                        
                        image = x_train[k, :, :] #input 28x28   
                        ConvOut1 = Convv2(image,W1) #in: 10000x28x28, 9x9x20 out: 20x20x20// input * weights 
                        ReLuOut2 = ReLU(ConvOut1) # in: 20x20x20 out: 20x20x20
                        PoolOut3 = Pool(ReLuOut2) # in: 20x20x20 out: 10x10x20                     
                        FlattenPool4 = np.reshape(PoolOut3, (-1, 1)) # in: 10x10x20 out: 2000x1 (10*10*20=2000) "Flatten"
                        Dense = np.matmul(W5, FlattenPool4) # in: 100x2000, 2000x1 out: 100x1 "Dense"
                        ReLuOut5 = ReLU(Dense) # in: 100x1 out: 100x1
                        x  = np.matmul(Wo, ReLuOut5) # in: 100x1, 10x100 out: 10x1
                        ans  = Softmax(x);
                         
                        # one-hot encoding
                        output = np.zeros((10, 1))
                        output[ans_train[k]][0] = 1 
                        
                #####__Back-Prop__#######
                        
                        # calcs error by subtracting 10x1 ans array (ans_train) by 
                        # the 10x1 output array that hold the guesses the network made 
                        e = output - ans
                        delta = e
                        e5 = np.matmul(Wo.T, delta)    # the goal is to get back to the dims of ReLuOut5
                                                       # in: Wo 10x100 transposed to 100x10, 
                                                       # delta 10x1
                                                       # out: 100x1 =100x10 * 10x1 (ReLuOut5 = Wo*T * y) 
                                                       
                        delta5 = (ReLuOut5 > 0) * e5  # Turns value "on" or "off"
                                                      # (ReLuOut5 > 0) = if value of pixel (x,y) is > 0 then that
                                                      # value will be represented as a 1 else 0.
                                                      # ReLuOut5 is a vector of all nodes that were active upon  
                                                      # the outputs "decision" 
                                                      # e5 represents the amount of error 
                                                      # ReLuOut5 and e5 are multiplyed to come up with the 
                                                      # adjustments to be made for the next layer
                        e4 = np.matmul(W5.T, delta5)
                       
                        e3 = np.reshape(e4, PoolOut3.shape) # undoes the reshape made after the initial pooling layer
                        
                        e2 = np.zeros_like(ReLuOut2)             # shape of what came out of our Convv layer
                        W3 = np.ones_like(ReLuOut2) / (2*2)      # 20x20x20 Mean tensor
                        
                        for c in range(e2.shape[2]):
                            # takes a 1x1 in e3 and copies it into a 2x2 
                            # thereby expanding a 10x10x20 back into a 20x20x20 
                            # then multiplies each pixel by 1/4 (W3) 20x20x20 * 20x20x20
                            e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]
                            
                        delta2 = (ReLuOut2 > 0) * e2
                        
                        delta1_x = np.zeros_like(W1)
                       
                        for c in range(20):
                                delta1_x[:, :, c] = sp.convolve2d(image[:, :], np.rot90(delta2[:, :, c], 2), 'valid')
                                
                                
                        dW1 = dW1 + delta1_x
                        dW5 = dW5 + np.matmul(delta5, FlattenPool4.T)
                        dWo = dWo + np.matmul(delta, ReLuOut5.T)       
                        
                        dW1 = dW1 / batchSize
                        dW5 = dW5 / batchSize
                        dWo = dWo / batchSize
                        
                        mm1 = learningRate*dW1 + beta*mm1
                        W1        = W1 + mm1
                        
                        mm5 = learningRate*dW5 + beta*mm5
                        W5        = W5 + mm5
                        
                        mm0 = learningRate*dWo + beta*mm0 
                        Wo        = Wo + mm0
                
    return W1, W5, Wo
                    
                    
                    
                    
                    