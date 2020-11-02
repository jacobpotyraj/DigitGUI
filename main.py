# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:12:01 2020

@author: potyraj
"""
from tkinter import *

import numpy as np
import cv2


from DigitsGPU import NeuralNet


import tensorflow as tf
import cv2 
import numpy as np

from CNN import *
from Convv2 import *
from ReLU import *
from Pool import *
from Softmax import *

height = 420
width = 420
canvas = np.ones((420,420), dtype="uint8") * 255
canvas[0:width,0:height] = 0


class main():
    
    def __init__(self,master):
        
        self.master = master
        self.color_paint = 'white' #forground of tkinter canvas
        self.color_bg = 'black' #background of tkinter canvas
        self.old_x = None
        self.old_y = None
        self.penwidth = 15 #thickness of paint brush for tkinter canvas
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint) #drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        
    #draws on the canvas
    def paint(self,paintBrush):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,paintBrush.x,paintBrush.y,width=self.penwidth,fill=self.color_paint,capstyle=ROUND,smooth=True)

        self.old_x = paintBrush.x
        self.old_y = paintBrush.y                                                  #line color 255 = white, line thinkness of cv2 canvas      
        cv2.line(canvas,(self.old_x,self.old_y),(paintBrush.x,paintBrush.y),255,15)


    #reseting x and y points of contact
    def reset(self,paintBrush):    
        self.old_x = None
        self.old_y = None
    
    #clearing both tkinter and hidden canvas    
    def clear(self):
        #tkinter 
        self.c.delete(ALL)
        
        #redraws boarder lines after clearing board
        self.c.create_line(60,60,60,360,width=1,fill=self.color_paint)
        self.c.create_line(60,60,360,60,width=1,fill=self.color_paint)
        self.c.create_line(360,360,60,360,width=1,fill=self.color_paint)
        self.c.create_line(360,360,360,60,width=1,fill=self.color_paint)
        canvas[0:width,0:height] = 0
        
    def start(self):
        self.net = NeuralNet()
        self.net.runCNN()
        Label(self.controls, text="Done Training",font=('arial 18')).grid(row=3,column=0) 
    #takes user drawing from hidden canvas and submits it through the trained network
    def submit(self):
        self.image = canvas
        self.result = self.net.predict(self.image)
        Label(self.controls, text="Your number is: %d" %self.result,font=('arial 18')).grid(row=8,column=0)
        print("PREDICTION : ", self.result)
        cv2.imshow("Test Canvas", self.image)

    #holds all the labels, boxes and buttons
    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 10)
        Label(self.controls, text="Hold left mouse button down\nto draw any number 0-9.",font=('arial 18')).grid(row=0,column=0)   
        Label(self.controls, text="Draw inside the box. ",font=('arial 18')).grid(row=1,column=0)
        Button(self.controls, text='Start Training',font=('arial 12'),command=self.start).grid(row=2,column=0)
        Label(self.controls, text=" ",font=('arial 18')).grid(row=3,column=0)
        Button(self.controls, text='Clear',font=('arial 18'),command=self.clear).grid(row=4,column=0)     
        Label(self.controls, text=" ",font=('arial 18')).grid(row=5,column=0)        
        Button(self.controls, text='Submit',font=('arial 18'),command=self.submit).grid(row=6,column=0)
        Label(self.controls, text=" ",font=('arial 18')).grid(row=7,column=0)        
        Label(self.controls, text="Your number is: ",font=('arial 18')).grid(row=8,column=0)

        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=width,height=height,bg=self.color_bg,)
        
        #draws boarder lines
        self.c.create_line(60,60,60,360,width=1,fill=self.color_paint)
        self.c.create_line(60,60,360,60,width=1,fill=self.color_paint)
        self.c.create_line(360,360,60,360,width=1,fill=self.color_paint)
        self.c.create_line(360,360,360,60,width=1,fill=self.color_paint)
        self.c.pack(fill=BOTH)
        




if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()