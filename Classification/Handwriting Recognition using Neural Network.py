#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:18:02 2017

@author: dhanashreepokale
"""

# Import libraries and dataset from Scikit-Learn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


############### LOAD DATA AND EXPLORE #################
# Import datasets
from sklearn import datasets
digits = datasets.load_digits()


# Analyse a sample image - each element represents the pixel of our greyscale image. 
import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()


# lets see how python sees this image
digits.images[0]
# each of this number is a grey scale pixel. 0 - black; 255- white
# the value ranges from 0 to 255 for an 8 bit pixel


# visualize first 15 images

images_labels = list(zip(digits.images,digits.target)) 
#images are the pictures of human handwriting & targets are the meaning of the numbers
plt.figure(figsize=(5,5))
# here we do a 5 by 5 matrix plot for 15 images
for index, (image, label) in enumerate(images_labels[:15]):
    plt.subplot(3,5,index + 1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r, interpolation = 'nearest')
    plt.title('%i' %label)


# we will now begin with model building part by first importing ensemble library of sklearn
import random
from sklearn import ensemble

# we define the variables first
n_samples = len(digits.images) # to keep track of how many images we have to perform classification on
x = digits.images.reshape((n_samples, -1)) # X are the images on which model would be trained
y = digits.target

############### SAMPLING ################
#create random indices

sample_index = random.sample(range(len(x)),len(x)//5) # 20 80 split sampling is done here. 20% in training and 80% in validation
valid_index = [i for i in range(len(x)) if i not in sample_index]

# sample & validation images
sample_images = [x[i] for i in sample_index]
valid_images = [x[i] for i in valid_index]

# sample & validation targets
sample_targets = [y[i] for i in sample_index]
valid_targets = [y[i] for i in valid_index]

############### MODEL BUILDIND - NEURAL NETWORK ################
# define neural networl model
# specify activation function at every node of hidden layer
# specify solver as stochastic gradient descent
# specify hidden layer size as 50 which is approximately 2/3 of total number of attributes(64)
nn = MLPClassifier(activation = 'logistic', solver = 'sgd', hidden_layer_sizes = (50), random_state =1)

# fit the defined model on sample/training dataset
nn.fit(sample_images,sample_targets)


############### PREDICTION #################
# prediction is made on validation data set
pred = nn.predict(valid_images)


# find length of validation target
len(valid_targets)

# initialize the counter for storing predicted values
count = 0

# start a loop that runs through all the predicted values 
# and increments the counter wherever predicted value 
# matches actual target value
for i in range(len(pred)):
    if pred[i] == valid_targets[i]:
        count = count+1

# print the count
count

# find the accuracy
count/len(valid_targets)

############### VERIFICATION #################
import pylab as pl
pl.gray()
pl.matshow(digits.images[0])
pl.show()
pl.gray()
pl.matshow(digits.images[i])
pl.show()
nn.predict(x[i])
