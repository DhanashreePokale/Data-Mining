#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:29:01 2017

@author: dhanashreepokale
"""

############# Random Forest for Handwriting Recognition #############3

######## Import Data and Explore ##############
# Import libraries and dataset from Scikit-Learn
import matplotlib.pyplot as plt
# Import datasets
from sklearn import datasets, svm, metrics
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


########### Sampling Data ############
# we will now begin with model building part by first importing ensemble library of sklearn
import random
from sklearn import ensemble

# we define the variables first
n_samples = len(digits.images) # to keep track of how many images we have to perform classification on
x = digits.images.reshape((n_samples, -1)) # X are the images on which model would be trained
y = digits.target

#create random indices

sample_index = random.sample(range(len(x)),len(x)//5) # 20 80 split sampling is done here. 20% in training and 80% in validation
valid_index = [i for i in range(len(x)) if i not in sample_index]

# sample & validation images
sample_images = [x[i] for i in sample_index]
valid_images = [x[i] for i in valid_index]

# sample & validation targets
sample_targets = [y[i] for i in sample_index]
valid_targets = [y[i] for i in valid_index]


############ Random Forest Classifier ##############

# using random tree classifier
classifier = ensemble.RandomForestClassifier()

# fit model with sample data
classifier.fit(sample_images, sample_targets)


############ Prediction ###########################
# prediction on validation data
score = classifier.score(valid_images, valid_targets)
print('Random Tree Classifier Output: \n')
print('Score\t' + str(score))

############## Verification #######################
# to verify this 
i=0

pl.gray()
pl.matshow(digits.images[i])
pl.show()
classifier.predict(x[i])


############## Misclassification ##################
# confusion matrix
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

