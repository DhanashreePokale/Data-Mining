#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 23:38:20 2017

@author: dhanashreepokale
"""

################ MOVIE REVIEWS - SENTIMENT ANALYSIS ##############

################ IMPORT DATA AND EXPLORE #########################
# Importing nltk & random package
import nltk
import random
# Imporing the dataset & stopwords corpus
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews


# Creating documents list which stores file name and its category as pos or neg
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# we randomly shuffle the documents before creating training and testing datasets
random.shuffle(documents)


# randomly chosing 40th document to see its content
print(documents[40])

# Listing categories
movie_reviews.categories()

# Listing unique file ids
movie_reviews.fileids()

# Finding out number of categories
len(movie_reviews.categories())
# Finding out number of distinct file ids
len(movie_reviews.fileids())

# here we define a function to find out useful words such that they are not stop words from the English language
# We also segregate these words and create our own dictionary to store all such words in a data object called dict
def create_word_features(words):
    useful_words = [word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


# Here we create a new vector for movies having negative reviews
# the for loop is for scanning every fileid that belongs to neg category
# iteratively for all neg file ids words are scanned & stored into 'words'
# all such words are then appended into the 'neg_reviews' vector
# As a result, we get a list of all words which are used in negative reviews
# We use the user defined function 'create_word_features' for storing only useful words and not stop words
neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

# Just as neg_reviews, we create a vector to store words from positive reviews
pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))


# To verify the vector contents we print the length of neg_reviews; 
# which should come out to be 1000 as there are exactly 1000 neg reviews
print(len(neg_reviews))

# to check the contents of neg_reviews, we run following command and we expect 1000 lists of lists of words
neg_reviews   

# Create training and testing dataset
train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]
print(len(train_set),  len(test_set))


#importing necessary packages
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
import os

#setting up working directory
os.getcwd()
os.chdir("/Users/dhanashreepokale/Downloads/Data Mining/tm/data")
df = pd.read_csv("movie_reviews_n.txt", sep='\t',names=['content','polarity'])
df.head()

#TFIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase = True, strip_accents = 'ascii', stop_words= stopset)

#in this case, our dependent variable witll be polarity as 0(din't like the movie) or 1 (liked the movie)
y=df.polarity

#convert df.tsv from text to features
X= vectorizer.fit_transform(df.content)

#observationsx unique words
print(y.shape)
print(X.shape)

#Test Train Split as usual
X_train, X_test, y_train,y_test = train_test_split(X,y, random_state=42)

#we will train a naive bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

################## PIPELINE #######################

#importing required packages
import sklearn.pipeline
from sklearn import ensemble
​
#specifying feature selection technique and its parameters
select = sklearn.feature_selection.SelectKBest(k=100)
#specifying the classifier
clf = sklearn.ensemble.RandomForestClassifier()
​

#creating a steps object to store above mentioned techniques
steps = [('feature_selection', select),
        ('random_forest', clf)]
​
# using pipeline for tightening up the steps code
pipeline = sklearn.pipeline.Pipeline(steps)


################## sampling #######################
X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)
​

################## MODEL FITTING & PREDICTION REPORT #######################
### fit your pipeline on X_train and y_train
pipeline.fit( X_train, y_train )
### call pipeline.predict() on your X_test data to make a set of test predictions
y_prediction = pipeline.predict( X_test )
### test your predictions using sklearn.classification_report()
report = sklearn.metrics.classification_report( y_test, y_prediction )
### and print the report
print(report)

######## GRID SEARCH CV ##########################
# importing the grid search package
import sklearn.grid_search

# defining the feature selection parameters and random forest estimators along with sample split
parameters = dict(feature_selection__k=[100, 200], 
              random_forest__n_estimators=[50, 100, 200],
              random_forest__min_samples_split=[2, 3, 4, 5, 10])

# using gridsearchcv and pipeline built above, we pass parameters defined in previous command
cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

# this cv is used to fit training data
cv.fit(X_train, y_train)
# same cv is used to predict test data
y_predictions = cv.predict(X_test)
# a report stores then the classification table
report = sklearn.metrics.classification_report( y_test, y_predictions )
