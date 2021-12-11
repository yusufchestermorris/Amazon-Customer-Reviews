# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:55:57 2021

@author: Yusuf
"""
class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    
class Review:
   def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #score of 4 or 5
            return Sentiment.POSITIVE
   
   def __init__(self, text, score):
       self.text = text
       self.score = score
       self.sentiment = self.get_sentiment()  
           
class ReviewContainer:
    def __init__(self,reviews):
        self.reviews = reviews
        
    def evenly_distribute(self):
        negative = filter(lambda x: x.sentiment == Sentiment.NEGATIVE,self.reviews)
        Positive = filter(lambda x: x.sentiment == Sentiment.POSITIVE,self.reviews)
        
        print (negative[0].text)
        print
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
file_name = './Books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall'])) 
        
reviews[5].text

### Splitting data from the sample to test and train!!

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

cont = ReviewContainer(training)
cont.evenly_distrivute()

## checking the values output by data split 
# print(training[0].text)
# print(training[0].senitment)

### seperating text from thee sentiment with list comprehension

train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]


## cheking text and senitment have been split
train_x[0]
train_y[0]

#### bag of words training model

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

#print(train_x[0])
#print(train_x_vectors[0])

train_x_vectors
train_y


clf_svm = svm.SVC(kernel = 'linear')

clf_svm.fit(train_x_vectors, train_y)
    
#test_x[0]
#test_x_vectors[0]

## result from classifier for bag of words on 0th index
print(clf_svm.predict(test_x_vectors[0]))


#### decision Tree Model

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

clf_dec.predict(test_x_vectors[0])

#### Naive Bayes Model
# converting sparse array to dense array for NB(this does take up more of a memory footprint!!)
train_x_darray = train_x_vectors.todense()
test_x_darray = test_x_vectors.todense()

clf_gnb = GaussianNB()

clf_gnb.fit(train_x_darray, train_y)
clf_gnb.predict(test_x_darray[0])


#### Logistic Regression


clf_LR = LogisticRegression()
clf_LR.fit(train_x_vectors, train_y)

clf_LR.predict(test_x_vectors[0])


#### Evaluation

# print(clf_svm.score(test_x_vectors, test_y))
# print(clf_dec.score(test_x_vectors, test_y))
# print(clf_gnb.score(test_x_darray, test_y))
# print(clf_LR.score(test_x_vectors, test_y))

# f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
# f1_score(test_y, clf_dec.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
# f1_score(test_y, clf_gnb.predict(test_x_darray), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
# f1_score(test_y, clf_LR.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])

### Comments the evaluation seems to suggest that there is an issue with the data sampled for the ML as each model have scored well in POSITIVE but not in NUETRAL and NAGATIVE

### checking the amount of POSITIVE reviews in data

train_y.count(Sentiment.POSITIVE)
train_y.count(Sentiment.NEGATIVE)
### out of 670 reviews there are 552 POSITIVE ones and only 47 NEGATIVE which means that the models are going to be more biased towards a POSITIVE result

### To improve this new data needs to be used to prevent bias


