# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.snowball import SnowballStemmer
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

# import the data
train_data = pd.read_csv('../data/kaggle/train_1000_xx.csv')
# train_data = pd.read_csv('../data/0_pusing_features.csv')
test_data = pd.read_csv('../data/kaggle/test.csv')


# print(train_data_1000.shape)
# print(test_data.shape)

# # explore the data
# x=train_data.iloc[:,3:].sum()
# rowsums=train_data.iloc[:,2:].sum(axis=1)
# no_label_count = 0
# for sum in rowsums.items():
#     if sum==0:
#         no_label_count +=1

# print("Total number of articles = ",len(train_data))
# print("Total number of articles without label = ",no_label_count)
# print("Total labels = ",x.sum())


# # check for missing value
# print("Check for missing values in Train dataset")
# print(train_data.isnull().sum().sum())
# print("Check for missing values in Test dataset")
# null_check=test_data.isnull().sum()
# print(null_check)

# gabungkan kolom title dan abstract
# train_data['Text']=train_data['TITLE']+' '+train_data['ABSTRACT']
# train_data.drop(columns=['TITLE','ABSTRACT'], inplace=True)
# print(train_data.head(10))


#Remove Stopwords
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

train_data['Text'] = train_data['Text'].apply(lambda x: remove_stopwords(x))


# stemming
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

train_data['Text'] = train_data['Text'].apply(stemming)


# define category
# categories=['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'] 
categories = ['acquiring','recommend','request','rating','relinquish','others']
# print(train_data[categories].head())


#split the data

x_train, x_test, y_train, y_test = train_test_split(train_data['Text'], train_data[categories], test_size=0.2, random_state=40, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(train_data['lemmatized_review'], train_data[categories], test_size=0.2, random_state=40, shuffle=True)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# using binary relevance

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', BinaryRelevance(MultinomialNB())),
            ])

# train
pipeline.fit(x_train, y_train)

# predict
predictions = pipeline.predict(x_test)


# from sklearn.metrics import accuracy_score
print('Accuracy = ', accuracy_score(y_test,predictions))
print('F1 score is ',f1_score(y_test, predictions, average="micro"))
print('Hamming Loss is ', hamming_loss(y_test, predictions))


# # using classifier chains with MultinomialNB


# pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', ClassifierChain(MultinomialNB())),
#             ])

# # train
# pipeline.fit(x_train, y_train)

# # predict
# predictions = pipeline.predict(x_test)

# print('Accuracy = ', accuracy_score(y_test,predictions))
# print('F1 score is ',f1_score(y_test, predictions, average="micro"))
# print('Hamming Loss is ', hamming_loss(y_test, predictions))