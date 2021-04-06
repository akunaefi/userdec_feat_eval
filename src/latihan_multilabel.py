# -*- coding: utf-8 -*-

'''
March 10, 2021

desc:
    Program ini untuk latihan problem multilabel classification menggunakan
    binary relevance, classifier chain, dan label powerset
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset


# PREPARE THE DATA

train_data = pd.read_csv('../data/train.csv')
#print(train_data.head())

# combine title and abstract into one column
train_data['Text']=train_data['TITLE']+' '+train_data['ABSTRACT']
train_data.drop(columns=['TITLE','ABSTRACT'], inplace=True)
# print(train_data.head(10))
print(train_data.columns)

# FUNCTIONS
stop_words = set(stopwords.words('english'))

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

#Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]"," ",text) 
    text = ' '.join(text.split()) 
    return text

#stemming
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

train_data['Text'] = train_data['Text'].apply(lambda x: remove_stopwords(x))
train_data['Text'] = train_data['Text'].apply(lambda x:clean_text(x))
train_data['Text'] = train_data['Text'].apply(stemming)


x_train, x_test, y_train, y_test = train_test_split(
train_data['Text'], train_data[train_data.columns[1:7]], test_size=0.3, random_state=0, shuffle=True)


# # Binary Relevance
# pipeline = Pipeline([
#                 ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 ('clf', BinaryRelevance(LogisticRegression(solver='sag'))),
#             ])
# pipeline.fit(x_train, y_train)
# predictions = pipeline.predict(x_test)

# Classifier Chain
pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', ClassifierChain(LogisticRegression(solver='sag'))),
            ])
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)


print('Accuracy = ', accuracy_score(y_test,predictions))
print('F1 score is ',f1_score(y_test, predictions, average="micro"))
print('Hamming Loss is ', hamming_loss(y_test, predictions))