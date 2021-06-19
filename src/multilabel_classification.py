# -*- coding: utf-8 -*-

'''
April 7, 2021

desc:
    Program ini untuk problem multilabel classification menggunakan
    binary relevance, classifier chain, dan label powerset
'''
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


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.field]

class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.field]]


# PREPARE THE DATA

def pick_dataset(dataset_name):
    if dataset_name == 'free':
        # df = pd.read_pickle('../data/1_freeapp_reviews_features.pkl')
        # df = pd.read_csv('../data/1_freeapp_reviews_features.csv')
        df = pd.read_csv('../data/0_pusing_features.csv')
    elif dataset_name == 'paid':
        df = pd.read_pickle('../data/2_paidapp_reviews_features.pkl')
    
    df = df.filter(['lemmatized_review','acquiring','recommend','request','rating',
                    'relinquish','others'])
    # df.fillna(0,inplace=True)
    
    return df        

    
df = pick_dataset('free')
# df = df.loc[:10,:]
print(df.shape)

# print(df.columns)
# df.to_csv('../data/1_freeapp_reviews_features.csv')

combined_features = ['lemmatized_review']
decision_labels = ['acquiring','recommend','request','rating','relinquish','others']
# decision_labels2 = ['acquiring']

x_train, x_test, y_train, y_test = train_test_split(df[combined_features], 
                                                    df[decision_labels], test_size=0.2, 
                                                    random_state=0, shuffle=True)

# print(y_train.columns)



# use feature combinations
# all_feats = FeatureUnion([
#         # ('text', Pipeline([
#         #     ('colext', TextSelector('lemmatized_review')),
#         #     ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
#         # ])),
#         # ('wordcombine', Pipeline([
#         #     ('colext', TextSelector('cleaned_review')),
#         #     ('vect', CustomVectorizer()),
#         #     ('tfidf2', TfidfTransformer()),
#         # ])),
#         ('rating', Pipeline([
#             ('wordext', NumberSelector('star')),
#             ('wscaler', MinMaxScaler()),
#         ])),
#         # ('words', Pipeline([
#         #     ('wordext', NumberSelector('total_words')),
#         #     ('wscaler', MinMaxScaler()),
#         # ])),  
#         ('senticom', Pipeline([
#             ('wordext', NumberSelector('sentiment')),
#             ('wscaler', MinMaxScaler()),
#         ])),       

# ])

# nb_classifier = naive_bayes.MultinomialNB()
# lr_classifier = LogisticRegression(solver='sag')
# svm_classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# rf_classifier = RandomForestClassifier(n_estimators=1000)

# Binary Relevance
stop_words = set(stopwords.words('english'))
pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', BinaryRelevance(LogisticRegression(solver='sag'))),
            ])
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

# Classifier Chain
# stop_words = set(stopwords.words('english'))
# # pipeline = Pipeline([
# #                 ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3))),
# #                 ('clf', ClassifierChain(LogisticRegression(solver='sag'))),
# #             ])

# pipeline = Pipeline([('feats', all_feats),
#                  ('clf', BinaryRelevance(lr_classifier)),
#                  ])

# pipeline.fit(x_train, y_train)
# predictions = pipeline.predict(x_test)


# print('Accuracy = ', accuracy_score(y_test,predictions))
# print('F1 score is ',f1_score(y_test, predictions, average="macro"))
# print('Hamming Loss is ', hamming_loss(y_test, predictions))