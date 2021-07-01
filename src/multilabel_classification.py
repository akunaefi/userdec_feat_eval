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
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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
from custom_vectorizer import CustomVectorizer, PunctVectorizer
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import multilabel_confusion_matrix

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

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=18):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes, cmap=plt.cm.Blues)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Conf. Matrix for " + class_label)
    
# import the data
train_data = pd.read_csv('../data/1_free_train_6000_worked.csv')
# train_data = pd.read_csv('../data/2_paid_train_6000_x.csv')

test_data = pd.read_csv('../data/kaggle/test.csv')


# print(train_data_1000.shape)
# print(test_data.shape)


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
combined_features = ['Text','star','sentiment','token','mark_exclam','mark_question','mark_dollar','mark_star']

#split the data

x_train, x_test, y_train, y_test = train_test_split(train_data[combined_features], train_data[categories], test_size=0.2, random_state=40, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(train_data['lemmatized_review'], train_data[categories], test_size=0.2, random_state=40, shuffle=True)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# use feature combinations
all_feats = FeatureUnion([
        ('text', Pipeline([
            ('colext', TextSelector('Text')),
            ('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3)))
        ])),
        ('wordcombine', Pipeline([
            ('colext', TextSelector('Text')),
            ('vect', CustomVectorizer()),
            ('tfidf2', TfidfTransformer()),
        ])),
        ('rating', Pipeline([
            ('wordext', NumberSelector('star')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('words', Pipeline([
            ('wordext', NumberSelector('token')),
            ('wscaler', MinMaxScaler()),
        ])),  
        ('senticom', Pipeline([
            ('wordext', NumberSelector('sentiment')),
            ('wscaler', MinMaxScaler()),
        ])),
        # # #--- punctuation features
        ('exclamation', Pipeline([
            ('wordext', NumberSelector('mark_exclam')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('question', Pipeline([
            ('wordext', NumberSelector('mark_question')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('dollar', Pipeline([
            ('wordext', NumberSelector('mark_dollar')),
            ('wscaler', MinMaxScaler()),
        ])),
        ('stars', Pipeline([
            ('wordext', NumberSelector('mark_star')),
            ('wscaler', MinMaxScaler()),
        ])),  
])

# classifiers
nb_classifier = naive_bayes.MultinomialNB()
lr_classifier = LogisticRegression(solver='sag')
svm_classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
rf_classifier = RandomForestClassifier(n_estimators=100)
mlp_classifier = MLPClassifier(random_state=1, max_iter=500)

# -----------------------
# using binary relevance
# -----------------------

# pipeline = Pipeline([
#                 # ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#                 # ('clf', BinaryRelevance(MultinomialNB())),
#                 ('feat',all_feats),
#                 ('clf',BinaryRelevance(svm_classifier)),
#             ])

# # train
# pipeline.fit(x_train, y_train)

# # predict
# predictions = pipeline.predict(x_test)


# # from sklearn.metrics import accuracy_score
# print('Accuracy = ', accuracy_score(y_test,predictions))
# print('F1 score is ',f1_score(y_test, predictions, average="micro"))
# print('Hamming Loss is ', hamming_loss(y_test, predictions))


# -------------------------
# # using classifier chains 
# -------------------------


pipeline = Pipeline([
                ('feat',all_feats),
                ('clf', ClassifierChain(rf_classifier)),
            ])

# train
pipeline.fit(x_train, y_train)

# predict
predictions = pipeline.predict(x_test)

print('Accuracy = ', accuracy_score(y_test,predictions))
print('F1 score is ',f1_score(y_test, predictions, average="micro"))
print('Hamming Loss is ', hamming_loss(y_test, predictions))

# confusion matrix

# fig, ax = plt.subplots(3, 2, figsize=(12, 7))
    
# for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, labels):
#     print_confusion_matrix(cfs_matrix, axes,['acq','rec','req','rate','rel','oth'], ["N", "Y"])
    
# fig.tight_layout()
# plt.show()

# print(y_test[:5])
# y_hat = pd.DataFrame(predictions)
# y_hat.to_csv('../data/predictions0.csv')

arr_pred = []
for index, entry in enumerate(predictions):

    arr_temp = [entry[0,0],entry[0,1],entry[0,2],entry[0,3],entry[0,4],entry[0,5]]
    arr_pred.append(arr_temp)

df_pred = pd.DataFrame(arr_pred,columns=['acquiring','recommend','request','rating','relinquish','others'])
# df_pred.to_csv('../data/predictions.csv')

# print(multilabel_confusion_matrix(y_test,df_pred))

vis_arr = multilabel_confusion_matrix(y_test,df_pred)
labels = ['acquiring','recommend','request','rating','relinquish','others']

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    
for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

fig.tight_layout()
plt.savefig('../chart/conf_matrix_free.eps')
plt.show()
