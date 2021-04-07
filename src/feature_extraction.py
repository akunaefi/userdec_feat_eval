#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:45:27 2021

@author: akunaefi

desc:
    this program extract features for training the classifiers. there are
    two types of features:
        a. statistical features:
            - review length
            - punctuation
            - rating
            - sentiment score
        b. lexical features:
            - word n-grams
            - word combination
            - word augmentation
"""

import pandas as pd
import numpy as np
import util
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors

from config import GOOGLE_NEWS_VECTOR, DEPENDABILITY, PERFORMANCE, SUPPORTABILITY, USABILITY

def pick_dataset(dataset_name):
    
    #free app dataset
    if dataset_name == 'free':
        df = pd.read_csv('../data/1_freeapp_reviews_preprocessed.csv')
        df.fillna(' ')
    
    elif dataset_name == 'paid':
        df = pd.read_csv('../data/2_paidapp_reviews_preprocessed.csv')
        df.fillna(' ')
    
    return df

def check_nfr(sentence):
    
    # cek apakah mengandung nfr
    # jika iya, maka catat word_idx, agar bisa di-augment
    word_idx = []
    IS_FOUND = False
    words = word_tokenize(sentence)
    for word in words:
        
        if word in DEPENDABILITY:
            word_idx.append(words.index(word))
            IS_FOUND = True
        if word in PERFORMANCE:
            word_idx.append(words.index(word))
            IS_FOUND = True
        if word in SUPPORTABILITY:
            word_idx.append(words.index(word))
            IS_FOUND = True
        if word in USABILITY:
            word_idx.append(words.index(word))
            IS_FOUND = True

    return word_idx

def load_word2vec():
    # load the model
    filename = '../model/w2v_gensim_on_fdroid_reviews.kv'
    word_vectors = KeyedVectors.load(filename)

    return word_vectors

def w2v_augment(word_idxs, sentence, k=1):
    
    # ganti kata2 yang ada dalam word_idxs dengan yang similar
    word2vec = load_word2vec()
    
    augmented_sent = ''
    words = word_tokenize(sentence)
    for idx in word_idxs:
       
        word = words[idx]
        # print('id=',idx)
        # print('akan diganti=',word)
        # print('similar with %s' % w9)
        try:
            if k==1:            
                aug_word = word2vec.most_similar(positive=word)[:3][0][0]
            else :
                word_list = []
                aug_words = word2vec.most_similar(positive=word)[:k]
                for i in range(k):
                    word_list.append(aug_words[i][0])
                aug_word = ' '.join(word_list)                    
        
        except:
            # word pengganti tidak ditemukan, maka tidak diganti
            aug_word = word
        
        else:
            
            # penggantian berhasil
            # print('diganti dengan=',aug_word)
            words[idx] = aug_word
    
    augmented_sent = ' '.join(words)
    return augmented_sent

#-----------------------------------------------------------------------
# main program
#-----------------------------------------------------------------------
df = pick_dataset('free')
# df = pick_dataset('paid')

# compute sentiment
sid = SentimentIntensityAnalyzer()
for index, sent in enumerate(df['lemmatized_review']):
    if (index % 100) == 0:
        print(index)
        # break
    score = sid.polarity_scores(sent)
    df.loc['sentiment'] = score['compound']
    
    # # coba full augmentasi
    # print('asli: ',sent)
    # words = word_tokenize(sent)
    # # word_idx = check_nfr(sent) # untuk partial augment
    # word_idx = list(range(0,len(words)))
    # augment_sent = w2v_augment(word_idx, sent)
    # print('tambahan: ', augment_sent)

# simpan di pickle
with open('../data/1_freeapp_reviews_features.pkl','wb') as f:
    pickle.dump(df,f)
    