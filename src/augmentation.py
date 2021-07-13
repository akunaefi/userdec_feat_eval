#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:06:54 2021

@author: akunaefi
"""

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
import sklearn
import gensim
from gensim.models import Word2Vec, KeyedVectors

def load_word2vec():
    # load the model
    filename = '../model/w2v_gensim_on_fdroid_reviews.kv'
    # filename = '../model/w2v_gensim_on_amazon_reviews.kv'

    word_vectors = KeyedVectors.load(filename)

    return word_vectors

def w2v_augment(word_idxs, sentence):
    
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
            aug_word = word2vec.most_similar(positive=word)[:3][0][0]
        
        except:
            # word pengganti tidak ditemukan, maka tidak diganti
            aug_word = word
        
        else:
            
            # penggantian berhasil
            # print('diganti dengan=',aug_word)
            words[idx] = aug_word
    
    augmented_sent = ' '.join(words)
    return augmented_sent


train_data = pd.read_csv('../data/1_free_train_6000_worked.csv')

categories = ['acquiring','recommend','request','rating','relinquish','others']
combined_features = ['Text','star','sentiment','token','mark_exclam','mark_question','mark_dollar','mark_star']

# # pembagian data dulu

x_train, x_test, y_train, y_test = train_test_split(train_data[combined_features], train_data[categories], test_size=0.2, random_state=40, shuffle=True)


# df_train = train_data.loc[x_train.index]
# print(df_train.shape)
# df_train.to_csv('../data/augment_train_0.csv')

# df_test = train_data.loc[x_test.index]
# print(df_test.shape)
# df_test.to_csv('../data/augment_test_0.csv')

#------------------
# proses augmentasi
#------------------
print(y_test.loc[:5,:])

results = []
for index, row in x_train.iterrows():
    line = row['Text']
    category1 = y_train.loc[index,'acquiring']
    category2 = y_train.loc[index,'recommend']
    category3 = y_train.loc[index,'request']
    category4 = y_train.loc[index,'rating']
    category5 = y_train.loc[index,'relinquish']
    category6 = y_train.loc[index,'others']
    # tambahkan dulu sentence asli
    array1 = []
    array1.append(line) #review
    array1.append(category1) #label
    array1.append(category2)
    array1.append(category3)
    array1.append(category4)
    array1.append(category5)
    array1.append(category6)
    results.append(array1)
    
    # full augment
    words = word_tokenize(line)
    word_idx = list(range(0,len(words)))
    augment_sent = w2v_augment(word_idx, line)
        
    # augmentasi          
    array2 = []
    array2.append(augment_sent) #review
    array2.append(category1) #label
    array2.append(category2)
    array2.append(category3)
    array2.append(category4)
    array2.append(category5)
    array2.append(category6)
    results.append(array2)

df_aug = pd.DataFrame(results, columns=['Text','acquiring','recommend','request','rating','relinquish','others'])
print(df_aug.shape)
df_aug.to_csv('../data/1_free_train_augmented.csv')
    
 
