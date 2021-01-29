#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:36:31 2021

@author: akunaefi

data statistics

free_app_reviews.csv : file hasil scrapping (raw)
free_app_reviews_base.csv : sudah difilter kolom tertentu saja dan rename

"""

import pandas as pd
import nltk
from nltk import word_tokenize
from util import text_cleaner

df_free = pd.read_csv('../data/free_app_reviews.csv')
df_paid = pd.read_csv('../data/paid_app_reviews.csv')

print('panjang review free= ', len(df_free))
print('panjang review paid= ', len(df_paid))

df_free = df_free.filter(['appId','at','content','score','thumbsUpCount'])
df_free = df_free.rename(columns={'at':'time','content':'review','score':'rating','thumbsUpCount':'helpful'})
print(df_free.columns)
df_paid = df_paid.filter(['appId','at','content','score','thumbsUpCount'])
df_paid = df_paid.rename(columns={'at':'time','content':'review','score':'rating','thumbsUpCount':'helpful'})
print(df_paid.columns)

df_free.to_csv('../data/free_app_reviews_base.csv')
df_paid.to_csv('../data/paid_app_reviews_base.csv')


# df_free = pd.read_csv('../data/free_app_reviews_base.csv')
# df_paid = pd.read_csv('../data/paid_app_reviews_base.csv')

# df_free['cleaned_review'] = df_free.apply(
#     lambda row: text_cleaner(row['content']))
# df_paid['cleaned_review'] = df_paid.apply(
#     lambda row: text_cleaner(row['content']))

# for index, review in enumerate(df_free['content']):
#     print(index)
#     cleaned = text_cleaner(review)
#     df_free.loc[index,'cleaned_review'] = cleaned
#     df_free.loc[index,'words_length'] = len(word_tokenize(cleaned))

# df_free.to_csv('../data/free_app_reviews_filtered_stats.csv')
    
# df_free['numberOfWords'] = df_free.apply(
#         lambda row: len(word_tokenize(row['cleaned_review'])), axis=1)
# df_paid['numberOfWords'] = df_paid.apply(
#         lambda row: len(word_tokenize(row['cleaned_review'])), axis=1)

