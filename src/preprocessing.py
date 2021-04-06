#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:32:34 2021

@author: akunaefi
"""

import pandas as pd
import numpy as np
from util import text_cleaner, text_preprocess

def pick_dataset(dataset_name):
    # df = pd.DataFrame
    if dataset_name == 'free':    
        df = pd.read_csv('../data/1_freeapp_reviews_labeled.csv')
    elif dataset_name == 'paid':
        df = pd.read_csv('../data/2_paidapp_reviews_labeled.csv')
    return df

data = pick_dataset('paid')
print(data.columns)

# hapus kolom yang tidak perlu
data.drop(columns=['id','appId','Unnamed: 11'], axis=1, inplace=True)
print(data.columns)
print(data.head())

# cek null values
null_index = data[data['review'] == ""].index 
data.drop(null_index,inplace=True) 


# text cleaning
data_preprocessed = text_preprocess(data)

data_preprocessed.to_csv('../data/2_paidapp_reviews_preprocessed.csv')
# print(data_preprocessed.columns)