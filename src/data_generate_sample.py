#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:48:34 2021

@author: akunaefi

create sample data
"""

import pandas as pd
import nltk

# untuk yang free
df_free = pd.read_csv('../data/a_free_annotated.csv')
print(df_free.shape)

df_free_sampled = df_free.sample(6000)
print(df_free_sampled.shape)
df_free_sampled.to_csv('../data/a_free_annotated_sampled_6000.csv')

# untuk yang paid
df_paid = pd.read_csv('../data/b_paid_annotated.csv')
print(df_paid.shape)

df_paid_sampled = df_paid.sample(6000)
print(df_paid_sampled.shape)
df_paid_sampled.to_csv('../data/b_paid_annotated_sampled_6000.csv')
