#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:09:58 2021

@author: akunaefi

based on this post:
    https://towardsdatascience.com/create-dataset-for-sentiment-analysis-by-scraping-google-play-app-reviews-using-python-ceaaa0e41c1

sebelumnya menggunakan crawler dan fetch review dengan metode crawling via chromedriver 
selalu gagal dan prosesnya lambat karena berbasis web. kali ini via text scrapper jauh lebih cepat.

"""

import json
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

# pip install google-play-scraper
from google_play_scraper import Sort, reviews, app

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)


def print_json(json_object):
  json_str = json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str
  )
  print(highlight(json_str, JsonLexer(), TerminalFormatter()))


free_app_packages = [
    'com.fitbit.FitbitMobile',
    'com.nianticlabs.pokemongo',
    'me.mycake',
    'com.linkedin.android',
    'com.piriform.ccleaner',
    'com.adsk.sketchbook',
    'com.soundcloud.android',
    'com.pinterest',
    'com.amazon.kindle',
    'com.adobe.lrmobile',
    'com.gps.navigation.maps.route.directions',
    'com.yelp.android',
    'notion.id',
    'com.tripadvisor.tripadvisor',
    'com.paypal.android.p2pmobile'
]

paid_app_packages = [
    
    ]

app_packages = free_app_packages

# scrapping app info
app_infos = []

for ap in tqdm(app_packages):
  info = app(ap, lang='en', country='us')
  del info['comments']
  app_infos.append(info)
  
  
# saving app info in csv
# print_json(app_infos[0])
# app_infos_df = pd.DataFrame(app_infos)
# app_infos_df.to_csv('apps_info.csv', index=None, header=True)


# scraping app reviews
app_reviews = []

for ap in tqdm(app_packages):
  for score in list(range(1, 6)):
    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
      rvs, _ = reviews(
        ap,
        lang='en',
        country='us',
        sort=sort_order,
        # count= 200 if score == 3 else 100,
        count = 100,
        filter_score_with=score
      )
      for r in rvs:
        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
        r['appId'] = ap
      app_reviews.extend(rvs)

# print_json(app_reviews[0])

# saving as csv
app_reviews_df = pd.DataFrame(app_reviews)
app_reviews_df.to_csv('../data/free_app_reviews.csv', index=None, header=True)
# app_reviews_df.to_csv('../data/paid_app_reviews.csv', index=None, header=True)

