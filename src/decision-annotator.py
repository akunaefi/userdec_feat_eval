#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:18:04 2020

@author: akunaefi

objective:
    program untuk melakukan anotasi keputusan pengguna (user decision) 

    using weight = 1/n (n=jml found word)
    label = argmax(w * r * c)
    c = context (-1 if it has negation)
    r = distance to the root (represent how important the word, if the word root, 
    then r = 1, if the distance 2 then r= 1/2, and so on )
    machine-labelled review using rule-based tagger
    
    rule:
        1. length of review >= 5
        2. it has causality links ('because','since','')
        3. it contains at least decision words (predetermined)
"""

import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
import spacy
import en_core_web_sm
from spacy import displacy
import matplotlib.pyplot as plt
from util import text_cleaner
import networkx as nx

def shortest_path_to_root(the_word, root_list, graph):
    num_path = 1000
    
    if the_word in root_list:
        num_path = 1
    else:
        for root in root_list:
            
            entity1 = the_word
            entity2 = root
            
            try:
                t = nx.shortest_path_length(graph, source=entity1, target=entity2)
            except:
                # no path
                t = 100
                
            if t < num_path:
                num_path = t
            
    return num_path


def pick_dataset(dataset_name, rule_step=0):
    
    if dataset_name == 'free':
        '''
        **** free app reviews
        '''
        if rule_step == 0:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/fdroid_reviews.csv')
            df = df.filter(['id','review','star'])
        elif rule_step == 1:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/fdroid_reviews_rule1.csv')
        elif rule_step == 2:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/fdroid_reviews_rule2.csv')
    
    elif dataset_name == 'amazon':
        '''
        ** paid app reviews dataset 
        '''
        if rule_step == 0:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/Amazon_Reviews/Apps_for_Android_5.csv')
            df = df.filter(['reviewerID','reviewText','rating'])
            df = df.rename(columns={'reviewText':'review'})
            df['review'].fillna(' ', inplace=True)
        elif rule_step == 1:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/Amazon_Reviews/Apps_for_Android5_rule1.csv')
        elif rule_step == 2:
            df = pd.read_csv('/home/akunaefi/PhDJourney/Dataset/Amazon_Reviews/Apps_for_Android5_rule2.csv')
            


    return df

# nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()



#"use it only for two months, then it stop working. deleted."


test_text = ['Tried uninstalling n installing but still not working. Deleted.',
             'it is hardly worked do not buy this app']

causality_links = ['therefore','thus','consequently','because',
                  'reason','furthermore','so that','so','actually',
                  'basically','however','nevertheless','alternatively',
                  'though','otherwise','instead','nonetheless',
                  'conversely','similarly','comparable','likewise',
                  'further','moreover','addition','additionally',
                  'then','besides','hence','therefore','accordingly',
                  'consequently','thereupon','as a result','since',
                  'whenever','hence','think','caused','cause','as']

# label based on kurtanovic: acquire, update, relinquish, switch, other

acquire_words = ['buy','purchase','subscribe','worth', 'price'
                 'money','pay','dollar','buck','acquire',
                 'trade','deal','invest','obtain','contract','pence',
                 'bargain','payment','pro version','premium','enjoy','useful',
                 'addictive']

recommend_words = ['recommend','should try','must have','must try','recommended',
                   'suggest','advise','check out','should download']

request_words = ['please','add','feature','request','update',
                 'need','option','upgrade','renew','improve',
                 'fix','enhance','repair','correct','new',
                 'function','wish','complaint']

rating_words = ['rate','star','give','five star','four star','three star']

relinquish_words = ['uninstall','remove','delete','trash','return',
                    'garbage','junk','crap','waste','erase',
                    'wipe','get rid of','dismiss','eliminate','cancel',
                    'abandoned','refund','bad','disappoint','crash','useless',
                    'hang','suck']



df = pick_dataset('bitdefender',2)

idx_remove = [] # index review yg akan dihapus krn tidak ditemukan decision


for index, review in enumerate(df['review']):
# for review in test_text:
    
    # display the progress
    if (index % 5000) == 0:
        print(index)
    
    # '''
    # RULE #1: LENGTH OF THE REVIEW SHOULD BE MORE THAN 5 WORDS
    # '''
    clean_review = text_cleaner(review)
    # tokens = word_tokenize(review)
    # if len(tokens) <= 10:
    #     idx_remove.append(index)
    
    
    # '''
    # RULE #2: SHOULD CONTAIN CAUSALITY LINK
    # '''
    # tokens = word_tokenize(clean_review)
    # flag_argument = False    
    # for token in tokens:
    #     if token in causality_links:
    #         flag_argument = True
    #         break
    
    # if flag_argument == False:
    #     idx_remove.append(index)
    
    '''
    RULE #3: SHOULD CONTAIN DECISION MARKER
    '''    
    # print(clean_review)
    doc = nlp(clean_review)
    w = 0
    c = 1
    s = 0    
    # displacy.serve(doc,style='dep')
    

    # for token in doc:
        # print(token.text, token.dep_, token.head.text, token.head.pos_,
              # [child for child in token.children])
    
    # for token in negation_head_tokens:
        # print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])
    
    count = [0,0,0,0,0]
    lst_acquire = []
    lst_recommend = []
    lst_request = []
    lst_rate = []
    lst_relinquish = []
    
    # 1. counting the words
    for token in doc:
        if token.lemma_ in acquire_words:
            count[0] += 1
            lst_acquire.append(token.text)
            # print('tertangkap acquire:',token,'>',token.lemma_)
        if token.lemma_ in recommend_words:
            count[1] += 1
            lst_recommend.append(token.text)
            # print('tertangkap recomm:',token,'>',token.lemma_)
        if token.lemma_ in request_words:
            count[2] +=1
            lst_request.append(token.text)
            # print('tertangkap request:',token,'>',token.lemma_)
        if token.lemma_ in rating_words:
            count[3] += 1
            lst_rate.append(token.text)
            # print('tertangkap rating:',token,'>',token.lemma_)
        if token.lemma_ in relinquish_words:
            count[4] += 1
            lst_relinquish.append(token.text)
            # print('tertangkap relinquish:',token,'>',token.lemma_)

    n_found = np.sum(count)
    # print('total_found = ', n_found)
    
    # 2. counting the weight w, word significance s, and context c
    value = [0,0,0,0,0]
    if n_found == 0: #jaga2 menghindari div by zero
        idx_remove.append(index)
        w = 0
    else:
        w = 1/n_found
    
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
    # print('negation tokens', negation_tokens)
    negation_head_tokens = [token.head.text for token in negation_tokens]
    # print('negation head tokens', negation_head_tokens)
    
    # generating graph tree
    root_tokens = [tok.text for tok in doc if tok.dep_ == 'ROOT']
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                      '{0}'.format(child.lower_)))

    graph = nx.Graph(edges)

    for entry in lst_acquire:
        # compute word significance s by counting the distance to the root
        d = shortest_path_to_root(entry, root_tokens, graph)
        # print(entry,'=> jarak root = ',d)
        s = 1/d
        if entry in negation_head_tokens:
            value[0] += w * s * -1
        else:
            value[0] += w * s * 1
    for entry in lst_recommend:
        # compute word significance s by counting the distance to the root
        d = shortest_path_to_root(entry, root_tokens, graph)
        # print(entry,'=> jarak root = ',d)
        s = 1/d

        if entry in negation_head_tokens:
            value[1] += w * s * -1
        else:
            value[1] += w * s * 1
    for entry in lst_request:
        # compute word significance s by counting the distance to the root
        d = shortest_path_to_root(entry, root_tokens, graph)
        # print(entry,'=> jarak root = ',d)
        s = 1/d

        if entry in negation_head_tokens:
            value[2] += w * s * -1
        else:
            value[2] += w * s * 1
    for entry in lst_rate:
        # compute word significance s by counting the distance to the root
        d = shortest_path_to_root(entry, root_tokens, graph)
        # print(entry,'=> jarak root = ',d)
        s = 1/d

        if entry in negation_head_tokens:
            value[3] += w * s * -1
        else:
            value[3] += w * s * 1
    for entry in lst_relinquish:
        # compute word significance s by counting the distance to the root
        d = shortest_path_to_root(entry, root_tokens, graph)
        # print(entry,'=> jarak root = ',d)
        s = 1/d

        # print(negation_head_tokens)
        if entry in negation_head_tokens:
            value[4] += w * s * -1
        else:
            value[4] += w * s * 1

    
    # assign label using argmax
    index_max = np.argmax(value, axis=0)
    # print(value)
    
    if value[index_max] == 0:
        # maka review ini bukan decision review, berarti hapus
        idx_remove.append(index)
        # print('bukan decision')
        # perlu juga jika nilai > 0 tapi lebih dari satu decision
        # sementara hapus juga untuk menghilangkan ambigu

        
    if index_max == 0 :
            # acquire_review.append(review)
            df.loc[index,'label'] = 'Acquire'
            # print('acquire')
    elif index_max == 1 :
            # recommend_review.append(review)
            df.loc[index,'label'] = 'Recommend'
            # print('recommend')
    elif index_max == 2 :
            # request_review.append(review)
            df.loc[index,'label'] = 'Request'
            # print('request')
    elif index_max == 3 :
            # rating_review.append(review)
            df.loc[index,'label'] = 'Rate'
            # print('rate')
    elif index_max == 4 :
            # relinquish_review.append(review)
            df.loc[index,'label'] = 'Relinquish'
            # print('relinquish')

    # print('gold = ', df.loc[index,'gold'])
    # print('label = ', df.loc[index, 'label'])

# beri penilaian
# benar = 0
# for index, row in df.iterrows():
#     if row['gold'].lower() == row['label'].lower():
#         benar += 1

# print('score = ', benar)

update_df = df.drop(idx_remove)
print('Bitdefender review ukuran setelah rule 3 = ',update_df.shape)
update_df.to_csv('../data/bitdefender_rule3.csv')
        

# cek each label distribution
fig, ax = plt.subplots()
fig.suptitle("distribution for each label", fontsize=12)
update_df["label"].reset_index().groupby("label").count().sort_values(by= 
        "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()

print(update_df[update_df['label'] == 'Acquire'].shape)
print(update_df[update_df['label'] == 'Recommend'].shape)
print(update_df[update_df['label'] == 'Rate'].shape)
print(update_df[update_df['label'] == 'Relinquish'].shape)
print(update_df[update_df['label'] == 'Request'].shape)