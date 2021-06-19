# -*- coding: utf-8 -*-
'''
author: akunaefi
email: akunaefi@st.cs.kumamoto-u.ac.jp

description:
    program util/library berisi functions multipurpose

'''

import pandas as pd
import nltk
from nltk import pos_tag, RegexpParser, Tree
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import words
nltk.download('words')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
from itertools import groupby, product
import enchant #pyenchant package
import string
import config
import pickle
from spellchecker import SpellChecker #pyspellchecker package
from textblob import TextBlob

lemmatizer = WordNetLemmatizer()

# mencoba menggunakan custom stopword sendiri
stopword = list(set(stopwords.words('english')).difference(config.REMOVE_FROM_STOPWORDS))
# stopword.extend(config.STOPWORDS_EXTEND)
# stopwords = set(w.rstrip() for w in open('stopwords.txt'))

indicators =  config.INDICATOR_LIST
spell = SpellChecker()

def text_cleaner(sent):
    
    # membersihkan tanda baca, dan mengoreksi singkatan (syntactical noise)
    sent = sent.lower()
    sent = re.sub(r"\'s", " is ", sent)
    sent = re.sub(r"\'", "", sent)
    sent = re.sub(r"@", " ", sent)
    sent = re.sub(r"\*", " star ", sent)
    sent = re.sub(r"\'ve", " have ", sent)
    sent = re.sub(r"can't", "cannot ", sent)
    sent = re.sub(r"can not", "cannot ", sent)
    sent = re.sub(r"cant", "cannot ", sent)
    sent = re.sub(r"won't", "would not ", sent)
    sent = re.sub(r"n\'t", " not ", sent)
    sent = re.sub(r"i\'m", "i am ", sent)
    sent = re.sub(r"\'re", " are ", sent)
    sent = re.sub(r"\'d", " would ", sent)
    sent = re.sub(r"\'ll", " will ", sent)
    sent = re.sub(r"full review", " ", sent)
    sent = re.sub(r"(\d+)(k)", r"\g<1>000", sent)
    # sent = re.sub('[%s]' % re.escape(string.punctuation), ' ', sent)
    sent = re.sub(r"[^A-Za-z0-9^\/'+-=]", " ", sent)

    # word extractor
    sent = re.sub(r"pls", "please ", sent)
    sent = re.sub(r"plz", "please ", sent)
    sent = re.sub(r"luv", "love ", sent)
    sent = re.sub(r"coz", "because ", sent)
    sent = re.sub(r"cos", "because ", sent)
    sent = re.sub(r"pwd", "password ", sent)
    sent = re.sub(r"thx", "thanks ", sent)
    sent = re.sub(r"osm", "awesome ", sent)
    sent = re.sub(r"fb", "facebook ", sent)
    sent = re.sub(r" i ", " user ", sent)
    
    

    return sent



def punctuation_extractor(sent):
    
    # meng-extract punctuation dari sentence
    punc_list = ['!', '"', '#', '$', '%', '&', '\'' ,'(' ,')', '*', '+', ',', '-', '.' ,'/' ,':' ,';' ,'' ,'?' ,'@' ,'[' ,'\\' ,']' ,'^' ,'_' ,'`' ,'{' ,'|' ,'}' ,'~']
    sent = sent.replace("\\r\\n"," ")
    for character in sent:
        if character not in punc_list:
            sent = sent.replace(character, "")
    return sent


def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def remove_consecutive_dups(s):
    return re.sub(r'(?i)(.)\1+', r'\1', s)

def all_consecutive_duplicates_edits(word, max_repeat=float('inf')):
    chars = [[c*i for i in range(min(len(list(dups)), max_repeat), 0, -1)]
             for c, dups in groupby(word)]
    return map(''.join, product(*chars))

# mengkoreksi kata hiperbolik seperti "loooove" menjadi "love"
# input parameter dataframe dengan yang memiliki kolom 'review'
def remove_elongated_word(line):
    words = enchant.Dict("en")
    is_known_word = words.check
    # dict = set(words.words())
    # is_known_word = dict.check
    #for index,line in enumerate(df['review']):
        #NOTE: unnecessary work, optimize if needed
    output = [next((e for e in all_consecutive_duplicates_edits(s)
                    if e and is_known_word(e)), remove_consecutive_dups(s))
              for s in re.split(r'(\W+)', line)]
    return ''.join(output)    
    # df.loc[index,'review'] = ''.join(output)
    #return df.copy()  

def spell_corrector(sent):
    dictionary = set(words.words())
    token = nltk.word_tokenize(sent)
    new_token = []
    for word in token:
        # print(spell.correction(word))
        if word not in dictionary:
            new_token.append(spell.correction(word))
        else:
            new_token.append(word)
    return ' '.join(new_token)

def lemmatize_sentence(sentence):
    words = set(nltk.corpus.words.words())
    
    # 1. elongated words
    sentence = remove_elongated_word(sentence)
    
    # 2. word correction
    # sentence = spell_corrector(sentence)
    sent = TextBlob(sentence).correct()
    # print(sent)
    
    nltk_tagged = pos_tag(word_tokenize(sentence))    
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if word in words and word not in stopword and word.isalpha(): # remove jika stopword
        # if word in words and word.isalpha(): # biarkan stopword
            if tag is None:                        
                res_words.append(word)
            else:
                res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


# def apply_clean_sentence(df):
#     df['cleaned_review'] = df.apply(
#         lambda row: text_cleaner(row['review']), axis=1)
#     return df

# def apply_lemmatize_review(df):
#     df['lemmatized_review'] = df.apply(
#         lambda row: lemmatize_sentence(row['cleaned_review']), axis=1)
#     return df

# def apply_count_words(df):
#     df['total_words'] = df.apply(
#         lambda row: len(word_tokenize(row['cleaned_review'])), axis=1)
#     return df
    
def text_preprocess(df):
    '''
    pro-processing data frame yang mengandung text
    terdiri dari tokenization, lemmatization
    
    input: df yang didalamnya ada kolom 'review'
    '''
    
    # print('lemmatizing...(expand contraction, remove elongated words)')
    lemmatizer = WordNetLemmatizer()
    
    # df['cleaned_review'] = df.apply(
    #     lambda row: text_cleaner(row['review']), axis=1)
    # print('cleaning text...done.')
    # df['lemmatized_review'] = df.apply(
    #     lambda row: lemmatize_sentence(row['cleaned_review']), axis=1)
    # print('lemmatize text...done.')
    # df['total_words'] = df.apply(
    #     lambda row: len(word_tokenize(row['cleaned_review'])), axis=1)
    # print('counting statistics...done.')


    # stopword = list(set(stopwords.words('english')).difference(config.REMOVE_FROM_STOPWORDS))
    # stopword.extend(config.STOPWORDS_EXTEND)
    # stopwords = set(w.rstrip() for w in open('stopwords.txt'))
    
    for index, entry in enumerate(df['review']):
        if (index % 100) == 0: 
            print(index)
        # puncts = punctuation_extractor(entry)
        cleaned_sent = text_cleaner(entry)
        lemmatized_sent = lemmatize_sentence(cleaned_sent)
        # print(lemmatized_sent)
        total_words = len(word_tokenize(cleaned_sent))
        df.loc[index,'cleaned_review'] = cleaned_sent
        df.loc[index,'lemmatized_review'] = lemmatized_sent
        df.loc[index,'total_words'] = total_words # for structural feature
        # df.loc[index,'punctuation'] = puncts

    #karena data yang di lemma banyak, maka perlu di pickle
    # df.to_pickle('../data/f_droid_reviews_lemmatized.pkl')
    # with open('../data/dataset_nfr_reviews_lemmatized.pkl', 'wb') as f:
    #     pickle.dump(df, f)
    return df.copy()

# program untuk mengambil review saja dari file review
# input_file: full path to the input file
# output_file: full path to the output file
def get_reviews_to_textfile(inputfile, outputfile, column_name):
    input_filename = inputfile

    output_filename = outputfile

    counter = 1
    df_reviews = pd.read_csv(input_filename)
    with open(output_filename, 'w') as writer:
        for index, row in df_reviews.iterrows():
            writer.writelines(str(row[column_name]) + '\n')
            print(row[column_name])
            counter += 1

    print('%s baris text telah dikirim ke file.' % counter)
    return counter

# if __name__ == '__main__':
    


    # df = pd.read_csv('../data/reviews_all.csv')
    # df_preprocessed = text_preprocess(df)
    # df_ori = pd.read_pickle('../data/feature_extraction_decision_sample.pkl')
    # df = df_ori.filter(['lemmatized_review','category','decision_category','total_words','punctuation','num_of_np','num_of_vp','num_of_md'],axis=1)
    # df.fillna(0,inplace=True)
    # print(df.columns.tolist())

    # cetak kalimat yang lebih dari 5 kata
    # for index, line in enumerate(df_ori['lemmatized_review']):
    #     if len(line)>5:
    #         print(line)
