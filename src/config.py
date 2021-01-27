# -*- coding: utf-8 -*-
'''
author: akunaefi@st.cs.kumamoto-u.ac.jp

description:
    file konfigurasi

'''

import nltk
from nltk.corpus import stopwords
# import util

DATASET_ANTIVIRUS = '/home/akunaefi/PycharmProjects/ReviewGraph/gen_arg_review/dataset/big_reviews.txt'
DATASET_FDROID = '/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/fdroid_reviews.txt'
GOOGLE_NEWS_VECTOR = '/home/akunaefi/PhDJourney/Dataset/GoogleNews-vectors-negative300.bin.gz'

MODEL_1 = ''
MODEL_2 = ''

IS_USING_STOPWORD = False

NUM_OF_SAMPLE = 100

# in stab's paper indicator is a list of argumentative keywords or
# in other term discourse marker
INDICATOR_LIST = ['therefore','thus','consequently','because',
                  'reason','furthermore','so that','so','actually',
                  'basically','however','nevertheless','alternatively',
                  'though','otherwise','instead','nonetheless',
                  'conversely','similarly','comparable','likewise',
                  'further','moreover','addition','additionally',
                  'then','besides','hence','therefore','accordingly',
                  'consequently','thereupon','as a result','since',
                  'whenever']

VERB_LIST = ['believe','think','agree']

ADVERB_LIST = ['also','often','really']

MODAL_LIST = ['should','could','would','might','must', 'can',
              'may','shall','will','ought to','need','have to',
              'used to']

STOPWORDS_EXTEND = ['app','apps','review','adobe','avlpro','bloomberg','fitbit','foursquared',
                 'gopro','kahoot','payoneer','smule','supermario','polaris','office',
                  'thanks','song','game',
                  'mario','great','phone','awesome','nintendo','wattpad','weather',
                  'really','okay','ever','photoshop','foursquare','underground','fitbit']

REMOVE_FROM_STOPWORDS = ['because','further','should','will','since','but','give','use','would','so']

GRAMMAR_LIST = ['']

DEPENDABILITY = ['accuracy','authenticate','bug','correct','count','crash','delete',
                 'disappear','error','fail','failure','fix','glitch','issue','log',
                 'login','lost','password','privacy','recognize','reliable','reset',
                 'restart','restore','shut','unstable','username','wrong']

PERFORMANCE = ['battery','buffer','delay','drain','fast','freeze','improve','jerk',
               'speed','lag','load','memory','notify','optimize','slow','stuck',
               'wait']

SUPPORTABILITY = ['dropbox','bluetooth','calibrate','cellular','cloud','compatible',
                  'compute','connect','ios','iphone','itunes','import','internet',
                  'ipad','language','laptop','offline','onedrive','phone','require',
                  'server','service','support','sync','update','upgrade','version',
                  'watch','wifi','github']

USABILITY = ['access','ad','alarm','autocorrect','bookmark','brush','button','change',
             'click','color','confuse','control','create','crop','difficult','draw',
             'figure','filter','format','hard','hear','highlight','icon','interface',
             'keyboard','landscape','listen','option','order','orient','pick','pixel',
             'portrait','rate','read','repetitive','replace','resize','scan','screen',
             'select','shape','shuffle','style','switch','tap','track','turn',
             'tutorial','type','unusable','verify','view','volume','website','widget',
             'zoom']

# if __name__ == '__main__':
    
    # stop = set(stopwords.words('english')).difference(REMOVE_FROM_STOPWORDS)
    
    # print(stop)

    # inputfile = '/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/reviews.csv'
    # outputfile = '/home/akunaefi/PhDJourney/Dataset/F_droid_feedback/fdroid_reviews.txt'

    # util.get_reviews_to_textfile(inputfile,outputfile,'review')
