import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import itertools  
import os
import sys

data_folder = '/home/chau/mva_link_prediction/data/'
code_folder = '/home/chau/mva_link_prediction/thai_chau/'
sys.path.append(code_folder)

dev_mode = True

#%%
import pickle
print('Load all features')
with open(code_folder + 'all_features.dat', 'rb') as f:
    train, test = pickle.load(f)
    

#%%
# Define features to use for classification
features = []
features += ['commonword']

#features += ['ngram_abstract_1_1','ngram_title_1_1', 'ngram_title_abstract_1_1',\
#             'ngram_abstract_2_2', 'ngram_title_2_2', 'ngram_title_abstract_2_2']
             
features += ['commonlink_temp26'] #High performance
#features += ['temp26_4']
#features += ['commonlink_temp25'] #Very low performance

features += ['is_same_journal', 'time_difference']

features += ['author_2_in_abstract_1', 'author_1_in_abstract_2']
features += ['author_1_in_author_2']
features += ['auth_cite_freq_1_median', 'auth_cite_freq_2_median']

            
to_predict = 'link' # Name of column to classify
# Create input data
X_train = train[features]
y_train = train[to_predict]

X_test = test[features]
if dev_mode:
    y_test = test[to_predict]


# Random forest classification
predictor = RandomForestClassifier(n_estimators=200, max_depth=9)
predictor.fit(X_train, y_train)
# Make predictions
y_train_pred = predictor.predict(X_train)
#y_train_pred_proba = predictor.predict_proba(X_train)
y_test_pred = predictor.predict(X_test)
#y_test_pred_proba = predictor.predict_proba(X_test)

# Compute accuracy
print ('[Train] Accuracy on random forest is {acc}'.format(acc=(y_train == y_train_pred).mean()))
if dev_mode:
    print ('[Test] Accuracy on random forest is {acc}'.format(acc=(y_test == y_test_pred).mean()))



# Write test prediction in data folder
if not dev_mode:
    write_submit(y_test_pred, name_suffix = '_5')