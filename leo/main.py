# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:41:52 2016

@author: work
"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords

import os

def pth(file_name):
    """Short access to file path in data folder"""
    return os.path.join('..', 'data', file_name)

# Load data
node_info = pd.read_csv(pth('node_information.xls'), header=None)
node_info.columns = ['id', 'date', 'title', 'authors', 'journal', 'abstract']

train = pd.read_csv(pth('training_set.txt'), sep=' ', header=None)
train.columns = ['id1', 'id2', 'link']

test = pd.read_csv(pth('testing_set.txt'), sep=' ', header=None)
test.columns = ['id1', 'id2']


# Split train into train and test
prop = 0.75
test = train.iloc[int(len(train)*prop):]
train = train.iloc[:int(len(train)*prop)]


# pre-process node_info 
if isinstance(node_info.authors.iloc[0], str):
    node_info.authors = node_info.authors.str.split(', ')
    node_info.loc[node_info.authors.isnull(), 'authors'] = node_info[node_info.authors.isnull()].apply(lambda x: [], axis=1)

# All abstract
#all_abstract = np.array(' '.join(node_info.abstract.as_matrix()).split())
#all_abstract = pd.Series(all_abstract)
#word_count = all_abstract.value_counts()
#my_stopwords = list(word_count[word_count >= word_count.quantile(0.999)].index)
my_stopwords = []
my_stopwords.extend(stopwords.words('english'))
my_stopwords = np.array(my_stopwords)


for word in my_stopwords:
    node_info.abstract = node_info.abstract.str.replace(' ' + word + ' ', ' ')
    node_info.abstract = node_info.abstract.str.rstrip(' ' + word)
    node_info.abstract = node_info.abstract.str.lstrip(word + ' ')
  
# Merge node_info with train and test
if train.shape[1] == 3:
    train = train.merge(node_info, how='left', left_on='id1', right_on='id').merge(node_info,\
                how='left', left_on='id2', right_on='id', suffixes=('_1', '_2'))
    test = test.merge(node_info, how='left', left_on='id1', right_on='id').merge(node_info,\
                how='left', left_on='id2', right_on='id', suffixes=('_1', '_2'))

train['temp1'] = train.apply(lambda x: len(set([y for y in x.abstract_1.split() if y in x.abstract_2.split()])), axis=1)
train['temp2'] = train.apply(lambda x: len(x.abstract_1.split() + x.abstract_2.split()), axis=1)
train['temp3'] = train.temp1.astype(float) / train.temp2


train['link_pred'] = train.temp3 >= train.temp3.median()
accuracy = (train.link_pred == train.link.astype(bool)).mean()

print 'Accuracy is {acc}'.format(acc=accuracy)
