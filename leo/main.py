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
if isinstance(node_info.authors.iloc[0], str) or isinstance(node_info.authors.iloc[0], float):
    node_info.authors = node_info.authors.str.split(', ')
    node_info.loc[node_info.authors.isnull(), 'authors'] = node_info[node_info.authors.isnull()].apply(lambda x: [], axis=1)



# All abstract
all_abstract = np.array(' '.join(node_info.abstract.as_matrix()).split())
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

# Count word co-occurence with repetition of 1
train['temp1'] = train.apply(lambda x: len(set([y for y in x.abstract_1.split() if y in x.abstract_2.split()])), axis=1)
train['temp2'] = train.apply(lambda x: len(x.abstract_1.split() + x.abstract_2.split()), axis=1)
train['temp3'] = train.temp1.astype(float) / train.temp2

# Count occurence of author 2 in abstract 1, author 1 in abstract 2 (and reversely)
train['temp4'] = train.apply(lambda x: sum([y in x.abstract_1 for y in x.authors_2]), axis=1)
train['temp5'] = train.apply(lambda x: sum([y in x.abstract_2 for y in x.authors_1]), axis=1)

# Count co_occurence of authors
train['temp6'] = train.apply(lambda x: sum([y in x.authors_1 for y in x.authors_2]), axis=1)

# Count occurence of abstract 2 in title 1, abstract 1 in title 2 (and reversely)
train['temp7'] = train.apply(lambda x: len(set([y for y in x.abstract_1.split() if y in x.title_2.split()])), axis=1)
train['temp8'] = train.apply(lambda x: len(set([y for y in x.abstract_2.split() if y in x.title_1.split()])), axis=1)
train['temp9'] = train.temp7 + train.temp8

# Count occurence of title 2 in abstract 1, title 1 in abstract 2 (and reversely)
train['temp10'] = train.apply(lambda x: len(set([y for y in x.title_2.split() if y in x.abstract_1.split()])), axis=1)
train['temp11'] = train.apply(lambda x: len(set([y for y in x.title_1.split() if y in x.abstract_2.split()])), axis=1)
train['temp12'] = train.temp10 + train.temp11

train['link_pred'] = train.temp3 >= train.temp3.median()
train['link_pred'] = train.link_pred | (train.temp6 >= 1)
train['link_pred'] = train.link_pred | (train.temp9 >= 4)
train['link_pred'] = train.link_pred | (train.temp12 >= 4)
train['link_pred'] = train.link_pred | (train.temp10 >= 4)
train['link_pred'] = train.link_pred | (train.temp11 >= 4)
accuracy = (train.link_pred == train.link.astype(bool)).mean()
print 'Accuracy is {acc}'.format(acc=accuracy)

assert False

# Use feature hasher for efficiency
from sklearn.feature_extraction.text import CountVectorizers


## Try word2vec train

import word2vec
from sklearn.metrics.pairwise import cosine_similarity as cosine

# Create txt file from node_info
all_abst_file_name = 'all_abstracts.txt'
all_phrases_file_name = 'all_abstracts_phrases.txt'
word2vec_out_file_name = 'all_abstracts.bin'

with open(pth(all_abst_file_name), 'w') as f:
    for abstract in node_info.abstract.as_matrix():
        f.write(abstract + '\n')
        
word2vec.word2phrase(pth(all_abst_file_name), pth(all_phrases_file_name), verbose=True)
word2vec.word2vec(pth(all_phrases_file_name), pth(word2vec_out_file_name), \
                    size=30, iter_=3, verbose=True)

model = word2vec.load(pth(word2vec_out_file_name))


indexes, metrics = model.cosine('applications', 20)


indexes, metrics = model.analogy(pos=['theorem', 'false'], neg=['true'], n=10)

model.vocab[indexes]