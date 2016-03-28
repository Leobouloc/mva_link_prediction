# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:41:52 2016

@author: work

"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools  
import os

def pth(file_name):
    """Short access to file path in data folder"""
    return os.path.join('..', 'data', file_name)

def load_data(dev_mode=True):
    '''Loads data: dev_mode=True splits the train set in train and test'''
    # Load data
    node_info = pd.read_csv(pth('node_information.xls'), header=None)
    node_info.columns = ['id', 'date', 'title', 'authors', 'journal', 'abstract']
    
    train = pd.read_csv(pth('training_set.txt'), sep=' ', header=None)
    train.columns = ['id1', 'id2', 'link']
    
    test = pd.read_csv(pth('testing_set.txt'), sep=' ', header=None)
    test.columns = ['id1', 'id2']
    
    
    # Split train into train and test
    if dev_mode:
        prop = 0.75
        idx_perm = np.random.permutation(range(len(train)))
        test = train.iloc[idx_perm[int(len(train)*prop):]]
        train = train.iloc[idx_perm[:int(len(train)*prop)]]
    
    # pre-process node_info 
    if isinstance(node_info.authors.iloc[0], str) or isinstance(node_info.authors.iloc[0], float):
        node_info.authors = node_info.authors.str.split(', ')
        node_info.loc[node_info.authors.isnull(), 'authors'] = node_info[node_info.authors.isnull()].apply(lambda x: [], axis=1)
  
    return node_info, train, test

def remove_stopwords(node_info):
    '''Removes stopwords FROM ABSTRACT (not titles) in node_info'''
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
    return node_info

def merge_data(train, test, node_info):
    '''Merge node_info with train and test'''
    if train.shape[1] == 3:
        train = train.merge(node_info, how='left', left_on='id1', right_on='id').merge(node_info,\
                    how='left', left_on='id2', right_on='id', suffixes=('_1', '_2'))
        test = test.merge(node_info, how='left', left_on='id1', right_on='id').merge(node_info,\
                    how='left', left_on='id2', right_on='id', suffixes=('_1', '_2'))
    return train, test


def create_features(table):
    '''Creates the features described in function'''
    
    # Count occurence of author 2 in abstract 1, author 1 in abstract 2 (and reversely)
    table['temp4'] = table.apply(lambda x: sum([y in x.abstract_1 for y in x.authors_2]), axis=1)
    table['temp5'] = table.apply(lambda x: sum([y in x.abstract_2 for y in x.authors_1]), axis=1)
    
    # Count co_occurence of authors
    table['temp6'] = table.apply(lambda x: sum([y in x.authors_1 for y in x.authors_2]), axis=1)
    
    # Count occurence of abstract 2 in title 1, abstract 1 in title 2 (and reversely)
    table['temp7'] = table.apply(lambda x: len(set([y for y in x.abstract_1.split() if y in x.title_2.split()])), axis=1)
    table['temp8'] = table.apply(lambda x: len(set([y for y in x.abstract_2.split() if y in x.title_1.split()])), axis=1)
    table['temp9'] = table.temp7 + table.temp8
 
    # Count occurence of title 2 in abstract 1, title 1 in abstract 2 (and reversely)
#    table['temp10'] = table.apply(lambda x: len(set([y for y in x.title_2.split() if y in x.abstract_1.split()])), axis=1)
#    table['temp11'] = table.apply(lambda x: len(set([y for y in x.title_1.split() if y in x.abstract_2.split()])), axis=1)
#    table['temp12'] = table.temp10 + table.temp11
    
    # Same journal (significant)
    table['temp13'] = table.journal_1 == table.journal_2
    
    # Year difference (significant)
    table['temp14'] = (table.date_1 - table.date_2).apply(abs)

    return table 


def kernel(x, word_mat_norm, node_id_order, _type='int'):    
    '''Intersection kernel'''    
    idx1 = node_id_order[x['id_1']]
    idx2 = node_id_order[x['id_2']] 
    if _type == 'int':
        val = word_mat_norm[idx1, :].minimum(word_mat_norm[idx2, :]).sum()
    elif _type == 'lin':
        val = word_mat_norm[idx1, :].dot(word_mat_norm[idx2, :].T).data
        if val:
            val = val[0]
        else:
            val = 0
    else:
        raise ValueError('This kernel type is not implemented')
    return val
      

    
def create_all_links(train):
    '''Returns a Pandas Serie that has as index a node ID and as value a list of nodes to which it is linked '''
    all_links1 = train.groupby('id1').apply(lambda x: list(x.loc[x.link == 1, 'id2']))
    all_links2 = train.groupby('id2').apply(lambda x: list(x.loc[x.link == 1, 'id1']))
    all_links = all_links1 + all_links2
    all_links.name = 'all_links'
    return all_links

def create_word_count(node_info):
    '''
    Create word_mat matrix (num_nodes x num_unique_words) that contains the
    number of occurences of each word in each abstract
    '''

    all_abstract = np.array(' '.join(node_info.abstract.as_matrix()).split())
    
    unique_words = np.unique(all_abstract)
    ind_to_words_dict = dict(zip(range(len(unique_words)), unique_words))
    words_to_ind_dict = dict(zip(unique_words, range(len(unique_words))))
    
    
    word_mat = lil_matrix((len(node_info), len(unique_words)))
    assert all(node_info.index == range(len(node_info)))
    # Fill matrix iteratively by looping on abstracts
    for (ind, abstract) in node_info.abstract.iteritems():
        if ind%200 == 0:
            print '[Creating Word Matrix] ind={ind}'.format(ind=ind)
        for word in abstract.split():
            word_mat[ind, words_to_ind_dict[word]] += 1
    
    
    # Normalise word_mat # 0: word occurence in corpus; 1: number of words in abstract
    word_count = word_mat.sum(0)
    # (by total number of occurence of each word)
    word_mat_norm = normalize(normalize(word_mat, norm='l1', axis=0))
    # (both)
    #word_mat_norm = normalize(normalize(word_mat, norm='l1', axis=0), norm='l1', axis=1) # 0: word occurence; 1: number of words
    return word_mat, word_mat_norm


def common_links_feature(table, all_links):
    '''(VERY INEFFECIENT) Computes the number of common ids each couple is linked to'''

    # Add the links to 1 and 2 as columns
    table = table.join(all_links, on = 'id1')
    table = table.join(all_links, on = 'id2', lsuffix='_1', rsuffix='_2')
    
    # Fill NaN's
    table.loc[table.all_links_1.isnull(), 'all_links_1'] = table[table.all_links_1.isnull()].apply(lambda x: [], axis=1)
    table.loc[table.all_links_2.isnull(), 'all_links_2'] = table[table.all_links_2.isnull()].apply(lambda x: [], axis=1)
    
    table['temp2'] = table.apply(lambda x: len(set([y for y in x['all_links_1'] if y in x['all_links_2']])), axis=1)
    return table


def make_authors_mat(train, node_info):
    all_authors = []
    for x in node_info.authors.as_matrix():
        for y in x:
            all_authors.append(y)
                
    all_authors = np.array(list(set(all_authors)))
    all_authors = np.array([x for x in all_authors if x != ''])
    num_authors = all_authors.shape[0]
    
    
    all_authors_idx = {x:i for i,x in enumerate(all_authors)}
    
    authors_mat = lil_matrix((num_authors, num_authors))
    
    for (idx, row) in train.iterrows():
        if idx%1000 == 0:
            print 'At idx {idx} / {len}'.format(idx=idx, len=len(train))
        authors_1 = [x for x in row.authors_1 if x != '']
        authors_2 = [x for x in row.authors_2 if x != '']
    
        author_pairs = list(itertools.product(authors_1, authors_2))
        for (a1, a2) in author_pairs:
            idx1 = all_authors_idx[a1]
            idx2 = all_authors_idx[a2]
            
            authors_mat[idx1, idx2] += 1
      
    # Make symetry
    authors_mat += authors_mat.T  
    
    # Normalise
    authors_mat_norm = normalize(authors_mat, norm='l1', axis=0)
    
    return authors_mat, authors_mat_norm, all_authors_idx


def author_cite_freq_metric(row, authors_mat_norm):
    authors_1 = [x for x in row.authors_1 if x != '']
    authors_2 = [x for x in row.authors_2 if x != '']

    author_pairs = list(itertools.product(authors_1, authors_2))
    author_cite_freq_1 = []
    author_cite_freq_2 = []
    for (a1, a2) in author_pairs:
        idx1 = all_authors_idx[a1]
        idx2 = all_authors_idx[a2]       
        
        author_cite_freq_1.append(authors_mat_norm[idx1, idx2])
        author_cite_freq_2.append(authors_mat_norm[idx2, idx1])
        
    return (author_cite_freq_1, author_cite_freq_2)

def make_authors_features(table, authors_mat_norm):
    table['auth_cite_freq'] = table.apply(lambda row: author_cite_freq_metric(row, authors_mat_norm), axis=1)
    table['auth_cite_freq_1'] = table['auth_cite_freq'].apply(lambda x: x[0])
    table['auth_cite_freq_2'] = table['auth_cite_freq'].apply(lambda x: x[1])
    
    sel = table.auth_cite_freq != ([], [])
    table['temp15'] = -1
    table['temp16'] = -1
    
    table['temp17'] = -1
    table['temp18'] = -1
    table.loc[sel, 'temp15'] = table.loc[sel, 'auth_cite_freq_1'].apply(lambda x: np.mean(x))
    table.loc[sel, 'temp16'] = table.loc[sel, 'auth_cite_freq_1'].apply(lambda x: np.max(x))
    
    table.loc[sel, 'temp17'] = table.loc[sel, 'auth_cite_freq_2'].apply(lambda x: np.mean(x))
    table.loc[sel, 'temp18'] = table.loc[sel, 'auth_cite_freq_2'].apply(lambda x: np.max(x))
    
    train['temp19'] = train.temp16.replace(-1, train.temp16.median())
    train['temp20'] = train.temp18.replace(-1, train.temp18.median())
    
    test['temp19'] = test.temp16.replace(-1, train.temp16.median())
    test['temp20'] = test.temp18.replace(-1, train.temp18.median())    
    
    return table

def write_submit(y_test_pred, name_suffix=None):
    '''Writes y_test_pred in the proper format in the data folder'''
    to_submit = pd.DataFrame()
    to_submit['id'] = range(len(test))
    to_submit['category'] = y_test_pred
    
    if name_suffix is not None:
        file_name = 'to_submit' + name_suffix + '.csv'
    else:
        file_name = 'to_submit.csv'
    to_submit.to_csv(pth(file_name), index=False, sep=',')
    print 'File written in: {path}'.format(path=pth(file_name))



# Set dev_mode to False for submission
dev_mode = False


# Load data
node_info, train, test = load_data(dev_mode=dev_mode)


# Remove stopwords from abstract
node_info = remove_stopwords(node_info)


# Merge node_info with train and test
train, test = merge_data(train, test, node_info)

 
# Make authors matrix (norm: each column is 1)
authors_mat, authors_mat_norm, all_authors_idx = make_authors_mat(train, node_info)    
print 'Computing authors features (long...)'
train = make_authors_features(train, authors_mat_norm)
test = make_authors_features(test, authors_mat_norm)



# Create abstract word similarity metric (temp1)
print 'Computing abstract word similarity'
word_mat, word_mat_norm = create_word_count(node_info)
node_id_order = dict(zip(node_info.id, node_info.index))
train['temp1'] = train.apply(lambda x: kernel(x, word_mat_norm, node_id_order), axis=1)
test['temp1'] = test.apply(lambda x: kernel(x, word_mat_norm, node_id_order), axis=1)
train['temp1_lin'] = train.apply(lambda x: kernel(x, word_mat_norm, node_id_order, 'lin'), axis=1)
test['temp1_lin'] = test.apply(lambda x: kernel(x, word_mat_norm, node_id_order, 'lin'), axis=1)

# Creating abstract tfidf simillarity (temp20)
vectorizer = TfidfVectorizer()
tfidf_mat = vectorizer.fit_transform(node_info.abstract.as_matrix())
#train['temp21'] = train.apply(lambda x: kernel(x, tfidf_mat, node_id_order), axis=1)
#test['temp21'] = test.apply(lambda x: kernel(x, tfidf_mat, node_id_order), axis=1)
train['temp22'] = train.apply(lambda x: kernel(x, tfidf_mat, node_id_order, 'lin'), axis=1)
test['temp22'] = test.apply(lambda x: kernel(x, tfidf_mat, node_id_order, 'lin'), axis=1)



# Create various features (temp6, temp13, temp14)
print 'Creating various features'
train = create_features(train)
test = create_features(test)


# Count links in common
print 'Counting links in common'
all_links = create_all_links(train) # all links per id
train = common_links_feature(train, all_links)
test = common_links_feature(test, all_links)






# Define features to use for classification
features = ['temp1', 'temp22', 'temp2', 'temp6', 'temp13', 'temp14'] + ['temp4', 'temp5', 'temp9'] + ['temp19', 'temp20']
to_predict = 'link' # Name of column to classify


# Random forest classification
predictor = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=13, \
                                min_samples_split=2, min_samples_leaf=1, \
                                min_weight_fraction_leaf=0.0, max_features='auto', \
                                max_leaf_nodes=None, bootstrap=True, oob_score=False, \
                                n_jobs=-1, random_state=None, verbose=0, \
                                warm_start=False, class_weight=None)


# Create input data
X_train = train[features]
y_train = train[to_predict]

X_test = test[features]
if dev_mode:
    y_test = test[to_predict]


# Train forest
predictor.fit(X_train, y_train)

# Make predictions
y_train_pred = predictor.predict(X_train)
#y_train_pred_proba = predictor.predict_proba(X_train)

y_test_pred = predictor.predict(X_test)
#y_test_pred_proba = predictor.predict_proba(X_test)



# Compute accuracy
print '[Train] Accuracy is {acc}'.format(acc=(y_train == y_train_pred).mean())
if dev_mode:
    print '[Test] Accuracy is {acc}'.format(acc=(y_test == y_test_pred).mean())


# Write test prediction in data folder
if not dev_mode:
    write_submit(y_test_pred, name_suffix = '_4')





assert False
# explore results

train['link_pred'] = y_train_pred
train['good_pred'] = train.link == train.link_pred

#train.to_csv(pth('my_train.csv'), index=False)
#test.to_csv(pth('my_test.csv'), index=False)

train = pd.read_csv(pth('my_train.csv'))
test = pd.read_csv(pth('my_test.csv'))









assert False
''' BELOW IS FOR EXPLORATION '''





assert False
train['link_pred'] = y_train_pred
train['link_pred_proba'] = y_train_pred_proba[:, 1]


#train['link_pred'] = (train.temp2 >= 1) | (train.temp1 >= train.temp1.quantile(0.7))
#accuracy = (train.link_pred == train.link.astype(bool)).mean()
#print 'Accuracy is {acc}'.format(acc=accuracy)

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