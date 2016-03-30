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

def pth(file_name):
    """Short access to file path in data folder"""
    return data_folder + file_name

def load_node_info():
    # Load data
    node_info = pd.read_csv(pth('node_information.csv'), header=None)
    node_info.columns = ['id', 'date', 'og_title', 'authors', 'journal', 'og_abstract']
    
    # pre-process node_info 
    if isinstance(node_info.authors.iloc[0], str) or isinstance(node_info.authors.iloc[0], float):
        node_info.authors = node_info.authors.str.split(', ')
        node_info.loc[node_info.authors.isnull(), 'authors'] = node_info[node_info.authors.isnull()].apply(lambda x: [], axis=1)
  
    return node_info

def create_train_test(dev_mode=True):
    '''Loads data: dev_mode=True splits the train set in train and test'''
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
        
    return train, test

def remove_stopwords(node_info):
    '''Removes stopwords FROM ABSTRACT (not titles) in node_info'''
    #all_abstract = np.array(' '.join(node_info.abstract.as_matrix()).split())
    #all_abstract = pd.Series(all_abstract)
    #word_count = all_abstract.value_counts()
    #my_stopwords = list(word_count[word_count >= word_count.quantile(0.999)].index)
    my_stopwords = ENGLISH_STOP_WORDS
    #    my_stopwords.extend(stopwords.words('english'))
    #    my_stopwords = np.array(my_stopwords)
    #    
    #    for word in my_stopwords:
    #        node_info.abstract = node_info.abstract.str.replace(' ' + word + ' ', ' ')
    #        node_info.abstract = node_info.abstract.str.rstrip(' ' + word)
    #        node_info.abstract = node_info.abstract.str.lstrip(word + ' ')
        
    node_info['abstract'] = node_info['og_abstract'].apply(lambda x: ' '.join([y for y in x.split() if y not in my_stopwords]))
    node_info['title'] = node_info['og_title'].apply(lambda x: ' '.join([y for y in x.split() if y not in my_stopwords]))
        
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
    table['author_2_in_abstract_1'] = table.apply(lambda x: sum([y in x.abstract_1 for y in x.authors_2]), axis=1)
    table['author_1_in_abstract_2'] = table.apply(lambda x: sum([y in x.abstract_2 for y in x.authors_1]), axis=1)
    
    # Count co_occurence of authors
    table['author_1_in_author_2'] = table.apply(lambda x: sum([y in x.authors_1 for y in x.authors_2]), axis=1)
    
    # Count occurence of abstract 2 in title 1, abstract 1 in title 2 (and reversely)
#    table['temp7'] = table.apply(lambda x: len(set([y for y in x.abstract_1.split() if y in x.title_2.split()])), axis=1)
#    table['temp8'] = table.apply(lambda x: len(set([y for y in x.abstract_2.split() if y in x.title_1.split()])), axis=1)
#    table['temp9'] = table.temp7 + table.temp8
 
    # Count occurence of title 2 in abstract 1, title 1 in abstract 2 (and reversely)
#    table['temp10'] = table.apply(lambda x: len(set([y for y in x.title_2.split() if y in x.abstract_1.split()])), axis=1)
#    table['temp11'] = table.apply(lambda x: len(set([y for y in x.title_1.split() if y in x.abstract_2.split()])), axis=1)
#    table['temp12'] = table.temp10 + table.temp11
    
    # Same journal (significant)
    table['is_same_journal'] = table.journal_1 == table.journal_2
    
    # Year difference (significant)
    table['time_difference'] = (table.date_1 - table.date_2).apply(abs)

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
    
    
    word_mat = lil_matrix((len(node_info), len(unique_words)), dtype=np.int32)
    assert all(node_info.index == range(len(node_info)))
    # Fill matrix iteratively by looping on abstracts
    for (ind, abstract) in node_info.abstract.iteritems():
        if ind%200 == 0:
            print('[Creating Word Matrix] ind={ind}'.format(ind=ind))
        for word in abstract.split():
            word_mat[ind, words_to_ind_dict[word]] += 1
#        if(word in synonym):
#            word_synonym = list(synonym[word].keys())
#            for vocab in word_synonym:
#                if(word != vocab)and(vocab in words_to_ind_dict):
#                    word_mat[ind, words_to_ind_dict[vocab]] += 1
    
    # Normalise word_mat # 0: word occurence in corpus; 1: number of words in abstract
    word_count = word_mat.sum(0)
    # (by total number of occurence of each word)
    word_mat_norm = normalize(normalize(word_mat, norm='l1', axis=0))
    # (both)
    #word_mat_norm = normalize(normalize(word_mat, norm='l1', axis=0), norm='l1', axis=1) # 0: word occurence; 1: number of words
    return word_mat, word_mat_norm


def make_link_mats(node_info, train):
    link_mat_idx = {id_: num for num, id_ in enumerate(node_info.id)}
    link_mat = lil_matrix((len(node_info), len(node_info)), dtype=np.int32)
    for (i, row) in train.iterrows():
        idx1 = link_mat_idx[row['id_1']]
        idx2 = link_mat_idx[row['id_2']]
        if bool(row['link']):
            link_mat[idx1, idx2] = 1
    #    else:
    #        link_mat[idx1, idx2] = -1
            
    link_mat = link_mat + link_mat.T
    link_mat_2 = link_mat.dot(link_mat)
    link_mat_3 = link_mat_2.dot(link_mat)
    link_mat_4 = link_mat_2.dot(link_mat)
    
    link_mat_3 = link_mat_4 - link_mat_3 - link_mat_2 - link_mat
    link_mat_3 = link_mat_3 - link_mat_2 - link_mat
    link_mat_2 = link_mat_2 - link_mat
    link_mat_2[range(len(node_info)), range(len(node_info))] = 0
    link_mat_3[range(len(node_info)), range(len(node_info))] = 0
        
    return link_mat_idx, link_mat, link_mat_2, link_mat_3, link_mat_4


def idxs1(row, link_mat_idx):
    idx1 = link_mat_idx[row['id_1']]
    return idx1
 
def idxs2(row, link_mat_idx):
    idx2 = link_mat_idx[row['id_2']]
    return idx2
    

    
def make_graph_features(table, link_mat_idx, link_mat_2, link_mat_3, link_mat_4):
    all_idxs1 = table.apply(lambda row: idxs1(row, link_mat_idx), axis=1)    
    all_idxs2 = table.apply(lambda row: idxs2(row, link_mat_idx), axis=1)   
       
    table['commonlink_temp25'] = link_mat_2[all_idxs1, all_idxs2].todense().T
    table['commonlink_temp26'] = link_mat_3[all_idxs1, all_idxs2].todense().T
    table['temp26_4'] = link_mat_4[all_idxs1, all_idxs2].todense().T
    return table

def common_links_feature(table, all_links):
    '''(VERY INEFFECIENT) Computes the number of common ids each couple is linked to'''

    # Add the links to 1 and 2 as columns
    table = table.join(all_links, on = 'id1')
    table = table.join(all_links, on = 'id2', lsuffix='_1', rsuffix='_2')
    
    # Fill NaN's
    table.loc[table.all_links_1.isnull(), 'all_links_1'] = table[table.all_links_1.isnull()].apply(lambda x: [], axis=1)
    table.loc[table.all_links_2.isnull(), 'all_links_2'] = table[table.all_links_2.isnull()].apply(lambda x: [], axis=1)
    
    table['commonlink_temp2'] = table.apply(lambda x: len(set([y for y in x['all_links_1'] if y in x['all_links_2']])), axis=1)
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
            print ('At idx {idx} / {len}'.format(idx=idx, len=len(train)))
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


def tfidf_ize(train, test, node_info):
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    vectorizer.fit(node_info.abstract.as_matrix())
    
    for table in [train, test]:
        table_tfidf_abstract_1 = vectorizer.transform(table.abstract_1.fillna(''))
        table_tfidf_abstract_2 = vectorizer.transform(table.abstract_2.fillna(''))
        table_tfidf_title_1 = vectorizer.transform(table.title_1.fillna(''))
        table_tfidf_title_2 = vectorizer.transform(table.title_2.fillna(''))
        
        #table['temp27'] = table_tfidf_abstract_1.multiply(table_tfidf_abstract_2).sum(1)
        table.loc[:, 'ngram_abstract_1_1'] = table_tfidf_abstract_1.minimum(table_tfidf_abstract_2).sum(1) # Intersection kernel
        table.loc[:, 'ngram_title_1_1'] = table_tfidf_title_1.minimum(table_tfidf_title_2).sum(1)
        table.loc[:, 'ngram_title_abstract_1_1'] = table_tfidf_abstract_1.minimum(table_tfidf_title_2).sum(1) \
                        + table_tfidf_abstract_2.minimum(table_tfidf_title_1).sum(1)
    
    vectorizer = TfidfVectorizer(ngram_range=(2,2))
    vectorizer.fit(node_info.abstract.as_matrix())
    
    for table in [train, test]:
        table_tfidf_abstract_1 = vectorizer.transform(table.abstract_1.fillna(''))
        table_tfidf_abstract_2 = vectorizer.transform(table.abstract_2.fillna(''))
        table_tfidf_title_1 = vectorizer.transform(table.title_1.fillna(''))
        table_tfidf_title_2 = vectorizer.transform(table.title_2.fillna(''))
        
        #table['temp27'] = table_tfidf_abstract_1.multiply(table_tfidf_abstract_2).sum(1)
        table.loc[:, 'ngram_abstract_2_2'] = table_tfidf_abstract_1.minimum(table_tfidf_abstract_2).sum(1) # Intersection kernel
        table.loc[:, 'ngram_title_2_2'] = table_tfidf_title_1.minimum(table_tfidf_title_2).sum(1)
        table.loc[:, 'ngram_title_abstract_2_2'] = table_tfidf_abstract_1.minimum(table_tfidf_title_2).sum(1) \
                        + table_tfidf_abstract_2.minimum(table_tfidf_title_1).sum(1)
    
    return train, test

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
    table['auth_cite_freq_1_mean'] = -1
    table['auth_cite_freq_1_max'] = -1
    
    table['auth_cite_freq_2_mean'] = -1
    table['auth_cite_freq_2_max'] = -1
    table.loc[sel, 'auth_cite_freq_1_mean'] = table.loc[sel, 'auth_cite_freq_1'].apply(lambda x: np.mean(x))
    table.loc[sel, 'auth_cite_freq_1_max'] = table.loc[sel, 'auth_cite_freq_1'].apply(lambda x: np.max(x))
    
    table.loc[sel, 'auth_cite_freq_2_mean'] = table.loc[sel, 'auth_cite_freq_2'].apply(lambda x: np.mean(x))
    table.loc[sel, 'auth_cite_freq_2_max'] = table.loc[sel, 'auth_cite_freq_2'].apply(lambda x: np.max(x))
    
    return table

def replace_authors_features(table, train):
    table['auth_cite_freq_1_median'] = table.auth_cite_freq_1_max.replace(-1, train.auth_cite_freq_1_max.median())
    table['auth_cite_freq_2_median'] = table.auth_cite_freq_2_max.replace(-1, train.auth_cite_freq_2_max.median())
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
    print ('File written in: {path}'.format(path=pth(file_name)))



# Set dev_mode to False for submission
dev_mode = True


#%%
# Load node information
node_info = load_node_info()
# Remove stopwords from abstract
node_info = remove_stopwords(node_info)


#%%
train, test = create_train_test(dev_mode=dev_mode)
# Merge node_info with train and test
train, test = merge_data(train, test, node_info)

#%%
# Make authors matrix (norm: each column is 1)
authors_mat, authors_mat_norm, all_authors_idx = make_authors_mat(train, node_info)    
print ('Computing authors features (long...)')
train = make_authors_features(train, authors_mat_norm)
test = make_authors_features(test, authors_mat_norm)

#%%
train = replace_authors_features(train, train)
test = replace_authors_features(test, train)

#%%
import pickle
print('Load synonym matrix')
with open(code_folder + 'synonym.dat', 'rb') as f:
    synonym = pickle.load(f)
    
#%%
# Create abstract word similarity metric (temp1)
print ('Computing abstract word similarity')
word_mat, word_mat_norm = create_word_count(node_info)
node_id_order = dict(zip(node_info.id, node_info.index))
train['commonword'] = train.apply(lambda x: kernel(x, word_mat_norm, node_id_order), axis=1)
test['commonword'] = test.apply(lambda x: kernel(x, word_mat_norm, node_id_order), axis=1)
#train['commonword_lin'] = train.apply(lambda x: kernel(x, word_mat_norm, node_id_order, 'lin'), axis=1)
#test['commonword_lin'] = test.apply(lambda x: kernel(x, word_mat_norm, node_id_order, 'lin'), axis=1)

#%%
# Creating abstract tfidf simillarity (temp20)
print ('In tfidf')
train, test = tfidf_ize(train, test, node_info)

#%%
# Create various features (temp6, temp13, temp14)
print ('Creating various features')
train = create_features(train)
test = create_features(test)

#%%
# Count links in common
print ('Counting links in common')
all_links = create_all_links(train) # all links per id
train = common_links_feature(train, all_links)
test = common_links_feature(test, all_links)

# Alternative method for links in common
link_mat_idx, link_mat, link_mat_2, link_mat_3, link_mat_4 = make_link_mats(node_info, train)
train = make_graph_features(train, link_mat_idx, link_mat_2, link_mat_3, link_mat_4)
test = make_graph_features(test, link_mat_idx, link_mat_2, link_mat_3, link_mat_4)

# Count link appearance
id_count = train.groupby('id_1').size() + train.groupby('id_2').size()
id_count.name = 'id_count'
train = train.join(id_count, on='id_1').join(id_count, on='id_2', lsuffix='_1', rsuffix='__2')
test = test.join(id_count, on='id_1').join(id_count, on='id_2', lsuffix='_1', rsuffix='__2')

#%%
import pickle
print('Save all features')
with open(code_folder + 'all_features.dat', 'wb') as f:
    pickle.dump((train, test), f)
    
    


    
#%%
import pickle
print('Load all features')
with open(code_folder + 'all_features.dat', 'rb') as f:
    train, test = pickle.load(f)
    





#%%
train = chau_features(train)
test = chau_features(test)
#%%
# Define features to use for classification
features = []
#features += ['commonword']

#features += ['ngram_abstract_1_1','ngram_title_1_1', 'ngram_title_abstract_1_1']
#features += ['ngram_abstract_2_2', 'ngram_title_2_2', 'ngram_title_abstract_2_2']
             
#features += ['commonlink_temp26'] #High performance
#features += ['commonlink_temp25'] #Very low performance

#features += ['is_same_journal', 'time_difference']

features += ['author_2_in_abstract_1', 'author_1_in_abstract_2']
features += ['author_1_in_author_2']
#features += ['auth_cite_freq_1_mean', 'auth_cite_freq_2_mean']
#features += ['auth_cite_freq_1_max', 'auth_cite_freq_2_max']
#features += ['auth_cite_freq_1_median', 'auth_cite_freq_2_median']

#features += ['temp6']
            
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