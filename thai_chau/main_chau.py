"""PLEASE RUN THE FILE TRAIN DOC2VEC.PY FIRST"""
import numpy as np
import pandas as pd
import os
from nltk.corpus import stopwords
import sys

data_folder = '/home/chau/mva_link_prediction/data/'
cur_folder = '/home/chau/mva_link_prediction/thai_chau/'
sys.path.append(cur_folder)
from labeledlinesentence import *

#%%
# Load data
node_info = pd.read_csv(data_folder + 'node_information.csv', header=None)
node_info.columns = ['id', 'date', 'title', 'authors', 'journal', 'abstract']

train = pd.read_csv(data_folder + 'training_set.txt', sep=' ', header=None)
train.columns = ['id1', 'id2', 'link']

test = pd.read_csv(data_folder + 'testing_set.txt', sep=' ', header=None)
test.columns = ['id1', 'id2']


#%%
# Create txt file from node_info
all_abst_file_name = cur_folder + 'all_abstracts.txt'
all_phrases_file_name = './mva_link_prediction/thai_chau/all_abstracts_phrases.txt'
word2vec_out_file_name = cur_folder + 'all_abstracts.bin'

with open(all_abst_file_name, 'w') as f:
    for abstract in node_info.abstract.as_matrix():
        f.write(abstract + '\n')

name = cur_folder + 'all_abstracts_phrases.txt'
sources = {name:'ALL_PHRASES'}

sentences = LabeledLineSentence(sources)

#%%
# logging
import logging
import os.path
import sys
#import cPickle as pickle

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

model = Doc2Vec(size=10, workers=1)

model.build_vocab(sentences.to_array())

for epoch in range(10):
    logger.info('Epoch %d' % epoch)
    model.train(sentences.sentences_perm())

model.save(cur_folder + 'doc2vect_10.dat')



#%%
model = Doc2Vec.load(cur_folder + 'doc2vect_10.dat')
n_abstract = model.docvecs.count
n_dim = len(model.docvecs[0])
abstract_vector = np.zeros((n_abstract, n_dim))
for i in range(n_abstract):
    abstract_vector[i] = model.docvecs[i]







#%%
## Try word2vec train
import gensim
model = gensim.models.word2vec.Word2Vec.load_word2vec_format(cur_folder + 'GoogleNews-vectors-negative300.bin', binary=True)#%%
#model = gensim.models.word2vec.Word2Vec.load_word2vec_format(cur_folder + 'glove.6B.50d.txt')
#%%
#import word2vec
#from sklearn.metrics.pairwise import cosine_similarity as cosine     
#word2vec.word2phrase(all_abst_file_name, all_phrases_file_name, verbose=True)
word2vec.word2vec(all_phrases_file_name, word2vec_out_file_name, size=30, verbose=True)

#%%
import word2vec
model = word2vec.load(word2vec_out_file_name)
indexes_1, metrics_1 = model.cosine('applications', 20)
indexes_2, metrics_2 = model.analogy(pos=['theorem', 'false'], neg=['true'], n=10)
model.vocab[indexes_2]