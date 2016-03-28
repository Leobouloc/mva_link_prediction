# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

##sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}
#sources = {'test-neg.txt':'TEST_NEG'}
#
#sentences = LabeledLineSentence(sources)
#a = sentences.to_array()
#
##%%
#model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
#
#model.build_vocab(sentences.to_array())
#
#for epoch in range(5):
#    logger.info('Epoch %d' % epoch)
#    model.train(sentences.sentences_perm())
#
#model.save('./imdb.d2v')
#
#
#
##%%
#from sklearn import linear_model
#
#train_arrays = numpy.zeros((250, 100))
#train_labels = numpy.zeros(250)
#
#for i in range(125):
#    prefix_train_pos = 'TRAIN_POS_' + str(i)
#    prefix_train_neg = 'TRAIN_NEG_' + str(i)
#    train_arrays[i] = model[prefix_train_pos]
#    train_arrays[125 + i] = model[prefix_train_neg]
#    train_labels[i] = 1
#    train_labels[12500 + i] = 0
#    
#    
#test_arrays = numpy.zeros((250, 100))
#test_labels = numpy.zeros(250)
#
#for i in range(125):
#    prefix_test_pos = 'TEST_POS_' + str(i)
#    prefix_test_neg = 'TEST_NEG_' + str(i)
#    test_arrays[i] = model[prefix_test_pos]
#    test_arrays[125 + i] = model[prefix_test_neg]
#    test_labels[i] = 1
#    test_labels[12500 + i] = 0
#
##%%
#classifier = linear_model.LinearRegression()
#classifier.fit(train_arrays, train_labels)
#
#classifier.score(test_arrays, test_labels)