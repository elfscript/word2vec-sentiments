# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression

import logging
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

import TaggedLineSentence  as mytag      

log.info('source load')
sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

log.info('TaggedDocument')
sentences = mytag.TaggedLineSentence(sources)


log.info('Load saved model')
model = Doc2Vec.load('./imdb.d2v')

log.info('Sentiment')
train_arrays = numpy.zeros((25000, 100))
train_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[12500 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

print train_labels

test_arrays = numpy.zeros((25000, 100))
test_labels = numpy.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i] = model.docvecs[prefix_test_pos]
    test_arrays[12500 + i] = model.docvecs[prefix_test_neg]
    test_labels[i] = 1
    test_labels[12500 + i] = 0

log.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print classifier.score(test_arrays, test_labels)

arr=['i like you','i like you so much', 'he is like his father , both are stingy', 'who can win this game?', 'what a wonderful world', 
'i like this movie  because  the real people portray themselves and their real life experience and do such a awesome job  the meaning of life while being angry at everyone especially the people who care for him most there s moments in the movie that will make you wanna cry because the family especially the father did such a good job']
s="this is a test"
x=s.split()
print(x)

test_arrays=[model.infer_vector(x.split()) for x in arr]
test_labels=[0, 1,0, 0, 1,1]
print classifier.score(test_arrays, test_labels)

print(classifier.predict(test_arrays))
