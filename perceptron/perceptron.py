#!/usr/bin/env python3
"""
ANLP A4: Perceptron

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)
"""
import sys, os, glob

from collections import Counter
from math import log
from numpy import mean
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer

from evaluation import Eval

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import nltk
#nltk.download('wordnet')

def load_docs(direc, lemmatize, labelMapFile='labels.csv'):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""


    labelMap = {}   # docID => gold label, loaded from mapping file
    with open(os.path.join(direc, labelMapFile)) as inF:
        for ln in inF:
            docid, label = ln.strip().split(',')
            assert docid not in labelMap
            labelMap[docid] = label

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        # open the file at file_path, construct a list of its word tokens,
        # and append that list to 'docs'.
        # look up the document's label and append it to 'labels'.
        with open(os.path.join(direc, filename)) as f:
            tokens = []
            for line in f:
                line = line.strip().split()
                tokens = tokens + line
            if lemmatize:
                docs.append([WordNetLemmatizer().lemmatize(word) for word in tokens])
            else:
                docs.append(tokens)    
        labels.append(labelMap[filename])

    return docs, labels

def extract_feats(doc, lowercase = False, ngram = False, n = 1):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    if lowercase:
        for idx, word in enumerate(doc):
            doc[idx] = word.lower()
            
    if not ngram:
        ff = Counter(set(doc))
        ff['BIAS'] = 1
    else:
        ff = Counter()
        for word1, word2 in (doc[i:i+n] for i in range(len(doc)-1)):
            ff[word1+" "+word2]+=1
        ff['BIAS'] = 1
    #print(ff)
    return ff
    
def load_featurized_docs(datasplit, lowercase = False, lemmatize = False, ngram = False, n = 1):
    
    rawdocs, labels = load_docs(datasplit, lemmatize) 
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in rawdocs:
        featdocs.append(extract_feats(d, lowercase,ngram,n))
        #print(d)
    return featdocs, labels

class Count:
    def __init__(self, train_docs, train_labels):
        # list of native language codes in the corpus
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.accuracy = {l: 0 for l in self.CLASSES}
        self.lan_count(train_docs, train_labels)

    def lan_count(self, docs, labels):

        labelCounts = {l: 0 for l in self.CLASSES}

        for i in range(0, len(labels)):
            l = labels[i]
            labelCounts[labels[i]] +=1

        print("Label\tLabel Count\tProbability", file=sys.stderr)
        for l in self.accuracy:
            self.accuracy[l] = np.divide(labelCounts[l], len(labels))
            print(l +"\t"+str(labelCounts[l])+"\t"+str(self.accuracy[l]), file=sys.stderr)
            
class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['ARA', 'DEU', 'FRA', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR', 'ZHO']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """
        print('Max iterations: {}, start learning...'.format(self.MAX_ITERATIONS))

        for iteration in range(self.MAX_ITERATIONS):
            update = 0 
            for i in range(len(train_docs)): 

                label = train_labels[i]
                pred = self.predict(train_docs[i]) 
                
                if pred != label:
                    for word in train_docs[i]: 
                        self.weights[label][word] += train_docs[i][word]
                        self.weights[pred][word] -= train_docs[i][word] 
                        print(self.weights)   
                    update += 1

            trainAcc = self.test_eval(train_docs, train_labels)
            devAcc = self.test_eval(dev_docs, dev_labels)
            print('iteration: ' + str(iteration) + ' updates: ' + str(update)+ ', trainAcc: ' + str(trainAcc) + ', devAcc: ' + str(devAcc), file=sys.stderr)

            if trainAcc == 1: 
                break


    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """
        score = 0
        for word in doc:
            score += self.weights[label][word] * doc[word] 
        return score

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        highest_label = self.CLASSES[0]
        highest_score = self.score(doc, highest_label)

        for l in self.CLASSES[1:]:
            current_score = self.score(doc, l)
            if current_score > highest_score:
                highest_score = current_score
                highest_label = l

        return highest_label


    def test_eval(self, test_docs, test_labels):
        
        pred_labels = [self.predict(d) for d in test_docs]    
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy()

    def error_anlysis(self, test_docs, test_labels):

        # error analysis
        pred_labels = [self.predict(d) for d in test_docs] 
        print(self.CLASSES) 
        con_matrix = confusion_matrix(test_labels, pred_labels, labels = self.CLASSES)
        #precision, recall, bias_weight, f1 = [], [], [], []
        bias_weight = []
        # 4 a)
        print(con_matrix, file=sys.stderr)

        # reference https://en.wikipedia.org/wiki/Precision_and_recall
        for l in range(len(self.CLASSES)):
            label_weights = self.weights[self.CLASSES[l]]
            bias_weight = str(label_weights['BIAS'])
        
            #most and least 10
            print(self.CLASSES[l]+' 10 highest-weighted:\n{}'.format(Counter(label_weights).most_common()[:10]))
            print(self.CLASSES[l]+' 10 lowest-weighted:\n{}'.format(Counter(label_weights).most_common()[:-11:-1]))

            print(bias_weight, file=sys.stderr)
        # recall precision f1
        print(classification_report(test_labels, pred_labels, labels = self.CLASSES, digits=5))


if __name__ == "__main__":

    args = sys.argv[1:]
    
    # python perceptron.py -lemma -lower -ngram 2 30
    #lemmatize
    if args[0] == '-lemma':
        lemmatize = True
        args = args[1:]
    else:
        lemmatize = False
   
    #lowercase
    if args[0] == '-lower':
        lowercase = True
        args = args[1:]
    else:
        lowercase = False

    #ngram
    if args[0] == '-ngram':
        ngram = True
        n = int(args[1])
        args = args[2:]
    else:
        ngram = False
        n = 1

    niters = int(args[0])

    train_docs, train_labels = load_featurized_docs('train',lowercase, lemmatize, ngram, n)
    print(len(train_docs), 'training docs with',
       sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('dev', lowercase, lemmatize, ngram, n)
    print(len(dev_docs), 'dev docs with',
       sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('test',lowercase, lemmatize, ngram, n)
    print(len(test_docs), 'test docs with',
       sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    # 1)
    #Count(dev_docs, dev_labels)
    # 2) 3)
    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc = ptron.test_eval(test_docs, test_labels)
    print(acc, file=sys.stderr)
    # 4)
    ptron.error_anlysis(test_docs, test_labels)
