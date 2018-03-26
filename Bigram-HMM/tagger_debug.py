#!/usr/bin/env python3
"""
ENLP A4: HMM for Part-of-Speech Tagging

Usage: 
  python tagger.py baseline
  python tagger.py hmm

(Nathan Schneider; adapted from Richard Johansson)
"""
from math import log, isfinite
from collections import Counter
from collections import defaultdict
import sys, os, time, platform, nltk
from nltk.tag.util import untag


def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


# utility function for color-coding in terminal
# https://gist.github.com/ssbarnea/1316877
def accepts_colors(handle=sys.stdout):
    if (hasattr(handle, "isatty") and handle.isatty()) or \
        ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
        if platform.system()=='Windows' and not ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
            return False #handle.write("Windows console, no ANSI support.\n")
        else:
            return True
    else:
        return False


def BuildTrainVocab(training):
    first_occur = set()
    s_occur = set()
    for sent in training:
        for token in untag(sent):
            if (token not in s_occur and token in first_occur):
                s_occur.add(token)
            if (token not in first_occur):
                first_occur.add(token)
    return s_occur


def PreprocessText(corpus, vocabulary):
    
    new_corp = []
    for sent in corpus: 
        new_sent = []
        new_sent.append((START , START))
        for token in sent:
            if (token[0] in vocabulary):
                new_sent.append(token)
            if (token[0] not in vocabulary):
                new_sent.append((UNK,token[1]))
        new_sent.append((END , END))
        new_corp.append(new_sent)

    return new_corp

# utility functions to read the corpus
START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

ALPHA = .1
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = defaultdict(int)
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}#len(allTagCounts)
emissionDists = {}
dictionary = defaultdict(set)
incorrectCounts=defaultdict(int)

def learn1(tagged_sentences):
    """
    Record the overall tag counts (allTagCounts) and counts for each word (perWordTagCounts) for baseline tagger.
    (These should not have pseudocounts and should only apply to observed words/tags, not START, END, or UNK.)
    
    Learn the parameters of an HMM with add-ALPHA smoothing (ALPHA = 0.1):
     - Store counts + pseudocounts of observed transitions (transitionCounts) and emissions (emissionCounts) for bigram HMM tagger. 
     - Also store a pseudocount for UNK for each distribution.
     - Normalize the distributions and store (natural) log probabilities in transitionDists and emissionDists.
    """

    # store training data counts in allTagCounts, perWordTagCounts, transitionCounts, emissionCounts
    # ...
    
    word = ""
    tag = ""
    previousTag = START
    
    for each_sentence in tagged_sentences:
        for each_pair in each_sentence:
            word = each_pair[0]
            tag = each_pair[1]
            if tag not in allTagCounts:
                allTagCounts[tag] = 0
            allTagCounts[tag] += 1
            
            if word not in perWordTagCounts:
                perWordTagCounts[word] = Counter()
                perWordTagCounts[word][tag] = 0
            elif tag not in perWordTagCounts[word]:
                perWordTagCounts[word][tag] = 0
            perWordTagCounts[word][tag] += 1
            
    # add pseudocounts in transitionCounts and emissionCounts, including for UNK
    #...
            if previousTag not in transitionCounts:
                transitionCounts[previousTag] = Counter()
                transitionCounts[previousTag][tag] = 0
            elif tag not in transitionCounts[previousTag]:
                transitionCounts[previousTag][tag] = 0
            transitionCounts[previousTag][tag] += 1
            
            previousTag = tag
            
            if tag not in emissionCounts:
                emissionCounts[tag] = Counter()
                emissionCounts[tag][word] = 0
            elif word not in emissionCounts[tag]:
                emissionCounts[tag][word] = 0
            emissionCounts[tag][word] += 1
        
        if previousTag not in transitionCounts:
            transitionCounts[previousTag] = Counter()
            transitionCounts[previousTag][END] = 0
        elif END not in transitionCounts[previousTag]:
            transitionCounts[previousTag][END] = 0
        transitionCounts[previousTag][END] += 1

    # normalize counts and store log probability distributions in transitionDists and emissionDists
    # ...
    for given_tag in transitionCounts:
        transitionCounts[given_tag][UNK] = 0
        total = sum(transitionCounts[given_tag].values(),0.0)
        length = len(transitionCounts[given_tag])
        transitionDists[given_tag] = {}
        for tag in transitionCounts[given_tag]:
            transitionDists[given_tag][tag] = log((transitionCounts[given_tag][tag]+ALPHA)/(total+ALPHA*length))
            
    for given_tag in emissionCounts:
        emissionCounts[given_tag][UNK] = 0
        total = sum(emissionCounts[given_tag].values(),0.0)
        length = len(emissionCounts[given_tag])
        emissionDists[given_tag] = {}
        for word in emissionCounts[given_tag]:
            emissionDists[given_tag][word] = log((emissionCounts[given_tag][word]+ALPHA)/(total+ALPHA*length))

    #print(perWordTagCounts)
    temp = {}    
    for i in perWordTagCounts:
        for j in perWordTagCounts[i]:
            temp[i, j] = perWordTagCounts[i][j]
    perWordTagCounts = temp

    temp={}
    for i in transitionDists:
        for j in transitionDists[i]:
            temp[i, j] = transitionDists[i][j]
    transitionDists = temp

    temp = {}          
    for i in emissionDists:
        for j in emissionDists[i]:
            temp[j, i] = emissionDists[i][j]
    emissionDists = temp    

def learn(tagged_sentences):
    """
    Record the overall tag counts (allTagCounts) and counts for each word (perWordTagCounts) for baseline tagger.
    (These should not have pseudocounts and should only apply to observed words/tags, not START, END, or UNK.)
    
    Learn the parameters of an HMM with add-ALPHA smoothing (ALPHA = 0.1):
     - Store counts + pseudocounts of observed transitions (transitionCounts) and emissions (emissionCounts) for bigram HMM tagger. 
     - Also store a pseudocount for UNK for each distribution.
     - Normalize the distributions and store (natural) log probabilities in transitionDists and emissionDists.
    """
    # store training data counts in allTagCounts, perWordTagCounts, transitionCounts, emissionCounts

    perWordTag = []
    for i in range(len(tagged_sentences)):
        for j in range(len(tagged_sentences[i])):
            allTagCounts[tagged_sentences[i][j][1]]+=1
            perWordTagCounts[tagged_sentences[i][j][0]][tagged_sentences[i][j][1]]

    perWordTagCounts = Counter(perWordTag)

    #transitionCounts
    last_tok = ("","")
    for sent in tagged_sentences:
        last_tok = ("","")
        for tok in sent:
            if tok[1] not in dictionary[tok[0]]:
                dictionary[tok[0]].add(tok[1])
            if (last_tok != ("","")):
                transitionCounts[(last_tok[1],tok[1])] += (1+ ALPHA)
            last_tok = tok

    #emissionCounts
    for k, v in perWordTagCounts.items():
        emissionCounts[k] = v + ALPHA
    
    # add pseudocounts in transitionCounts and emissionCounts, including for UNK

    # normalize counts and store log probability distributions in transitionDists and emissionDists
    for k, v in perWordTagCounts.items():
        emissionDists[k] = log( v / allTagCounts[k[1]] )
    for k, v in transitionCounts.items():
        transitionDists[k] = log( (v) / (allTagCounts[k[0]] + (ALPHA*len(allTagCounts)))) # not sure V and alpha

    return perWordTagCounts

def baseline_tag_sentence(sentence,perWordTagCounts):
    """
    Tag the sentence with a most-frequent-tag baseline: 
    For each word, if it has been seen in the training data, 
    choose the tag it was seen most often with; 
    otherwise, choose the overall most frequent tag in the training data.
    Hint: Use the most_common() method of the Counter class.
    
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """

    pred_sent = []
    for token in sentence:
        max_key = FindMax(token, perWordTagCounts)
        pred_sent.append((token[0],max_key[1]))
 
    return pred_sent

def FindMax(token, perWordTagCounts):

    maxlist = {}
    for key,value in perWordTagCounts.items():
        if (key[0] == token[0]):
            maxlist[key] = value

    return max(maxlist, key=lambda i: maxlist[i])

# testing all in one viterbi
def Viterbi(sentence, perWordTagCounts):

    tagged_sent = defaultdict(str)
    viterbi = defaultdict(float)
    backpointers = {} 

    for tag in dictionary[sentence[1][0]]:
        viterbi[(tag,1)] = transitionDists[(sentence[0][1], tag)] + emissionDists[(sentence[1][0], tag)]
        backpointers[(tag,1)] = (START, 0)
    viterbi[(START, 0)] = 0

    #viterbi algo step
    counter = 1
    for token in sentence[2:]:
        counter += 1
        for tag in dictionary[token[0]]:
            find_max = defaultdict(float)
            for state in viterbi:
                if (state[1] == counter-1):
                    #if (state[0],tag) in transitionDists and (token[0],tag) in emissionDists:
                        #print(state[0]+tag)
                    find_max[state]= viterbi[state] + transitionDists[(state[0],tag)] + emissionDists[(token[0],tag)]
            if len(find_max.values()) >= 1:
                viterbi[(tag, counter)] = max(find_max.values())
                backpointers[(tag, counter)] = max(find_max, key=lambda i: find_max[i])

    #retrace
    lookstate = (END,len(sentence)-1)
    #print(backpointers.keys())
    for k in backpointers.keys():
        tagged_sent[backpointers[lookstate][1]] = backpointers[lookstate][0]
        if backpointers[lookstate] != (START, 0):
            lookstate = backpointers[lookstate]
    tagged_sent[0] = START

    final_tagged = []
    for count, k in enumerate(tagged_sent.values()):
        final_tagged.append((sentence[count][0], k))
    final_tagged.append((END,END))

    return final_tagged

def joint_prob(sentence):
    """Compute the joint probability of the given words and tags under the HMM model."""
    p = 0   # joint log prob. of words and tags
    last_tok = ("","")
    productA = 0.0
    productB = 0.0
    for token in sentence:
        if last_tok != ("",""):
            if (token) in emissionDists:   
                productB += emissionDists[token]
            if (last_tok[1],token[1]) in transitionDists:
                productA += transitionDists[(last_tok[1],token[1])]
        last_tok = token
    p = productA + productB

    assert isfinite(p) and p<0,p  # Should be negative
    return p

def count_correct(gold_sentence, pred_sentence):
    """Given a gold-tagged sentence and the same sentence with predicted tags,
    return the number of tokens that were tagged correctly overall, 
    the number of OOV tokens tagged correctly, 
    and the total number of OOV tokens."""
    assert len(gold_sentence)==len(pred_sentence),gold_sentence
    correct_sentences = 0.0
    OOV_correct_count = 0.0
    OOV_count = 0.0

    for count, sent in enumerate(gold_sentence):
        if sent == pred_sentence[count]:
            correct_sentences += 1
        else:
            incorrectCounts[sent[1]] += 1
        if sent[0] == UNK:
            OOV_count+=1
            if sent[1] == pred_sentence[count][1]:
                OOV_correct_count+=1
    #nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent
    return correct_sentences, OOV_correct_count, OOV_count

def count_correct1(gold_sentence, pred_sentence):
    """Given a gold-tagged sentence and the same sentence with predicted tags,
    return the number of tokens that were tagged correctly overall, 
    the number of OOV tokens tagged correctly, 
    and the total number of OOV tokens."""
    assert len(gold_sentence)==len(pred_sentence)
    # ...
    correct = 0
    correctOOV = 0
    OOV = 0
    
    for idx, gold_pair in enumerate(gold_sentence):
        word = gold_pair[0]
        gold_label = gold_pair[1]
        pred_label = pred_sentence[idx][1]
        if gold_label == pred_label:
            correct += 1
            if word not in perWordTagCounts:
                correctOOV += 1
        if word not in perWordTagCounts:
            OOV += 1
            
    return correct, correctOOV, OOV

TRAIN_DATA = 'en-ud-train.upos.tsv'
TEST_DATA = 'en-ud-test.upos.tsv'

train_sentences = read_tagged_corpus(TRAIN_DATA)
# vocabulary = BuildTrainVocab(train_sentences)
# training_set_prep = PreprocessText(train_sentences, vocabulary)


# train the bigram HMM tagger & baseline tagger in one fell swoop
trainingStart = time.time()
perWordTagCounts = learn1(train_sentences)
trainingStop = time.time()
trainingTime = trainingStop - trainingStart


# decide which tagger to evaluate
if len(sys.argv)<=1:
    assert False,"Specify which tagger to evaluate: 'baseline' or 'hmm'"
if sys.argv[1]=='baseline':
    tagger = baseline_tag_sentence
elif sys.argv[1]=='hmm':
    tagger = Viterbi
else:
    assert False,'Invalid command line argument'



if accepts_colors():
    class bcolors:  # terminal colors
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
else:
    class bcolors:
        HEADER = ''
        OKBLUE = ''
        OKGREEN = ''
        WARNING = ''
        FAIL = ''
        ENDC = ''
        BOLD = ''
        UNDERLINE = ''


def render_gold_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.WARNING + gold + bcolors.ENDC if gold!=pred else gold)
    
def render_pred_tag(x):
    (w,gold),(w,pred) = x
    return w + '/' + (bcolors.FAIL + pred + bcolors.ENDC if gold!=pred else pred)



test_sentences = read_tagged_corpus(TEST_DATA)
#test_set_prep = PreprocessText(test_sentences, vocabulary)

nTokens = nCorrect = nOOV = nCorrectOOV = nPerfectSents = nPGoldGreater = nPPredGreater = 0

taggingTime = 0

#testing list

for sent in test_sentences:
    taggerStart = time.time()
    pred_tagging = tagger(sent, perWordTagCounts)
    taggerStop = time.time()
    taggingTime += taggerStop - taggerStart

    nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent = count_correct(sent, pred_tagging)
    
    acc = nCorrectThisSent/len(sent)
    
    pHMMGold = joint_prob(sent)
    pHMMPred = joint_prob(pred_tagging)
    sent_print = [list(x) for x in sent[1:-1]]
    pred_tagging_print = [list(x) for x in  pred_tagging[1:-1]]

    for i in range(len(sent_print)):
        if test_sentences[test_set_prep.index(sent)][i][0] != sent_print[i][0]:
            sent_print[i][0] = test_sentences[test_set_prep.index(sent)][i][0]
            pred_tagging_print[i][0] = test_sentences[test_set_prep.index(sent)][i][0]

    print(pHMMGold, ' '.join(map(render_gold_tag, zip(sent_print,pred_tagging_print))))
    print(pHMMPred, ' '.join(map(render_pred_tag, zip(sent_print,pred_tagging_print))), '{:.0%}'.format(acc))
    print('{:.0%}'.format(acc))
    if pHMMGold > pHMMPred:
        nPGoldGreater += 1
        # assert False
    elif pHMMGold < pHMMPred:
        nPPredGreater += 1
    
    nCorrect += nCorrectThisSent
    nCorrectOOV += nCorrectOOVThisSent
    nOOV += nOOVThisSent
    nTokens += len(sent)
    if pred_tagging==sent:
        nPerfectSents += 1

print('TAGGING ACCURACY BY TOKEN: {}/{} = {:.1%}   OOV TOKENS: {}/{} = {:.1%}   PERFECT SENTENCES: {}/{} = {:.1%}   #P_HMM(GOLD)>P_HMM(PRED): {}   #P_HMM(GOLD)<P_HMM(PRED): {}'.format(nCorrect, nTokens, nCorrect/nTokens, 
            nCorrectOOV, nOOV, nCorrectOOV/nOOV,
            nPerfectSents, len(test_sentences), nPerfectSents/len(test_sentences), 
            nPGoldGreater, nPPredGreater))
print('RUNTIME: TRAINING = {:.2}s, TAGGING = {:.2}s'.format(trainingTime, taggingTime))
print('Sorted Incorrect Tag Counts:\n{}'.format(sorted( ((v,k) for k,v in incorrectCounts.items()), reverse=True)))
