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

# utility functions to read the corpus

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


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

ALPHA = .1
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}
emissionDists = {}

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
    
    word = ""
    tag = ""
    previousTag = START
    
    for each_sentence in tagged_sentences:
        for each_pair in each_sentence:
            allTagCounts[each_pair[1]] += 1
            if each_pair[0] not in perWordTagCounts:
                perWordTagCounts[each_pair[0]] = Counter()
                perWordTagCounts[each_pair[0]][each_pair[1]] = 0
            elif each_pair[1] not in perWordTagCounts[each_pair[0]]:
                perWordTagCounts[each_pair[0]][each_pair[1]] = 0
            perWordTagCounts[each_pair[0]][each_pair[1]] += 1
            
    # add pseudocounts in transitionCounts and emissionCounts, including for UNK
            if previousTag not in transitionCounts:
                transitionCounts[previousTag] = Counter()
                transitionCounts[previousTag][each_pair[1]] = 0
            elif each_pair[1] not in transitionCounts[previousTag]:
                transitionCounts[previousTag][each_pair[1]] = 0
            transitionCounts[previousTag][each_pair[1]] += 1
            
            previousTag = each_pair[1]
            
            if each_pair[1] not in emissionCounts:
                emissionCounts[each_pair[1]] = Counter()
                emissionCounts[each_pair[1]][each_pair[0]] = 0
            elif each_pair[0] not in emissionCounts[each_pair[1]]:
                emissionCounts[each_pair[1]][each_pair[0]] = 0
            emissionCounts[each_pair[1]][each_pair[0]] += 1
        
        if previousTag not in transitionCounts:
            transitionCounts[previousTag] = Counter()
            transitionCounts[previousTag][END] = 0
        elif END not in transitionCounts[previousTag]:
            transitionCounts[previousTag][END] = 0
        transitionCounts[previousTag][END] += 1

    # normalize counts and store log probability distributions in transitionDists and emissionDists
    for given_tag in transitionCounts:
        transitionCounts[given_tag][UNK] = 0
        total = sum(transitionCounts[given_tag].values(),0.0)
        length = len(transitionCounts[given_tag])
        print(length)
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

def baseline_tag_sentence(sentence):
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
        max_key = FindMax(token)
        pred_sent.append((token[0],max_key[1]))
 
    return pred_sent

def FindMax(token):

    maxlist = {}
    for key,value in perWordTagCounts.items():
        if (key[0] == token[0]):
            maxlist[key] = value

    return max(maxlist, key=lambda i: maxlist[i])

def hmm_tag_sentence(sentence):
    """
    Tag the sentence with the bigram HMM using the Viterbi algorithm.
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    # fill in the Viterbi chart
 
    end_item = viterbi(sentence)
    
    # then retrace your steps from the best way to end the sentence, following backpointers
 
    tags = retrace(end_item, len(sentence))
    
    # finally return the list of tagged words
    tagged_list = []
    for idx, each_pair in enumerate(sentence):
        word = each_pair[0]
        tagged_list.append((word, tags[idx]))
    
    return tagged_list



def viterbi(sentence):
    """
    Creates the Viterbi chart, column by column. 
    Each column is a list of tuples representing cells.
    Each cell ("item") holds: the tag being scored at the current position; 
    a reference to the corresponding best item from the previous position; 
    and a log probability. 
    This function returns the END item, from which it is possible to 
    trace back to the beginning of the sentence.
    """
    # make a dummy item with a START tag, no predecessor, and log probability 0
    # current list = [ the dummy item ]
    current = [(START, [], 0)]
    # for each word in the sentence:
    #    previous list = current list
    #    current list = []        
    #    determine the possible tags for this word
    #  
    #    for each tag of the possible tags:
    #         add the highest-scoring item with this tag to the current list
    
 
    for each_pair in sentence:
        word = each_pair[0]
        previous_list = current_list
        current_list = []
        possibleTags = []
        if word in perWordTagCounts:
            possibleTags = perWordTagCounts[word]
        else:
            possibleTags = allTagCounts
        for tag in possibleTags:
            Max = find_best_item(word, tag, previous_list)
            
            current.append(Max)
        

    # end the sequence with a dummy: the highest-scoring item with the tag END
    tag = END
    maxscore = -float("inf")
    Max = ()
    previous_list = current_list
    current_list = []
    score = 0
    for each_turple in previous_list:
        given_tag = each_turple[0]
        predecessor = each_turple[1]
        previous_score = each_turple[2]
        t = transitionDists[given_tag][UNK]
        if tag in transitionDists[given_tag]:
            t = transitionDists[given_tag][tag]
        score = previous_score + t
        if score>maxscore:
            maxscore = score
            Max = (None, predecessor+[given_tag], score)
    current_list.append(Max)

    return Max
    
def find_best_item(word, tag, possible_predecessors):    
    # determine the emission probability: 
    #  the probability that this tag will emit this word
    
    # ...
    
    # find the predecessor that gives the highest total log probability,
    #  where the total log probability is the sum of
    #    1) the log probability of the emission,
    #    2) the log probability of the transition from the tag of the 
    #       predecessor to the current tag,
    #    3) the total log probability of the predecessor

    maxscore = -float("inf")
    Max = ()
    score = 0
    for each_turple in possible_predecessors:
        given_tag = each_turple[0]
        predecessor = each_turple[1]
        previous_score = each_turple[2]
        t = transitionDists[given_tag][UNK]
        if tag in transitionDists[given_tag]:
            t = transitionDists[given_tag][tag]
            
        e = emissionDists[tag][UNK]
        if word in emissionDists[tag]:
            e = emissionDists[tag][word]
        score = previous_score + t + e
        if score>maxscore:
            maxscore = score
            Max = (tag, predecessor+[given_tag], score)
    
    # return a new item (tag, best predecessor, best total log probability)
    return Max

def retrace(end_item, sentence_length):
    # tags = []
    # item = predecessor of end_item
    # while the tag of the item isn't START:
    #     add the tag of item to tags
    #     item = predecessor of item
    # reverse the list of tags and return it
    # ...
    tags = end_item[1][1:]
    assert len(tags)==sentence_length
    return tags

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

TRAIN_DATA = 'en-ud-train.upos.tsv'
TEST_DATA = 'en-ud-test.upos.tsv'

train_sentences = read_tagged_corpus(TRAIN_DATA)


# train the bigram HMM tagger & baseline tagger in one fell swoop
trainingStart = time.time()
learn(train_sentences)
trainingStop = time.time()
trainingTime = trainingStop - trainingStart


# decide which tagger to evaluate
if len(sys.argv)<=1:
    assert False,"Specify which tagger to evaluate: 'baseline' or 'hmm'"
if sys.argv[1]=='baseline':
    tagger = baseline_tag_sentence
elif sys.argv[1]=='hmm':
    tagger = hmm_tag_sentence
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

nTokens = nCorrect = nOOV = nCorrectOOV = nPerfectSents = nPGoldGreater = nPPredGreater = 0

taggingTime = 0

for sent in test_sentences:
    taggerStart = time.time()
    pred_tagging = tagger(sent)
    taggerStop = time.time()
    taggingTime += taggerStop - taggerStart
    nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent = count_correct(sent, pred_tagging)
    
    acc = nCorrectThisSent/len(sent)
    
    pHMMGold = joint_prob(sent)
    pHMMPred = joint_prob(pred_tagging)
    print(pHMMGold, ' '.join(map(render_gold_tag, zip(sent,pred_tagging))))
    print(pHMMPred, ' '.join(map(render_pred_tag, zip(sent,pred_tagging))), '{:.0%}'.format(acc))
    
    if pHMMGold > pHMMPred:
        nPGoldGreater += 1
        #assert False
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


