#!/usr/bin/env python3
"""
ENLP A4: HMM for Part-of-Speech Tagging

Usage: 
  python tagger.py baseline
  python tagger.py hmm

(Nathan Schneider; adapted from Richard Johansson)
"""
from math import log, isfinite,inf
import math
from collections import Counter
from collections import defaultdict
import sys, os, time, platform, nltk
from nltk.tag.util import untag

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

START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"

ALPHA = .1
allTagCounts = Counter()
# use Counters inside these
# initial 0
perWordTagCounts = defaultdict(int)
transitionCounts = defaultdict(int) #tag tag
emissionCounts = defaultdict(int) # tag word
# log probability distributions: do NOT use Counters inside these because missing Counter entries default to 0, not log(0)
transitionDists = {}
emissionDists = {}
incorrectOOV = defaultdict(int)

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
    # add pseudocounts in transitionCounts and emissionCounts, including for UNK

    for sentence in tagged_sentences:
        if START not in transitionCounts:
            transitionCounts[START] = Counter()
            transitionCounts[START][sentence[0][1]] = 0
        transitionCounts[START][sentence[0][1]] += 1
        last_tag = ""
        for token in sentence:
            allTagCounts[token[1]] += 1
            #set counter          
            if token[0] not in perWordTagCounts:
                perWordTagCounts[token[0]] = Counter()
                perWordTagCounts[token[0]][token[1]] = 0
            perWordTagCounts[token[0]][token[1]] += 1
            # first
            if last_tag not in transitionCounts:
                transitionCounts[last_tag] = Counter()
                transitionCounts[last_tag][token[1]] = 0
            transitionCounts[last_tag][token[1]] += 1
            last_tag = token[1]
            if token[1] not in emissionCounts:
                emissionCounts[token[1]] = Counter()
                emissionCounts[token[1]][token[0]] = 0
            emissionCounts[token[1]][token[0]] += 1
        # END
        if last_tag not in transitionCounts:
            transitionCounts[last_tag] = Counter()
            transitionCounts[last_tag][END] = 0
        transitionCounts[last_tag][END] += 1
    del transitionCounts[""]
    # normalize counts and store log probability distributions in transitionDists and emissionDists
    # 2 tables
    for tag_t in transitionCounts:
        # set UNK
        transitionCounts[tag_t][UNK] = 0
        transitionDists[tag_t] = {}
        for tag in transitionCounts[tag_t]:
            transitionDists[tag_t][tag] = log((transitionCounts[tag_t][tag]+ALPHA)/
                (sum(transitionCounts[tag_t].values())
                    +ALPHA*len(transitionCounts[tag_t]))) 
    for tag_e in emissionCounts:
        # set UNK
        emissionCounts[tag_e][UNK] = 0
        emissionDists[tag_e] = {}
        for word in emissionCounts[tag_e]:
            emissionDists[tag_e][word] = log((emissionCounts[tag_e][word]+ALPHA)/
                (sum(emissionCounts[tag_e].values())
                    +ALPHA*len(emissionCounts[tag_e])))

def old_learn(tagged_sentences):
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
            perWordTag.append(tagged_sentences[i][j])

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
        max_key,tag = FindMax(token)
        pred_sent.append((max_key[0],tag))
 
    return pred_sent

def FindMax(token):
    #print(perWordTagCounts)
    tag = None

    if token[0] in perWordTagCounts:
        tag = perWordTagCounts[token[0]].most_common(1)[0][0]
    else:
        tag = allTagCounts.most_common(1)[0][0]

    return token,tag

def hmm_tag_sentence(sentence):
    """
    Tag the sentence with the bigram HMM using the Viterbi algorithm.
    Do NOT modify the contents of 'sentence'.
    Return a list of (word, predicted_tag) pairs.
    """
    # fill in the Viterbi chart
    last = viterbi(sentence)
    
    # then retrace your steps from the best way to end the sentence, following backpointers
    sent_lenth = len(sentence)
    tags = retrace(last, sent_lenth)
    
    # finally return the list of tagged words
    tagged_list = []
    j = 0
    for i in sentence:
        j += 1
        tagged_list.append((i[0],tags[j-1]))
    
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
    # current list = [ (current tag, previous, prob)]

    # for each word in the sentence:
    #    previous list = current list
    #    current list = []        
    #    determine the possible tags for this word
    #  
    #    for each tag of the possible tags:
    #         add the highest-scoring item with this tag to the current list
    
    # viterbi algo step
    current = [(START, [], 0)]

    for token in sentence:

        previous = current
        current = []
        possible_Tags = []
        if token[0] in perWordTagCounts:
            possible_Tags = perWordTagCounts[token[0]]
        else:
            possible_Tags = allTagCounts
        # get score
        for each_tag in possible_Tags:
            maxscore = -math.inf
            score = 0
            Max = ()
            for cell in previous:

                tran_score = transitionDists[cell[0]][UNK]
                if each_tag in transitionDists[cell[0]]:
                    tran_score = transitionDists[cell[0]][each_tag]
                emi_score = emissionDists[each_tag][UNK]
                if token[0] in emissionDists[each_tag]:
                    emi_score = emissionDists[each_tag][token[0]]
                
                score = cell[2] + tran_score + emi_score
                # update
                if score > maxscore:
                    maxscore = score
                    Max = (each_tag, cell[1] + [cell[0]], score)
            
            current.append(Max)
            # print(current)

    # end the sequence with a dummy: the highest-scoring item with the tag END
 
    END_tag = END
    maxscore = -math.inf
    score = 0
    previous = current
    current = []
    Max = ()
    for cell in previous:
        
        tran_score = transitionDists[cell[0]][UNK]
        if END_tag in transitionDists[cell[0]]:
            tran_score = transitionDists[cell[0]][END_tag]
        # emi_score = emissionDists[each_tag][UNK]
        # if token[0] in emissionDists[each_tag]:
        #     emi_score = emissionDists[each_tag][token[0]]

        score = cell[2] + tran_score + emi_score
        # update
        if score > maxscore:
            maxscore = score
            Max = ("", cell[1]+[cell[0]], score)
    current.append(Max)

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
    
    ...
    
    # return a new item (tag, best predecessor, best total log probability)


def retrace(end_item, sentence_length):
    # tags = []
    # item = predecessor of end_item
    # while the tag of the item isn't START:
    #     add the tag of item to tags
    #     item = predecessor of item
    # reverse the list of tags and return it

    #ignore start
    tags = []
    for i in end_item[1][1:]:
        tags.append(i)

    assert len(tags)==sentence_length, len(tags)

    return tags

# old idea
def old_Viterbi(sentence, perWordTagCounts):

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

    last_tag = START

    for token in sentence:
        
        tran_score = transitionDists[last_tag][UNK]
        if token[1] in transitionDists[last_tag]:
            tran_score = transitionDists[last_tag][token[1]]
        #print('{} tran:{}'.format(token,tran_score))
        emi_score = emissionDists[token[1]][UNK]
        if token[0] in emissionDists[token[1]]:
            emi_score = emissionDists[token[1]][token[0]] 
        #print('{} emi:{}'.format(token,emi_score))
        p += (tran_score + emi_score)
        last_tag = token[1]

    tran_score = transitionDists[last_tag][UNK]
    if END in transitionDists[last_tag]:
        tran_score = transitionDists[last_tag][END] 
    
    p += tran_score

    assert isfinite(p) and p<0  # Should be negative
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
        if sent[1] == pred_sentence[count][1]:
            correct_sentences += 1
            if sent[0] not in perWordTagCounts:
                OOV_correct_count +=1
        else:
            incorrectOOV[sent[1]] += 1        
                
        if sent[0] not in perWordTagCounts:
            OOV_count += 1

    #nCorrectThisSent, nCorrectOOVThisSent, nOOVThisSent
    return correct_sentences, OOV_correct_count, OOV_count


TRAIN_DATA = 'en-ud-train.upos.tsv'
TEST_DATA = 'en-ud-test.upos.tsv'

train_sentences = read_tagged_corpus(TRAIN_DATA)
# vocabulary = BuildTrainVocab(train_sentences)
# training_set_prep = PreprocessText(train_sentences, vocabulary)

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
# test_set_prep = PreprocessText(test_sentences, vocabulary)

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
print('Sorted Incorrect Tag Counts:\n{}'.format(sorted( ((v,k) for k,v in incorrectOOV.items()), reverse=True)))


