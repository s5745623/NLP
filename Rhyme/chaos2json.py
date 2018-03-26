#!/usr/bin/env python3

"""
Converts chaos.html into JSON. A sample of the input:

<xxx1><p>Dearest <i>creature</i> in <i>creation</i><br>
<xxx2>Studying English <i>pronunciation</i>,<br>
<xxx3><tt>&nbsp;&nbsp;&nbsp;</tt>I will teach you in my <i>verse</i><br>
<xxx4><tt>&nbsp;&nbsp;&nbsp;</tt>Sounds like <i>corpse</i>, <i>corps</i>, <i>horse</i> and <i>worse</i>.</p>

A hand-formatted portion of the output (note that indentation, line breaks,
order of dict entries, etc. don't matter as long as the data matches):

[
    {"stanza": 1,
     "lines": [
          {"lineId": "1-1", "lineNum": 1, "text": "Dearest creature in creation",
           "tokens": ["Dearest", "creature", "in", "creation"],
           "rhymeWord": "creation"},  
          {"lineId": "1-2", "lineNum": 2, "text": "Studying English pronunciation,",
           "tokens": ["Studying", "English", "pronunciation", ","],
           "rhymeWord": "pronunciation"},
          ...
     ]},
    {"stanza": 2,
     "lines": [
          {"lineId": "2-1", "lineNum": 1, "text": "I will keep you, Susy, busy,",
           "tokens": ["I", "will", "keep", "you", ",", "Susy", ",", "busy", ","],
           "rhymeWord": "busy"},
          ...
     ]},
     ...
]
#tokenize
nltk.word_tokenize(string)

#rythm
words = nltk.word_tokenize(string)
words=[word for word in words if word.isalpha()]
rythm = words[-1]
"""


import json, re
import nltk 
import codecs

# return stanzas and xxx1-xxx4 collection list
def input():
  
  f = codecs.open("chaos.html", 'r', 'utf-8').read()
  #xxx1
  xxx_list = ['xxx1','xxx2','xxx3','xxx4']
  for i in range(1,5):
    regex = '<xxx'+ str(i) +'>(.+)'
    reg = re.compile(regex)
    xxx_list[i-1] = re.findall(reg, f)
    xxx_list[i-1] = cleanhtml(xxx_list[i-1])
  #print(nltk.word_tokenize(xxx_list[0][0]))

  stanzas=[]
  for k in range(50):
    for j in range(4):
      #print(xxx_list[j][k])
      stanzas.append(xxx_list[j][k])

    #print('\n')
  #print(stanzas[0:4])

  return stanzas,xxx_list

#clean html to list
def cleanhtml(a):

  cleanr = re.compile('<.*?>')
  xxx4 = re.compile('&nbsp;&nbsp;&nbsp;')
  hyphens = re.compile('-')
  cleantext_1 = []
  cleantext_2 = []
  cleantext = []
  for i in range(len(a)):
    cleantext_1.append(re.sub(cleanr, '', a[i]))
    cleantext_2.append(re.sub(hyphens, ' - ', cleantext_1[i]))
    cleantext.append(re.sub(xxx4, '', cleantext_2[i]))

  return cleantext

# rhythms token list
def token(stanzas):

  import string

  translator = str.maketrans('', '', string.punctuation)
  rhymes = []
  tokens = []
  for i in range(len(stanzas)):
    tokens.append(nltk.word_tokenize(stanzas[i]))
  for i in range(len(stanzas)):
    rhymes.append(nltk.word_tokenize(stanzas[i].translate(translator))[-1])

  return tokens,rhymes

def tojson(tokens, rhymes,stanzas):

  final = [ { "stanza":i+1, 
              "lines":[ 
              {
              "lineId":"%d-%d"%(i+1,j+1), 
              "lineNum":j+1, 
              "text":stanzas[4*i+j],
              "tokens":tokens[4*i+j],
              "rhymeWord":rhymes[4*i+j]
              } 
              for j in range(4) ] 
            } 
              for i in range(int(len(tokens)/4)) 
          ]

  json.dump(final, open("chaos.json", "w"), indent=2)

  return final

if __name__ == '__main__':

  stanzas, xxx_list = input()
  tokens, rhymes = token(stanzas)
  final = tojson(tokens, rhymes, stanzas)
  #print(final)

# TODO: read from chaos.html, construct data structure, write to chaos.json
