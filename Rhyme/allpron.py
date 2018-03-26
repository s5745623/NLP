#!/usr/bin/env python3

import json, re
from collections import defaultdict
import nltk # need to have downloaded the data through NLTK
import json

def LoadJson():

	data = json.load(open('chaos.json'))

	return data


def Prons(data):

	entries = nltk.corpus.cmudict.entries()
	
	#data[i]['lines'][j]['rhymeWord'] i=0-49  j=0-3
	print('It may take a while to run through the entire cmudict\n')
	prons = []
	for j in range(len(data)):
		for k in range(len(data[j]['lines'])):
			for i in range(len(entries)):
				if entries[i][0]== data[j]['lines'][k]['rhymeWord'].lower():
					#clean prons list
					prons.append(' '.join(entries[i][1]))
					#add to json
					data[j]['lines'][k]['rhymeProns'] =  prons
			prons = []


	#add to json

	return data


def OOVs(data):
	
	OOV = []
	for j in range(len(data)):
		for k in range(len(data[j]['lines'])):
			if len(data[j]['lines'][k]) == 5:	
				OOV.append(data[j]['lines'][k]['rhymeWord'])

	print('Total out-of-vocabulary word count:',len(OOV))
	OOV_10 = ', '.join(OOV[0:10])
	print('10 examples of the out-of-vocabulary words:',OOV_10)

	return OOV

def toJson(data):

	json.dump(data, open("chaos.pron.json", "w"), indent=2)
	
if __name__ == '__main__':

	data = LoadJson()
	data_prons = Prons(data)
	OOV = OOVs(data_prons)

	toJson(data_prons)


""" Sample part of the output (hand-formatted):

{"lineId": "1-1", "lineNum": 1,
 "text": "Dearest creature in creation",
 "tokens": ["Dearest", "creature", "in", "creation"],
 "rhymeWord": "creation",
 "rhymeProns": ["K R IY0 EY1 SH AH0 N"]},

{"lineId": "4-1", "lineNum": 1,
 "text": "Previous, precious, fuchsia, via",
 "tokens": ["Previous", ",", "precious", ",", "fuchsia", ",", "via"],
 "rhymeWord": "via",
 "rhymeProns": ["V AY1 AH0", "V IY1 AH0"]
},
"""

# Load the cmudict entries into a data structure.
# Store each pronunciation as a STRING of phonemes (separated by spaces).
...

# Load chaos.json
...

# For each line of the poem, add a "rhymeProns" entry
# which is a list parallel with "rhymeWords".
# For each word, it contains a list of possible pronunciations.
...

# Write the enhanced data to chaos.pron.json
...

"""
TODO: Answer the question:

- How many rhyme words are NOT found in cmudict (they are "out-of-vocabulary", or "OOV")?
Give some examples.
	total out-of-vocabulary word count: 17
	10 examples of the out-of-vocabulary words: ague, Terpsichore, reviles, endeavoured, tortious, clamour, clangour, hygienic, inveigle, mezzot
	
	Improvment:
	if could use nltk.corpus.cmudict.dict()['rhymeWord'] to search Prons to implement faster algorithm.
"""
