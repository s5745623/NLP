#!/usr/bin/env python3

import json, re
from collections import defaultdict
import nltk 
from nltk.corpus import cmudict # need to have downloaded the data through NLTK


def LoadJson():

	data = json.load(open('chaos.pron.json'))

	return data


def isExactRhyme(data):

	Rhyme = 0
	MultiProns = 0
	
	for j in range(len(data)):
	#for j in range(len(data)):
		
		k = 0
		flag = False # for breaking outside for
		# 2-syllable
		#line 1,2
		for q in range(len(data[j]['lines'][k]['rhymeProns'])):
			for g in range(len(data[j]['lines'][k+1]['rhymeProns'])):
				if nltk.word_tokenize(''.join(data[j]['lines'][k]['rhymeProns'][q]))[-2:] == nltk.word_tokenize(''.join(data[j]['lines'][k+1]['rhymeProns'][g]))[-2:]:
					if len(data[j]['lines'][k+1]['rhymeProns']) >= 2  or len(data[j]['lines'][k]['rhymeProns']) >= 2:
						MultiProns += 1
					Rhyme += 1
					flag = True
					break
			if flag == True:
				break		
		flag = False		

		#line 3,4
		for x in range(len(data[j]['lines'][k+2]['rhymeProns'])):
			for y in range(len(data[j]['lines'][k+3]['rhymeProns'])):
				if nltk.word_tokenize(''.join(data[j]['lines'][k+2]['rhymeProns'][x]))[-2:] == nltk.word_tokenize(''.join(data[j]['lines'][k+3]['rhymeProns'][y]))[-2:]:
					if len(data[j]['lines'][k+2]['rhymeProns']) >= 2 or len(data[j]['lines'][k+3]['rhymeProns']) >= 2:
						MultiProns += 1
					Rhyme += 1
					flag = True
					break
			if flag == True:
				break
		flag = False		


	return Rhyme, MultiProns# TODO: whether pronunciations p1 and p2 rhyme

def PreProcessJson(data):

	for j in range(len(data)):
		for k in range(len(data[j]['lines'])):
			if len(data[j]['lines'][k]) == 5:
				# giving OOV_1 to line 1 ...
				OOV_line = 'OOV_' + str(k)	
				data[j]['lines'][k]['rhymeProns'] = [OOV_line]

	return data
if __name__ == '__main__':

	data = LoadJson()
	clean_data = PreProcessJson(data)
	TotalRhyme,MultiProns = isExactRhyme(clean_data)
	print('TotalRhyme: ',TotalRhyme)
	print('MultiProns: ',MultiProns)


# Load chaos.pron.json
...

# For each pair of lines that are supposed to rhyme,
# check whether there are any pronunciations of the words that
# make them rhyme according to cmudict and your heuristic.
# Print the rhyme words with their pronunciations and whether
# they are deemed to rhyme or not
# so you can examine the effects of your rhyme detector.
# Count how many pairs are deemed to rhyme vs. not.
...

"""
TODO: Answer the questions briefly:

- How many pairs of lines that are supposed to rhyme actually have rhyming pronunciations
according to your heuristic and cmudict?

... 60

- For how many lines does having the rhyming line help you disambiguate
between multiple possible pronunciations?

... 23

- What are some reasons that lines supposed to rhyme do not,
according to your rhyme detector? Give examples.

... The algorithm I used for checking rhyme lines is by using 'Syllable-timed Rhythm', 
more specifically is 2-syllable rhythm which is by matching the last two syllables. 
The program can not match or detect the 'Stress-timed Rhythm'. which means; 
[ iron: "AY1 ER0 N"; lion: "L AY1 AH0 N" ] this pair or other rhythmword won't 
match as rhythm lines in my program.

"""
