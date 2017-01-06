#!/usr/bin/python2.7

import gzip
import nltk
from nltk.tokenize import RegexpTokenizer
from random import shuffle

## read data from dataset
def read_review():
	rev_file = gzip.open('reviews_Musical_Instruments_5.json.gz','r')
	for line in rev_file:
		yield eval(line)


def get_data(data):
	rev = {}
	tokens = []
	n = 0
	for lines in data:
		product_id = lines['asin']
		review = lines['reviewText']
		rating = lines['overall']
						
		## create a list with tokenized review text, each sublist in list is a review
		token = RegexpTokenizer(r'\w+').tokenize(review)
		token = [i.lower() for i in token]
		tokens.append(token)
		
		### Create dictionary with key=rating, value = review
		try:
			rev[rating].append(token)
		except KeyError:
			rev[rating] = [token]
		
		n +=1
		if n >= 500:
			break
			
	return rev,tokens

## pos tagging the words to get informative words
def pos_tagger(tokens):
	#tagged = []
	#tagged_info = []
	featss = []
	for lines in tokens:
		for rev in tokens[lines]:
			info = []
			tag = nltk.pos_tag(rev)
			#tagged.append(tag)
			
			for i in tag:
				## keep informative words, words like: good,nice are JJ
				## NN: quality,perfect,love,nice; RB: smoothly,nicely,very,not; VBN: satisfied,disappointed
				if i[1] == 'JJ' or i[1] == 'NN' or i[1] == 'RB' or i[1] == 'JJR' or i[1] == 'VBN':
					info.append(i[0])
			#print(info)
			#tagged_info.append(info)
			feat = dict([(word, True) for word in info])
			featss.append((feat,str(lines)))
			
	#print(tagged_info)
	return featss
	
		

"""code comes from assignment 1"""
# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(feats, split=0.9):
	train_feats = []
	test_feats = []

	shuffle(feats) # randomise dataset before splitting into train and test
	cutoff = int(len(feats) * split)
	train_feats, test_feats = feats[:cutoff], feats[cutoff:]	
	
	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Test set: %i" % len(test_feats))
	return train_feats, test_feats

def train(train_feats):
	classifier = nltk.classify.DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=300)
	return classifier
	
	

dataset = list(read_review())
reviews,tokens = get_data(dataset)
featss = pos_tagger(reviews)
train_rev, test_set = split_train_test(featss)
#print(train_rev)
classifier = train(train_rev)
print(nltk.classify.accuracy(classifier, test_set))
