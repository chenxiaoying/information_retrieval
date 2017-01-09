#!/usr/bin/python2.7

import gzip
import nltk
from nltk.tokenize import RegexpTokenizer
from random import shuffle
import collections, itertools
from ast import literal_eval as make_tuple


## read data from dataset
def read_review():
	rev_file = gzip.open('reviews_Musical_Instruments_5.json.gz','r')
	for line in rev_file:
		yield eval(line)


def get_data(data):
	rev = {}
	rev['pos'] = []
	rev['neg'] = []
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

		if lines['overall'] == 5.0 or lines['overall'] == 4.0:
			rev['pos'].append(token)
		else:
			rev['neg'].append(token)	
		n +=1
		#if n >= 5000:
		#	break
		
	print(' reviews read')			
	return rev,tokens

## pos tagging the words to get informative words
def pos_tagger(tokens):
	featss = []
	for lines in tokens:
		for rev in tokens[lines]:
			info = []
			tag = nltk.pos_tag(rev)
			
			for i in tag:
				## keep informative words, words like: good,nice are JJ
				## NN: quality,perfect,love,nice; RB: smoothly,nicely,very,not; VBN: satisfied,disappointed
				if i[1] == 'JJ' or i[1] == 'NN' or i[1] == 'RB' or i[1] == 'JJR' or i[1] == 'VBN':
					info.append(i[0])

			feat = dict([(word, True) for word in info])
			featss.append((feat,str(lines)))
			
	#print(featss)
	
	print('\n pos tagger done')
	return featss
	
## read the pos tagged reviews from the file
def pos_tag_file(p_file):
	feats = []
	for lists in p_file:
		lists = lists.rstrip()
		lists = make_tuple(lists)
		feats.append(lists)
		
		
	print('features with pos tagger read')
	return feats


## words without pos tag
def bag_words(rev):
	featss = []
	for i in rev:
		for reviews in rev[i]:
			feat = dict([(word, True) for word in reviews])
			featss.append((feat,str(i)))
	
	print('\n words into bag done')		
	return featss
	
## create bigrams of the reviews
def bigram(rev):
	bigrams = []
	for ratings in rev:
		for reviews in rev[ratings]:
			bigr = []
			for n in range(len(reviews)):
				try:
					bi = reviews[n] + ' ' + reviews[n+1]
					bigr.append(bi)
				except IndexError:
					pass
			feat = dict([(word, True) for word in bigr])
			bigrams.append((feat,str(ratings)))
	#print(bigrams)
	
	print('\n bigrams done')
	return bigrams
	

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
	## decision tree
	print('\n #########start classify########### \n')
	classifier = nltk.classify.DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=300)
	
	#classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)

	return classifier

def precision_recall(classifier, testfeats):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)
	
	for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)
	
	precisions = {}
	recalls = {}
	
	for label in classifier.labels():
		precisions[label] = nltk.metrics.precision(refsets[label], testsets[label])
		recalls[label] = nltk.metrics.recall(refsets[label], testsets[label])
	
	return precisions, recalls

# prints accuracy, precision and recall, f-score
def evaluation(classifier, test_feats, score):
	print ("\n##### Evaluation...")
	print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
	precisions, recalls = precision_recall(classifier, test_feats)  
	
	f_scores = {}

	print(" |---------------|---------------|---------------|---------------|")
	print(" |%-15s|%-15s|%-15s|%-15s|" % ("score","precision","recall","F-measure"))
	print(" |---------------|---------------|---------------|---------------|")
	for score,scores in zip(precisions,recalls):
		if precisions[score] is not None and recalls[score] is not None and precisions[score] != 0.0 and recalls[score] != 0.0:
			f_score = (2 * precisions[score] * recalls[score]) / (precisions[score] + recalls[score])
			f_scores[score] = f_score
		try:	
			print(" |%-15s|%-15s|%-15s|%-15s|" % (score,precisions[score],recalls[score],f_scores[score]))
			print(" |---------------|---------------|---------------|---------------|")
		except KeyError:
			print(" |%-15s|%-15s|%-15s|%-15s|" % (score,precisions[score],recalls[score],'f_score'))
			print(" |---------------|---------------|---------------|---------------|")
			
# show informative features
def analysis(classifier):
	print("\n##### Analysis...")
	classifier.show_most_informative_features(10)

if __name__ == "__main__":
	"""
	#########read reviews
	dataset = list(read_review())
	reviews,tokens = get_data(dataset)
	"""
	
	"""
	######### write pos tagger to file
	f = open('pos.txt','w+')
	for i in feats:
		i = str(i) + '\n'
		f.write(str(i))
	f.close()"""
	
	"""
	#featss = bigram(reviews)
	#featss = pos_tagger(reviews)
	#featss = bag_words(reviews)
	"""
	
	p_file = open('pos.txt','r')
	featss = pos_tag_file(p_file)
	acc = []
	score = ['neg','pos']
	for n in range(1):
		train_rev, test_set = split_train_test(featss)
		classifier = train(train_rev)
		evaluation(classifier,test_set,score)
		accuracy = nltk.classify.accuracy(classifier, test_set)
		acc.append(accuracy)
		analysis(classifier)
	for i in acc:
		print(i)
	average = sum(acc) / len(acc)
	print(average)
