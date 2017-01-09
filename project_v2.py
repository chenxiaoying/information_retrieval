#!/usr/bin/python2.7
import nltk.classify
from nltk.tokenize import word_tokenize
import nltk
#dl = nltk.downloader.Downloader("http://nltk.github.com/nltk_data/")
#dl.download()
from featx import bag_of_words, high_information_words, bag_of_words_in_set
from classification import precision_recall

from nltk.tokenize import RegexpTokenizer

from random import shuffle
from os import listdir # to read files
from os.path import isfile, join # to read files
import sys
import gzip

## read data from dataset
def open_review():
	rev_file = gzip.open('reviews_Musical_Instruments_5.json.gz','r')
	for line in rev_file:
		yield eval(line)

def read_data(data):
	feats = list()
	num_files = 0
	for entry in data:
		review = entry['reviewText']
		score = entry['overall']
		tokens = RegexpTokenizer(r'\w+').tokenize(review)
		bag = bag_of_words(tokens)
		feats.append((bag,str(score)))
		num_files += 1
	print("%i reviews read." % (num_files))
	return feats

# prints accuracy, precision and recall
def evaluation(classifier, test_feats, categories):
	print ("\n##### Evaluation...")
	print("  Accuracy: %f" % nltk.classify.accuracy(classifier, test_feats))
	precisions, recalls = precision_recall(classifier, test_feats)
	f_measures = calculate_f(precisions, recalls)  
	
	print(" |-----------|-----------|-----------|-----------|")
	print(" |%-11s|%-11s|%-11s|%-11s|" % ("category","precision","recall","F-measure"))
	print(" |-----------|-----------|-----------|-----------|")
	for category in categories:
		print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], f_measures[category]))
	print(" |-----------|-----------|-----------|-----------|")
	

# show informative features
def analysis(classifier):
	print("\n##### Analysis...")
	classifier.show_most_informative_features(10)

# trains a classifier
def train(train_feats):
	#nb_classifier = NaiveBayesClassifier.train(train_feats)

	classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)
	# DecisionTreeClassifier is too slow, even with an entropy cut off of 2
	#classifier = nltk.classify.DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=2, depth_cutoff=100, support_cutoff=10)
	return classifier

	# I tried this code, but the average accuracy decreases when using this

	#from nltk.probability import LaplaceProbDist
	#classifier = nltk.classify.NaiveBayesClassifier.train(train_feats, estimator=LaplaceProbDist)
	#return classifier


def calculate_f(precisions, recalls):
	f_measures = {}
	#calculate the f measure for each category using as input the precisions and recalls
	for category in precisions:
		f_measures[category] = 2 * ((precisions[category]*recalls[category])/(precisions[category]+recalls[category]))
	
	return f_measures

# splits a labelled dataset into two disjoint subsets train and test
def split_train_test(feats, split=0.9):
	train_feats = []
	test_feats = []

	shuffle(feats) # randomise dataset before splitting into train and test
	cutoff = int(len(feats) * split)
	train_feats, test_feats = feats[:cutoff], feats[cutoff:]	
	"""
	print("\n##### Splitting datasets...")
	print("  Training set: %i" % len(train_feats))
	print("  Test set: %i" % len(test_feats))
	"""
	return train_feats, test_feats

def main(argv):
	categories = ['1.0','2.0','3.0','4.0','5.0']
	data = list(open_review())
	feats = read_data(data)
	
	print("\n#### Accuracy scores:\n")
	accuracy_sum = 0
	for N in range(1): # towards n-fold cross validation?
		train_feats, test_feats = split_train_test(feats)
		classifier = train(train_feats)
		accuracy = nltk.classify.accuracy(classifier, test_feats)
		accuracy_sum += accuracy
		print(accuracy)
		evaluation(classifier, test_feats, categories)
		analysis(classifier)
	

if __name__ == "__main__":
	main(sys.argv)