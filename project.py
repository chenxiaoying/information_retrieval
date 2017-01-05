#!/usr/bin/python2.7

import gzip
import nltk
from nltk.tokenize import RegexpTokenizer

## read data from dataset
def read_review():
	rev_file = gzip.open('reviews_Musical_Instruments_5.json.gz','r')
	for line in rev_file:
		yield eval(line)


def get_data(data):
	rev = {}
	tokens = []
	for lines in data:
		product_id = lines['asin']
		review = lines['reviewText']
		rating = lines['overall']
		## create dictionary with key = product_id, value = review,rating
		try:
			rev[product_id].append((review,rating))
		except KeyError:
			rev[product_id] = [(review,rating)]
			
		## create a list with tokenized review text, each sublist in list is a review
		token = RegexpTokenizer(r'\w+').tokenize(review)
		token = [i.lower() for i in token]
		tokens.append(token)
		
		## remove reviewerID, reviewerName, helpful, review time
		del lines['reviewerID']
		del lines['helpful']
		del lines['unixReviewTime']
		del lines['reviewTime']
		try:
			del lines['reviewerName']		
		except KeyError:
			pass
			
	return data,rev,tokens

## pos tagging the words to get informative words
def pos_tagger(tokens):
	tagged = []
	tagged_info = []
	for lines in tokens:
		info = []
		tag = nltk.pos_tag(lines)
		tagged.append(tag)
		
		for i in tag:
			## keep informative words, words like: good,nice are JJ
            ## NN: quality,perfect,love,nice; RB: smoothly,nicely,very,not; VBN: satisfied,disappointed
			if i[1] == 'JJ' or i[1] == 'NN' or i[1] == 'RB' or i[1] == 'JJR' or i[1] == 'VBN':
				info.append(i)
		tagged_info.append(info)
		print(info)
	
	return tagged, tagged_info
	

dataset = list(read_review())
data,reviews,tokens = get_data(dataset)
pos_tagger(tokens)
