#!/usr/bin/python2.7

import gzip
from nltk.tokenize import RegexpTokenizer

## read data from dataset
def read_review():
	rev_file = gzip.open('reviews_Musical_Instruments_5.json.gz','r')
	for line in rev_file:
		yield eval(line)


## remove reviewerID, reviewerName, helpful, review time
## create dictionary with key = product_id, value = review,rating
## create a list with tokenized review text, each sublist in list is a review
def get_data(data):
	rev = {}
	tokens = []
	for lines in data:
		product_id = lines['asin']
		review = lines['reviewText']
		rating = lines['overall']
		try:
			rev[product_id].append((review,rating))
		except KeyError:
			rev[product_id] = [(review,rating)]
		token = RegexpTokenizer(r'\w+').tokenize(review)
		token = [i.lower() for i in token]
		tokens.append(token)
		del lines['reviewerID']
		del lines['helpful']
		del lines['unixReviewTime']
		del lines['reviewTime']
		try:
			del lines['reviewerName']		
		except KeyError:
			pass
			
	return data,rev,tokens



dataset = list(read_review())
data,reviews,tokens = get_data(dataset)
for i in tokens:
	print(i)
