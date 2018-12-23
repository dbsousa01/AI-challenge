# Task 1 is to train an algorithm in order to distinguish en, es, pt

## TF-IDF Score at Word level implementation
## TF-IDF score represents the relative importance of a term in the document and the entire corpus. 
#TF-IDF score is composed by two terms: the first computes the normalized Term Frequency (TF), the second term is the 
#Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by 
#the number of documents where the specific term appears.

#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)

from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import pandas as pd 
import numpy as np 
import os

PATH = os.getcwd()
texts_pt = []
labels = []
texts_en = []
texts_es = []
X_test = []
#Open file and populate arrays to create dataset - pt
with open(PATH + '/langid-data/task1/data.pt') as fp:
	for i, line in enumerate(fp):
		texts_pt.append(line.split("\n")[0])
		if i == 100000: #might be enough for a dataset to this problem - ? Should introduce a shuffle read in the future
			break

fp.close()

#Open file and populate arrays to create dataset - en
with open(PATH + '/langid-data/task1/data.en') as fp:
	for i, line in enumerate(fp):
		texts_en.append(line.split("\n")[0])
		if i == 100000: #might be enough for a dataset to this problem - ?
			break

fp.close()

#Open file and populate arrays to create dataset - es
with open(PATH + '/langid-data/task1/data.es') as fp:
	for i, line in enumerate(fp):
		texts_es.append(line.split("\n")[0])
		if i == 100000: #might be enough for a dataset to this problem - ?
			break

fp.close()

## Test set
with open(PATH + "/langid.test") as fp:
	for i, line in enumerate(fp):
			X_test.append(line.split("\n")[0])
			
fp.close()

label_pt = ['pt'] * len(texts_pt)
label_en = ['en'] * len(texts_en)
label_es = ['es'] * len(texts_es)

texts = texts_pt
texts.extend(texts_en)
texts.extend(texts_es)
labels = label_pt
labels.extend(label_en)
labels.extend(label_es)

#print(len(texts))
#print(len(labels))

trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
#print(trainDF)


# split the dataset into training and validation datasets 
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_valid = encoder.fit_transform(y_valid)

#Calculate the TF-IDF words vector
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=1000000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(X_train)
xvalid_tfidf =  tfidf_vect.transform(X_valid)
xtest_tfidf = tfidf_vect.transform(X_test)

# Naive Bayes on Word Level TF IDF Vectors
clf = MultinomialNB()
clf.fit(xtrain_tfidf, y_train)

predictions = clf.predict(xvalid_tfidf)

acc = metrics.accuracy_score(predictions, y_valid)
print("Accuracy of the model:", acc) # It's around 86%

#Run with the test set
predictions = clf.predict(xtest_tfidf)
predictions = encoder.inverse_transform(predictions)

np.savetxt('task1-result.txt', predictions, fmt='%s')