import pandas as pd
import numpy as np
import re
import gensim
import stop_words

from gensim import corpora
from gensim import models
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import STOPWORDS
from time import time
import string
import csv

import nltk
nltk.download()
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

pd.options.mode.chained_assignment = None  # default='warn'

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.head())

####################
####################
# Take care of the training target stuff
# Read training data
train_ids = list()
y_train = list()
with open('train.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        t = line.split(',')
        train_ids.append(t[0])
        y_train.append(t[1][:-1])
        if i<5:
            print("t[0]: ", t[0])
            print("t[1][:-1]: ", t[1][:-1]);
            i += 1
#####################
#####################

train_abstracts = list()
train_titles = list()
train_authors = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    train_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
    train_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])

#ALLabstracts = pd.concat([x['abstract'] for x in [df]])
ALLabstracts = pd.Series(v for v in train_abstracts)

#ALLtitles = pd.concat([x['title'] for x in [df]])
ALLtitles = pd.Series(v for v in train_titles)

ALLabstracts.head(5)
ALLtitles.head(5)

#ALLauthors = pd.concat([x['authors'] for x in [df]])
ALLauthors = pd.Series(v for v in train_authors)

rm_punct = string.punctuation.replace('-','')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in rm_punct])

ALLabstracts = ALLabstracts.str.replace(RE_PUNCTUATION,' ')
ALLtitles = ALLtitles.str.replace(RE_PUNCTUATION,' ')
ALLauthors = ALLauthors.str.replace(RE_PUNCTUATION,' ')

ALLabstracts_split = ALLabstracts.str.split()
ALLtitles_split = ALLtitles.str.split()
ALLauthors_split = ALLauthors.str.split()

stopwords = set(STOPWORDS)
ALLabstracts_split = ALLabstracts_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])
ALLabstracts_split.head(10)

ALLtitles_split = ALLtitles_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])

ALLabstracts_split = ALLabstracts_split.apply(lambda tokens: [token for token in tokens if len(token)>1])
ALLtitles_split = ALLtitles_split.apply(lambda tokens: [token for token in tokens if len(token)>1])

# Replace first all the NaNs with a character as they keep getting in the way
ALLauthors_split = ['n' if x is np.nan else x for x in ALLauthors_split]
ALLauthors_split = ALLauthors.str.split()

for i in range(len(ALLauthors_split)):
    if ALLauthors_split[i] != ALLauthors_split[i]:
        ALLauthors_split[i] = ['n']

ALLauthors_split = ALLauthors_split.apply(lambda tokens: [token for token in tokens if len(token)>1])

#########################################################
#########################################################

ALLauthors_split.head(5)
ALLtitles_split.head(5)

## I'll try merging the abstracts and titles
ALLabstitles = [a + b for a, b in zip(ALLabstracts_split, ALLtitles_split)]
ALLabstitles_j = [' '.join(x) for x in ALLabstitles]

# This is now a list of lists. To see the first 2 let's say entries
ALLabstitles_j[:2]

########
# PIPELINE
########
# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(decode_error='ignore',stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__min_df': (0, 0.1),
    'vect__max_df': (0.5, 0.8, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__norm': (None, 'l1', 'l2'),
    'clf__loss': ('log','modified_huber'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

grid_search.fit(ALLabstitles_j, y_train)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

#######
# Until here all good
#######

#########################################################
#########################################################
## I want to do the same for the test data now.
#########################################################
#########################################################
# Read test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Extract the abstract of each test article from the dataframe
n_test = len(test_ids)
test_abstracts = list()
test_titles = list()
for i in test_ids:
    test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    test_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])

TESTabstracts = pd.Series(v for v in test_abstracts)
TESTtitles = pd.Series(v for v in test_titles)

TESTabstracts = TESTabstracts.str.replace(RE_PUNCTUATION,' ')
TESTtitles = TESTtitles.str.replace(RE_PUNCTUATION,' ')

TESTabstracts_split = TESTabstracts.str.split()
TESTtitles_split = TESTtitles.str.split()

TESTabstracts_split = TESTabstracts_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])
TESTtitles_split = TESTtitles_split.apply(lambda tokens: [token for token in tokens if token not in stopwords])

TESTabstracts_split = TESTabstracts_split.apply(lambda tokens: [token for token in tokens if len(token)>1])
TESTtitles_split = TESTtitles_split.apply(lambda tokens: [token for token in tokens if len(token)>1])

TESTabstitles = [a + b for a, b in zip(TESTabstracts_split, TESTtitles_split)]
TESTabstitles_j = [' '.join(x) for x in TESTabstitles]

y_pred = grid_search.predict_proba(TESTabstitles_j)

# Write predictions to a file - LOGISTIC
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = text_clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)













