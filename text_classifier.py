import pandas as pd
import numpy as np
import re
import string
import csv
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from gensim.parsing.preprocessing import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_union
from sklearn.metrics import make_scorer, log_loss

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

import gensim
from gensim.models import Word2Vec, Doc2Vec

from feature_union_sklearn import ItemSelector, TextStats, AbstractTitleAuthorExtractor

import pdb

def read_corpus(docs, tokens_only=False):
    for i, line in enumerate(docs):
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

pd.options.mode.chained_assignment = None  # default='warn'

def get_csv(filename):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        result = list(reader)
    return result

# train = pd.read_csv("datasets/train/abstitauth.csv",header=None)
# test = pd.read_csv("datasets/test/abstitauth.csv",header=None)

train_abs = pd.read_csv("datasets/train/abstracts.csv",header=None)
train_titles = pd.read_csv("datasets/train/titles.csv",header=None)
train_authors = pd.read_csv("datasets/train/authors.csv", header=None)
#train_abs = np.array(get_csv("datasets/train/abstracts.csv"))
#train_titles = np.array(get_csv("datasets/train/titles.csv"))
#train_authors = np.array(get_csv("datasets/train/authors.csv"))

#pdb.set_trace()

all_train = np.dstack([train_abs,train_titles,train_authors])
all_train = np.array([t[0] for t in all_train])

#pdb.set_trace()

test_abs = pd.read_csv("datasets/test/abstracts.csv",header=None)
test_titles = pd.read_csv("datasets/test/titles.csv",header=None)
test_authors = pd.read_csv("datasets/test/authors.csv",header=None)

# Get the labels from the train dataset
y_train = list()
with open('train.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        t = line.split(',')
        y_train.append(t[1][:-1])

labels = np.unique(y_train)


########
# PIPELINE
########
##############################################################################
pipeline = Pipeline([
    # ('vect', CountVectorizer(decode_error='ignore',stop_words='english',max_df=0.6,ngram_range=(1,2),min_df=0.0)),
    ('vect', CountVectorizer(decode_error='ignore',stop_words='english')),
    ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
    #('sgd', SGDClassifier(max_iter=5,tol=None)),
    #('mnb', MultinomialNB(alpha=0.1)),
    #('logr', LogisticRegression()),
    ('logr', LogisticRegression(penalty='l2',tol=1e-05)),
    #('rfc', RandomForestClassifier()),
    #('knn', KNeighborsClassifier()),
    # ('NuSVC', NuSVC(probability=True)), # This one SUCKS
    # ('dtc', DecisionTreeClassifier()),
    # ('clf', MLPClassifier(hidden_layer_sizes=(5,3),verbose=True)),
])

# clf = MLPClassifier(  hidden_layer_sizes=(100, ),
#                             activation='relu',
#                             solver='adam',
#                             alpha=0.0001,
#                             momentum=0.9,
#                             verbose=True
#                         )


#################
# FEATURE UNION #
#################

# This is painful because transform (feature selection) using estimators has been
# deprecated in scikit learn. So we are going to go with SelectFromModel
logr_abs = LogisticRegression(penalty='l2',tol=1e-05)
logr_tit = LogisticRegression(penalty='l2',tol=1e-05)
logr_aut = LogisticRegression(penalty='l2',tol=1e-05)

pipeline = Pipeline([
    ('abstracttitleauthor', AbstractTitleAuthorExtractor()),

    # Use FeatureUnion to combine the features from abstracts, titles and authors
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from abstracts
            ('abstract', Pipeline([
                ('selector', ItemSelector(key='abstract')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001)),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs', SelectFromModel(logr_abs)),
            ])),

            # Pipeline for pulling features from titles
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.18, min_df=0)),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('logr', SelectFromModel(logr_tit)),
            ])),

            # Pipeline for pulling features from authors
            ('author', Pipeline([
                ('selector', ItemSelector(key='author')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.03, min_df=0)),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('logr', SelectFromModel(logr_aut)),
            ])),

        ],

        # weight components in FeatureUnion
        # transformer_weights={
        #     'subject': 0.8,
        #     'body_bow': 0.5,
        #     'body_stats': 1.0,
        # },
    )),

    # Use Logistic Regression again on the combined features
    ('logr', LogisticRegression(penalty='l2', tol=1e-05)),

    # Use a SVC classifier on the combined features
    # ('svc', SVC(kernel='linear')),
])

# vect_abs = CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001)
# vect_tit = CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.18, min_df=0)
# vect_aut = CountVectorizer(decode_error='ignore', max_df=0.03, min_df=0)

# tr_abs = vect_abs.fit_transform([t[0] for t in train_abs.values])
# tr_tit = vect_tit.fit_transform([t[0] for t in train_titles.values])
# tr_aut = vect_aut.fit_transform([t[0] for t in train_authors.values])

# tfidf_abs = TfidfTransformer(norm='l2',sublinear_tf=True)
# tfidf_tit = TfidfTransformer(norm='l2',sublinear_tf=True)
# tfidf_aut = TfidfTransformer(norm='l2',sublinear_tf=True)

# tr_abs = tfidf_abs.fit_transform(tr_abs)
# tr_tit = tfidf_tit.fit_transform(tr_tit)
# tr_aut = tfidf_aut.fit_transform(tr_aut)

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
# parameters = {
#     'vect__min_df': (0, 0.00001),
#     'vect__max_df': (0.2, 0.22, 0.25, 0.28, 0.3, 0.6, 0.65),
#     # 'vect__ngram_range': ((1,1),(1,2),(1,3)),
#     # 'tfidf__sublinear_tf': (False,True),
#     #'sgd__loss': ('modified_huber','log'),
#     #'mnb__alpha': (0.0001,0.001,0.01,0.1,1),
#     #'logr__penalty': ('l1','l2'),
#     #'logr__tol': (0.0001,0.00001),
#     #'logr__C': (1.0,0.8,0.5,0.1), # defaults to 1.0 which performs way better
#     #'logr__class_weight': (None, 'balanced'), # defaults to None which performs way better than balanced
#     #'clf__activation': ('relu','logistic'),
#     #'clf__solver': ('adam','sgd'),
#     #'clf__alpha': (0.000001, 0.0000001),
#     #'clf__penalty': ('l2', 'elasticnet'),
#     #'clf__momentum': (0.9,0.8),
#     #'rfc__n_estimators': (10,4,8),
#     #'rfc__criterion': ('gini','entropy'),
#     #'knn__n_neighbors': (5,25,101),
#     #'knn__weights': ('uniform','distance'),
#     #'knn__p': (1,2),
#     # 'NuSVC__nu': (0.01,0.4),
#     # 'NuSVC__kernel': ('sigmoid','rbf'),
#     # 'dtc__criterion': ('gini','entropy'),
#     # 'dtc__class_weight': (None,'balanced')
# }

#log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
#grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10,scoring=log_loss_scorer)

# logr = LogisticRegression(penalty='l2',tol=1e-05)
pipeline.fit(all_train,y_train)

# Get the input and output in the appropriate format 
# x_train = [x[0] for x in train.values]
# x_test  = [x[0] for x in test.values]


# WORD EMBEDDINGS
# vocab = [s.split() for s in x_train]
# vocab_y = [s.split() for s in y_train]
#embeddings = Word2Vec(vocab,min_count=1)
#embeddings.wv['george']

# DOC EMBEDDINGS
# train_corpus = list(read_corpus(x_train))
# test_corpus = list(read_corpus(x_test,tokens_only=True))
# doc_embeddings = Doc2Vec(min_count=1)
# doc_embeddings.build_vocab(train_corpus)
# doc_embeddings.train(train_corpus,total_examples=doc_embeddings.corpus_count,epochs=55)
# doc_embeddings.infer_vector(['physics','good'])

# x_train_emb = [doc_embeddings.infer_vector(x) for x in vocab]
# y_train_emb = [doc_embeddings.infer_vector(y) for y in vocab_y]

#x_train = np.dstack([tr_abs,tr_tit,tr_aut])
x_test = np.dstack([test_abs,test_titles,test_authors])
x_test = np.array([t[0] for t in x_test])

# x_train = np.dstack([train_abs,train_titles])
# x_test = np.dstack([test_abs,test_titles])
#pdb.set_trace()
# logr.fit(tr_abs,y_train)
#grid_search.fit(x_train,y_train)
#grid_search.fit(x_train_emb, y_train)

# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))


####################
# Get test IDs too #
####################
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])


#y_pred = grid_search.predict_proba(x_test)
y_pred = pipeline.predict_proba(x_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    #lst = grid_search.classes_.tolist()
    lst = pipeline.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
