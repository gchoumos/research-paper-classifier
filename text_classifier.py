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
from sklearn.feature_extraction import DictVectorizer
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
from sklearn.svm import NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier

import gensim
from gensim.models import Word2Vec, Doc2Vec

from feature_union_sklearn import ItemSelector, TextStats, AbstractTitleAuthorExtractor, GraphProperties

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

train_abs = pd.read_csv("datasets/train/abstracts.csv",header=None)
train_titles = pd.read_csv("datasets/train/titles.csv",header=None)
train_authors = pd.read_csv("datasets/train/authors.csv", header=None)
train_citations_in = pd.read_csv("datasets/train/incoming_citations.csv", header=None)
train_citations_out = pd.read_csv("datasets/train/outgoing_citations.csv", header=None)
train_outdeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[0])
train_indeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[1])
train_avg_neigh_deg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[2])

all_train = np.dstack([
                        train_abs,
                        train_titles,
                        train_authors,
                        train_citations_in,
                        train_citations_out,
                        train_outdeg,
                        train_indeg,
                        train_avg_neigh_deg
                      ])
all_train = np.array([t[0] for t in all_train])

test_abs = pd.read_csv("datasets/test/abstracts.csv",header=None)
test_titles = pd.read_csv("datasets/test/titles.csv",header=None)
test_authors = pd.read_csv("datasets/test/authors.csv",header=None)
test_citations_in = pd.read_csv("datasets/test/incoming_citations.csv", header=None)
test_citations_out = pd.read_csv("datasets/test/outgoing_citations.csv", header=None)
test_outdeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[0])
test_indeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[1])
test_avg_neigh_deg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[2])

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
# pipeline = Pipeline([
#     # ('vect', CountVectorizer(decode_error='ignore',stop_words='english',max_df=0.6,ngram_range=(1,2),min_df=0.0)),
#     ('vect', CountVectorizer(decode_error='ignore',stop_words='english')),
#     ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
#     #('sgd', SGDClassifier(max_iter=5,tol=None)),
#     #('mnb', MultinomialNB(alpha=0.1)),
#     #('logr', LogisticRegression()),
#     ('logr', LogisticRegression(penalty='l2',tol=1e-05)),
#     #('rfc', RandomForestClassifier()),
#     #('knn', KNeighborsClassifier()),
#     # ('NuSVC', NuSVC(probability=True)), # This one SUCKS
#     # ('dtc', DecisionTreeClassifier()),
#     # ('clf', MLPClassifier(hidden_layer_sizes=(5,3),verbose=True)),
# ])

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
logr_cit_in = LogisticRegression(penalty='l2',tol=1e-05)
logr_cit_out = LogisticRegression(penalty='l2',tol=1e-05)
logr_outdeg = LogisticRegression(penalty='l2',tol=1e-05)
logr_indeg = LogisticRegression(penalty='l2',tol=1e-05)
logr_avg_neigh_deg = LogisticRegression(penalty='l2',tol=1e-05)


thres_all = None
pipeline = Pipeline([
    ('abstracttitleauthor', AbstractTitleAuthorExtractor()),

    # Use FeatureUnion to combine the features from abstracts, titles and authors
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from abstracts
            ('abstract', Pipeline([
                ('selector', ItemSelector(key='abstract')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001,ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

            # Pipeline for pulling features from titles
            ('title', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.18, min_df=0)),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_tit', SelectFromModel(logr_tit,threshold=thres_all)),
            ])),

            # Pipeline for pulling features from authors
            ('author', Pipeline([
                ('selector', ItemSelector(key='author')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.03, min_df=0)),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_aut', SelectFromModel(logr_aut,threshold=0.55)),
            ])),

            # Pipeline for pulling features from authors
            ('incoming_citations', Pipeline([
                ('selector', ItemSelector(key='cit_in')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_aut', SelectFromModel(logr_cit_in,threshold=0.15)),
            ])),

            ('outgoing_citations', Pipeline([
                ('selector', ItemSelector(key='cit_out')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_aut', SelectFromModel(logr_cit_out,threshold=0.1)),
            ])),

            # consider having all the graph properties to a single logistic regression
            ('gprops', Pipeline([
                ('selector', ItemSelector(key='graph_props')),
                ('props', GraphProperties()),
                ('vect_graph_props', DictVectorizer()),
            ])),

            # Pipeline for abstract stats
            ('abs_stats', Pipeline([
                ('selector', ItemSelector(key='abstract')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect_abs_stats', DictVectorizer()),
            ])),

            # Pipeline for title stats
            ('aut_stats', Pipeline([
               ('selector', ItemSelector(key='author')),
               ('stats', TextStats()),  # returns a list of dicts
               ('vect_titles_stats', DictVectorizer()),
            ])),

    
        ],

        # weight components in FeatureUnion - Check the result notes file for more
        transformer_weights={
            'abstract': 1.40,
            'title': 0.50,
            'author': 1.2,
            #'abs_stats': 0.9,
            'incoming_citations': 1.3,
            'outgoing_citations': 1.3,
			#'gprops': 0.6,
            #'aut_stats': 0.7,
        },
    )),

    # Use Logistic Regression again on the combined features
    #('sgd', SGDClassifier(loss='modified_huber')),
	# ####################
    # Default solver is liblinear, however I am trying more (with the best parameters from above).
    # newton-cg --> score 1.997
    # sag --> score 2.745 lol
    # saga --> score 2.73 lol
    # lbfgs --> score not good
    # newton-cg with multi_class: multinomial --> score --> took much time. Try again
    ('logr', LogisticRegression(penalty='l2',tol=0.0001)),
    # Use an SVC classifier on the combined features
    #('svc', SVC(verbose=False,kernel='linear',probability=True)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
#     'vect__min_df': (0, 0.00001),
#     'vect__max_df': (0.2, 0.22, 0.25, 0.28, 0.3, 0.6, 0.65),
#     # 'vect__ngram_range': ((1,1),(1,2),(1,3)),
#     # 'tfidf__sublinear_tf': (False,True),
#     #'sgd__loss': ('modified_huber','log'),
#     #'mnb__alpha': (0.0001,0.001,0.01,0.1,1),
#     #'logr__penalty': ('l1','l2'),
#     'logr__tol': (0.0001,0.00001),
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
}

log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10,scoring=log_loss_scorer)

grid_search.fit(all_train,y_train)

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
x_test = np.dstack([
                        test_abs,
                        test_titles,
                        test_authors,
                        test_citations_in,
                        test_citations_out,
                        test_outdeg,
                        test_indeg,
                        test_avg_neigh_deg
                  ])
x_test = np.array([t[0] for t in x_test])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

####################
# Get test IDs too #
####################
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

y_pred = grid_search.predict_proba(x_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = grid_search.classes_.tolist()
    #lst = pipeline.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
