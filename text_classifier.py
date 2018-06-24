import pandas as pd
import numpy as np
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import log_loss, make_scorer
from feature_union_sklearn import (
                            GraphProperties,
                            ItemSelector,
                            MainExtractor,
                            NodeEmbeddingsVectorizer,
                            TextStats,AuthorStats,
                            WordEmbeddingsVectorizer
                        )
from data_preprocessor import DataPreprocessor
from collections import Counter

import math
import random

pd.options.mode.chained_assignment = None  # default='warn'

ch = False
while not ch:
    c = input("Do you want to run the data preprocessor? (y/n):")
    if c in ['Y','y','N','n']:
        ch = True
    else:
        print("Invalid choice!")

if c in ['Y','y']:
    data_prep = DataPreprocessor()
    data_prep.preprocess()

tr_abs = pd.read_csv("datasets/train/abstracts.csv", header=None)
tr_titles = pd.read_csv("datasets/train/titles.csv", header=None)
tr_authors = pd.read_csv("datasets/train/authors.csv", header=None)
tr_cit_in = pd.read_csv("datasets/train/incoming_citations.csv", header=None)
tr_cit_out = pd.read_csv("datasets/train/outgoing_citations.csv", header=None)
tr_outdeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[0])
tr_indeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[1])
tr_avg_ndeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[2])
tr_comm = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[3])
tr_embs = pd.read_csv("datasets/train/node_embeddings.csv", header=None)
tr_w_embs = pd.read_csv("datasets/train/word_embeddings.csv", header=None)

all_train = np.dstack([
                        tr_abs,
                        tr_titles,
                        tr_authors,
                        tr_cit_in,
                        tr_cit_out,
                        tr_outdeg,
                        tr_indeg,
                        tr_avg_ndeg,
                        tr_comm,
                      ])

all_train = np.array([t[0] for t in all_train])
all_train = np.hstack((all_train, tr_embs))
all_train = np.hstack((all_train, tr_w_embs))

te_abs = pd.read_csv("datasets/test/abstracts.csv", header=None)
te_titles = pd.read_csv("datasets/test/titles.csv", header=None)
te_authors = pd.read_csv("datasets/test/authors.csv", header=None)
te_cit_in = pd.read_csv("datasets/test/incoming_citations.csv", header=None)
te_cit_out = pd.read_csv("datasets/test/outgoing_citations.csv", header=None)
te_outdeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[0])
te_indeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[1])
te_avg_ndeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[2])
te_comm = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[3])
te_embs = pd.read_csv("datasets/test/node_embeddings.csv", header=None)
te_w_embs = pd.read_csv("datasets/test/word_embeddings.csv", header=None)

# Get the labels from the train dataset
y_train = list()
with open('train.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        t = line.split(',')
        y_train.append(t[1][:-1])

labels = np.unique(y_train)

#################
# FEATURE UNION #
#################
# This is painful because transform (feature selection) using
# estimators has been deprecated in scikit learn. So we are
# going to go with SelectFromModel
logr_abs = LogisticRegression(penalty='l2', tol=1e-05)
logr_tit = LogisticRegression(penalty='l2', tol=1e-05)
logr_aut = LogisticRegression(penalty='l2', tol=1e-05)
logr_cit_in = LogisticRegression(penalty='l2', tol=1e-05)
logr_cit_out = LogisticRegression(penalty='l2', tol=1e-05)
logr_gprops = LogisticRegression(penalty='l2', tol=1e-05)
logr_comm = LogisticRegression(penalty='l2', tol=1e-05)
logr_embs = LogisticRegression(penalty='l2', tol=1e-05)
logr_w_embs = LogisticRegression(penalty='l2', tol=1e-05)

thres_all = None
pipeline = Pipeline([
    ('main_extractor', MainExtractor()),

    # Use FeatureUnion to combine the features from abstracts, titles and authors
    ('union', FeatureUnion(
        transformer_list=[
            # ('abstract_title', Pipeline([
            #     ('selector', ItemSelector(key='abstract_title')),
            #     ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001,ngram_range=(1,2))),
            #     ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
            #     ('sfm_abs', SelectFromModel(logr_abs,threshold=thres_all)),
            # ])),

            # I decided to weight them separately as it led to a predicition performance boost
            ('abstract_title_uni', Pipeline([
                ('selector', ItemSelector(key='abstract_title')), # min: None, max: 0.5 --> 1.924 (started with min_df: 0.001 max_df: 0.6)
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.5, min_df=0,ngram_range=(1,1))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs_uni', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

            ('abstract_title_bi', Pipeline([
                ('selector', ItemSelector(key='abstract_title')), # --> 1.924 (started with min: 0.001, max: 0.6)
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001,ngram_range=(2,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs_bi', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

            ('abstract_title_tri', Pipeline([
                ('selector', ItemSelector(key='abstract_title')), # --> 1.91869
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.0001,ngram_range=(3,3))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs_tri', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

            ('abstract_title_quad', Pipeline([
                ('selector', ItemSelector(key='abstract_title')), # --> 1.91869
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.0001,ngram_range=(4,4))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs_quad', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

            # Pipeline for pulling features from authors
            ('author', Pipeline([
                ('selector', ItemSelector(key='author')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.03, min_df=0,ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_aut', SelectFromModel(logr_aut,threshold=0.55)),
            ])),

            # Pipeline for pulling features from Incoming Citations
            ('incoming_citations', Pipeline([
                ('selector', ItemSelector(key='cit_in')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=False)),
                ('sfm_citi', SelectFromModel(logr_cit_in,threshold=0.15)),
            ])),

            # Outgoing Citations
            ('outgoing_citations', Pipeline([
                ('selector', ItemSelector(key='cit_out')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=False)),
                ('sfm_cito', SelectFromModel(logr_cit_out,threshold=0.10)),
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
               ('stats', AuthorStats()),  # returns a list of dicts
               ('vect_aut_stats', DictVectorizer()),
            ])),

            # Word embeddings from abstracts give a nice boost as well
            ('abs_embeddings', Pipeline([
                ('selector', ItemSelector(key='w_embeddings')),
                ('word_embs_vect', WordEmbeddingsVectorizer()),
                ('sfm_w_embs', SelectFromModel(logr_w_embs,threshold=thres_all)),
            ])),

            # Node Embeddings attempt
            ('node_embeddings', Pipeline([
               ('selector', ItemSelector(key='embeddings')),
               ('node_embs_vect', NodeEmbeddingsVectorizer()),
               ('sfm_embs', SelectFromModel(logr_embs,threshold=thres_all)),
            ])),

            # No improvement was offered by the communities
            # ('communities', Pipeline([
            #     ('selector', ItemSelector(key='comm')),
            #     ('vect_comms', DictVectorizer()),
            #     #('sfm_aut', SelectFromModel(logr_comm,threshold=0.1)),
            # ])),
        ],

        # Weight components in FeatureUnion - Here are the optimals
        transformer_weights={
            'abstract_title_uni':   1.30,
            'abstract_title_bi':    0.75,
            'abstract_title_tri':   1.05,
            'abstract_title_quad':  0.75,
            'author':               1.60,
            'incoming_citations':   1.20,
            'outgoing_citations':   1.30,
            'gprops':               0.70,
            'node_embeddings':      0.35,
            'abs_embeddings':       0.35,
        },
    )),

    ('logr', LogisticRegression(penalty='l2', tol=0.0001)),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    # This is now empty, as I kept the best parameters. It is still needed though
    # as it is a mandatory argumetn for the grid search below.
}

# The default scorer is the accuracy, but we want the log loss in our case.
log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10,scoring=log_loss_scorer)
grid_search.fit(all_train,y_train)

x_test = np.dstack([
                        te_abs,
                        te_titles,
                        te_authors,
                        te_cit_in,
                        te_cit_out,
                        te_outdeg,
                        te_indeg,
                        te_avg_ndeg,
                        te_comm,
                  ])
x_test = np.array([t[0] for t in x_test])
x_test = np.hstack((x_test, te_embs))
x_test = np.hstack((x_test, te_w_embs))

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:") # Will output nothing now that the parameters  dict is empty.
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Get test IDs too
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
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
