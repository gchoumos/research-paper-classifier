import pandas as pd
import numpy as np
import csv
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer
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
from feature_union_sklearn import ItemSelector, TextStats, AuthorStats, MainExtractor, GraphProperties, NodeEmbeddingsVectorizer, WordEmbeddingsVectorizer
import pdb

from data_preprocessor import DataPreprocessor

pd.options.mode.chained_assignment = None  # default='warn'

import math
import random
from collections import Counter

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

#######################
# MANUAL OVERSAMPLING #
#######################
# y_t_counts = Counter(y_train)
# print("Class Frequencies:")
# for i, j in enumerate(y_t_counts):
#     # Print proportions (percentage)
#     print("{1:.2f}: {0} - {2}".format(j,y_t_counts[j]*100/len(y_train),math.ceil(y_t_counts[j]*100/len(y_train))))

# y_t_indices = list()
# for i,j in enumerate(y_t_counts):
#     y_t_indices.append(np.random.choice([k for k,l in enumerate(y_train) if j == l],4*math.ceil(y_t_counts[j]*100/len(y_train))))

# # Now flatten the list
# indices = [x for sublist in y_t_indices for x in sublist]
# print("We are adding {0} extra datapoints".format(len(indices)))

# More training data with resampling (add k lines)
# indices = np.random.randint(0,all_train.shape[0],100) # 1.913 (train)  -- 1.81707  (kaggle)
# balanced: 114 datapoints - 1.913 (train)  -- not submitted (kaggle)
#indices = np.random.randint(0,all_train.shape[0],200) # 1.902 (train)  -- 1.81743  (kaggle)
# balanced: 228 datapoints - 1.898 (train)  -- not submitted (kaggle)
#indices = np.random.randint(0,all_train.shape[0],400) # 1.879 (train)   -- 1.81339  (kaggle)
# balanced: 456 datapoints - 1.866 (train)  -- 1.81954 (kaggle)
#indices = np.random.randint(0,all_train.shape[0],800) # 1.835 (train)   -- 1.81956  (kaggle)
#indices = random.sample(range(all_train.shape[0]),800) # 1.828 (train)  -- 1.81782  (kaggle)
#indices = np.random.randint(0,all_train.shape[0],1000) # lol = 1.809 (train)

#all_train = np.append(all_train,all_train[indices],axis=0)
# Same for y_train
#y_train = y_train + [y_train[i] for i in indices]

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

            ###### REMOVE THIS ########
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
            ####### UNTIL HERE ########

            # Pipeline for pulling features from authors
            ('author', Pipeline([
                ('selector', ItemSelector(key='author')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.03, min_df=0,ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_aut', SelectFromModel(logr_aut,threshold=0.55)),
            ])),

            # Pipeline for pulling features from authors
            ('incoming_citations', Pipeline([
                ('selector', ItemSelector(key='cit_in')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=False)),
                ('sfm_citi', SelectFromModel(logr_cit_in,threshold=0.15)),
            ])),

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

            # ('communities', Pipeline([
            #     ('selector', ItemSelector(key='comm')),
            #     ('vect_comms', DictVectorizer()),
            #     #('sfm_aut', SelectFromModel(logr_comm,threshold=0.1)),
            # ])),
        ],

        # weight components in FeatureUnion
        transformer_weights={
            'abstract': 1.65,
            'abstract_title': 1.65,
            'abstract_title_uni': 1.30, # 1.30 --> 1.910245
            'abstract_title_bi': 0.75,  # 0.75 --> 1.910245
            'abstract_title_tri': 1.05, # 1.05 --> 1.910245
            'abstract_title_quad': 0.75,# 0.75 --> 1.910245
            'author': 1.60,             # 1.60 --> 1.910245
            # 'abs_stats': 0.3,         # None --> 1.910245
            'incoming_citations': 1.20, # 1.20 --> 1.910245
            'outgoing_citations': 1.30, # 1.30 --> 1.910245
            'gprops': 0.70,             # None --> 1.910245
            # 'aut_stats': 0.30,        # None --> 1.910245
            #'communities': 8,
            'node_embeddings': 0.35,    # 0.35 --> 1.910245
            'abs_embeddings': 0.35,     # 0.35 --> 1.910245
            # best node embs with
            #   num_walks       10
            #   walk length     20
            #   epochs (iter):  4
        },
    )),

    ('logr', LogisticRegression(penalty='l2', tol=0.0001)), # C= r
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
}

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
print("Best parameters set:")
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
