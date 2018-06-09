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
from feature_union_sklearn import ItemSelector, TextStats, MainExtractor, GraphProperties
import pdb

pd.options.mode.chained_assignment = None  # default='warn'

tr_abs = pd.read_csv("datasets/train/abstracts.csv", header=None)
tr_titles = pd.read_csv("datasets/train/titles.csv", header=None)
tr_authors = pd.read_csv("datasets/train/authors.csv", header=None)
tr_cit_in = pd.read_csv("datasets/train/incoming_citations.csv", header=None)
tr_cit_out = pd.read_csv("datasets/train/outgoing_citations.csv", header=None)
tr_outdeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[0])
tr_indeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[1])
tr_avg_ndeg = pd.read_csv("datasets/train/graph_properties.csv", header=None, usecols=[2])

all_train = np.dstack([
                        tr_abs,
                        tr_titles,
                        tr_authors,
                        tr_cit_in,
                        tr_cit_out,
                        tr_outdeg,
                        tr_indeg,
                        tr_avg_ndeg
                      ])
all_train = np.array([t[0] for t in all_train])

te_abs = pd.read_csv("datasets/test/abstracts.csv", header=None)
te_titles = pd.read_csv("datasets/test/titles.csv", header=None)
te_authors = pd.read_csv("datasets/test/authors.csv", header=None)
te_cit_in = pd.read_csv("datasets/test/incoming_citations.csv", header=None)
te_cit_out = pd.read_csv("datasets/test/outgoing_citations.csv", header=None)
te_outdeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[0])
te_indeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[1])
te_avg_ndeg = pd.read_csv("datasets/test/graph_properties.csv", header=None, usecols=[2])

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

thres_all = None
pipeline = Pipeline([
    ('main_extractor', MainExtractor()),

    # Use FeatureUnion to combine the features from abstracts, titles and authors
    ('union', FeatureUnion(
        transformer_list=[
            ('abstract_title', Pipeline([
                ('selector', ItemSelector(key='abstract_title')),
                ('vect', CountVectorizer(decode_error='ignore', stop_words='english', max_df=0.6, min_df=0.001,ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=True)),
                ('sfm_abs', SelectFromModel(logr_abs,threshold=thres_all)),
            ])),

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
                ('sfm_aut', SelectFromModel(logr_cit_in,threshold=0.15)),
            ])),

            ('outgoing_citations', Pipeline([
                ('selector', ItemSelector(key='cit_out')),
                ('vect', CountVectorizer(decode_error='ignore')),
                ('tfidf', TfidfTransformer(norm='l2',sublinear_tf=False)),
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

        # weight components in FeatureUnion
        transformer_weights={
            'abstract': 1.65,
            'abstract_title': 1.65,
            #'title': 0.50,
            'author': 1.5,
            #'abs_stats': 1.2,
            'incoming_citations': 1.2,
            'outgoing_citations': 1.3,
			#'gprops': 0.6,
            #'aut_stats': 0.7,
        },
    )),

    ('logr', LogisticRegression(penalty='l2',tol=0.0001)), # C=1.25 results in a slight improvement
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
                        te_avg_ndeg
                  ])
x_test = np.array([t[0] for t in x_test])

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
