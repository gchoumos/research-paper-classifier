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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, log_loss

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

pd.options.mode.chained_assignment = None  # default='warn'


train   = pd.read_csv("datasets/train/abstitauth.csv",header=None)
test    = pd.read_csv("datasets/test/abstitauth.csv",header=None)

# train_abs = pd.read_csv("datasets/train/abstracts.csv",header=None)
# train_titles = pd.read_csv("datasets/train/titles.csv",header=None)

# test_abs = pd.read_csv("datasets/test/abstracts.csv",header=None)
# test_titles = pd.read_csv("datasets/test/titles.csv",header=None)

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
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer(decode_error='ignore',stop_words='english')),
    ('tfidf', TfidfTransformer(norm='l2')),
    # ('sgd', SGDClassifier(loss='log',max_iter=5,loss='l1',tol=None)),
    # ('mnb', MultinomialNB()),
    ('logr', LogisticRegression()),
    # ('rfc', RandomForestClassifier()),
    # ('knn', KNeighborsClassifier()),
    # ('NuSVC', NuSVC(probability=True)), # This one SUCKS
    # ('dtc', DecisionTreeClassifier()),
])

# clf = MLPClassifier(  hidden_layer_sizes=(100, ),
#                             activation='relu',
#                             solver='adam',
#                             alpha=0.0001,
#                             momentum=0.9,
#                             verbose=True
#                         )

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__min_df': (0, 0.01, 0.1),
    'vect__max_df': (0.5, 0.6, 0.7),
    'vect__ngram_range': ((1,1),(1,2)),
    # 'mnb__alpha': (0.0001,0.001,0.01,0.1,1),
    'logr__penalty': ('l1','l2'),
    'logr__tol': (0.0001,0.00001),
    # 'logr__C': (1.0,0.8,0.5,0.1), # defaults to 1.0 which performs way better
    # 'logr__class_weight': (None, 'balanced'), # defaults to None which performs way better than balanced
    # 'clf__activation': ('relu','logistic'),
    # 'clf__solver': ('adam','sgd'),
    # 'clf__alpha': (0.00001, 0.000001, 0.0000001),
    # 'clf__penalty': ('l2', 'elasticnet'),
    # 'clf__momentum': (0.7,0.9),
    # 'rfc__n_estimators': (10,4,8),
    # 'rfc__criterion': ('gini','entropy'),
    # 'knn__n_neighbors': (5,25,101),
    # 'knn__weights': ('uniform','distance'),
    # 'knn__p': (1,2),
    # 'NuSVC__nu': (0.01,0.4),
    # 'NuSVC__kernel': ('sigmoid','rbf'),
    # 'dtc__criterion': ('gini','entropy'),
    # 'dtc__class_weight': (None,'balanced')
}

log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10,scoring=log_loss_scorer)


# Get the input and output in the appropriate format 
x_train = [x[0] for x in train.values]
x_test  = [x[0] for x in test.values]

# x_train = np.dstack([train_abs,train_titles])
# x_test = np.dstack([test_abs,test_titles])

grid_search.fit(x_train, y_train)

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
    # lst = clf.classes_.tolist()
    lst.insert(0, "Article")
    writer.writerow(lst)
    for i,test_id in enumerate(test_ids):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_id)
        writer.writerow(lst)
