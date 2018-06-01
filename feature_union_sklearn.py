# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause

# Modified by George Choumos

from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    # Number of Digit words?
    def transform(self, lines):
        return [{'length': len(line),
                 'num_words': line.count(' '),
                 'num_digits': sum(c.isdigit() for c in line),
                 'words_dash': len([w2 for w2 in line if '-' in w2])
                 }
                for line in lines]

# ################ #
# Don't forget!
# I'll change this later because I want to be sure that I am able to weight
# the various properties separately
class GraphProperties(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    # Number of Digit words?
    def transform(self, lines):
        return [{'outdeg': float(line[0]),
                 'indeg': float(line[1]),
                 'avg_neigh_deg': float(line[2])}
                for line in lines]


class AbstractTitleAuthorExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        features = np.recarray(shape=(len(lines),),
                               dtype=[
							       ('abstract', object),
							       ('title', object),
								   ('author', object),
								   ('cit_in', object),
								   ('cit_out', object),
								   ('graph_props', object),
							    ])
        for i, line in enumerate(lines):
            abstract, title, author, cit_in, cit_out = line[0], line[1], line[2], line[3], line[4]
            graph_props = [float(line[5]), float(line[6]), float(line[7])]

            features['abstract'][i] = abstract if abstract==abstract else ''
            features['title'][i] = title if title==title else ''
            features['author'][i] = author if author==author else ''
            features['cit_in'][i] = cit_in if cit_in==cit_in else ''
            features['cit_out'][i] = cit_out if cit_out==cit_out else ''
            features['graph_props'][i] = graph_props if graph_props==graph_props else [0, 0, 0]

        print("MYSELF: Features shape is {0}".format(features.shape))
        print("MYSELF: {0}".format(features[0]))
        #import pdb
        #pdb.set_trace()
        return features
