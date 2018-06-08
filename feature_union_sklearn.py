# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
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
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    # Number of Digit words?
    def transform(self, lines):
        return [{'length': len(line),
                 #'num_words': line.count(' '),
                 'num_digits': sum(c.isdigit() for c in line),
                 #'words_dash': len([w2 for w2 in line if '-' in w2]),
                 #'avg_word_len': sum([len(word) for word in line.split()])/len(line.split()),
                 }
                for line in lines]

# ################ #
# Don't forget!
# I'll change this later because I want to be sure that I am able to weight
# the various properties separately
class GraphProperties(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    # Number of Digit words?
    def transform(self, lines):
        return [{'outdeg': float(line[0]),
                 'indeg': float(line[1]),
                 'avg_neigh_deg': float(line[2]),
                 }
                for line in lines]


class AbstractTitleAuthorExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
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
								   #('indeg', object),
								   #('avg_neigh_deg', object),
                                   ('year', object),
                                   ('abstract_title', object)
							    ])
        for i, line in enumerate(lines):
            abstract, title, author, cit_in, cit_out = line[0], line[1], line[2], line[3], line[4]
            graph_props = [float(line[5]), float(line[6]), float(line[7])]
            year = int(line[8])
            abs_tit = str(line[0]) + ' ' + str(line[1])

            features['abstract'][i] = abstract if abstract==abstract else ''
            features['title'][i] = title if title==title else ''
            features['author'][i] = author if author==author else ''
            features['cit_in'][i] = cit_in if cit_in==cit_in else ''
            features['cit_out'][i] = cit_out if cit_out==cit_out else ''
            features['graph_props'][i] = graph_props if graph_props==graph_props else [0, 0, 0]
            features['year'][i] = {'year': int(year)} if year==year else {'year': 0}
            features['abstract_title'][i] = abs_tit if abs_tit==abs_tit else ''

        #print("MYSELF: Features shape is {0}".format(features.shape))
        #print("MYSELF: {0}".format(features[0]))
        #import pdb
        #pdb.set_trace()
        return features
