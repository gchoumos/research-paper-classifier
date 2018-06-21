# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause

# Modified by George Choumos

from sklearn.base import BaseEstimator, TransformerMixin
from settings import SETTINGS
import numpy as np

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

class AuthorStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        return [{
                    'length': len(line),
                } for line in lines]

class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        return [{
                    'length': len(line),
                    'num_digits': sum(c.isdigit() for c in line),
                } for line in lines]


class GraphProperties(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        return [{
                    'outdeg': float(line[0]),
                    'indeg': float(line[1]),
                    'avg_neigh_deg': float(line[2]),
                    #'comm': int(line[3]),
                } for line in lines]

class NodeEmbeddingsVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        return [[float(x) for x in line.split()] for line in lines]

class WordEmbeddingsVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        return [[float(x) for x in line.split()] for line in lines]

class MainExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, lines):
        features = np.recarray(
                    shape=(len(lines),),
                        dtype=[
                                ('abstract', object),
                                ('title', object),
                                ('abstract_title', object),
                                ('author', object),
                                ('cit_in', object),
                                ('cit_out', object),
                                ('graph_props', object),
                                #('comm', object),
                                ('embeddings', object),
                                ('w_embeddings', object),
							  ])
        for i, line in enumerate(lines):
            abstract = line[0]
            title = line[1],
            abs_tit = str(line[0]) + ' ' + str(line[1])
            author = line[2]
            cit_in = line[3]
            cit_out = line[4]
            graph_props = [float(line[5]), float(line[6]), float(line[7])]#, int(line[8])]
            # comm = {'comm': int(line[8])}
            embs = ' '.join([str(x) for x in line[9:9+SETTINGS['node_embs_dim']]])
            w_embs = ' '.join([str(x) for x in line[9+SETTINGS['node_embs_dim']:]])

            features['abstract'][i] = abstract if abstract==abstract else ''
            features['title'][i] = title if title==title else ''
            features['author'][i] = author if author==author else ''
            features['cit_in'][i] = cit_in if cit_in==cit_in else ''
            features['cit_out'][i] = cit_out if cit_out==cit_out else ''
            features['graph_props'][i] = graph_props if graph_props==graph_props else [0, 0, 0]
            features['abstract_title'][i] = abs_tit if abs_tit==abs_tit else ''
            # features['comm'][i] = comm if comm==comm else -1
            features['embeddings'][i] = embs
            features['w_embeddings'][i] = w_embs

        # print("Features shape is {0}".format(features.shape))
        # print("{0}".format(features[0]))
        return features
