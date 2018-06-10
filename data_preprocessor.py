"""
    TODO:
        - Try removing the words that consist of digits only.
        - We might want to remove the 'University' word as it appears
          in a lot of documents.
        - The names should be removed... They create too much noise

    Common Names:
        David
        Lee
        Kim
        Alex
        Michael
        Martin
        Park
        Sergei
        John
        Lu
        Institute
        er
        Yu
"""

from gensim.parsing.preprocessing import STOPWORDS
from settings import SETTINGS
import networkx as nx
import pandas as pd
import numpy as np
import inflect
import string
import csv
import re

class DataPreprocessor(object):
    """ """
    def __init__(self):
        self.data_file = SETTINGS['main_data']
        self.train_file = SETTINGS['train_file']
        self.test_file = SETTINGS['test_file']
        self.citations_file = SETTINGS['citations_file']

        self.stopwords = set(STOPWORDS)
        self.punct = '|'.join([re.escape(x) for x in string.punctuation.replace('-','')])

        # train and test graph properties list
        self.tr_gproperties = list()
        self.te_gproperties = list()

        self.train_ids = list()
        self.y_train = list()
        self.labels = list()
        self.test_ids = list()

        # Training stuff
        self.tr_abst = list()
        self.tr_titl = list()
        self.tr_auth = list()
        self.tr_citi = list()
        self.tr_cito = list()

        # Test stuff
        self.te_abst = list()
        self.te_titl = list()
        self.te_auth = list()
        self.te_citi = list()
        self.te_cito = list()


    def write_to_file(self, data, filename, numbers=False):
        with open(filename,"w") as f:
            if numbers == True:
                for l in data:
                    f.write(str(l))
                    f.write("\n")
            else:
                for l in data:
                 f.write(l if len(l)>0 else "\"\"")
                 f.write("\n")

    def read_main_data(self):
        """ Read the main dataset (node information) """
        self.main_df = pd.read_csv(self.data_file)
        #print(main_df.head())

    def read_citations(self,as_network=False):
        """ Read the citations """
        if as_network:
            # Load the citations as a directed network graph
            self.G = nx.read_edgelist(self.citations_file,
                                      delimiter='\t',
                                      create_using=nx.DiGraph())
            print("Nodes: ", self.G.number_of_nodes())
            print("Edges: ", self.G.number_of_edges())
        else:
            self.cit_df = pd.read_csv(self.citations_file,
                                 delimiter='\t',
                                 header=None,
                                 names=['paper_id','cites'])
        #print(cit_df.head())

    def read_train_data(self):
        """ Read data from the train file """
        with open(self.train_file, 'r') as f:
            next(f)
            for line in f:
                t = line.split(',')
                self.train_ids.append(t[0])
                self.y_train.append(t[1][:-1])
        self.labels = np.unique(self.y_train)

    def read_test_data(self):
        """ Read data from the test file """
        with open(self.test_file, 'r') as f:
            next(f)
            for line in f:
                self.test_ids.append(line[:-2])

    def compute_graph_properties(self, func='train'):
        """
            Compute the graph properties for the train/test graph.
        """

        # Choose between train and test ids depending on the func arg
        id_list = self.train_ids if func == 'train' else self.test_ids
        # The graph baseline is actually used here
        # (1) out-degree, (2) in-degree (3) average degree of neighborhood
        avg_n_deg = nx.average_neighbor_degree(self.G, nodes=id_list)
        gprop_list = list()
        for i in range(len(id_list)):
            outdeg = self.G.out_degree(id_list[i])
            indeg = self.G.in_degree(id_list[i])
            avg_ndeg = avg_n_deg[id_list[i]]
            gprop_list.append([outdeg,indeg,avg_ndeg])

        if func == 'train':
            self.tr_gproperties = gprop_list
        else:
            self.te_gproperties = gprop_list


    def fill_dataframes(self, func='train'):

        if func == 'train':
            for i in self.train_ids:
                self.tr_abst.append(self.main_df.loc[self.main_df['id'] == int(i)]['abstract'].iloc[0])
                self.tr_titl.append(self.main_df.loc[self.main_df['id'] == int(i)]['title'].iloc[0])
                self.tr_auth.append(self.main_df.loc[self.main_df['id'] == int(i)]['authors'].iloc[0])
                self.tr_citi.append(' '.join([str(x) for x in list(self.cit_df.loc[self.cit_df['paper_id'] == int(i)]['cites'].values)]))
                self.tr_cito.append(' '.join([str(x) for x in list(self.cit_df.loc[self.cit_df['cites'] == int(i)]['paper_id'].values)]))
        else:
            for i in self.test_ids:
                self.te_abst.append(self.main_df.loc[self.main_df['id'] == int(i)]['abstract'].iloc[0])
                self.te_titl.append(self.main_df.loc[self.main_df['id'] == int(i)]['title'].iloc[0])
                self.te_auth.append(self.main_df.loc[self.main_df['id'] == int(i)]['authors'].iloc[0])
                self.te_citi.append(' '.join([str(x) for x in list(self.cit_df.loc[self.cit_df['paper_id'] == int(i)]['cites'].values)]))
                self.te_cito.append(' '.join([str(x) for x in list(self.cit_df.loc[self.cit_df['cites'] == int(i)]['paper_id'].values)]))

    # rm_sw:    Remove stopwords
    # rm_smw:   Remove small words
    # rm_dg:    Remove words that consist of digits only
    # mdash:    Merge dash including words if the no-dash version exists in the data
    def transform_data(self,
                       data,
                       stopwords,
                       punct,
                       rm_sw=True,
                       rm_smw=True,
                       rm_dg=True,
                       mdash=False,
                       mplural=False,
                       singulars=True,
                       ing_ed=True,
                       auth=False):
        # Convert to Pandas series
        data = pd.Series(d for d in data)
        # Replace the punctuation with space apart from the dashes 
        data = data.str.replace(punct,' ')
        # Split it
        data = data.str.split()
        # Are there any Nan Values? Replace them
        data = pd.Series([[] if x is np.nan else x for x in data])
        # Remove the stopwords
        if rm_sw == True:
            data = pd.Series([[word for word in d if word not in stopwords] for d in data])
        # Remove words with length 1 (many occured after the punctuation removal)
        if rm_smw == True:
            data = pd.Series([[word for word in d if len(word)>1] for d in data])
        # Remove digit-only words:
        if rm_dg == True:
            data = pd.Series([[word for word in d if not word.isdigit()] for d in data])
        # Some special handling for the authors
        # - Remove universit*
        if auth == True:
            data = pd.Series([[word for word in d if not word.lower().startswith('univ')] for d in data])

            # Extra stuff for common names
            #common_names = ['david','lee','kim','alex','michael','martin','park','sergei','john','lu','institute','er','yu']
            #data = pd.Series([[word for word in d if word.lower() not in common_names] for d in data])
        # Merge dash-including words together if the combined version exists
        # This one takes some time so by default it is not performed
        if mdash == True:
            # Get the list of dash-including-words
            dash_words = set()
            non_dashed = set()
            for line in data:
                for word in line:
                    if "-" in word and word not in dash_words:
                        dash_words.add(word)
                        non_dashed.add(word.replace('-',''))

            # Replace those that exist
            for j, line in enumerate(data):
                print("line {0}".format(j))
                for k, word in enumerate(line):
                    if word in dash_words and word.replace('-','') in non_dashed:
                        data[j][k] = word.replace('-','')

        if mplural == True:
            # This will only recognize cases of a final s.
            candidates = set()
            singulars = set()
            for line in data:
                for word in line:
                    if len(word) > 4 and word.endswith('s'):
                        candidates.add(word)
                    elif len(word) > 3:# that's intented - just think
                        singulars.add(word)

            # Replace the simple plurals with the singulars
            for k, line in enumerate(data):
                print("Plurals - Line {0}".format(k))
                for l, word in enumerate(line):
                    if word in candidates and word[:-1] in singulars:
                        data[k][l] = word[:-1]

        if singulars == True:
            p = inflect.engine()
            for k,line in enumerate(data):
                print("Plurals inflect - Line {0}".format(k))
                for l, word in enumerate(line):
                    if p.singular_noun(word) == False:
                        continue
                    else:
                        data[k][l] = p.singular_noun(word)

        if ing_ed == True:
            candidates = set()
            not_inged = set()
            for line in data:
                for word in line:
                    if len(word) > 4 and (word.endswith('ed') or word.endswith('ing')):
                        candidates.add(word)
                    elif len(word) > 2:
                        not_inged.add(word)

            for k, line in enumerate(data):
                print("inged - line {0}".format(k))
                for l, word in enumerate(line):
                    if word in candidates and word.endswith('ed'):
                        if word[:-2] in not_inged:
                            data[k][l] = word[:-2]
                        elif word[:-1] in not_inged:
                            data[k][l] = word[:-1]
                    elif word in candidates and word.endswith('ing'):
                        if word[:-3] in not_inged:
                            data[k][l] = word[:-3]

        # This was needed to make the data compatible with the FeatureUnion format
        data = [' '.join(x) for x in data]

        return data


    def preprocess(self):
        self.read_train_data()
        self.read_test_data()
        self.read_main_data()
        self.read_citations()
        self.read_citations(as_network=True)
        self.compute_graph_properties(func='train')
        self.fill_dataframes(func='train')
        self.compute_graph_properties(func='test')
        self.fill_dataframes(func='test')

        # Transform the train data
        self.tr_abst = self.transform_data(self.tr_abst, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True)
        self.tr_titl = self.transform_data(self.tr_titl, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True)
        self.tr_auth = self.transform_data(self.tr_auth, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True, auth=True)
        # Transform the test data
        self.te_abst = self.transform_data(self.te_abst, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True)
        self.te_titl = self.transform_data(self.te_titl, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True)
        self.te_auth = self.transform_data(self.te_auth, self.stopwords, self.punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=True, auth=True)

        # Write to train files for the main program to use
        self.write_to_file(self.tr_abst,"datasets/train/abstracts.csv")
        self.write_to_file(self.tr_titl,"datasets/train/titles.csv")
        self.write_to_file(self.tr_auth,"datasets/train/authors.csv")
        self.write_to_file(self.tr_citi,"datasets/train/incoming_citations.csv")
        self.write_to_file(self.tr_cito,"datasets/train/outgoing_citations.csv")

        # Use the csv writer for the graph properties
        with open("datasets/train/graph_properties.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.tr_gproperties)

        # Write to test files for the main program to use
        self.write_to_file(self.te_abst,"datasets/test/abstracts.csv")
        self.write_to_file(self.te_titl,"datasets/test/titles.csv")
        self.write_to_file(self.te_auth,"datasets/test/authors.csv")
        self.write_to_file(self.te_citi,"datasets/test/incoming_citations.csv")
        self.write_to_file(self.te_cito,"datasets/test/outgoing_citations.csv")

        # Use the csv writer for the graph properties
        with open("datasets/test/graph_properties.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.te_gproperties)
