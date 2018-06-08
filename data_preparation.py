"""
    TODO:
        - Try removing the words that consist of digits only.
        - We might want to remove the 'University' word as it appears
          in a lot of documents.
        - The names should be removed... They create too much noise
Name:
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
import pandas as pd
import numpy as np
import networkx as nx
import string
import re
import csv
import inflect
p = inflect.engine()

from gensim.parsing.preprocessing import STOPWORDS

def write_to_file(data,filename,numbers=False):
    with open(filename,"w") as f:
        if numbers == True:
            for l in data:
                f.write(str(l))
                f.write("\n")
        else:
            for l in data:
             f.write(l if len(l)>0 else "\"\"")
             f.write("\n")

stopwords = set(STOPWORDS)
# We'll use this to replace the punctuation with space apart from the dashes
punctuation = '|'.join([re.escape(x) for x in string.punctuation.replace('-','')])

# rm_sw:    Remove stopwords
# rm_smw:   Remove small words
# rm_dg:    Remove words that consist of digits only
# mdash:    Merge dash including words if the no-dash version exists in the data
def transform_data(dataset,
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
    dataset = pd.Series(d for d in dataset)
    # Replace the punctuation with space apart from the dashes 
    dataset = dataset.str.replace(punct,' ')
    # Split it
    dataset = dataset.str.split()
    # Are there any Nan Values? Replace them
    dataset = pd.Series([[] if x is np.nan else x for x in dataset])
    # Remove the stopwords
    if rm_sw == True:
        dataset = pd.Series([[word for word in d if word not in stopwords] for d in dataset])
    # Remove words with length 1 (many occured after the punctuation removal)
    if rm_smw == True:
        dataset = pd.Series([[word for word in d if len(word)>1] for d in dataset])
    # Remove digit-only words:
    if rm_dg == True:
        dataset = pd.Series([[word for word in d if not word.isdigit()] for d in dataset])
    # Some special handling for the authors
    # - Remove universit*
    if auth == True:
        dataset = pd.Series([[word for word in d if not word.lower().startswith('univ')] for d in dataset])

        # Extra stuff for common names
        #common_names = ['david','lee','kim','alex','michael','martin','park','sergei','john','lu','institute','er','yu']
        #dataset = pd.Series([[word for word in d if word.lower() not in common_names] for d in dataset])
    # Merge dash-including words together if the combined version exists
    # This one takes some time so by default it is not performed
    if mdash == True:
        # Get the list of dash-including-words
        dash_words = set()
        non_dashed = set()
        for line in dataset:
            for word in line:
                if "-" in word and word not in dash_words:
                    dash_words.add(word)
                    non_dashed.add(word.replace('-',''))

        # Check which of them exist without a dash
        #found_words = set()
        #for line in dataset:
        #    for word in line:
        #        if word in dash_words and word.replace('-','') in non_dashed:
        #            found_words.add(word)

        # Replace those that exist
        for j, line in enumerate(dataset):
            print("line {0}".format(j))
            for k, word in enumerate(line):
                if word in dash_words and word.replace('-','') in non_dashed:
                    dataset[j][k] = word.replace('-','')

    if mplural == True:
        # This will only recognize cases of a final s.
        candidates = set()
        singulars = set()
        for line in dataset:
            for word in line:
                if len(word) > 4 and word.endswith('s'):
                    candidates.add(word)
                elif len(word) > 3:# that's intented - just think
                    singulars.add(word)

        # Replace the simple plurals with the singulars
        for k, line in enumerate(dataset):
            print("Plurals - Line {0}".format(k))
            for l, word in enumerate(line):
                if word in candidates and word[:-1] in singulars:
                    dataset[k][l] = word[:-1]

    if singulars == True:
        for k,line in enumerate(dataset):
            print("Plurals inflect - Line {0}".format(k))
            for l, word in enumerate(line):
                if p.singular_noun(word) == False:
                    continue
                else:
                    dataset[k][l] = p.singular_noun(word)

    if ing_ed == True:
        candidates = set()
        not_inged = set()
        for line in dataset:
            for word in line:
                if len(word) > 4 and (word.endswith('ed') or word.endswith('ing')):
                    candidates.add(word)
                elif len(word) > 2:
                    not_inged.add(word)

        for k, line in enumerate(dataset):
            print("inged - line {0}".format(k))
            for l, word in enumerate(line):
                if word in candidates and word.endswith('ed'):
                    if word[:-2] in not_inged:
                        dataset[k][l] = word[:-2]
                    elif word[:-1] in not_inged:
                        dataset[k][l] = word[:-1]
                elif word in candidates and word.endswith('ing'):
                    if word[:-3] in not_inged:
                        dataset[k][l] = word[:-3]

    # This was needed to make the dataset compatible with the FeatureUnion format
    dataset = [' '.join(x) for x in dataset]

    return dataset

# Read training data
train_ids = list()
y_train = list()
with open('train.csv', 'r') as f:
    next(f)
    i = 0
    for line in f:
        t = line.split(',')
        train_ids.append(t[0])
        y_train.append(t[1][:-1])

labels = np.unique(y_train)

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.head())

# Load the citations too
df2 = pd.read_csv("Cit-HepTh.txt",delimiter='\t', header=None, names=['paper_id','cites'])
print(df2.head())

# Load the citations again, but this time as a directed network graph
G = nx.read_edgelist('Cit-HepTh.txt', delimiter='\t', create_using=nx.DiGraph())
print("Nodes: ", G.number_of_nodes())
print("Edges: ", G.number_of_edges())

# The graph baseline is actually used here
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
bc_k = 10240
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_ids)
#bw_centrality = nx.betweenness_centrality(G)#,k=bc_k)# normalized=False)
train_graph_properties = list()
for i in range(len(train_ids)):
    train_outdeg = G.out_degree(train_ids[i])
    train_indeg = G.in_degree(train_ids[i])
    train_avg_neighbour_deg = avg_neig_deg[train_ids[i]]
    #train_bw_centr = bw_centrality[train_ids[i]]
    train_graph_properties.append([train_outdeg,train_indeg,train_avg_neighbour_deg])#,train_bw_centr])

train_abstracts = list()
train_titles    = list()
train_authors   = list()
train_citations_in = list()
train_citations_out = list()
train_years = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    train_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
    train_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])
    train_citations_in.append(' '.join([str(x) for x in list(df2.loc[df2['paper_id'] == int(i)]['cites'].values)]))
    train_citations_out.append(' '.join([str(x) for x in list(df2.loc[df2['cites'] == int(i)]['paper_id'].values)]))
    train_years.append(df.loc[df['id'] == int(i)]['year'].iloc[0])

ALLabstracts = transform_data(train_abstracts,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
ALLtitles    = transform_data(train_titles,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
ALLauthors   = transform_data(train_authors,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True,auth=True)


# Same for the test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Create the test matrix. Use the same 3 features as above
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_ids)
#bw_centrality = nx.betweenness_centrality(G)#,k=bc_k)# normalized=False)
test_graph_properties = list()
for i in range(len(test_ids)):
    test_outdeg = G.out_degree(test_ids[i])
    test_indeg = G.in_degree(test_ids[i])
    test_avg_neighbour_deg = avg_neig_deg[test_ids[i]]
    #test_bw_centr = bw_centrality[test_ids[i]]
    test_graph_properties.append([test_outdeg,test_indeg,test_avg_neighbour_deg])#,test_bw_centr])

# Extract the abstract of each test article from the dataframe
test_abstracts = list()
test_titles = list()
test_authors = list()
test_citations_in = list()
test_citations_out = list()
test_years = list()
for i in test_ids:
    test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    test_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
    test_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])
    test_citations_in.append(' '.join([str(x) for x in list(df2.loc[df2['paper_id'] == int(i)]['cites'].values)]))
    test_citations_out.append(' '.join([str(x) for x in list(df2.loc[df2['cites'] == int(i)]['paper_id'].values)]))
    test_years.append(df.loc[df['id'] == int(i)]['year'].iloc[0])

TESTabstracts = transform_data(test_abstracts,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
TESTtitles    = transform_data(test_titles,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
TESTauthors   = transform_data(test_authors,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True,auth=True)


# Save to files - The main program will use them
write_to_file(ALLabstracts,"datasets/train/abstracts.csv")
write_to_file(ALLtitles,"datasets/train/titles.csv")
write_to_file(ALLauthors,"datasets/train/authors.csv")
write_to_file(train_citations_in,"datasets/train/incoming_citations.csv")
write_to_file(train_citations_out,"datasets/train/outgoing_citations.csv")
write_to_file(train_years, "datasets/train/years.csv", numbers=True)
# Use the csv writer for the graph properties
with open("datasets/train/graph_properties.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(train_graph_properties)

write_to_file(TESTabstracts,"datasets/test/abstracts.csv")
write_to_file(TESTtitles,"datasets/test/titles.csv")
write_to_file(TESTauthors,"datasets/test/authors.csv")
write_to_file(test_citations_in,"datasets/test/incoming_citations.csv")
write_to_file(test_citations_out,"datasets/test/outgoing_citations.csv")
write_to_file(test_years, "datasets/test/years.csv", numbers=True)
# Use the csv writer for the graph properties
with open("datasets/test/graph_properties.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(test_graph_properties)