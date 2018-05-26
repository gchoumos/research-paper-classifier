"""
    TODO:
        - Try removing the words that consist of digits only.
        - We might want to remove the 'University' word as it appears
          in a lot of documents.
"""
import pandas as pd
import numpy as np
import string
import re
from gensim.parsing.preprocessing import STOPWORDS

def write_to_file(data,filename):
    with open(filename,"w") as f:
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
def transform_data(dataset, stopwords, punct, rm_sw=True, rm_smw=True, rm_dg=True, mdash=False):
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
        for j,line in enumerate(dataset):
            print("line {0}".format(j))
            for k,word in enumerate(line):
                if word in dash_words and word.replace('-','') in non_dashed:
                    dataset[j][k] = word.replace('-','')

    # This was needed to make the dataset compatible with the FeatureUnion format
    dataset = [' '.join(x) for x in dataset]

    return dataset

# Load data about each article in a dataframe
df = pd.read_csv("node_information.csv")
print(df.head())

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

train_abstracts = list()
train_titles    = list()
train_authors   = list()
for i in train_ids:
    train_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    train_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
    train_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])

ALLabstracts = transform_data(train_abstracts,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
ALLtitles    = transform_data(train_titles,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
ALLauthors   = transform_data(train_authors,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)

# Same for the test data
test_ids = list()
with open('test.csv', 'r') as f:
    next(f)
    for line in f:
        test_ids.append(line[:-2])

# Extract the abstract of each test article from the dataframe
n_test = len(test_ids)
test_abstracts = list()
test_titles = list()
test_authors = list()
for i in test_ids:
    test_abstracts.append(df.loc[df['id'] == int(i)]['abstract'].iloc[0])
    test_titles.append(df.loc[df['id'] == int(i)]['title'].iloc[0])
    test_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])

TESTabstracts = transform_data(test_abstracts,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
TESTtitles    = transform_data(test_titles,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)
TESTauthors   = transform_data(test_authors,stopwords,punctuation,rm_sw=True,rm_smw=True,rm_dg=True,mdash=True)


# Save to files - The main program will use them
write_to_file(ALLabstracts,"datasets/train/abstracts.csv")
write_to_file(ALLtitles,"datasets/train/titles.csv")
write_to_file(ALLauthors,"datasets/train/authors.csv")

write_to_file(TESTabstracts,"datasets/test/abstracts.csv")
write_to_file(TESTtitles,"datasets/test/titles.csv")
write_to_file(TESTauthors,"datasets/test/authors.csv")