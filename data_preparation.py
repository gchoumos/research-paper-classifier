"""
    TODO:
        - Try removing the words that consist of digits only.
        - I believe we should also remove the 'University' word as it appears
          in a lot of documents.
"""

import pandas as pd
import numpy as np
import string
import re
from gensim.parsing.preprocessing import STOPWORDS

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

ALLabstracts = pd.Series(v for v in train_abstracts)
ALLtitles    = pd.Series(v for v in train_titles)
ALLauthors   = pd.Series(v for v in train_authors)

ALLabstracts.head(5)
ALLtitles.head(5)
ALLauthors.head(5)

rm_punct = string.punctuation.replace('-','')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in rm_punct])

ALLabstracts = ALLabstracts.str.replace(RE_PUNCTUATION,' ')
ALLtitles    = ALLtitles.str.replace(RE_PUNCTUATION,' ')
ALLauthors   = ALLauthors.str.replace(RE_PUNCTUATION,' ')

ALLabstracts_split = ALLabstracts.str.split()
ALLtitles_split    = ALLtitles.str.split()
ALLauthors_split   = ALLauthors.str.split()

stopwords = set(STOPWORDS)
ALLabstracts_split = pd.Series([[token for token in abstract if token not in stopwords] for abstract in ALLabstracts_split])
ALLtitles_split    = pd.Series([[token for token in title if token not in stopwords] for title in ALLtitles_split])

ALLabstracts_split = pd.Series([[token for token in abstract if len(token)>1] for abstract in ALLabstracts_split])
ALLtitles_split    = pd.Series([[token for token in title if len(token)>1] for title in ALLtitles_split])

# Remove all the digit words
ALLabstracts_split = pd.Series([[token for token in abstract if not token.isdigit()] for abstract in ALLabstracts_split])
ALLtitles_split = pd.Series([[token for token in title if not token.isdigit()] for title in ALLtitles_split])

# Replace first all the NaNs with a character as they keep getting in the way
ALLauthors_split = pd.Series([[] if x is np.nan else x for x in ALLauthors_split])
# Then remove the single characters from the authors as well
ALLauthors_split = pd.Series([[token for token in author if len(token)>1] for author in ALLauthors_split])
# We can also remove the stopwords from the authors. It will remove common words such us "of"
# that appear as part of "University of X" cases.
ALLauthors_split = pd.Series([[token for token in author if token not in stopwords] for author in ALLauthors_split])
# also remove the of word from the authors (University of N)
ALLauthors_split = pd.Series([[token for token in author if token != 'of'] for author in ALLauthors_split])


##################################

ALLabstracts_split.head(2)
ALLauthors_split.head(5)
ALLtitles_split.head(5)

##################################


## I'll try merging the abstracts and titles
ALLabstitles   = pd.Series([a + b for a, b in zip(ALLabstracts_split, ALLtitles_split)])
ALLabstitles_j = pd.Series([' '.join(x) for x in ALLabstitles])

# Now I'll merge the authors too
ALLabstitlesauth   = pd.Series([a+b+c for a,b,c in zip (ALLabstracts_split,ALLtitles_split, ALLauthors_split)])
ALLabstitlesauth_j = pd.Series([' '.join(x) for x in ALLabstitlesauth])

#########################################################
#########################################################
## I want to do the same for the test data now.
#########################################################
#########################################################
# Read test data
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

TESTabstracts = pd.Series(v for v in test_abstracts)
TESTtitles    = pd.Series(v for v in test_titles)
TESTauthors   = pd.Series(v for v in test_authors)

TESTabstracts = TESTabstracts.str.replace(RE_PUNCTUATION,' ')
TESTtitles    = TESTtitles.str.replace(RE_PUNCTUATION,' ')
TESTauthors   = TESTauthors.str.replace(RE_PUNCTUATION,' ')

TESTabstracts_split = TESTabstracts.str.split()
TESTtitles_split    = TESTtitles.str.split()
TESTauthors_split   = TESTauthors.str.split()

TESTabstracts_split = pd.Series([[token for token in abstract if token not in stopwords] for abstract in TESTabstracts_split])
TESTtitles_split    = pd.Series([[token for token in title if token not in stopwords] for title in TESTtitles_split])

TESTabstracts_split = pd.Series([[token for token in abstract if len(token)>1] for abstract in TESTabstracts_split])
TESTtitles_split    = pd.Series([[token for token in title if len(token)>1] for title in TESTtitles_split])

# Remove the digit words
TESTabstracts_split = pd.Series([[token for token in abstract if not token.isdigit()] for abstract in TESTabstracts_split])
TESTtitles_split    = pd.Series([[token for token in title if not token.isdigit()] for title in TESTtitles_split])


# Replace first all the NaNs with a character as they keep getting in the way
TESTauthors_split = pd.Series([[] if x is np.nan else x for x in TESTauthors_split])
# Then remove the single characters from the authors as well
TESTauthors_split = pd.Series([[token for token in author if len(token)>1] for author in TESTauthors_split])
TESTauthors_split = pd.Series([[token for token in author if token not in stopwords] for author in TESTauthors_split])

# also remove the of word from the authors (University of N)
TESTauthors_split = pd.Series([[token for token in author if token != 'of'] for author in TESTauthors_split])

TESTabstitles       = pd.Series([a+b for a,b in zip(TESTabstracts_split, TESTtitles_split)])
TESTabstitlesauth   = pd.Series([a+b+c for a,b,c in zip(TESTabstracts_split,TESTtitles_split,TESTauthors_split)])
TESTabstitles_j     = pd.Series([' '.join(x) for x in TESTabstitles])
TESTabstitlesauth_j = pd.Series([' '.join(x) for x in TESTabstitlesauth])


#####################
# Save the datasets #
#####################
ALLabstracts_split.to_csv("datasets/train/abstracts.csv",index=False)
ALLtitles_split.to_csv("datasets/train/titles.csv",index=False)
ALLauthors_split.to_csv("datasets/train/authors.csv",index=False)
ALLabstitlesauth_j.to_csv("datasets/train/abstitauth.csv",index=False)

TESTabstracts_split.to_csv("datasets/test/abstracts.csv",index=False)
TESTtitles_split.to_csv("datasets/test/titles.csv",index=False)
TESTauthors_split.to_csv("datasets/test/authors.csv",index=False)
TESTabstitlesauth_j.to_csv("datasets/test/abstitauth.csv",index=False)