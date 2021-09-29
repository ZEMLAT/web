#!/usr/bin/env python
# coding: utf-8

# # Text classification and sentiment analysis: Twitter

# Once text data has been converted into numerical features using the natural language processing techniques discussed in the previous sections, text classification works just like any other classification task.
# 
# In this notebook, we will apply these preprocessing technique to news articles, product reviews, and Twitter data and teach various classifiers to predict discrete news categories, review scores, and sentiment polarity.

# ## Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import pyarrow as pa   
import pyarrow.parquet as pq
from fastparquet import ParquetFile 
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import seaborn as sns

# spacy, textblob and nltk for language processing
from textblob import TextBlob, Word

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.externals import joblib

import lightgbm as lgb

import json
from time import clock, time


# In[2]:


plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ## Twitter Sentiment

# We use a dataset that contains 1.6 million training and 350 test tweets from 2009 with algorithmically assigned binary positive and negative sentiment scores that are fairly evenly split.

# Download the data from [here](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip).

# Extract the content of the compressed file, move to 'data/sentiment140/' and rename the files:
# - `training.1600000.processed.noemoticon.csv` to `train.csv`, and
# - `testdata.manual.2009.06.14.csv` to `test.csv`

# - 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive); training data has no neutral tweets
# - 1 - the id of the tweet (2087)
# - 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# - 3 - the query (lyx). If there is no query, then this value is NO_QUERY. (only test data uses query)
# - 4 - the user that tweeted (robotickilldozr)
# - 5 - the text of the tweet (Lyx is cool)

# ### Read train/test data

# We move the data to the faster parqu3et

# In[3]:


names = ['polarity', 'id', 'date', 'query', 'user', 'text']
train = (pd.read_csv('data/sentiment140/train.csv',
                     low_memory=False,
                     encoding='latin1',
                     header=None,
                     names=names,
                     parse_dates=['date'])
         .drop(['id', 'query'], axis=1)
         .drop_duplicates(subset=['polarity', 'text']))

train = train[train.text.str.len()<=140]
train.polarity = (train.polarity>0).astype(int)


# In[4]:


train.info(null_counts=True)


# In[5]:


train.to_parquet('data/sentiment140/train.parquet')


# In[7]:


test = (pd.read_csv('data/sentiment140/test.csv',
                    low_memory=False,
                    encoding='latin1',
                    header=None,
                    names=names,
                    parse_dates=['date'])
        .drop(['id', 'query'], axis=1)
        .drop_duplicates(subset=['polarity', 'text']))
test = test[(test.text.str.len()<=140) & (test.polarity.isin([0,4]))]


# In[8]:


test.info()


# In[9]:


test.to_parquet('data/sentiment140/test.parquet')


# In[10]:


train = pd.read_parquet('data/sentiment140/train.parquet')
test = pd.read_parquet('data/sentiment140/test.parquet')


# ### Explore data

# In[11]:


train.head()


# In[12]:


train.polarity = (train.polarity>0).astype(int)
train.polarity.value_counts()


# In[13]:


test.polarity = (test.polarity>0).astype(int)
test.polarity.value_counts()


# In[14]:


sns.distplot(train.text.str.len(), kde=False);


# In[15]:


train.date.describe()


# In[16]:


train.user.nunique()


# In[17]:


train.user.value_counts().head(10)


# ### Create text vectorizer

# We create a document-term matrix with 934 tokens as follows:

# In[18]:


vectorizer = CountVectorizer(min_df=.001, max_df=.8, stop_words='english')
train_dtm = vectorizer.fit_transform(train.text)


# In[19]:


train_dtm


# In[20]:


test_dtm = vectorizer.transform(test.text)


# ### Train Naive Bayes Classifier

# In[21]:


nb = MultinomialNB()
nb.fit(train_dtm, train.polarity)


# ### Predict Test Polarity

# In[22]:


predicted_polarity = nb.predict(test_dtm)


# ### Evaluate Results

# In[24]:


accuracy_score(test.polarity, predicted_polarity)


# ### TextBlob for Sentiment Analysis

# In[25]:


sample_positive = train.text.loc[256332]
print(sample_positive)
parsed_positive = TextBlob(sample_positive)
parsed_positive.polarity


# In[26]:


sample_negative = train.text.loc[636079]
print(sample_negative)
parsed_negative = TextBlob(sample_negative)
parsed_negative.polarity


# In[27]:


def estimate_polarity(text):
    return TextBlob(text).sentiment.polarity


# In[28]:


train[['text']].sample(10).assign(sentiment=lambda x: x.text.apply(estimate_polarity)).sort_values('sentiment')


# ### Compare with TextBlob Polarity Score

# We also obtain TextBlob sentiment scores for the tweets and note (see left panel in below figure) that positive test tweets receive a significantly higher sentiment estimate. We then use the MultinomialNB ‘s model .predict_proba() method to compute predicted probabilities and compare both models using the respective Area Under the Curve (see right panel below).

# In[29]:


test['sentiment'] = test.text.apply(estimate_polarity)


# In[30]:


accuracy_score(test.polarity, (test.sentiment>0).astype(int))


# #### ROC AUC Scores

# In[31]:


roc_auc_score(y_true=test.polarity, y_score=test.sentiment)


# In[32]:


roc_auc_score(y_true=test.polarity, y_score=nb.predict_proba(test_dtm)[:, 1])


# In[33]:


fpr_tb, tpr_tb, _ = roc_curve(y_true=test.polarity, y_score=test.sentiment)
roc_tb = pd.Series(tpr_tb, index=fpr_tb)
fpr_nb, tpr_nb, _ = roc_curve(y_true=test.polarity, y_score=nb.predict_proba(test_dtm)[:, 1])
roc_nb = pd.Series(tpr_nb, index=fpr_nb)


# The Naive Bayes model outperforms TextBlob in this case.

# In[34]:


fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
sns.boxplot(x='polarity', y='sentiment', data=test, ax=axes[0])
axes[0].set_title('TextBlob Sentiment Scores')
roc_nb.plot(ax=axes[1], label='Naive Bayes', legend=True, lw=1, title='ROC Curves')
roc_tb.plot(ax=axes[1], label='TextBlob', legend=True, lw=1)
fig.tight_layout();

