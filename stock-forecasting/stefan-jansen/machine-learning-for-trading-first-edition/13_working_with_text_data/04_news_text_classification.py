#!/usr/bin/env python
# coding: utf-8

# # Text classification and sentiment analysis

# Once text data has been converted into numerical features using the natural language processing techniques discussed in the previous sections, text classification works just like any other classification task.
# 
# In this notebook, we will apply these preprocessing technique to news articles, product reviews, and Twitter data and teach various classifiers to predict discrete news categories, review scores, and sentiment polarity.

# ## Imports

# In[3]:


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


# In[4]:


plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ## News article classification

# We start with an illustration of the Naive Bayes model for news article classification using the BBC articles that we read as before to obtain a DataFrame with 2,225 articles from 5 categories.

# ### Read BBC articles

# In[5]:


path = Path('data', 'bbc')
files = path.glob('**/*.txt')
doc_list = []
for i, file in enumerate(files):
    topic = file.parts[-2]
    article = file.read_text(encoding='latin1').split('\n')
    heading = article[0].strip()
    body = ' '.join([l.strip() for l in article[1:]])
    doc_list.append([topic, heading, body])


# In[6]:


docs = pd.DataFrame(doc_list, columns=['topic', 'heading', 'body'])
docs.info()


# ### Create stratified train-test split

# We split the data into the default 75:25 train-test sets, ensuring that the test set classes closely mirror the train set:

# In[7]:


y = pd.factorize(docs.topic)[0]
X = docs.body
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# ### Vectorize text data

# We proceed to learn the vocabulary from the training set and transforming both dataset using the CountVectorizer with default settings to obtain almost 26,000 features:

# In[8]:


vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)


# In[9]:


X_train_dtm.shape, X_test_dtm.shape


# ### Train Multi-class Naive Bayes model

# In[10]:


nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)


# ### Evaluate Results

# We evaluate the multiclass predictions using accuracy to find the default classifier achieved almost 98%:

# #### Accuracy

# In[12]:


accuracy_score(y_test, y_pred_class)


# #### Confusion matrix

# In[13]:


pd.DataFrame(confusion_matrix(y_true=y_test, y_pred=y_pred_class))

