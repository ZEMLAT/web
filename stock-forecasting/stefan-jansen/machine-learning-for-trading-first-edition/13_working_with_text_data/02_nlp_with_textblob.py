#!/usr/bin/env python
# coding: utf-8

# # NLP with TextBlob

# TextBlob is a python library that provides a simple API for common NLP tasks and builds on the Natural Language Toolkit (nltk) and the Pattern web mining libraries. TextBlob facilitates part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and others.

# ## Imports & Settings

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# spacy, textblob and nltk for language processing
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer

# sklearn for feature extraction & modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB         # Naive Bayes
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib


# In[2]:


np.random.seed(42)
pd.set_option('float_format', '{:,.2f}'.format)


# ## Load BBC Data

# To illustrate the use of TextBlob, we sample a BBC sports article with the headline ‘Robinson ready for difficult task’. Similar to spaCy and other libraries, the first step is to pass the document through a pipeline represented by the TextBlob object to assign annotations required for various tasks.

# In[3]:


path = Path('data', 'bbc')
files = path.glob('**/*.txt')
doc_list = []
for i, file in enumerate(files):
    topic = file.parts[-2]
    article = file.read_text(encoding='latin1').split('\n')
    heading = article[0].strip()
    body = ' '.join([l.strip() for l in article[1:]]).strip()
    doc_list.append([topic, heading, body])


# In[4]:


docs = pd.DataFrame(doc_list, columns=['topic', 'heading', 'body'])
docs.info()


# ## Introduction to TextBlob
# 
# You should already have downloaded TextBlob, a Python library used to explore common NLP tasks.

# ### Select random article

# In[5]:


article = docs.sample(1).squeeze()


# In[6]:


print(f'Topic:\t{article.topic.capitalize()}\n\n{article.heading}\n')
print(article.body.strip())


# In[7]:


parsed_body = TextBlob(article.body)


# ### Tokenization

# In[8]:


parsed_body.words


# ### Sentence boundary detection

# In[9]:


parsed_body.sentences


# ### Stemming

# To perform stemming, we instantiate the SnowballStemmer from the nltk library, call its .stem() method on each token and display tokens that were modified as a result:

# In[10]:


# Initialize stemmer.
stemmer = SnowballStemmer('english')

# Stem each word.
[(word, stemmer.stem(word)) for i, word in enumerate(parsed_body.words) 
 if word.lower() != stemmer.stem(parsed_body.words[i])]


# ### Lemmatization

# In[11]:


[(word, word.lemmatize()) for i, word in enumerate(parsed_body.words) 
 if word != parsed_body.words[i].lemmatize()]


# Lemmatization relies on parts-of-speech (POS) tagging; `spaCy` performs POS tagging, here we make assumptions, e.g. that each token is verb.

# In[12]:


[(word, word.lemmatize(pos='v')) for i, word in enumerate(parsed_body.words) 
 if word != parsed_body.words[i].lemmatize(pos='v')]


# ### Sentiment & Polarity

# TextBlob provides polarity and subjectivity estimates for parsed documents using dictionaries provided by the Pattern library. These dictionaries lexicon map adjectives frequently found in product reviews to sentiment polarity scores, ranging from -1 to +1 (negative ↔ positive) and a similar subjectivity score (objective ↔ subjective).
# 
# The .sentiment attribute provides the average for each over the relevant tokens, whereas the .sentiment_assessments attribute lists the underlying values for each token

# In[15]:


parsed_body.sentiment


# In[14]:


parsed_body.sentiment_assessments


# ### Combine Textblob Lemmatization with `CountVectorizer`

# In[13]:


def lemmatizer(text):
    words = TextBlob(text.lower()).words
    return [word.lemmatize() for word in words]


# In[14]:


vectorizer = CountVectorizer(analyzer=lemmatizer, decode_error='replace')

