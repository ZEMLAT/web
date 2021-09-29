#!/usr/bin/env python
# coding: utf-8

# ## Imports & Settings

# In[1]:


from time import time
import warnings
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


analogies_path = Path('data', 'analogies', 'analogies-en.txt')


# ## Convert GloVE Vectors to gensim format

# The various GloVE vectors are available [here](https://nlp.stanford.edu/projects/glove/). Download link for the [wikipedia](http://nlp.stanford.edu/data/glove.6B.zip) version. Unzip and store in `data/glove`.

# ### WikiPedia

# In[4]:


glove_path = Path('data/glove')
glove_wiki_file= glove_path / 'glove.6B.300d.txt'
word2vec_wiki_file = glove_path / 'glove.wiki.gensim.txt'


# In[ ]:


glove2word2vec(glove_input_file=glove_wiki_file, word2vec_output_file=word2vec_wiki_file)


# ### Twitter Data

# In[18]:


glove_twitter_file= glove_path / 'glove.twitter.27B.200d.txt'
word2vec_twitter_file = glove_path / 'glove.twitter.gensim.txt'


# In[19]:


glove2word2vec(glove_input_file=glove_twitter_file, word2vec_output_file=word2vec_twitter_file)


# ### Common Crawl

# In[26]:


glove_crawl_file= glove_path / 'glove.840B.300d.txt'
word2vec_crawl_file = glove_path / 'glove.crawl.gensim.txt'


# In[27]:


glove2word2vec(glove_input_file=glove_crawl_file, word2vec_output_file=word2vec_crawl_file)


# ## Evaluate embeddings

# In[37]:


def eval_analogies(file_name, vocab=30000):
    model = KeyedVectors.load_word2vec_format(file_name, binary=False)
    accuracy = model.wv.accuracy(analogies_path,
                                 restrict_vocab=vocab,
                                 case_insensitive=True)
    return (pd.DataFrame([[c['section'],
                           len(c['correct']),
                           len(c['incorrect'])] for c in accuracy],
                         columns=['category', 'correct', 'incorrect'])
            .assign(samples=lambda x: x.correct.add(x.incorrect))
            .assign(average=lambda x: x.correct.div(x.samples))
            .drop(['correct', 'incorrect'], axis=1))


# In[40]:


result = eval_analogies(word2vec_twitter_file, vocab=100000)


# ### twitter result

# In[41]:


result


# ### wiki result

# In[39]:


result


# ### Common Crawl result

# In[33]:


result


# In[16]:


result


# In[17]:


result.to_csv(glove_path / 'accuracy.csv', index=False)


# In[ ]:




