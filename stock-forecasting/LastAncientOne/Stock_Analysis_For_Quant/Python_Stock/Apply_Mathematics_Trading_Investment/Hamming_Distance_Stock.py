#!/usr/bin/env python
# coding: utf-8

# # Hamming Distance

# ## Hamming distance between two vectors is simply the sum of corresponding elements that differ between the vectors.

# In[1]:


import numpy as np
from scipy.spatial.distance import hamming

import warnings
warnings.filterwarnings("ignore") 

# yfinance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


symbol = 'AMD'

start = '2018-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbol,start,end)

# View Columns
dataset.head()


# In[3]:


dataset.tail()


# In[4]:


dataset = dataset.drop(['Adj Close', 'Volume'], axis=1)
dataset.head()


# In[5]:


def hamming_distance(x, y):

    hamming_d = hamming(x, y) * len(x)
    return hamming_d


# In[6]:


Open = np.array(dataset['Open'])


# In[7]:


Close = np.array(dataset['Close'])


# In[8]:


Open


# In[9]:


Close


# In[10]:


hamming_distance(Open, Close)

