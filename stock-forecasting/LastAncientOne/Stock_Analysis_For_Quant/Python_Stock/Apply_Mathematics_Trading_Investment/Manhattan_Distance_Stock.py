#!/usr/bin/env python
# coding: utf-8

# # Manhattan Distance

# ## Manhattan distance between two vectors, A and B
# 
# ## Formula: Σ|Ai – Bi|
# 
# ### where i is the ith element in each vector. Manhattan distance is calculated as the sum of the absolute differences between the two vectors. The Manhattan distance is related to the L1 vector norm and the sum absolute error and mean absolute error metric
# 
# ### This distance is used to measure the dissimilarity between two vectors and is commonly used in machine learning algorithms.

# In[1]:


import numpy as np

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


Open = np.array(dataset['Open'])
Close = np.array(dataset['Adj Close'])


# In[4]:


Open


# In[5]:


Close


# In[6]:


def manhattan_distance(a, b):
    manhattan = sum(abs(value1-value2) for value1, value2 in zip(a,b))
    return manhattan


# In[7]:


manhattan_distance(Open, Close)

