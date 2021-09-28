#!/usr/bin/env python
# coding: utf-8

# # Euclidean Distance

# ## Euclidean distance between two vectors, A and B
# 
# ## Formula: √Σ(Ai-Bi)^2

# In[1]:


import numpy as np
from numpy.linalg import norm

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


norm(Close - Open)

