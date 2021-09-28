#!/usr/bin/env python
# coding: utf-8

# # Chebyshev Distance

# ### Chebyshev distance (or Tchebychev distance), maximum metric, or L∞ metric. is a metric defined on a vector space where the distance between two vectors is the greatest of their differences along any coordinate dimension (wikipeida).
# 
# ## Formula: max(|xA - xB|, |yA - yB|)

# In[1]:


import numpy as np
from scipy.spatial import distance

import matplotlib.pyplot as plt

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
High = np.array(dataset['High'])
Low = np.array(dataset['Low'])


# In[4]:


Open


# In[5]:


Close


# In[6]:


max(Close)


# In[7]:


distance.chebyshev(Open, Close)


# In[8]:


x = Low
y = High
p = np.polynomial.Chebyshev.fit(x, y, 90)

plt.plot(x, y, 'r.')
plt.plot(x, p(x), 'k-', lw=3)
plt.show()

