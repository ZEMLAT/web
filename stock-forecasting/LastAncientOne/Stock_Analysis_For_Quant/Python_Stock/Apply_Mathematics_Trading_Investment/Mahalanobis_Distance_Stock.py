#!/usr/bin/env python
# coding: utf-8

# # Mahalanobis Distance

# ## Mahalanobis distance is the distance between two points in a multivariate space. It’s  used in statistical analyses to find outliers that involve serval variables.
# 

# ## Formula: d(p,q) = √(p1-q1)^2 + (p2-q2)^2

# In[1]:


import numpy as np
import scipy as stats
from scipy.stats import chi2

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


def mahalanobis_distance(x=None, data=None, cov=None):

    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


# In[6]:


df = mahalanobis_distance(x=dataset, data=dataset)
df


# In[7]:


dataset = dataset.reset_index(drop=True)


# In[8]:


dataset.head()


# In[9]:


dataset['mahalanobis'] = mahalanobis_distance(x=dataset, data=dataset[['Open', 'High', 'Low', 'Close']])
dataset.head()


# In[10]:


dataset['p'] = 1 - chi2.cdf(dataset['mahalanobis'], 4)
dataset.head()

