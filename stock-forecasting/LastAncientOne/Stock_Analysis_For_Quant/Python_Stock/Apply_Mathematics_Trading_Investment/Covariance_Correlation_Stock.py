#!/usr/bin/env python
# coding: utf-8

# # Variance, Covariance, and Correlation

# In[1]:


import numpy as np

import warnings
warnings.filterwarnings("ignore") 

# yfinance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


symbol = 'AMD'
market = '^GSPC'
start = '2018-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbol,start,end)['Adj Close']
benchmark = yf.download(market,start,end)['Adj Close']

# View Columns
dataset.head()


# In[3]:


benchmark.head()


# ### Math for variance

# In[4]:


variance = ((dataset - dataset.mean())**2).sum() / len(dataset)


# In[5]:


print("The Variance for " + symbol + ":", variance)


# ### Math for covariance

# In[6]:


covariance = ((dataset - dataset.mean()) * (dataset - dataset.mean())).sum() / (len(dataset) - 1)


# In[7]:


print("The Covariance for " + symbol + ":", covariance)


# ### Math for correlation coefficient

# In[8]:


upper = ((dataset - dataset.mean()) * (benchmark - benchmark.mean())).sum()
lower = np.sqrt((((dataset - dataset.mean())**2).sum()) * (((benchmark - benchmark.mean())**2).sum()))

correlation_coefficient = upper/lower


# In[9]:


print("The Correlation Coefficient for " + symbol + ":", correlation_coefficient)


# In[10]:


r_square = correlation_coefficient**2


# In[11]:


print("The R-Square for " + symbol + ":", r_square)

