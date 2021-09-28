#!/usr/bin/env python
# coding: utf-8

# # Stock Covariance & Correlations

# #### Covariance measures the directional relationship between the returns on two assets. A positive covariance means that asset returns move together while a negative covariance means they move inversely. Covariance is calculated by analyzing at-return surprises (standard deviations from the expected return) or by multiplying the correlation between the two variables by the standard deviation of each variable. (https://www.investopedia.com/terms/c/covariance.asp)
# 
# #### Stock correlation explained the relationship that exists between two stocks and their respective price movements which has a value that must fall between -1.0 and +1.0.    
# 
# #### A perfect positive correlation means that the correlation coefficient is exactly 1. This implies that as one security moves, either up or down, the other security moves in lockstep, in the same direction. A perfect negative correlation means that two assets move in opposite directions, while a zero correlation implies no relationship at all. (https://www.investopedia.com/terms/c/correlation.asp)

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# ## Two Securities Correlation

# In[2]:


# input
symbols = ['AMD','INTC']
start = '2012-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbols,start,end)['Adj Close']

# View Columns
dataset.head()


# In[3]:


stocks_returns = np.log(dataset / dataset.shift(1))


# In[4]:


AMD = stocks_returns['AMD'].var() 
AMD


# In[5]:


INTC = stocks_returns['INTC'].var() 
INTC


# In[6]:


AMD = stocks_returns['AMD'].var() * 250
AMD


# In[7]:


INTC = stocks_returns['INTC'].var() * 250
INTC


# In[8]:


cov_matrix = stocks_returns.cov()
cov_matrix


# In[9]:


print('Covariance Matrix')
cov_matrix = stocks_returns.cov()*250
cov_matrix


# In[10]:


print('Correlation Matrix')
corr_matrix = stocks_returns.corr()*250
corr_matrix


# ## Four Securities Correlation

# In[11]:


# input
symbols = ['AAPL','MSFT','AMD','NVDA']
start = '2012-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbols,start,end)['Adj Close']

# View Columns
dataset.head()


# In[12]:


stocks_returns = np.log(dataset / dataset.shift(1))


# In[13]:


AAPL = stocks_returns['AAPL'].var() 
AAPL


# In[14]:


MSFT = stocks_returns['MSFT'].var() 
MSFT


# In[15]:


AMD = stocks_returns['AMD'].var() 
AMD


# In[16]:


NVDA = stocks_returns['NVDA'].var() 
NVDA


# In[17]:


AAPL = stocks_returns['AAPL'].var() * 250
AAPL


# In[18]:


MSFT = stocks_returns['MSFT'].var() * 250 
MSFT


# In[19]:


AMD = stocks_returns['AMD'].var() * 250 
AMD


# In[20]:


NVDA = stocks_returns['NVDA'].var() * 250 
NVDA


# In[21]:


cov_matrix = stocks_returns.cov()
cov_matrix


# In[22]:


print('Covariance Matrix')
cov_matrix = stocks_returns.cov()*250
cov_matrix


# In[23]:


print('Correlation Matrix')
corr_matrix = stocks_returns.corr()*250
corr_matrix

