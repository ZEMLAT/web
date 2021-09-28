#!/usr/bin/env python
# coding: utf-8

# # Stock Skewness Chart

# In[1]:


# Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs

import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[2]:


start = '2016-01-01' #input
end = '2020-07-01' #input
symbol = 'AMD'


# In[3]:


df = yf.download("AMD", start, end)


# In[4]:


returns = df['Adj Close'].pct_change()[1:].dropna()


# In[5]:


def skewness(returns):
    stock_skewness = scs.skew(returns)
    return stock_skewness


# In[6]:


# Compute the running skewness
running = [skewness(returns[i-90:i]) for i in range(90, len(returns))]

# Plot running skewness up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([df.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Skewness')
plt.xlabel('Date')
plt.ylabel('Skewness')


# In[7]:


sk = skewness(returns)
sk


# In[8]:


running = [skewness(returns[i-90:i]) for i in range(90, len(returns))]
running

