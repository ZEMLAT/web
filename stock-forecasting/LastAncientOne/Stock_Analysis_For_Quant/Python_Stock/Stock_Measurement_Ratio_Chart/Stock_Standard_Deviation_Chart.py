#!/usr/bin/env python
# coding: utf-8

# # Stock Standard Deviation Chart

# In[1]:


# Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def std(returns):
    stock_std = returns.std()
    return stock_std


# In[6]:


# Compute the running Standard Deviation
running = [std(returns[i-90:i]) for i in range(90, len(returns))]

# Plot running Standard Deviation up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([df.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')


# In[7]:


sd = std(returns)
sd


# In[8]:


running = [std(returns[i-90:i]) for i in range(90, len(returns))]
running

