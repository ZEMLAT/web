#!/usr/bin/env python
# coding: utf-8

# # Stock Lower Partial Moment Chart

# In[1]:


# Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import statistics

import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[2]:


start = '2019-01-01' #input
end = '2020-07-01' #input
symbol = 'AMD' #input


# In[3]:


stocks = yf.download(symbol, start=start, end=end)['Adj Close']


# In[4]:


stocks_returns = stocks.pct_change().dropna()


# In[5]:


def lpm(stock_returns):
    threshold=0.0
    order=1
    threshold_array = np.empty(len(stock_returns))
    threshold_array.fill(threshold)
    diff = threshold_array - stock_returns
    diff = diff.clip()
    return np.sum(diff ** order) / len(stock_returns)


# In[6]:


# Compute the running Lower Partial Moment
running = [lpm(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Lower Partial Moment up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Lower Partial Moment')
plt.xlabel('Date')
plt.ylabel('Lower Partial Moment')


# In[7]:


stock_lpm = lpm(stocks_returns)
stock_lpm


# In[8]:


running = [lpm(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]
running

