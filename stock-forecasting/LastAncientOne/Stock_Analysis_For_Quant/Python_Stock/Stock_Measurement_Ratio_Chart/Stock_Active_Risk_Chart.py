#!/usr/bin/env python
# coding: utf-8

# # Stock Active Risk Chart

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


start = '2019-01-01' #input
end = '2020-07-01' #input
symbol1 = '^GSPC' #input
symbol2 = 'AMD' #input


# In[3]:


market = yf.download(symbol1, start=start, end=end)['Adj Close']
stocks = yf.download(symbol2, start=start, end=end)['Adj Close']


# In[4]:


market_returns = market.pct_change().dropna()
stocks_returns = stocks.pct_change().dropna()


# In[5]:


def active_risk(stock_returns, market_returns):
    N = 14  
    ar = np.sqrt((sum((stock_returns - market_returns)**2)/(N-1)))
    return ar


# In[6]:


# Compute the running Active Risk
running = [active_risk(stocks_returns[i-90:i], market_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Active Risk up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol2 + ' Active Risk')
plt.xlabel('Date')
plt.ylabel('Active Risk')


# In[7]:


stock_ar = active_risk(stocks_returns, market_returns)
stock_ar


# In[8]:


running = [active_risk(stocks_returns[i-90:i], market_returns[i-90:i]) for i in range(90, len(stocks_returns))]
running

