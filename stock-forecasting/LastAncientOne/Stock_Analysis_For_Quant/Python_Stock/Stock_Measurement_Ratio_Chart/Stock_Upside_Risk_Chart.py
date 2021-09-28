#!/usr/bin/env python
# coding: utf-8

# # Stock Upside Risk Chart

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
symbol = 'AMD' #input


# In[3]:


stocks = yf.download(symbol, start=start, end=end)['Adj Close']


# In[4]:


stocks_returns = stocks.pct_change().dropna()


# In[5]:


def upside_risk(stock_returns):
    ur = stock_returns[stock_returns > stock_returns.mean()].std(skipna = True) * np.sqrt(252)
    return ur


# In[6]:


# Compute the running Upside Risk
running = [upside_risk(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Upside Risk up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Upside Risk')
plt.xlabel('Date')
plt.ylabel('Upside Risk')


# In[7]:


stock_ur = upside_risk(stocks_returns)
stock_ur


# In[8]:


running = [upside_risk(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]
running

