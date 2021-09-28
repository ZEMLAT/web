#!/usr/bin/env python
# coding: utf-8

# # Stock Downside Risk Chart

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


def downside_risk(stock_returns):
    dr = stock_returns[stock_returns < stock_returns.mean()].std(skipna = True) * np.sqrt(252)
    return dr


# In[6]:


# Compute the running Downside Risk
running = [downside_risk(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Downside Risk up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Downside Risk')
plt.xlabel('Date')
plt.ylabel('Downside Risk')


# In[7]:


stock_dr = downside_risk(stocks_returns)
stock_dr


# In[8]:


running = [downside_risk(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]
running

