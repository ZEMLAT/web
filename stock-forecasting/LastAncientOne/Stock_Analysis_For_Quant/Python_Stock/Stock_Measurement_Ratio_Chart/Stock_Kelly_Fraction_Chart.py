#!/usr/bin/env python
# coding: utf-8

# # Stock Kelly Fraction Chart

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


def kelly_fraction(stock_returns):
    # returns = np.array(stock_returns)
    wins = stock_returns[stock_returns > 0]
    losses = stock_returns[stock_returns <= 0]
    W = len(wins) / len(stock_returns)
    R = np.mean(wins) / np.abs(np.mean(losses))
    kelly_f = W - ( (1 - W) / R )
    return kelly_f


# In[6]:


# Compute the running Kelly Fraction
running = [kelly_fraction(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Kelly Fraction up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Kelly Fraction')
plt.xlabel('Date')
plt.ylabel('Kelly Fraction')


# In[7]:


kf = kelly_fraction(stocks_returns)
kf


# In[8]:


running

