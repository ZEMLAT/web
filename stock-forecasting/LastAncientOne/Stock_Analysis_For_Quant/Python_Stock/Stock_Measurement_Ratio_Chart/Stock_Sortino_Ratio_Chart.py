#!/usr/bin/env python
# coding: utf-8

# # Stock Sortino Ratio Chart

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


def sortino_ratio(symbol):
    numer = pow((1 + symbol.mean()), 252) - 1
    annual_volatility = symbol.std() * np.sqrt(252)
    denom = annual_volatility

    if denom > 0.0:
         sortino_ratio = numer / denom
    else:
        print('none')
    return sortino_ratio


# In[6]:


# Compute the running Sortino Ratio
running_sharpe = [sortino_ratio(returns[i-90:i]) for i in range(90, len(returns))]

# Plot running Sortino Ratio up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(returns)-100), running_sharpe[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([df['Adj Close'].index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Sortino Ratio')
plt.xlabel('Date')
plt.ylabel('Sortino Ratio')

