#!/usr/bin/env python
# coding: utf-8

# # Stock Adjusted Sharpe Ratio Chart

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


# risk free
rf = yf.download('BIL', start=start, end=end)['Adj Close'].pct_change()[1:]


# In[6]:


def adj_sharpe_ratio(stock_returns, rf):
    sharpe_ratio = (stock_returns.mean() - rf) / stock_returns.std()
    stock_skewness = scs.skew(stock_returns)
    stock_kurtosis = stock_returns.kurtosis()
    Adj_SR = sharpe_ratio * (1 + (stock_skewness / 6.0) * sharpe_ratio + (stock_kurtosis - 3) / 24.0 * sharpe_ratio**2)
    return Adj_SR


# In[7]:


# Compute the running Adjusted Sharpe ratio
running_sharpe = [adj_sharpe_ratio(returns[i-90:i], rf[i-90:i]) for i in range(90, len(returns))]

# Plot running Adjusted Sharpe ratio up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(returns)-100), running_sharpe[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([df['Adj Close'].index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Adjusted Sharpe Ratio')
plt.xlabel('Date')
plt.ylabel('Adjusted Sharpe Ratio')


# In[8]:


adj_sharpe_ratio(returns, rf)

