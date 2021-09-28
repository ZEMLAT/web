#!/usr/bin/env python
# coding: utf-8

# # Stock Kelly Arithmetic Chart

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

def expected_arith(stock_returns):
    expected_arith = np.mean(stock_returns)
    return expected_arith

def kelly_arithmetic(stock_returns):
    bounded_rets = stock_returns / np.abs(np.min(stock_returns))
    kelly_f = kelly_fraction(bounded_rets) / np.abs(np.min(stock_returns))
    exp_arith_kelly = expected_arith(bounded_rets * kelly_f)
    return exp_arith_kelly


# In[6]:


# Compute the running Kelly Arithmetic
running = [kelly_arithmetic(stocks_returns[i-90:i]) for i in range(90, len(stocks_returns))]

# Plot running Kelly Arithmetic up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(stocks_returns)-100), running[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([stocks.index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Kelly Arithmetic')
plt.xlabel('Date')
plt.ylabel('Kelly Arithmetic')


# In[7]:


ka = kelly_arithmetic(stocks_returns)
ka


# In[8]:


running


# In[9]:


print('Expected Value (Arithmetic): {}%'.format(np.round(kelly_arithmetic(stocks_returns) * 100, 5)))

