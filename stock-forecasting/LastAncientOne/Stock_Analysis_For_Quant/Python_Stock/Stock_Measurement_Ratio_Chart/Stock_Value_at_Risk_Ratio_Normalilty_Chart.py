#!/usr/bin/env python
# coding: utf-8

# # Stock Value-at-Risk Ratio Normality Chart

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


# risk free
rf = yf.download('BIL', start=start, end=end)['Adj Close'].pct_change()[1:]


# In[6]:


def var_ratio_normality(symbol, rf):
    sr = np.mean(symbol - rf)/np.std(symbol - rf)
    t = 2.33
    var_n = sr  / (t - sr)
    return var_n


# In[7]:


# Compute the running Value-at-Risk Ratio Normality
running_sharpe = [var_ratio_normality(returns[i-90:i], rf[i-90:i]) for i in range(90, len(returns))]

# Plot running Value-at-Risk Ratio Normality up to 100 days before the end of the data set
_, ax1 = plt.subplots(figsize=(12,8))
ax1.plot(range(90, len(returns)-100), running_sharpe[:-100])
ticks = ax1.get_xticks()
ax1.set_xticklabels([df['Adj Close'].index[int(i)].date() for i in ticks[:-1]]) # Label x-axis with dates
plt.title(symbol + ' Value-at-Risk Ratio Normality')
plt.xlabel('Date')
plt.ylabel('Value-at-Risk Ratio Normality')


# In[8]:


var_ratio_normality(returns, rf)

