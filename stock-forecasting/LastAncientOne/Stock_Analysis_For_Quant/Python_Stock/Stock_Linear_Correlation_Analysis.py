#!/usr/bin/env python
# coding: utf-8

# # Stock Linear Correlation Analysis

# Correlation is a measure of the strength of linear between 2 or more stocks. However, when 2 stocks is highly correlated, they tend to move in teh same direction. 

# In[1]:


# Library
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


start = '2016-01-01'
end = '2019-01-01'

market = 'SPY'
symbol1 = 'AAPL'
symbol2 = 'MSFT'
bench = yf.download(market, start=start, end=end)['Adj Close']
stock1 = yf.download(symbol1, start=start, end=end)['Adj Close']
stock2 = yf.download(symbol2, start=start, end=end)['Adj Close']


# In[3]:


plt.figure(figsize=(12,8))
plt.scatter(stock1,stock2)
plt.xlabel(symbol1)
plt.ylabel(symbol2)
plt.title('Stock prices from ' + start + ' to ' + end)


# In[4]:


plt.figure(figsize=(12,8))
plt.scatter(stock1,bench)
plt.xlabel(symbol1)
plt.ylabel(market)
plt.title('Stock prices from ' + start + ' to ' + end)


# In[5]:


plt.figure(figsize=(12,8))
plt.scatter(stock2,bench)
plt.xlabel(symbol2)
plt.ylabel(market)
plt.title('Stock prices from ' + start + ' to ' + end)


# In[6]:


print("Correlation coefficients")
print(symbol1 + ' and ' + symbol2 + ':', np.corrcoef(stock1,stock2)[0,1])
print(symbol1 + ' and ' + market + ':', np.corrcoef(stock1,bench)[0,1])
print(market + ' and ' + symbol2 + ':', np.corrcoef(bench,stock2)[0,1])


# In[7]:


rolling_correlation = stock1.rolling(60).corr(stock2)
plt.figure(figsize=(12,8))
plt.plot(rolling_correlation)
plt.xlabel('Day')
plt.ylabel('60-day Rolling Correlation')

