#!/usr/bin/env python
# coding: utf-8

# # Portfolio Functions

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


def get_historical_price(ticker, start_date, end_date):
    df = yf.download(ticker, start_date, end_date)['Adj Close']
    return df


# In[3]:


symbols = ['FB','JNJ','LMT']
start = '2012-01-01'
end = '2019-01-01'


# In[4]:


closes = get_historical_price(symbols, start, end)


# In[5]:


closes[:5]


# In[6]:


def calc_daily_returns(closes):
    return np.log(closes/closes.shift(1))


# In[7]:


daily_returns = calc_daily_returns(closes)
daily_returns = daily_returns.dropna()
daily_returns[:5]


# In[8]:


def calc_month_returns(daily_returns):
    monthly = np.exp(daily_returns.groupby(lambda date: date.month).sum())-1
    return monthly


# In[9]:


month_returns = calc_month_returns(daily_returns)
month_returns


# In[10]:


def calc_annual_returns(daily_returns):
    grouped = np.exp(daily_returns.groupby(lambda date: date.year).sum())-1
    return grouped


# In[11]:


annual_returns = calc_annual_returns(daily_returns)
annual_returns


# In[12]:


def calc_portfolio_var(returns, weights=None):
    if (weights is None):
        weights = np.ones(returns.columns.size) / returns.columns.size
    sigma = np.cov(returns.T,ddof=0)
    var = (weights * sigma * weights.T).sum()
    return var


# In[13]:


calc_portfolio_var(annual_returns)


# In[14]:


def Sharpe_ratio(returns, weights = None, risk_free_rate = 0.001):
    n = returns.columns.size
    if (weights is None): 
        weights = np.ones(n)/n
        var = calc_portfolio_var(returns, weights)
        means = returns.mean()
        sr = (means.dot(weights) - risk_free_rate)/np.sqrt(var)
        return sr


# In[15]:


Sharpe_ratio(daily_returns)

