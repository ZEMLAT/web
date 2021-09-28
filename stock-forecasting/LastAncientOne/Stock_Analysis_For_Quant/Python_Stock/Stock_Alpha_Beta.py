#!/usr/bin/env python
# coding: utf-8

# # Stock Alpha & Beta

# Alpha is a measurement of performance. A positive alpha of 1.0 means the fund or stock has outperformed its benchmark index by 1 percent. A negative alpha of 1.0 would indicate an underperformance of 1 percent.
# 
# Beta is a measurement of volatile. A beta of less than 1 means that the security will be less volatile than the market.

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
ticker = "AMD"
spx = "^GSPC"
start = '2014-01-01'
end = '2019-01-01'

# Read data 
stock = yf.download(ticker,start,end)
market = yf.download(spx, start, end)


# In[3]:


# View columns 
stock.head()


# In[4]:


# View columns 
market.head()


# In[5]:


prices = stock['Adj Close']
values = market['Adj Close']


# In[6]:


#ret = prices.pct_change(1)[1:]
#ret = np.log(prices/prices.shift(1))
ret = (np.log(prices) - np.log(prices.shift(1))).dropna()


# In[7]:


ret.head()


# In[8]:


mrk = values.pct_change(1).dropna()


# In[9]:


mrk.head()


# In[10]:


from scipy import stats

beta, alpha, r_value, p_value, std_err = stats.linregress(ret, mrk)


# In[11]:


print("Beta: 			%9.6f" % beta)
print("Alpha: 			%9.6f" % alpha)
print("R-Squared: 		%9.6f" % r_value)
print("p-value: 		%9.6f" % p_value)
print("Standard Error: 	%9.6f" % std_err)

