#!/usr/bin/env python
# coding: utf-8

# # Black Scholes Stock Puts Inputs

# In[1]:


import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


dfo = yf.Ticker("AAPL")


# In[3]:


dfo.options


# In[4]:


dfo_exp = dfo.option_chain('2020-05-28')


# In[5]:


dfo_exp.puts


# In[6]:


symbol = 'AAPL'
start = '2019-12-01'
end = '2020-04-02'


# In[7]:


df = yf.download(symbol,start,end)


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


returns = df['Adj Close'].pct_change().dropna()


# In[11]:


from datetime import datetime
from dateutil import relativedelta

d1 = datetime.strptime(start, "%Y-%m-%d")
d2 = datetime.strptime('2020-05-28', "%Y-%m-%d")
delta = relativedelta.relativedelta(d2,d1)
print('How many years of investing?')
print('%s years' % delta.years)


# In[12]:


maturity_days = (df.index[-1] - df.index[0]).days
print('%s days' % maturity_days)


# In[13]:


S0 = df['Adj Close'][-1]
K = dfo_exp.puts['strike'][6]
r = 0.1
sigma = returns.std()
T = maturity_days/252


# In[14]:


print("S0\tCurrent Stock Price:", S0)
print("K\tStrike Price:", K)
print("r\tContinuously compounded risk-free rate:", r)
print("sigma\tVolatility of the stock price per year:", sigma)
print("T\tTime to maturity in trading years:", T)


# In[15]:


def d1(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    return d1


# In[16]:


def d2(S0, K, r, sigma, T):
    d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return d2


# In[17]:


def BlackScholesCall(S0, K, r, sigma, T):
    BSC = S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    return BSC       


# In[18]:


def BlackScholesPut(S0, K, r, sigma, T):
    BSP = K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T)) 
    return BSP


# In[19]:


Put_BS = BlackScholesPut(S0, K, r, sigma, T)
Put_BS

