#!/usr/bin/env python
# coding: utf-8

# # Payoff for a Call Option

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


dfo = yf.Ticker("AAPL")


# In[3]:


dfo.options


# In[4]:


dfo_exp = dfo.option_chain('2020-04-16')


# In[5]:


dfo_exp.calls


# In[6]:


df = yf.download("AAPL")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df['Adj Close'][-1]


# In[11]:


dfo_exp.calls['strike'][31]


# In[12]:


def payoff_call(St, X):
    poc = (St-X+abs(St-X)) / 2
    return poc


# In[13]:


St = df['Adj Close'][-1] # Current Price
X = dfo_exp.calls['strike'][31] # Strike Price


# In[14]:


payoff_call(St, X)


# In[15]:


St = df['Adj Close'][-1] # Current Price
X = dfo_exp.calls['strike'][29] # Strike Price


# In[16]:


payoff_call(St, X)

