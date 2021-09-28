#!/usr/bin/env python
# coding: utf-8

# # Payoff for a Put Option

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


dfo_exp.puts


# In[6]:


df = yf.download("AAPL")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df['Adj Close'][-1]


# In[10]:


dfo_exp.puts['strike'][29]


# In[11]:


def payoff_puts(St, X):
    pop = (X-St+abs(X-St)) / 2
    return pop


# In[12]:


St = df['Adj Close'][-1] # Current Price
X = dfo_exp.puts['strike'][29] # Strike Price


# In[13]:


payoff_puts(St, X)


# In[14]:


St = df['Adj Close'][-1] # Current Price
X = dfo_exp.puts['strike'][30] # Strike Price


# In[15]:


payoff_puts(St, X)

