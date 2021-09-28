#!/usr/bin/env python
# coding: utf-8

# # Butterfly Spread

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


dfo = yf.Ticker("AAPL")


# In[3]:


dfo.options


# In[4]:


dfo_exp = dfo.option_chain('2020-11-26')


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


# In[10]:


K1 = dfo_exp.calls['strike'][11]
K2 = dfo_exp.calls['strike'][13]
K3 = dfo_exp.calls['strike'][15]


# In[11]:


CurrentStockPrice = df['Adj Close'][-1]


# In[12]:


i = 0
while i < 10:
    CurrentStockPrice = CurrentStockPrice + 25
    payoff = 0
    if (CurrentStockPrice > K1 and CurrentStockPrice <= K2):
         payoff = CurrentStockPrice - K1
    elif (CurrentStockPrice > K2 and CurrentStockPrice < K3):
         payoff = K3 - CurrentStockPrice
    elif (CurrentStockPrice <= K1):
         payoff = 0
    elif (CurrentStockPrice >= K3):
         payoff  = 0       
    else:
         payoff  = 0      
    print("Stock price: %2.2f"%CurrentStockPrice)
    print("Payoff from butterfly spread: %2.2f"%payoff)
    print("-"*35)
    i = i + 1
    
while i >= 0:
    CurrentStockPrice = CurrentStockPrice - 25
    payoff = 0
    if (CurrentStockPrice > K1 and CurrentStockPrice <= K2):
        payoff = CurrentStockPrice - K1
    elif (CurrentStockPrice > K2 and CurrentStockPrice < K3):
        payoff = K3 - CurrentStockPrice
    elif (CurrentStockPrice <= K1):
        payoff  = 0
    elif (CurrentStockPrice >= K3):
        payoff  = 0     
    else:
        payoff  = 0     

    print("Stock price: %2.2f"%CurrentStockPrice)
    print("Payoff from butterfly spread: %2.2f"%payoff)
    print("-"*35)
    i = i - 1

