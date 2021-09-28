#!/usr/bin/env python
# coding: utf-8

# # Put Option

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


dfo = yf.Ticker("MSFT")


# In[3]:


dfo.options


# In[4]:


dfo_exp = dfo.option_chain('2020-03-19')


# In[5]:


dfo_exp.puts


# In[6]:


df = yf.download("MSFT")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


df['Adj Close'][-1]


# In[10]:


spot_price = df['Adj Close'][-1] # current price
share_price = np.arange(0.9*spot_price,1.1*spot_price)
strike_price = dfo_exp.puts['strike'][37] # exercise price of an options that is fixed price
put_price = dfo_exp.puts['lastPrice'][37] # price of an option or premium 


# In[11]:


def put_option(share_price, strike_price, put_price):
    pnl = np.where(share_price < strike_price, strike_price - share_price, 0)  
    return pnl - put_price


# In[12]:


payoff_long_put = put_option(share_price, strike_price, put_price)
# Plot the graph
plt.subplots(figsize=(16,8))
plt.gca().spines['bottom'].set_position('zero')
plt.plot(share_price, payoff_long_put,label='Put option buyer payoff',color='g')
plt.xlabel('Range Stock Price')
plt.ylabel('Profit and loss')
plt.grid(which='both')
plt.legend()
plt.show()


# In[13]:


payoff_short_put = payoff_long_put * -1.0
# Plot
plt.subplots(figsize=(16,8))
plt.gca().spines['bottom'].set_position('zero')
plt.plot(share_price,payoff_short_put,label='Short 147 Strike Put',color='r')
plt.xlabel('Range Stock Price')
plt.ylabel('Profit and loss')
plt.grid(which='both')
plt.legend()
plt.show()

