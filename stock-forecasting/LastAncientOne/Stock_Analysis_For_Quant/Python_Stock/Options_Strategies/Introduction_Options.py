#!/usr/bin/env python
# coding: utf-8

# # Introduction to Options

# ## Put-Call Parity

# # Call Options

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[4]:


dfo = yf.Ticker("AAPL")


# In[9]:


dfo.options


# In[12]:


dfo_exp = dfo.option_chain('2020-04-16')


# In[13]:


dfo_exp.calls


# In[14]:


dfo_exp.puts


# In[6]:


df = yf.download("AAPL")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[22]:


price = df['Adj Close']
strike = dfo_exp.calls['strike'][0]
premium = dfo_exp.calls['lastPrice'][0]


# In[23]:


payoff = [max(-premium, i - strike-premium) for i in price]


# In[24]:


payoff = [max(-premium, i - strike-premium) for i in price]


# In[25]:


plt.plot(price, payoff)
plt.xlabel('Price at T S_T ($)')
plt.ylabel('payoff')
plt.title('Call option Payoff at Expiry')
plt.grid(True)


# # Put Option

# In[27]:


price = df['Adj Close']
strike = dfo_exp.puts['strike'][0]
premium = dfo_exp.puts['lastPrice'][0]


# In[28]:


payoff = [max(-premium, strike - i -premium) for i in price]


# In[29]:


plt.plot(price, payoff)
plt.xlabel('Price at T S_T ($)')
plt.ylabel('payoff')
plt.title('Put option Payoff at Expiry')
plt.grid(True)

