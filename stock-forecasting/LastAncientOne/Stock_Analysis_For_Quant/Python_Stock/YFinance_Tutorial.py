#!/usr/bin/env python
# coding: utf-8

# In[2]:


import yfinance as yf


# In[4]:


aapl = yf.Ticker('AAPL')


# In[5]:


aapl.info


# In[6]:


df  =  aapl.history(start="2010-01-01",  end="2020-10-01")


# In[7]:


df.head()


# In[8]:


df.tail()


# In[10]:


aapl.actions


# In[11]:


aapl.dividends


# In[12]:


aapl.splits


# In[13]:


aapl.sustainability


# In[15]:


aapl.recommendations


# In[16]:


aapl.calendar


# In[19]:


aapl.options

