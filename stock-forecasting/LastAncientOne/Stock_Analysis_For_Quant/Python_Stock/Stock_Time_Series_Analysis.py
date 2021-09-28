#!/usr/bin/env python
# coding: utf-8

# # Stock Time Series Analysis

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# fetch yahoo data
import yfinance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AMD'
start = '2014-01-01'
end = '2018-08-27'

# Read data 
dataset = yf.download(symbol,start,end)

# Only keep close columns 
dataset.head()


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


len(dataset['Adj Close'].loc[:'2018-01-01'])


# In[6]:


weekly_Monday = dataset.asfreq('W-Mon')


# In[7]:


weekly_Monday


# In[8]:


fig, ax = plt.subplots(figsize=(16, 4))
weekly_Monday['Adj Close'].plot(title='Weekly Stock Adj Close for Monday', ax=ax)


# In[9]:


weekly_avg = dataset.resample('W').mean()


# In[10]:


weekly_avg


# In[11]:


fig, ax = plt.subplots(figsize=(16, 4))
weekly_avg['Adj Close'].plot(title='Weekly Stock Average for Monday', ax=ax)


# In[12]:


weekly_first = dataset.resample('W').first()


# In[13]:


fig, ax = plt.subplots(figsize=(16, 4))
weekly_first['Adj Close'].plot(title='First Weekly Stock', ax=ax)


# In[20]:


fig, ax = plt.subplots(figsize=(16, 4))
(dataset.groupby(pd.Grouper(freq='W'))[['Low','High']]).mean().plot(color=['Red', 'Green'], ax=ax, title='First Weekly Stock')


# In[15]:


business_monthly = dataset.resample('BM')


# In[16]:


fig, ax = plt.subplots(figsize=(16, 4))
business_monthly['Adj Close'].plot(title='Stock Close Price monthly', ax=ax)


# In[17]:


business_monthly['Adj Close'].plot(title='Stock Close Price monthly', ax=ax)

