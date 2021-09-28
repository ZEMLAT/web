#!/usr/bin/env python
# coding: utf-8

# # Stock Interactive Candlestick Chart

# In[1]:


# Library
import pandas as pd
import numpy as np
import plotly  
import plotly.graph_objs as go

import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[9]:


plotly.__version__


# In[2]:


start = '2016-01-01' #input
end = '2020-07-01' #input
symbol = 'AMD'


# In[3]:


df = yf.download("AMD", start, end)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


fig = go.Figure(data=[go.Candlestick(x=df.index,
                       open=df['Open'],
                       high=df['High'],
                       low=df['Low'],
                       close=df['Close'])])                     


# In[7]:


fig.show()


# In[10]:


fig = go.Figure(data=[go.Candlestick(x=df.index,
open=df['Open'],
high=df['High'],
low=df['Low'],
close=df['Close'],
increasing_line_color='red',
decreasing_line_color = 'blue'
)])
fig.show()

