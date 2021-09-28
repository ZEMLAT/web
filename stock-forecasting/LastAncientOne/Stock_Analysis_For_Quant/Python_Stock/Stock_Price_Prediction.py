#!/usr/bin/env python
# coding: utf-8

# # Stock Price Predictions
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
yf.pdr_override()


# In[2]:


symbol = 'AAPL'
start = '2020-01-01' 
end = '2021-01-01'
df = yf.download(symbol, start, end)
df = df.reset_index()


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


X_train = df[df.columns[1:5]] # data_aal[['open', 'high', 'low', 'close']]
Y_train = df['Adj Close']


# In[6]:


X_train = X_train.values[:-1]
Y_train = Y_train.values[1:]


# In[7]:


lr = LinearRegression()


# In[8]:


lr.fit(X_train, Y_train)


# In[9]:


X_test = df[df.columns[1:5]].values[:-1]
Y_test = df['Adj Close'].values[1:]


# In[10]:


lr.score(X_test, Y_test)


# In[11]:


opening_price = float(input('Open: '))
high = float(input('High: '))
low = float(input('Low: '))
close = float(input('Close: '))
print('My Prediction the opening price will be:', lr.predict([[opening_price, high, low, close]])[0])

