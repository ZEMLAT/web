#!/usr/bin/env python
# coding: utf-8

# # RSI & Bollinger Bands Strategy

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AAPL'
start = '2018-08-01'
end = '2018-12-31'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


# Simple Line Chart
plt.figure(figsize=(14,10))
plt.plot(df['Adj Close'])
plt.legend(loc='best')
plt.title('Stock '+ symbol +' Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# ## RSI

# In[4]:


import talib as ta


# In[5]:


rsi = ta.RSI(df['Adj Close'], timeperiod=14)
rsi = rsi.dropna()
rsi


# ## Bollinger Bands

# In[6]:


# Create Bollinger Band
# https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_bands
df['20 Day MA'] = df['Adj Close'].rolling(window=20).mean()
df['20 Day STD'] = df['Adj Close'].rolling(window=20).std()
df['Upper Band'] = df['20 Day MA'] + (df['20 Day STD'] * 2)
df['Lower Band'] = df['20 Day MA'] - (df['20 Day STD'] * 2)


# In[7]:


df[['Adj Close', '20 Day MA', 'Upper Band', 'Lower Band']].plot(figsize=(14,8))
plt.title('30 Day Bollinger Band for Facebook')
plt.ylabel('Price')
plt.legend(loc='best')
plt.show()


# In[8]:


dfc = df.copy()
dfc = dfc.reset_index()


# In[9]:


from matplotlib import dates as mdates
import datetime as dt

dfc['Date'] = mdates.date2num(dfc['Date'].astype(dt.date))


# In[10]:


# This one has not date and is convert to number
from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(1,1,1)
candlestick_ohlc(ax,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
plt.title('Candlestick Chart of Stock')
plt.ylabel('Price')
plt.show()


# In[11]:


# Plot Candlestick with dates
fig = plt.figure(figsize=(20,10))
ax = plt.subplot(1,1,1)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
candlestick_ohlc(ax,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
plt.title('Candlestick Chart of Stock')
plt.ylabel('Price')
plt.show()


# ## Combine RSI and Bollinger Bands

# In[12]:


fig = plt.figure(figsize=(20,18))
ax = plt.subplot(2,1,2)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.plot(df[['20 Day MA', 'Upper Band', 'Lower Band']], label=('20 Day MA', 'Upper Band', 'Lower Band'))
candlestick_ohlc(ax,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
plt.title('RSI & Bollinger Bands')
plt.ylabel('Price')

plt.plot(rsi, '-', label='RSI')
plt.text(s='Overbought', x=rsi.index[0], y=80, fontsize=14)
plt.text(s='OverSold', x=rsi.index[0], y=20, fontsize=14)
ax.axhline(y=80,color='r')
ax.axhline(y=20,color='r')
plt.xlabel('Date')
plt.legend(loc='best')
plt.show()

