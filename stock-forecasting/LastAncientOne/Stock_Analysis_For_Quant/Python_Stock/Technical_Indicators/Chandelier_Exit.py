#!/usr/bin/env python
# coding: utf-8

# # Chandelier Exit

# https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chandelier_exit

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
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


import talib as ta


# In[4]:


df['ATR'] = ta.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=22)


# In[5]:


df['High_22'] = df['High'].rolling(22).max()
df['Low_22'] = df['Low'].rolling(22).min()


# In[6]:


df['CH_Long'] = df['High_22'] - df['ATR'] * 3 
df['CH_Short'] = df['Low_22'] + df['ATR'] * 3


# In[7]:


df = df.dropna()
df.head()


# In[8]:


plt.figure(figsize=(16,10))
plt.plot(df['Adj Close'])
plt.plot(df['CH_Long'])
plt.title('Chandelier Exit for Long')
plt.legend(loc='best')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()


# In[9]:


plt.figure(figsize=(16,10))
plt.plot(df['Adj Close'])
plt.plot(df['CH_Short'])
plt.title('Chandelier Exit for Short')
plt.legend(loc='best')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()


# In[10]:


plt.figure(figsize=(16,10))
plt.plot(df['Adj Close'])
plt.plot(df['CH_Long'])
plt.plot(df['CH_Short'])
plt.title('Chandelier Exit for Long & Short')
plt.legend(loc='best')
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()


# ## Candlestick with Chandelier Exit

# In[11]:


from matplotlib import dates as mdates
import datetime as dt


df['VolumePositive'] = df['Open'] < df['Adj Close']
df = df.dropna()
df = df.reset_index()
df['Date'] = mdates.date2num(df['Date'].astype(dt.date))
df.head()


# In[12]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot(111)
candlestick_ohlc(ax1,df.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df.Date, df['CH_Long'])
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#ax1.axhline(y=dfc['Adj Close'].mean(),color='r')
ax1v = ax1.twinx()
colors = df.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(df.Date, df['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Chandelier Exit for Long')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')
ax1.legend(loc='best')


# In[13]:


fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot(111)
candlestick_ohlc(ax1,df.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df.Date, df['CH_Short'], color='Orange')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#ax1.axhline(y=dfc['Adj Close'].mean(),color='r')
ax1v = ax1.twinx()
colors = df.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(df.Date, df['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Chandelier Exit for Short')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')
ax1.legend(loc='best')


# In[14]:


fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot(111)
candlestick_ohlc(ax1,df.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df.Date, df['CH_Long'])
ax1.plot(df.Date, df['CH_Short'])
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#ax1.axhline(y=dfc['Adj Close'].mean(),color='r')
ax1v = ax1.twinx()
colors = df.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(df.Date, df['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Chandelier Exit for Long & Short')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')
ax1.legend(loc='best')

