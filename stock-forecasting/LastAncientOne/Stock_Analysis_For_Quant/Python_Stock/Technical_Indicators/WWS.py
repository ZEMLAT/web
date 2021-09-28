#!/usr/bin/env python
# coding: utf-8

# # Welles Wilder’s Smoothing Average (WWS)

# https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/welles-wilders-smoothing-average-wws/
# 
# http://etfhq.com/blog/2010/08/19/wilders-smoothing/

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


n = 14
df['WSMA'] = (df['Adj Close'].rolling(n).sum())/n
df['WWS'] = (df['Adj Close'].rolling(n).sum()-df['WSMA']+df['Adj Close'])/n
df = df.dropna()
df.head()


# In[4]:


plt.figure(figsize=(16,10))
plt.plot(df['Adj Close'])
plt.plot(df['WWS'])
plt.title('Welles Wilder’s Smoothing Average for Stock')
plt.legend(loc='best')
plt.xlabel('Price')
plt.ylabel('Date')
plt.show()


# ## Candlestick with WWS

# In[5]:


from matplotlib import dates as mdates
import datetime as dt


df['VolumePositive'] = df['Open'] < df['Adj Close']
df = df.dropna()
df = df.reset_index()
df['Date'] = mdates.date2num(df['Date'].astype(dt.date))
df.head()


# In[6]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(16,8))
ax1 = plt.subplot(111)
candlestick_ohlc(ax1,df.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df.Date, df['WWS'])
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
#ax1.axhline(y=dfc['Adj Close'].mean(),color='r')
ax1v = ax1.twinx()
colors = df.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(df.Date, df['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')
ax1.legend(loc='best')

