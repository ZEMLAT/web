#!/usr/bin/env python
# coding: utf-8

# # RSI(2)

# https://stockcharts.com/school/doku.php?id=chart_school:trading_strategies:rsi2

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
start = '2018-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


# Simple way to do RSI
import talib as ta

df['MA5'] = df['Adj Close'].rolling(5).mean()
df['MA200'] = df['Adj Close'].rolling(200).mean()
df['RSI2'] = ta.RSI(df['Adj Close'], timeperiod=2)
df.head()


# In[4]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
ax1.plot(df['MA5'], label='MA5')
ax1.plot(df['MA200'], label='MA200')
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.legend(loc='best')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['RSI2'], label='Relative Strengths Index')
ax2.text(s='Overbought', x=df.RSI2.index[30], y=70, fontsize=14)
ax2.text(s='Oversold', x=df.RSI2.index[30], y=30, fontsize=14)
ax2.axhline(y=70, color='red')
ax2.axhline(y=30, color='red')
ax2.fill_between(df.index, y1=30, y2=70, color='#adccff', alpha='0.3')
ax2.axhline(y=95, color='darkblue')
ax2.axhline(y=5, color='darkblue')
ax2.grid()
ax2.set_ylabel('RSI2')
ax2.set_xlabel('Date')


# ## Candlestick with RSI2 Strategy

# In[5]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = mdates.date2num(dfc['Date'].astype(dt.date))
dfc.head()


# In[6]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['MA5'], label='MA5')
ax1.plot(df['MA200'], label='MA200')
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax1.grid(True, which='both')
ax1.minorticks_on()
ax1v = ax1.twinx()
colors = dfc.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(dfc.Date, dfc['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.legend(loc='best')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['RSI2'], label='Relative Strengths Index')
ax2.text(s='Overbought', x=df.RSI2.index[30], y=70, fontsize=14)
ax2.text(s='Oversold', x=df.RSI2.index[30], y=30, fontsize=14)
ax2.axhline(y=70, color='red')
ax2.axhline(y=30, color='red')
ax2.fill_between(dfc.Date, y1=30, y2=70, color='#adccff', alpha='0.3')
ax2.axhline(y=95, color='darkblue')
ax2.axhline(y=5, color='darkblue')
ax2.grid()
ax2.set_ylabel('RSI2')
ax2.set_xlabel('Date')

