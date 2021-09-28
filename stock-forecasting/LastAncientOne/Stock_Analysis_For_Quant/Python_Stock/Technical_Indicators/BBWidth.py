#!/usr/bin/env python
# coding: utf-8

# # Bollinger BandWidth

# https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:bollinger_band_width

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
start = '2017-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


n = 20
MA = pd.Series(df['Adj Close'].rolling(n).mean())
STD = pd.Series(df['Adj Close'].rolling(n).std())
bb1 = MA + 2*STD
df['Upper Bollinger Band'] = pd.Series(bb1)
bb2 = MA - 2*STD
df['Lower Bollinger Band'] = pd.Series(bb2)
df['SMA'] = df['Adj Close'].rolling(n).mean()


# In[4]:


df['BBWidth'] = (df['Upper Bollinger Band'] - df['Lower Bollinger Band'])/df['SMA'] * 100


# In[5]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
ax1.plot(df['Upper Bollinger Band'])
ax1.plot(df['Lower Bollinger Band'])
ax1.plot(df['Adj Close'].rolling(20).mean(), label='Mean Average', linestyle='--')
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.legend(loc='best')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['BBWidth'], label='BB Width', color='black')
ax2.plot(df['BBWidth'].rolling(20).mean(), label='200 Moving Average', color='darkblue')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('BB Width')
ax2.set_xlabel('Date')


# ## Candlestick with BB Width

# In[6]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = mdates.date2num(dfc['Date'].astype(dt.date))
dfc.head()


# In[7]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df['Upper Bollinger Band'], label='Upper Bollinger Band')
ax1.plot(df['Lower Bollinger Band'], label='Lower Bollinger Band')
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
ax1.legend(loc='best')
ax1.set_ylabel('Price')
ax1.set_xlabel('Date')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['BBWidth'], label='BB Width', color='black')
ax2.plot(df['BBWidth'].rolling(20).mean(), label='200 Moving Average', color='darkblue')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('BB Width')
ax2.set_xlabel('Date')

