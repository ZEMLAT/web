#!/usr/bin/env python
# coding: utf-8

# # Elder Ray Index

# https://www.investopedia.com/terms/e/elderray.asp
# 
# https://library.tradingtechnologies.com/trade/chrt-ti-elder-ray-index.html

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


n = 13
df['EMA'] = df['Adj Close'].ewm(ignore_na=False,span=n,min_periods=n,adjust=True).mean()
df['BullPower'] = [i-j for (i,j) in zip(df['High'],df['EMA'])]
df['BearPower'] = [i-j for (i,j) in zip(df['EMA'],df['Low'])]


# In[4]:


df.head(20)


# In[5]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')

ax2 = plt.subplot(2, 1, 2)
ax2.bar(df.index, df['BullPower'], label='Bull Power', color='green')
ax2.bar(df.index, df['BearPower'], label='Bear Power', color='red')
ax2.grid()
ax2.set_ylabel('Elder Ray Index')
ax2.set_xlabel('Date')
ax2.legend(loc='best')


# ## Candlestick with Elder Ray Index

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

ax2 = plt.subplot(2, 1, 2)
ax2.bar(df.index, df['BullPower'], label='Bull Power', color='green')
ax2.bar(df.index, df['BearPower'], label='Bear Power', color='red')
ax2.grid()
ax2.set_ylabel('Elder Ray Index')
ax2.set_xlabel('Date')
ax2.legend(loc='best')

