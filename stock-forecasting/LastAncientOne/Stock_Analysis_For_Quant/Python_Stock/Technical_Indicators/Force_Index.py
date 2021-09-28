#!/usr/bin/env python
# coding: utf-8

# # Force Index

# https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

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
start = '2016-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


n = 13
df['FI_1'] = (df['Adj Close'] - df['Adj Close'].shift())*df['Volume']
df['FI_13'] = df['FI_1'].ewm(ignore_na=False,span=n,min_periods=n,adjust=True).mean()


# In[4]:


df.head(20)


# In[5]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df['Adj Close'])
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')

ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['FI_1'], label='1-Period Force Index', color='black')
ax2.axhline(y=0, color='blue', linestyle='--')
ax2.grid()
ax2.set_ylabel('1-Period Force Index')
ax2.legend(loc='best')

ax3 = plt.subplot(3, 1, 3)
ax3.plot(df['FI_13'], label='13-Period Force Index', color='black')
ax3.axhline(y=0, color='blue', linestyle='--')
ax3.fill_between(df.index, df['FI_13'], where=df['FI_13']>0, color='green')
ax3.fill_between(df.index, df['FI_13'], where=df['FI_13']<0, color='red')
ax3.grid()
ax3.set_ylabel('13-Period Force Index')
ax3.set_xlabel('Date')
ax3.legend(loc='best')


# ## Candlestick with Force Index

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
ax1 = plt.subplot(3, 1, 1)
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

ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['FI_1'], label='1-Period Force Index', color='black')
ax2.axhline(y=0, color='blue', linestyle='--')
ax2.grid()
ax2.set_ylabel('1-Period Force Index')
ax2.legend(loc='best')

ax3 = plt.subplot(3, 1, 3)
ax3.plot(df['FI_13'], label='13-Period Force Index', color='black')
ax3.axhline(y=0, color='blue', linestyle='--')
ax3.fill_between(df.index, df['FI_13'], where=df['FI_13']>0, color='green')
ax3.fill_between(df.index, df['FI_13'], where=df['FI_13']<0, color='red')
ax3.grid()
ax3.set_ylabel('13-Period Force Index')
ax3.set_xlabel('Date')
ax3.legend(loc='best')

