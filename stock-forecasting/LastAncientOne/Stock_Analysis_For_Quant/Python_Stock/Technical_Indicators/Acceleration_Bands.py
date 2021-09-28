#!/usr/bin/env python
# coding: utf-8

# # Acceleration Bands (ABANDS)

# https://www.tradingtechnologies.com/xtrader-help/x-study/technical-indicator-definitions/acceleration-bands-abands/

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
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


n = 7
UBB = df['High'] * ( 1 + 4 * (df['High'] - df['Low']) / (df['High'] + df['Low']))
df['Upper_Band'] = UBB.rolling(n, center=False).mean()
df['Middle_Band'] = df['Adj Close'].rolling(n).mean()
LBB = df['Low'] * ( 1 - 4 * (df['High'] - df['Low']) / (df['High'] + df['Low']))
df['Lower_Band'] = LBB.rolling(n, center=False).mean()


# In[4]:


df.head(20)


# In[5]:


plt.figure(figsize=(14,10))
plt.plot(df['Adj Close'])
plt.plot(df['Upper_Band'])
plt.plot(df['Middle_Band'])
plt.plot(df['Lower_Band'])
plt.ylabel('Price')
plt.xlabel('Date')
plt.title('Stock Closing Price of ' + str(n) + '-Day Acceleration Bands')
plt.legend(loc='best')


# In[6]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = pd.to_datetime(dfc['Date'])
dfc['Date'] = dfc['Date'].apply(mdates.date2num)
dfc.head()


# In[7]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(18,14))
ax1 = plt.subplot(2, 1, 1)
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df['Upper_Band'], label='Upper Band')
ax1.plot(df['Middle_Band'], label='Middle Band')
ax1.plot(df['Lower_Band'], label='Lower Band')
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

