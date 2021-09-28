#!/usr/bin/env python
# coding: utf-8

# # Stochastic Oscillator

# https://www.investopedia.com/terms/s/stochasticoscillator.asp

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
start = '2018-09-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


n = 14
smin = df['Low'].rolling(n).min()
smax = df['High'].rolling(n).max()
df['stoch_k'] = 100 * (df['Adj Close'] - smin) / (smax - smin)
d_n = 3
df['stoch_d'] = df['stoch_k'].rolling(d_n).mean()


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.legend(loc='best')
ax1.tick_params(axis='x', rotation=45)

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['stoch_k'], label='Stoch %K')
ax2.plot(df['stoch_d'], label='Stoch %D')
ax2.legend(loc='best')
ax2.set_ylabel('Stochastic Oscillator')
ax2.set_xlabel('Date')
ax2.tick_params(axis='x', rotation=45)


# ## Candlestick with Stochastic Oscillator

# In[7]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = pd.to_datetime(dfc['Date'])
dfc['Date'] = dfc['Date'].apply(mdates.date2num)
dfc.head()


# In[8]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(16,10))
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
ax1.set_xlabel('Date')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['stoch_k'], label='Stoch %K')
ax2.plot(df['stoch_d'], label='Stoch %D')
ax2.legend(loc='best')
ax2.set_ylabel('Stochastic Oscillator')
ax2.set_xlabel('Date')
ax2.tick_params(axis='x', rotation=45)

