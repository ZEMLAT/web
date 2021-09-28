#!/usr/bin/env python
# coding: utf-8

# # Average Directional Movement Rating (ADXR)

# https://www.fmlabs.com/reference/default.htm
# 
# https://www.linnsoft.com/techind/adxr-avg-directional-movement-rating
# 
# https://www.marketvolume.com/technicalanalysis/adxr.asp

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


import talib as ta


# In[4]:


df['ADX'] = ta.ADX(df['High'], df['Low'],df['Adj Close'], timeperiod=4)


# In[5]:


n = 7
df['ADXR'] = (df['ADX'][n] + df['ADX'][n+7:]) / 2
df


# In[6]:


# Line Chart
fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
#ax1.grid(True, which='both')
ax1.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax1.minorticks_on()
ax1.legend(loc='best')
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')


ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['ADXR'], '-', label='ADXR')
#ax2.text(s='Strong Trend', x=df['ADXR'].index[0], y=50, fontsize=14)
#ax2.text(s='Weak Trend', x=df['ADXR'].index[0], y=20, fontsize=14)
#ax2.axhline(y=50,color='r')
#ax2.axhline(y=20,color='r')
ax2.set_xlabel('Date')
ax2.legend(loc='best')


# ## Candlestick with ADX

# In[7]:


# Candlestick
dfc = df.copy()

from matplotlib import dates as mdates
import datetime as dt

dfc['ADX'] = ta.ADX(dfc['High'], dfc['Low'],dfc['Adj Close'], timeperiod=14)
dfc = dfc.dropna()
dfc.head()


# In[8]:


dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = pd.to_datetime(dfc['Date'])
dfc['Date'] = dfc['Date'].apply(mdates.date2num)
dfc.head()


# In[9]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax1.grid(True, which='both')
#ax1.grid(which='minor', linestyle='-', linewidth='0.5', color='black')
#ax1.grid(which='major', linestyle='-', linewidth='0.5', color='red')
ax1.minorticks_on()
#ax1.legend(loc='best')
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')


ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['ADXR'], '-', label='ADXR')
#ax2.text(s='Strong Trend', x=df['ADXR'].index[0], y=50, fontsize=14)
#ax2.text(s='Weak Trend', x=df['ADXR'].index[0], y=20, fontsize=14)
#ax2.axhline(y=50,color='r')
#ax2.axhline(y=20,color='r')
ax2.set_ylabel('Average Directional Movement Rating')
ax2.set_xlabel('Date')
ax2.legend(loc='best')

