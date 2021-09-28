#!/usr/bin/env python
# coding: utf-8

# # Rainbow Charts

# http://www.binarytribune.com/forex-trading-indicators/rainbow-charts

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


# R=red, O=orange, Y=yellow, G=green, B=blue, I = indigo, and V=violet
df['Red'] = df['Adj Close'].rolling(2).mean()
df['Orange'] = df['Red'].rolling(2).mean()
df['Yellow'] = df['Orange'].rolling(2).mean() 
df['Green'] = df['Yellow'].rolling(2).mean()
df['Blue'] = df['Green'].rolling(2).mean()
df['Indigo'] = df['Blue'].rolling(2).mean()
df['Violet'] = df['Indigo'].rolling(2).mean()
df = df.dropna()


# In[4]:


colors = ['k','r', 'orange', 'yellow', 'g', 'b', 'indigo', 'violet']
df[['Adj Close','Red','Orange','Yellow','Green','Blue','Indigo','Violet']].plot(colors=colors, figsize=(18,12))
plt.fill_between(df.index, df['Low'], df['High'], color='grey', alpha=0.4)
plt.plot(df['Low'], c='darkred', linestyle='--', drawstyle="steps")
plt.plot(df['High'], c='forestgreen', linestyle='--', drawstyle="steps")
plt.title('Rainbow Charts')
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# ## Candlestick with Rainbow

# In[5]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = mdates.date2num(dfc['Date'].astype(dt.date))
dfc.head()


# In[8]:


from mpl_finance import candlestick_ohlc

fig, ax1 = plt.subplots(figsize=(20,12))
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
#colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
#labels = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
for i in dfc[['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']]:
    ax1.plot(dfc['Date'], dfc[i], color=i, label=i)
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
ax1.legend(loc='best')

