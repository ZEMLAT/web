#!/usr/bin/env python
# coding: utf-8

# # Smoothed Moving Average (SMMA)

# https://www.metatrader5.com/en/terminal/help/indicators/trend_indicators/ma#smma

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


df.shape


# In[4]:


n = 10
SMMA = np.array([np.nan] * len(df['Adj Close']))
SMMA[n - 2] = df['Adj Close'][:n - 1].mean()
for i in range(n - 1, len(df['Adj Close'])):
    SMMA[i] = (SMMA [i - 1] * (n - 2) + 2 * df['Adj Close'][i]) / n


# In[5]:


SMMA


# In[6]:


x = SMMA.reshape(-1,1)
data = pd.DataFrame.from_records(x)
data


# In[7]:


df['SMMA'] = data.values
df.head(10)


# In[8]:


# Line Chart
plt.figure(figsize=(14,8))
plt.plot(df['Adj Close'])
plt.plot(df['SMMA'])
plt.title('Guppy Multiple Moving Averages of EMA')
plt.legend(loc='best')
plt.show()


# ## Candlestick with SMMA

# In[9]:


from matplotlib import dates as mdates
import datetime as dt

dfc = df.copy()
dfc['VolumePositive'] = dfc['Open'] < dfc['Adj Close']
#dfc = dfc.dropna()
dfc = dfc.reset_index()
dfc['Date'] = mdates.date2num(dfc['Date'].astype(dt.date))
dfc.head()


# In[10]:


from mpl_finance import candlestick_ohlc

fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(111)
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df['SMMA'], color='orange')
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax1.grid(True, which='both')
ax1.minorticks_on()
ax1v = ax1.twinx()
colors = dfc.VolumePositive.map({True: 'g', False: 'r'})
ax1v.bar(dfc.Date, dfc['Volume'], color=colors, alpha=0.4)
ax1v.axes.yaxis.set_ticklabels([])
ax1v.set_ylim(0, 3*df.Volume.max())
ax1.set_title('Stock '+ symbol +' Closing Price of SMMA')
ax1.set_ylabel('Price')

