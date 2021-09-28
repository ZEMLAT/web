#!/usr/bin/env python
# coding: utf-8

# # Positive Volume Index (PVI)

# https://www.investopedia.com/terms/p/pvi.asp

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


returns = df['Adj Close'].pct_change()
vol_increase = (df['Volume'].shift(1) < df['Volume'])

pvi = pd.Series(data=np.nan, index=df['Adj Close'].index, dtype='float64')

pvi.iloc[0] = 1000
for i in range(1,len(pvi)):
    if vol_increase.iloc[i]:
        pvi.iloc[i] = pvi.iloc[i - 1] * (1.0 + returns.iloc[i])
    else:
        pvi.iloc[i] = pvi.iloc[i - 1]

pvi = pvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

df['PVI'] = pd.Series(pvi)


# In[4]:


df.head()


# In[5]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(df['Adj Close'])
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')
ax1.legend(loc='best')

ax2 = plt.subplot(2, 1, 2)
ax2.plot(df['PVI'], label='Positive Volume Index', color='green')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('Positive Volume Index')
ax2.set_xlabel('Date')


# ## Candlestick with Postive Volume Index

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
ax2.plot(df['PVI'], label='Positive Volume Index', color='green')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('Positive Volume Index')
ax2.set_xlabel('Date')

