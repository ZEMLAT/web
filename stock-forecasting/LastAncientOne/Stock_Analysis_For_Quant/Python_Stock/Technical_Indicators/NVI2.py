#!/usr/bin/env python
# coding: utf-8

# # Negative Volume Index (NVI)

# https://www.investopedia.com/terms/n/nvi.asp

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
vol_decrease = (df['Volume'].shift(1) > df['Volume'])

nvi = pd.Series(data=np.nan, index=df['Adj Close'].index, dtype='float64')

nvi.iloc[0] = 1000
for i in range(1,len(nvi)):
    if vol_decrease.iloc[i]:
        nvi.iloc[i] = nvi.iloc[i - 1] * (1.0 + returns.iloc[i])
    else:
        nvi.iloc[i] = nvi.iloc[i - 1]

nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(1000)

df['NVI'] = pd.Series(nvi)


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
ax2.plot(df['NVI'], label='Negative Volume Index', color='red')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('Negative Volume Index')
ax2.set_xlabel('Date')


# ## Candlestick with Negative Volume Index

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
ax2.plot(df['NVI'], label='Negative Volume Index', color='red')
ax2.grid()
ax2.legend(loc='best')
ax2.set_ylabel('Negative Volume Index')
ax2.set_xlabel('Date')

