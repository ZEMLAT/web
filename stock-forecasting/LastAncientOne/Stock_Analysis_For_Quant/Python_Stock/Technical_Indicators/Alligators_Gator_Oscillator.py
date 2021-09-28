#!/usr/bin/env python
# coding: utf-8

# # Gator Oscillator

# https://www.metatrader5.com/en/terminal/help/indicators/bw_indicators/go

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


def SMMA(price, n, m=3):
    SMMA = np.array([np.nan] * len(price))
    SMMA[n - 2] = price[:n - 1].mean()
    for i in range(n - 1, len(price)):
        SMMA[i] = (SMMA [i - 1] * (n - 2) + 2 * price[i]) / n
    return SMMA


# In[4]:


medianprice = (df['High']/2) + (df['Low']/2)
df['Jaw'] = SMMA(medianprice,13,8)
df['Teeth'] = SMMA(medianprice,8 ,5)
df['Lips']  = SMMA(medianprice,5 ,3) 


# In[5]:


df


# https://mahifx.com/mfxtrade/indicators/gator-oscillator
# 
# Top bars of histogram (Above zero) = Absolute value (Jaw – Teeth)
# 
# Bottom bars of histogram (Below zero) = - {Absolute value of (Teeth – Lips)}

# In[6]:


df['Top_Bars'] = abs(df['Jaw'] - df['Teeth'])
df['Bottom_Bars'] = -(abs(df['Teeth'] - df['Lips']))


# In[7]:


df


# In[8]:


fig = plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(df['Adj Close'],lw=1)
ax1.plot(df['Jaw'],color='blue')
ax1.plot(df['Teeth'],color='red')
ax1.plot(df['Lips'],color='green')
ax1.set_title(symbol + ' Close Price')
ax1.set_ylabel('Stock price')
ax1.set_xlabel('Date')
ax1.grid(True)
ax1.legend(loc='best')

ax2 = fig.add_subplot(2, 1, 2)
df['Positive_T'] = df.Top_Bars > df.Top_Bars.shift(1)
df['Positive_B'] = df.Bottom_Bars > df.Bottom_Bars.shift(1)
ax2.bar(df.index, df['Top_Bars'], color=df.Positive_T.map({True: 'g', False: 'r'}), label='Top')
ax2.bar(df.index, df['Bottom_Bars'], color=df.Positive_B.map({True: 'g', False: 'r'}), label='Bottom')
#ax2.bar(df.index, df['Top_Bars'],label='Top')
#ax2.bar(df.index, df['Bottom_Bars'],label='Bottom')
ax2.legend(loc=2,prop={'size':8})
ax2.grid(True)


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
ax1 = plt.subplot(2, 1, 1)
candlestick_ohlc(ax1,dfc.values, width=0.5, colorup='g', colordown='r', alpha=1.0)
ax1.plot(df['Jaw'],color='blue')
ax1.plot(df['Teeth'],color='red')
ax1.plot(df['Lips'],color='green')
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
ax1.legend(loc='best')

ax2 = fig.add_subplot(2, 1, 2)
df['Positive_T'] = df.Top_Bars > df.Top_Bars.shift(1)
df['Positive_B'] = df.Bottom_Bars > df.Bottom_Bars.shift(1)
ax2.bar(df.index, df['Top_Bars'], color=df.Positive_T.map({True: 'g', False: 'r'}), label='Top')
ax2.bar(df.index, df['Bottom_Bars'], color=df.Positive_B.map({True: 'g', False: 'r'}), label='Bottom')
ax2.legend(loc=2,prop={'size':8})
ax2.grid(True)

