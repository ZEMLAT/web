#!/usr/bin/env python
# coding: utf-8

# # Coppock Curve

# https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:coppock_curve

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
start = '2017-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


n = [11,14]
for nd in n:
  df['ROC_'+str(nd)] = ((df['Adj Close'] - df['Adj Close'].shift(nd))/df['Adj Close'].shift(nd)) * 100


# In[4]:


df['COPP'] = (df['ROC_14']+df['ROC_11']).ewm(ignore_na=False,span=10,min_periods=10,adjust=True).mean()


# In[5]:


fig = plt.figure(figsize=(14,10))
ax1 = plt.subplot(4, 1, 1)
ax1.plot(df['Adj Close'])
ax1.set_title('Stock '+ symbol +' Closing Price')
ax1.set_ylabel('Price')

ax2 = plt.subplot(4, 1, 2)
ax2.plot(df['COPP'], label='Coppock Curve', color='black')
ax2.axhline(y=0, color='red')
ax2.grid()
ax2.set_ylabel('Coppock Curve')
ax2.legend(loc='best')

ax3 = plt.subplot(4, 1, 3)
ax3.plot(df['ROC_11'], label='ROC 11', color='black')
ax3.axhline(y=0, color='red')
ax3.grid()
ax3.set_ylabel('11 Month Rate-of-Change')
ax3.legend(loc='best')

ax4 = plt.subplot(4, 1, 4)
ax4.plot(df['ROC_14'], label='ROC 14', color='black')
ax4.axhline(y=0, color='red')
ax4.grid()
ax4.set_ylabel('14 Month Rate-of-Change')
ax4.legend(loc='best')


# ## Candlestick with Coppock Curve

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
ax1 = plt.subplot(4, 1, 1)
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

ax2 = plt.subplot(4, 1, 2)
ax2.plot(df['COPP'], label='Coppock Curve', color='black')
ax2.axhline(y=0, color='red')
ax2.grid()
ax2.set_ylabel('Coppock Curve')
ax2.legend(loc='best')

ax3 = plt.subplot(4, 1, 3)
ax3.plot(df['ROC_11'], label='ROC 11', color='black')
ax3.axhline(y=0, color='red')
ax3.grid()
ax3.set_ylabel('11 Month Rate-of-Change')
ax3.legend(loc='best')

ax4 = plt.subplot(4, 1, 4)
ax4.plot(df['ROC_14'], label='ROC 14', color='black')
ax4.axhline(y=0, color='red')
ax4.grid()
ax4.set_ylabel('14 Month Rate-of-Change')
ax4.legend(loc='best')

