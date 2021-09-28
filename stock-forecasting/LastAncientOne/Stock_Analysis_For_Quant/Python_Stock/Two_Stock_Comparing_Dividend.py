#!/usr/bin/env python
# coding: utf-8

# Simple Stock Analysis

# Stock analysis is the evaluation or examination of the stock market. There are many trading tools to use to analysis stocks such as fundamental and technical analysis. Fundamental analysis is more focused on data from the financial statements, economic reports, and company assets. Technical analysis is based on the study of the past of historical price to predict the future price movement. However, this tutorial is not to get rich quick. Therefore, do not use your money to trade based on this stock analysis. Please do not use this method to invest with your money and I am not responsible for you loss.   
# 
# Simple stock is a basic stock analysis tutorial. There are 7 parts in this tutorial. 
# 1. Import Libraries
# 2. Get data from Yahoo
# 3. Analysis Data
# 4. Understand the Data based on Statistics
# 5. Calculate Prices
# 6. Plot Charts
# 7. Calculate Holding Period Return

# I. Import Libraries

# In[1]:


# Libaries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

import fix_yahoo_finance as yf
yf.pdr_override()


# II. Get Data from Yahoo!
# 
# This section we will pull the data from the website in Yahoo. We will be using the company of Apple and the symbol is 'AAPL'. Also, we will have a starting date and ending date.

# In[2]:


stock = 'AAPL'
start = '2015-01-01' 
end = '2017-01-01'
df = yf.download(stock, start, end)


# III. Analysis Data

# In[3]:


df.head() # the first 5 rows


# In[4]:


df.tail() # the last 5 rows


# In[5]:


df.shape # (rows, columns)


# In[6]:


df.columns # Shows names of columns


# In[7]:


df.dtypes # Shows data types


# In[8]:


df.info() # Shows information about DataFrame


# In[9]:


df.describe() # Shows summary statistics based on stock data 


# IV. Understand the Data based on Statistics

# We will be using "Adj. Closing" price to find the minimum, maximum, average and standard deviation prices. The reason we are using "Adj. Closing" because is mostly use for historical returns. Also, the Adjusting Prices is change where the stock was accounts for the dividend and splits. However, the "Closing" price was not including with dividend and splits.

# In[10]:


# Use only Adj. Closing
# Find the minimum
df['Adj Close'].min()


# In[11]:


# Find the maximum
df['Adj Close'].max()


# In[12]:


# Find the average
df['Adj Close'].mean()


# In[13]:


# Find the standard deviation
df['Adj Close'].std()


# V. Calculate the Prices

# This section, we will be calculating the daily returns, log returns, and other technical indicators such as RSI(Relative Strength Index), MA(Moving Average), SMA(Simple Moving Averga), EMA(Exponential Moving Average), and VWAP(Voume Weighted Average Price). Also, we will calculate drawdowns.

# In[14]:


# Daily Returns
# Formula: (Today Price / Yesterday Price) - 1 
df['Daily_Returns'] = df['Adj Close'].shift(1) / df['Adj Close']  - 1
df['Daily_Returns'].head()


# In[15]:


# Another way of calculating Daily Returns in simple way
DR = df['Adj Close'].pct_change(1) # 1 is for "One Day" in the past
DR.head()


# In[16]:


# Log Returns
# Formula: log(Today Price/ Yesterday Price)
df['Log_Returns'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))


# In this part of this section, we will be using the library of technical analysis. This packages has many different types of technical indicators. However, it does not have every single technical indicators.
# We do not need to do calculation since the library has done it for us.
# https://mrjbq7.github.io/ta-lib/doc_index.html

# In[17]:


import talib as ta

# Creating Indicators
n=30 # number of periods

# RSI(Relative Strength Index)
# RSI is technical analysis indicator
# https://www.investopedia.com/terms/r/rsi.asp
df['RSI']=ta.RSI(np.array(df['Adj Close'].shift(1)), timeperiod=n)

# MA(Moving Average)
# https://www.investopedia.com/terms/m/movingaverage.asp
df['MA']=ta.MA(np.array(df['Adj Close'].shift(1)), timeperiod=n, matype=0)

# SMA(Simple Moving Average)
# https://www.investopedia.com/terms/s/sma.asp
df['SMA']=ta.SMA(np.array(df['Adj Close'].shift(1)))

# EMA(Exponential Moving Average)
# https://www.investopedia.com/terms/e/ema.asp
df['EMA']=ta.EMA(np.array(df['Adj Close'].shift(1)), timeperiod=n)


# In[18]:


# Volume Weighted Average Price - VWAP
# http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vwap_intraday
df['VWAP'] = round(np.cumsum(df['Volume']*(df['High']+df['Low'])/2) / np.cumsum(df['Volume']), 2)
df.head()


# In[19]:


# Drawdown
# Drawdown shows the decline price since the stock began trading
# https://www.investopedia.com/terms/d/drawdown.asp
# There are 252 trading day in a year
window = 252

# Calculate the maximum drawdown
# Use the min_period of 1 (1 is the least valid observations) for the first 252 day in the data
Maximum_Drawdown = df['Adj Close'].rolling(window, min_periods=1).max()
Daily_Drawdown = df['Adj Close']/Maximum_Drawdown - 1.0

# Calculate the negative drawdown
Negative_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()


# VI. Plot Charts

# In[20]:


# Plot Simple Line Chart
# Plot Adj Close

plt.figure(figsize=(16,10))
df['Adj Close'].plot(grid=True)
plt.title("Stock Adj Close Price", fontsize=18, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price",fontsize=12)
plt.show()


# In[21]:


# Plot High, Low, Adj Close
df[['High', 'Low', 'Adj Close']].plot(figsize=(16,10), grid=True)
plt.title("Stock Adj Close Price", fontsize=18, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.show()


# In[22]:


# Plot Daily Returns
df['Daily_Returns'].plot(figsize=(12,6))
plt.title("Daily Returns",fontsize=18, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.show()


# In[23]:


# Plot Log Returns
df['Log_Returns'].plot(figsize=(12,6))
plt.title("Log Returns", fontsize=18, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.show()


# In[24]:


# Histogram of Daily Returns
# Histogram is distribution of numerical data and has a rectangle whose area is prportional to the frequency of a variable. 
plt.figure(figsize=(16,10))
plt.hist(df['Daily_Returns'].dropna(), bins=100, label='Daily Returns data') # Drop NaN
plt.title("Histogram of Daily Returns", fontsize=18, fontweight='bold')
plt.axvline(df['Daily_Returns'].mean(), color='r', linestyle='dashed', linewidth=2) # Shows the average line
plt.xlabel("Date", fontsize=12)
plt.ylabel("Daily Returns", fontsize=12)
plt.show()


# In[25]:


# Plot Drawdown
plt.figure(figsize=(16,10))
Daily_Drawdown.plot()
Negative_Drawdown.plot(color='r',grid=True) 
plt.title("Maximum Drawdown", fontsize=18, fontweight='bold')
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.show()


# VII. Holding Period Return(HPR)
# 
# Holding period return (HPR) is the rate of return on an individual stocks or portfolio over the whole period during the time it was held and it a measurement of investment performance. 

# In[26]:


# https://www.investopedia.com/exam-guide/series-65/quantitative-methods/holding-period-return.asp
# Formula: (Ending Value of Investment + Dividend - Beginning Value of Investment) / Beginning Value of Investment
# To get dividend in Yahoo!
DIV = yf.download(stock, start, end, actions=True)['Dividends']


# In[27]:


# See how much dividends and splits was given during the time period
DIV


# In[28]:


# Add all the dividend
Total_Dividend = DIV.sum()
Total_Dividend


# In[29]:


# You invest beginning 2015 and sold it end of 2017
HPR = (df['Adj Close'][502] + Total_Dividend - df['Adj Close'][0]) / df['Adj Close'][0]
HPR


# In[30]:


# You can use round for 4 decimal points
print('Holding Period Return: ', str(round(HPR,4)*100)+"%")


# We going to pick another stocks that is Microsoft and we will compare it to Apple.

# In[31]:


MSFT =  yf.download('MSFT', start, end)['Adj Close'] # Use Adj Close only
MSFT_DIV = yf.download('MSFT', start, end, actions=True)['Dividends']


# In[32]:


MSFT.head() # Shows only Date and Adj Close


# In[33]:


MSFT_DIV # Shows how much dividend was given


# In[34]:


MSFT_Dividend = MSFT_DIV.sum()
MSFT_Dividend


# In[35]:


# You invest beginning 2015 and sold it end of 2017
MSFT_HPR = (MSFT[502] + Total_Dividend - MSFT[0]) / MSFT[0]
MSFT_HPR


# In[36]:


# You can use round for 4 decimal points
print('Apple Holding Period Return: ', str(round(HPR,4)*100)+"%")
print('Microsoft Holding Period Return: ', str(round(MSFT_HPR,4)*100)+"%")


# In the conclusion, we use 2 stocks to compare holding period return. Therefore, Microsoft had higher holding period return than Apple. Therefore, I would invest in Microsoft based on the stock analysis. However, if you comparing 2 stocks or 2 portfolio. You would pick the ones with the highest rate of return. 
