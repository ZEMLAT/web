#!/usr/bin/env python
# coding: utf-8

# # Technical Indicators
# Adding technical indicator values

# ### Importing Libraries and Data

# In[8]:


import pandas as pd
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[9]:


df = pd.read_csv('BTC-USD.csv')

# Viewing the DF
df


# ### Datetime Conversion

# In[10]:


# Datetime conversion
df['Date'] = pd.to_datetime(df.Date)

# Setting the index
df.set_index('Date', inplace=True)


# ## Charting Values

# In[11]:


# Viewing the Chart of Closing Values
df['Close'].plot(figsize=(15,7), title='BTC Closing Prices');


# ## Creating the Technical Indicators

# In[12]:


def SMA(df, periods=50):
    """
    Calculating the Simple Moving Average for the past n days
    
    **Values must be descending**
    """
    lst = []
        
    for i in range(len(df)):
        if i < periods:
            
            # Appending NaNs for instances unable to look back on
            lst.append(np.nan)
            
        else:
            # Calculating the SMA
            lst.append(round(np.mean(df[i:periods+i]), 2))
        
    return lst


# In[13]:


def Stoch(closes, lows, highs, periods=14, d_periods=3):
    """
    Calculating the Stochastic Oscillator for the past n days
    
    **Values must be descending**
    """
    k_lst = []
    
    d_lst = []
    
    for i in range(len(closes)):
        if i < periods:
            
            # Appending NaNs for instances unable to look back on
            k_lst.append(np.nan)
            
            d_lst.append(np.nan)
            
        else:
            
            # Calculating the Stochastic Oscillator
            
            # Calculating the %K line
            highest = max(highs[i:periods+i])
            lowest = min(lows[i:periods+i])
            
            k = ((closes[i] - lowest) / (highest - lowest)) * 100
            
            k_lst.append(round(k, 2))
            
            # Calculating the %D line
            if len(k_lst) < d_periods:
                d_lst.append(np.nan)
            else:
                d_lst.append(round(np.mean(k_lst[-d_periods-1:-1])))
    
    return k_lst, d_lst
    


# In[14]:


def RSI(df, periods=14):
    """
    Calculates the Relative Strength Index
    
    **Values must be descending**
    """
    
    df = df.diff()
    
    lst = []
    
    for i in range(len(df)):
        if i < periods:
            
            # Appending NaNs for instances unable to look back on
            lst.append(np.nan)
            
        else:
            
            # Calculating the Relative Strength Index          
            avg_gain = (sum([x for x in df[i:periods+i] if x >= 0]) / periods)
            avg_loss = (sum([abs(x) for x in df[i:periods+i] if x <= 0], .00001) / periods)

            rs = avg_gain / avg_loss

            rsi = 100 - (100 / (1 + rs))

            lst.append(round(rsi, 2))

            
    return lst


# ## Creating New Values and Plotting
# Based on the Indicators

# #### RSI Indicator

# In[15]:


df['RSI'] = RSI(df.Close)

# Plotting
df['RSI'][-30:].plot(figsize=(15,6), title='BTC RSI');


# #### Stochastic Oscillator

# In[16]:


df['Stoch_k'], df['Stoch_d'] = Stoch(df.Close, df.Low, df.High)

# Plotting
df[-30:].plot(y=['Stoch_k', 'Stoch_d'], figsize=(15,6), title='BTC Stochastic Oscillator');


# #### Simple Moving Average

# In[17]:


df['SMA'] = SMA(df.Close, 6)

# Plotting
df['SMA'][-30:].plot(figsize=(15,6), title='BTC Simple Moving Average');


# In[18]:


df.tail(10)


# ## Using a Technical Analysis Python Library

# In[19]:


# Importing Library
import ta


# In[20]:


# TA's RSI
df['ta_rsi'] = ta.momentum.rsi(df.Close)

# TA's Stochastic Oscillator
df['ta_stoch_k'] = ta.momentum.stoch(df.High, df.Low, df.Close)
df['ta_stoch_d'] = ta.momentum.stoch_signal(df.High, df.Low, df.Close)


# In[21]:


# Showing our own custom indicators plus the libraries
df.drop(df.columns[:6], axis=1).tail(10)


# ### Using only the TA library's Indicators
# Dropping our own custom indicators

# In[25]:


df.drop(df.columns[6:], axis=1, inplace=True)


# In[40]:


# Adding all the indicators
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)


# In[42]:


# Dropping everything else besides 'Close' and the Indicators
df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis= 1, inplace=True)
df


# In[ ]:


df

