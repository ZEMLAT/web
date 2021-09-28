#!/usr/bin/env python
# coding: utf-8

# # Stock Information

# In[1]:


# Libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import warnings
warnings.filterwarnings("ignore")

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[2]:


# Check versions of modules used 
print("numpy: {}".format(np.__version__))
print("pandas: {}".format(pd.__version__))
print("matplotlib: {}".format(matplotlib.__version__))
print("seaborn: {}".format(sns.__version__))
print("yahoo_finance: {}".format(yf.__version__))
print("python: {}".format(sys.version))


# In[3]:


stock = 'AMD'
start = '2015-01-01' 
end = '2018-01-01'
data = pdr.get_data_yahoo(stock, start, end)


# In[4]:


# Inspect the index 
data.index


# In[5]:


# Inspect the columns
data.columns


# In[6]:


# Type of data
type(data)


# In[7]:


data = data.reset_index() # Date has a column


# In[8]:


data.head() # First 5 rows


# In[9]:


data.tail() # Last 5 rows


# In[10]:


data.describe() # Statistics


# In[11]:


prices = data['Adj Close']
features = data.drop(['Date','Adj Close', 'Close'], axis = 1)


# In[12]:


features.tail()


# In[13]:


print("Stock dataset has {} data points with {} variables each.".format(*data.shape))


# In[14]:


# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Stock dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

