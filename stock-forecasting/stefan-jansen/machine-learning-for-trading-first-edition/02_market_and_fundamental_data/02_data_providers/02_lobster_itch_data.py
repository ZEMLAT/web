#!/usr/bin/env python
# coding: utf-8

# # Tick Data from LOBSTER

# LOBSTER (Limit Order Book System - The Efficient Reconstructor) is an [online](https://lobsterdata.com/info/WhatIsLOBSTER.php) limit order book data tool to provide easy-to-use, high-quality limit order book data.
# 
# Since 2013 LOBSTER acts as a data provider for the academic community, giving access to reconstructed limit order book data for the entire universe of NASDAQ traded stocks. 
# 
# More recently, it has started to make the data available on a commercial basis.

# ## Imports

# In[7]:


from pathlib import Path
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Orderbook Data

# We will illustrate the functionality using a free sample.

# Obtain data here: https://lobsterdata.com/info/DataSamples.php; [this](https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip) is the link to the 10-level file

# The code assumes the file has been extracted into a `data` subfolder of the current directory.

# In[2]:


list(chain(*[('Ask Price {0},Ask Size {0},Bid Price {0},Bid Size {0}'.format(i)).split(',') for i in range(10)]))


# In[4]:


price = list(chain(*[('Ask Price {0},Bid Price {0}'.format(i)).split(',') for i in range(10)]))
size = list(chain(*[('Ask Size {0},Bid Size {0}'.format(i)).split(',') for i in range(10)]))
cols = list(chain(*zip(price, size)))


# In[8]:


path = Path('data')
order_data = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
orders = pd.read_csv(path / order_data, header=None, names=cols)


# In[9]:


orders.info()


# In[9]:


orders.head()


# ### Message Data

# Message Type Codes:
# 
#     1: Submission of a new limit order
#     2: Cancellation (Partial deletion 
#        of a limit order)
#     3: Deletion (Total deletion of a limit order)
#     4: Execution of a visible limit order			   	 
#     5: Execution of a hidden limit order
#     7: Trading halt indicator 				   
#        (Detailed information below)

# In[10]:


types = {1: 'submission',
         2: 'cancellation',
         3: 'deletion',
         4: 'execution_visible',
         5: 'execution_hidden',
         7: 'trading_halt'}


# In[11]:


trading_date = '2012-06-21'
levels = 10


# In[12]:


message_data = 'AMZN_{}_34200000_57600000_message_{}.csv'.format(trading_date, levels)
messages = pd.read_csv(path / message_data, header=None, names=['time', 'type', 'order_id', 'size', 'price', 'direction'])
messages.info()


# In[13]:


messages.head()


# In[14]:


messages.type.map(types).value_counts()


# In[15]:


messages.time = pd.to_timedelta(messages.time, unit='s')
messages['trading_date'] = pd.to_datetime(trading_date)
messages.time = messages.trading_date.add(messages.time)
messages.drop('trading_date', axis=1, inplace=True)
messages.head()


# In[16]:


data = pd.concat([messages, orders], axis=1)
data.info()


# In[17]:


ex = data[data.type.isin([4, 5])]


# In[18]:


ex.head()


# In[19]:


cmaps = {'Bid': 'Blues','Ask': 'Reds'}


# In[20]:


fig, ax=plt.subplots(figsize=(14, 8))
time = ex['time'].dt.to_pydatetime()
for i in range(10):
    for t in ['Bid', 'Ask']:
        y, c = ex['{} Price {}'.format(t, i)], ex['{} Size {}'.format(t, i)]
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0));


# In[21]:


fig, ax=plt.subplots(figsize=(14, 8))
time = data['time'].dt.to_pydatetime()
for i in range(10):
    for t in ['Bid', 'Ask']:
        y, c = data['{} Price {}'.format(t, i)], data['{} Size {}'.format(t, i)]
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0));


# In[ ]:




