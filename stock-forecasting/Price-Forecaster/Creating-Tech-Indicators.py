#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries and Data

# In[4]:


import pandas as pd
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import ta


# In[2]:


df = pd.read_csv('BTC-USD.csv')

# Viewing the DF
df


# ### Datetime Conversion

# In[3]:


# Datetime conversion
df['Date'] = pd.to_datetime(df.Date)

# Setting the index
df.set_index('Date', inplace=True)


# ## Creating the Indicators

# In[5]:


# Adding all the indicators
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)


# In[6]:


# Dropping everything else besides 'Close' and the Indicators
df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis= 1, inplace=True)
df


# ## Exporting the Indicator DF

# In[8]:


with open("df_indicators.pkl", 'wb') as fp:
    pickle.dump(df, fp)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




