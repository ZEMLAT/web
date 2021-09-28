#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt
import pandas as pd
import statistics as st

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

# For reading stock data from yahoo
import yfinance as yf
yf.pdr_override()


# In[2]:


start = '2020-01-01'
end = '2020-12-31'

symbol = 'AMD'


# In[3]:


df = yf.download(symbol, start, end)


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


monthly = df.copy()


# In[7]:


monthly.set_index(monthly.index, inplace=True)
monthly.index = pd.to_datetime(monthly.index)
monthly = monthly.resample('M').mean()


# In[8]:


data = monthly['Adj Close']
data


# In[9]:


month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Jul', 'Aug', 'Sep', 'Oct', 'Sep', 'Nov', 'Dec']

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# In[10]:


fig, ax = plt.subplots(nrows=1, ncols=1)

ax.set_title("Measures of Center")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.scatter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], data)

ax.plot([st.mean(data)], [st.mean(data)], color='r', marker="o", markersize=15)
ax.plot([st.median(data)], [st.median(data)], color='g', marker="o", markersize=15)
#ax.plot([st.mode(data)], [st.mode(data)], color='k', marker="o", markersize=15)

plt.annotate("Mean", (st.mean(data), st.mean(data)+0.3), color='r')
plt.annotate("Median", (st.median(data), st.median(data)+0.3), color='g')
#plt.annotate("Mode", (st.mode(data), st.mode(data)+0.3), color='k')
plt.show()

