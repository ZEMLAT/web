#!/usr/bin/env python
# coding: utf-8

# # Anscombe's Quartet Stock Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore") 

# yfinance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AMD'
start = '2019-12-01'
end = '2020-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


df = df.astype('float64')
df.head()


# In[4]:


df = df[['Open', 'High', 'Low', 'Adj Close']]
df.head()


# In[5]:


df.shape


# In[6]:


for i in df.values:
  print(np.array(i))


# In[7]:


quartets = np.array([df['Open'], df['High'], df['Low'], df['Adj Close']])


# In[8]:


quartets


# In[9]:


quartets[0]


# In[10]:


quartets.shape[0]


# In[11]:


for quartet in range(quartets.shape[0]):
    x = np.array(quartet)
    print(x)


# In[12]:


for names in df.columns:
    print(names)


# In[13]:


for name in df.columns: print(name)


# In[14]:


for name in df.columns: 
    print("Next")
    print("Adj Close vs ", name)


# In[15]:


roman = ['I', 'II', 'III', 'IV']


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(16,12))
fig.suptitle("Anscombe's Quartets", fontsize=14)
axes = fig.subplots(2, 2, sharex= True, sharey = True)
n = len(df.index)

for name, quartet in zip(df.columns, range(quartets.shape[0])):
    x = quartets[quartet]
    y = np.array(df['Adj Close'])
    coef = np.polyfit(x, y, 1)
    reg_line = np.poly1d(coef)
    ax = axes[quartet // 2, quartet % 2]
    ax.plot(x, y, 'ro', x, reg_line(x), '--k')
    ax.set_title(roman[quartet])
   
    print("Quartet:", roman[quartet])
    print("Adj Close vs", name)
    print("Mean X:", x.mean())
    print("Variance X:", x.var())
    print("Mean Y:", y.mean())
    print("Variance Y:", y.var())
    print("Pearson's correlation coef.:", round(np.corrcoef(x, y)[0][1], 2))
    print('-'*40)

plt.show()

    

