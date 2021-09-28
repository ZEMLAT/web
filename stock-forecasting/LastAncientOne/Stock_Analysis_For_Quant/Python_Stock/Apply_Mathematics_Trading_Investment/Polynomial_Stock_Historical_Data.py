#!/usr/bin/env python
# coding: utf-8

# # Polynomial Stock of Historical Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.chebyshev import chebfit,chebval
import pandas as pd

import warnings
warnings.filterwarnings("ignore") 

# yfinance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AMD'
start = '2017-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbol,start,end)

# View Columns
dataset.head()


# In[3]:


dataset.tail()


# In[4]:


y = np.array(dataset['Adj Close'])


# In[5]:


len(y)


# In[6]:


x = np.arange(len(y))
c = chebfit(x, y, 30)


# In[7]:


p = []
for i in np.arange(len(y)):
    p.append(chebval(i, c))


# In[8]:


df = pd.DataFrame(data={'x': x, 'y': y, 'p': p})
df['diff'] = df['y'] - df['p']


# In[9]:


sns.set(rc={'figure.figsize':(14,10)})
sns.pointplot(x = 'x', y = 'y', data=df, color='green')
sns.pointplot(x = 'x', y = 'p', data=df, color='red')
sns.pointplot(x = 'x', y = 'diff', data=df, color='blue')

