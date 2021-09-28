#!/usr/bin/env python
# coding: utf-8

# # Straight Line of Stock Histocial Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
dataset = yf.download(symbol,start,end)['Adj Close']

# View Columns
dataset.head()


# In[3]:


df = dataset.reset_index()


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


max_p = df['Adj Close'].max()
min_p = df['Adj Close'].min()
avg_p = df['Adj Close'].mean()


# In[16]:


data = df.drop(['Date'], axis=1)
data


# In[17]:


data = data.reset_index()


# In[18]:


data.as_matrix()


# In[8]:


from numpy import ones,vstack
from numpy.linalg import lstsq


# In[19]:


points = data.as_matrix()


# In[20]:


x_coords, y_coords = zip(*points)
A = vstack([x_coords,ones(len(x_coords))]).T
m, c = lstsq(A, y_coords)[0]


# In[21]:


print("Line Equation is y = {m}x + {c}".format(m=m,c=c))


# In[25]:


equation_of_line = print("y = {m}x + {c}".format(m=m,c=c))


# In[28]:


plt.figure(figsize=(16,8))
plt.plot(dataset)
plt.title('Line of Equation', equation_of_line)
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='best')
plt.grid()
plt.show()

