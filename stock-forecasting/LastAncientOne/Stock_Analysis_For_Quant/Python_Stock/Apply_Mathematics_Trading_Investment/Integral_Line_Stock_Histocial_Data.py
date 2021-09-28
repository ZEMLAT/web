#!/usr/bin/env python
# coding: utf-8

# # Intergal using Line Equation from Stock Histocial Data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy import *

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


# In[7]:


data = df.drop(['Date'], axis=1)
data


# In[8]:


data = data.reset_index()


# In[9]:


data.values


# In[10]:


from numpy import ones,vstack
from numpy.linalg import lstsq


# In[11]:


points = data.values


# In[12]:


x_coords, y_coords = zip(*points)
A = vstack([x_coords,ones(len(x_coords))]).T
m, c = lstsq(A, y_coords)[0]


# In[13]:


print("Line Equation is y = {m}x + {c}".format(m=m,c=c))


# In[14]:


equation_of_line = print("y = {m}x + {c}".format(m=m,c=c))


# In[15]:


equation = print("{m}*x + {c}".format(m=m,c=c))


# In[16]:


x = Symbol('x')


# In[17]:


integrate(0.021718614923358828*x+9.372574584656501, x)


# In[18]:


integrate(0.0108593074616794*x**2 + 9.3725745846565 * x, x)


# # Univariate roots and fixed points

# In[19]:


def f(x):
    return 0.00361976915389313*x**3 + 4.68628729232825 * x**2


# In[20]:


x = df['Adj Close']


# In[21]:


plt.axhline((f(x)).mean(), c='red')
plt.plot(x, f(x))

