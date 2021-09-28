#!/usr/bin/env python
# coding: utf-8

# # Derivative Linear Equation Stock Data

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


# In[7]:


data = df.drop(['Date'], axis=1)
data


# In[8]:


data = data.reset_index()


# In[9]:


data.as_matrix()


# In[10]:


from numpy import ones,vstack
from numpy.linalg import lstsq


# In[11]:


points = data.as_matrix()


# In[12]:


x_coords, y_coords = zip(*points)
A = vstack([x_coords,ones(len(x_coords))]).T
m, c = lstsq(A, y_coords)[0]


# In[13]:


print("Line Equation is y = {m}x + {c}".format(m=m,c=c))


# In[14]:


equation_of_line = print("y = {m}x + {c}".format(m=m,c=c))


# In[15]:


plt.figure(figsize=(16,8))
plt.plot(dataset)
plt.title('Line of Equation', equation_of_line)
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[16]:


from sympy import *


# In[17]:


x = Symbol('x')


# In[18]:


y = 0.021718614923358824*x + 9.372574584656498


# In[19]:


yder = y.diff(x)
yder


# In[20]:


y =  0.021718614923358824*(df.index) + 9.372574584656498


# In[21]:


y


# In[22]:


pd.DataFrame(y, columns=['Forecast'])


# In[23]:


dataset


# In[24]:


forecast = pd.DataFrame(y, columns=['Forecast'])
forecast


# In[25]:


df = dataset.reset_index()


# In[26]:


df = df.join(forecast)


# In[27]:


df


# In[29]:


plt.figure(figsize=(16,8))
plt.plot(df.Date, df['Adj Close'])
plt.plot(df.Date, df['Forecast'])
plt.title('Line of Equation', equation_of_line)
plt.xlabel('Date', color='#1C2833')
plt.ylabel('Price', color='#1C2833')
plt.legend(loc='best')
plt.grid()
plt.show()

