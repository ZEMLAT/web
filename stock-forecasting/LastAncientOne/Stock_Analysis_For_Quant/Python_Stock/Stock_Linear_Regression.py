#!/usr/bin/env python
# coding: utf-8

# # Stock Linear Regression

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbols = 'AMD'
start = '2012-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbols,start,end)

# View Columns
dataset.head()


# In[3]:


dataset['Returns'] = np.log(dataset['Adj Close'] / dataset['Adj Close'].shift(1))


# In[4]:


dataset = dataset.dropna()


# In[5]:


X = dataset['Open']
Y = dataset['Adj Close']


# In[6]:


plt.scatter(X,Y)
plt.xlabel('Open')
plt.ylabel('Adj Close')
plt.title('Stock Linear Regression')
plt.show()


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[9]:


X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)


# In[10]:


X_train.shape


# In[11]:


from sklearn.linear_model import LinearRegression

linregression=LinearRegression()
linregression.fit(X_train,y_train)


# In[12]:


y_pred = linregression.predict(X_test)
y_pred


# In[13]:


print('Intercept')
linregression.intercept_


# In[14]:


print('Slope')
linregression.coef_


# In[15]:


import matplotlib.pyplot as plt

plt.scatter(X_train,y_train)
plt.plot(X_train,linregression.predict(X_train),'r')
plt.xlabel('Open')
plt.ylabel('Adj Close')
plt.title('Stock Linear Regression')
plt.show()

