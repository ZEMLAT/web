#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression for stock using scikit-learn
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


stock = 'AAPL'
start = '2016-01-01' 
end = '2018-01-01'
data = yf.download(stock, start, end)
data.head()


# In[3]:


df = data.reset_index()
df.head()


# In[4]:


X = df.drop(['Date','Close'], axis=1, inplace=True)
y = df[['Adj Close']]


# In[5]:


df = df.as_matrix()


# In[6]:


from sklearn.model_selection import train_test_split

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25,  random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[8]:


intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[9]:


regression_model.score(X_test, y_test)


# In[10]:


from sklearn.metrics import mean_squared_error

y_predict = regression_model.predict(X_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

regression_model_mse


# In[11]:


math.sqrt(regression_model_mse)


# In[12]:


# input the latest Open, High, Low, Close, Volume
# predicts the next day price
regression_model.predict([[167.81, 171.75, 165.19, 166.48, 37232900]])

