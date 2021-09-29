#!/usr/bin/env python
# coding: utf-8

# # How to prepare the data

# We use a simplified version of the data set constructed in Chapter 4, Alpha factor research. It consists of daily stock prices provided by Quandl for the 2010-2017 period and various engineered features. 
# 
# The decision tree models in this chapter are not equipped to handle missing or categorical variables, so we will apply dummy encoding to the latter after dropping any of the former.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import os
from pathlib import Path
import quandl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, precision_recall_curve
from sklearn.preprocessing import Imputer
import statsmodels.api as sm
from scipy.interpolate import interp1d, interp2d


# In[2]:


warnings.filterwarnings('ignore')
plt.style.use('ggplot')


# ## Get Data

# In[3]:


with pd.HDFStore('../data/assets.h5') as store:
    print(store.info())
    prices = store['quandl/wiki/prices'].adj_close.unstack('ticker')
    stocks = store['us_equities/stocks']


# In[4]:


shared = prices.columns.intersection(stocks.index)
prices = prices.loc['2010': '2018', shared]
stocks = stocks.loc[shared, ['marketcap', 'ipoyear', 'sector']]


# In[5]:


stocks.info()


# In[6]:


prices.info()


# ### Create monthly return series

# Remove outliers

# In[8]:


returns = prices.resample('M').last().pct_change().stack().swaplevel()
returns = (returns[returns.between(left=returns.quantile(.05), 
                                   right=returns.quantile(.95))].to_frame('returns'))


# ### Lagged Returns

# In[9]:


for t in range(1, 13):
    returns[f't-{t}'] = returns.groupby(level='ticker').returns.shift(t)
returns = returns.dropna()


# ### Time Period Dummies

# In[10]:


# returns = returns.reset_index('date')
dates = returns.index.get_level_values('date')
returns['year'] = dates.year
returns['month'] = dates.month
returns = pd.get_dummies(returns, columns=['year', 'month'])


# In[11]:


returns.info()


# ### Get stock characteristics

# #### Create age proxy

# In[12]:


stocks['age'] = pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6))).astype(float).fillna(0).astype(int)
stocks = stocks.drop('ipoyear', axis=1)


# #### Create size proxy

# In[15]:


stocks.info()


# In[16]:


stocks.marketcap.head()


# In[18]:


stocks['size'] = pd.qcut(stocks.marketcap, q=10, labels=list(range(1, 11)))
stocks = stocks.drop(['marketcap'], axis=1)


# #### Create Dummy variables

# In[19]:


stocks.info()


# In[20]:


stocks = pd.get_dummies(stocks, 
                        columns=['size', 'age',  'sector'], 
                        prefix=['size', 'age', ''], 
                        prefix_sep=['_', '_', ''])
stocks.info()


# ### Combine data

# In[21]:


data = (returns
        .reset_index('date')
        .merge(stocks, left_index=True, right_index=True)
        .dropna()
        .set_index('date', append=True))

s = len(returns.columns)
data.iloc[:, s:] = data.iloc[:, s:].astype(int).apply(pd.to_numeric, downcast='integer')
data.info()


# ### Store data

# In[22]:


with pd.HDFStore('data.h5') as store:
    store.put('data', data)

