#!/usr/bin/env python
# coding: utf-8

# # Train a Deep NN to predict Asset Price movements

# ## Setup Docker for GPU acceleration

# `docker run -it -p 8889:8888 -v /path/to/machine-learning-for-trading/16_convolutions_neural_nets/cnn:/cnn --name tensorflow tensorflow/tensorflow:latest-gpu-py3 bash`

# ## Imports & Settings

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import os
from pathlib import Path
from importlib import reload
from joblib import dump, load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from keras.models import Sequential
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint


# In[3]:


np.random.seed(42)


# ## Build Dataset

# We load the Quandl adjusted stock price data:

# In[4]:


prices = (pd.read_hdf('../data/assets.h5', 'quandl/wiki/prices')
          .adj_close
          .unstack().loc['2007':])
prices.info()


# ### Resample to weekly frequency

# We start by generating weekly returns for close to 2,500 stocks without missing data for the 2008-17 period, as follows:

# In[5]:


returns = (prices
           .resample('W')
           .last()
           .pct_change()
           .loc['2008': '2017']
           .dropna(axis=1)
           .sort_index(ascending=False))
returns.info()


# In[6]:


returns.head().append(returns.tail())


# ### Create & stack 52-week sequences

# We'll use 52-week sequences, which we'll create in a stacked format:

# In[7]:


n = len(returns)
T = 52 # weeks
tcols = list(range(T))


# In[8]:


data = pd.DataFrame()
for i in range(n-T-1):
    if i % 50 == 0:
        print(i, end=' ', flush=True)
    df = returns.iloc[i:i+T+1]
    data = pd.concat([data, (df
                             .reset_index(drop=True)
                             .transpose()
                             .reset_index()
                             .assign(year=df.index[0].year,
                                     month=df.index[0].month))],
                     ignore_index=True)
data.info()


# ### Create categorical variables

# We create dummy variables for different time periods, namely months and years:

# In[9]:


data[tcols] = (data[tcols].apply(lambda x: x.clip(lower=x.quantile(.01),
                                                  upper=x.quantile(.99))))
data.ticker = pd.factorize(data.ticker)[0]
data['label'] = (data[0] > 0).astype(int)
data['date'] = pd.to_datetime(data.assign(day=1)[['year', 'month', 'day']])
data = pd.get_dummies((data.drop(0, axis=1)
                       .set_index('date')
                       .apply(pd.to_numeric)),
                      columns=['year', 'month']).sort_index()
data.info()


# In[10]:


data.to_hdf('data.h5', 'returns_daily')


# In[11]:


data.shape

