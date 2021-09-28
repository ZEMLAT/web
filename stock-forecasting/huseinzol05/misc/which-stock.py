#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[2]:


directory = '../dataset/'
ori_name = ['AMD.csv', 'FB.csv', 'FSV.csv', 'INFY.csv', 'KNX.csv',
          'MONDY.csv', 'MTDR.csv', 'SINA.csv', 'TMUS.csv', 'TSLA.csv', 'TWTR.csv']
stocks = [directory + s for s in ori_name]
stocks


# In[3]:


dfs = [pd.read_csv(s)[['Date', 'Close']] for s in stocks]


# In[4]:


from functools import reduce
data = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs).iloc[:, 1:]
data.head()


# In[5]:


returns = data.pct_change()
mean_daily_returns = returns.mean()
volatilities = returns.std()


# In[6]:


mean_daily_returns * 252


# In[7]:


volatilities * 252


# In[8]:


combine = pd.DataFrame({'returns': mean_daily_returns * 252,
                       'volatility': volatilities * 252})


# In[9]:


g = sns.jointplot("volatility", "returns", data=combine, kind="reg",height=7)

for i in range(combine.shape[0]):
    plt.annotate(ori_name[i].replace('.csv',''), (combine.iloc[i, 1], combine.iloc[i, 0]))
    
plt.text(0, -1.5, 'SELL', fontsize=25)
plt.text(0, 1.0, 'BUY', fontsize=25)
    
plt.show()


# In[ ]:




