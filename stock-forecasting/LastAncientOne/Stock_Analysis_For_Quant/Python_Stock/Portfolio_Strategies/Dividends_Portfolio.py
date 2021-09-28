#!/usr/bin/env python
# coding: utf-8

# # Stocks Dividends Portfolio

# ## Stocks with Dividend

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# fetch dividend
import yfinance as yfd
# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbols = ['ALX','BLK','SPG','LMT']
start = '2007-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbols,start,end)['Adj Close']

# View Columns
df.head()


# In[3]:


df.tail()


# In[4]:


from datetime import datetime
from dateutil import relativedelta

d1 = datetime.strptime(start, "%Y-%m-%d")
d2 = datetime.strptime(end, "%Y-%m-%d")
delta = relativedelta.relativedelta(d2,d1)
print('How many years of investing?')
print('%s years' % delta.years)


# In[5]:


for s in symbols: 
    df[s].plot(label = s, figsize = (15,10))
plt.legend()


# In[6]:


for s in symbols:
    print(s + ":",  df[s].max())


# In[7]:


for s in symbols:
    print(s + ":",  df[s].min())


# In[8]:


returns = pd.DataFrame()
for s in symbols: 
    returns[s + " Return"] = (np.log(1 + df[s].pct_change())).dropna()
    
returns.head(4)


# In[9]:


sns.pairplot(returns[1:])


# In[10]:


# dates each bank stock had the best and worst single day returns. 
print('Best Day Returns')
print('-'*20)
print(returns.idxmax())
print('\n')
print('Worst Day Returns')
print('-'*20)
print(returns.idxmin())


# In[11]:


plt.figure(figsize=(17,13))

for r in returns:
    sns.kdeplot(returns.ix["2011-01-01" : "2011-12-31 "][r])


# In[12]:


returns.corr()


# In[13]:


# Heatmap for return of all the banks
plt.figure(figsize=(15,10))
sns.heatmap(returns.corr(), cmap="cool",linewidths=.1, annot= True)

sns.clustermap(returns.corr(), cmap="Wistia",linewidths=.1, annot= True)


# In[14]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="hot",linewidths=.1, annot= True)

sns.clustermap(df.corr(), cmap="copper",linewidths=.1, annot= True)


# In[15]:


Cash = 100000
print('Percentage of invest:')
percent_invest = [0.25, 0.25, 0.25, 0.25]
for i, x in zip(df.columns, percent_invest):
    cost = x * Cash
    print('{}: {}'.format(i, cost))


# In[16]:


print('Number of Shares:')
percent_invest = [0.25, 0.25, 0.25, 0.25]
for i, x, y in zip(df.columns, percent_invest, df.iloc[0]):
    cost = x * Cash
    shares = int(cost/y)
    print('{}: {}'.format(i, shares))


# In[17]:


print('Beginning Value:')
percent_invest = [0.25, 0.25, 0.25, 0.25]
for i, x, y in zip(df.columns, percent_invest, df.iloc[0]):
    cost = x * Cash
    shares = int(cost/y)
    Begin_Value = round(shares * y, 2)
    print('{}: ${}'.format(i, Begin_Value))


# In[18]:


print('Current Value:')
percent_invest = [0.25, 0.25, 0.25, 0.25]
for i, x, y, z in zip(df.columns, percent_invest, df.iloc[0], df.iloc[-1]):
    cost = x * Cash
    shares = int(cost/y)
    Current_Value = round(shares * z, 2)
    print('{}: ${}'.format(i, Current_Value))


# In[19]:


result = []
percent_invest = [0.25, 0.25, 0.25, 0.25]
for i, x, y, z in zip(df.columns, percent_invest, df.iloc[0], df.iloc[-1]):
    cost = x * Cash
    shares = int(cost/y)
    Current_Value = round(shares * z, 2)
    result.append(Current_Value)
print('Total Value: $%s' % round(sum(result),2))


# In[22]:


stock = yfd.Tickers('ALX BLK SPG LMT')
stock


# In[25]:


s1_dividend = stock.ALX.dividends['2007-01-01':].sum()
s2_dividend = stock.BLK.dividends['2007-01-01':].sum()
s3_dividend = stock.SPG.dividends['2007-01-01':].sum()
s4_dividend = stock.LMT.dividends['2007-01-01':].sum()


# In[26]:


data = [s1_dividend, s2_dividend, s3_dividend, s4_dividend]


# In[27]:


print('Total Dividends:')
data = [s1_dividend, s2_dividend, s3_dividend, s4_dividend]
for i, x in zip(df.columns, data):
    print('{}: {}'.format(i, x))


# In[39]:


print('Dividends with Shares:')
percent_invest = [0.25, 0.25, 0.25, 0.25]
data = [s1_dividend, s2_dividend, s3_dividend, s4_dividend]
for i, x, y in zip(df.columns, percent_invest, data):
    cost = x * Cash
    shares = int(cost/y)
    total_dividend_cost = shares * y
    print('{}: ${}'.format(i, round(total_dividend_cost,2)))


# In[40]:


dividend = []
percent_invest = [0.25, 0.25, 0.25, 0.25]
data = [s1_dividend, s2_dividend, s3_dividend, s4_dividend]
for i, x, y in zip(df.columns, percent_invest, data):
    cost = x * Cash
    shares = int(cost/y)
    total_dividend_cost = shares * y
    dividend.append(total_dividend_cost)
print('Total Dividends: $%s' % round(sum(dividend),2))


# In[52]:


print('Total Money: $%s' % round((sum(dividend) + sum(result)),2))
print('Total Profit: $%s' % (round((sum(dividend) + sum(result)),2) - Cash))

