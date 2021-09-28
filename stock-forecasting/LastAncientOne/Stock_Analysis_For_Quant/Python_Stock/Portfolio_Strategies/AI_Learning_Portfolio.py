#!/usr/bin/env python
# coding: utf-8

# # Artificial Intelligence Portfolio Risk and Returns

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


# input
# A.I. Stock
symbols = ['TCEHY','NVDA','MSFT','TWLO','AMZN','GOOGL','NFLX','CRM','DE','SPLK','VERI','SNPS','QUIK','BRN.AX']
start = '2016-01-01'
end = '2019-12-28'


# In[3]:


df = pd.DataFrame()
for s in symbols:
    df[s] = yf.download(s,start,end)['Adj Close']


# In[4]:


from datetime import datetime
from dateutil import relativedelta

d1 = datetime.strptime(start, "%Y-%m-%d")
d2 = datetime.strptime(end, "%Y-%m-%d")
delta = relativedelta.relativedelta(d2,d1)
print('How many years of investing?')
print('%s years' % delta.years)


# In[5]:


number_of_years = delta.years


# In[6]:


days = (df.index[-1] - df.index[0]).days
days


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


plt.figure(figsize=(12,8))
plt.plot(df)
plt.title('Artificial Intelligence Stocks Closing Price')
plt.legend(labels=df.columns)


# In[10]:


# Normalize the data
normalize = (df - df.min())/ (df.max() - df.min())


# In[11]:


plt.figure(figsize=(18,12))
plt.plot(normalize)
plt.title('Artificial Intelligence Stocks Normalize')
plt.legend(labels=normalize.columns)


# In[12]:


stock_rets = df.pct_change().dropna()


# In[13]:


plt.figure(figsize=(12,8))
plt.plot(stock_rets)
plt.title('Artificial Intelligence Stocks Returns')
plt.legend(labels=stock_rets.columns)


# In[14]:


plt.figure(figsize=(12,8))
plt.plot(stock_rets.cumsum())
plt.title('Artificial Intelligence Stocks Returns Cumulative Sum')
plt.legend(labels=stock_rets.columns)


# In[15]:


sns.set(style='ticks')
ax = sns.pairplot(stock_rets, diag_kind='hist')

nplot = len(stock_rets.columns)
for i in range(nplot) :
    for j in range(nplot) :
        ax.axes[i, j].locator_params(axis='x', nbins=6, tight=True)


# In[16]:


ax = sns.PairGrid(stock_rets)
ax.map_upper(plt.scatter, color='purple')
ax.map_lower(sns.kdeplot, color='blue')
ax.map_diag(plt.hist, bins=30)
for i in range(nplot) :
    for j in range(nplot) :
        ax.axes[i, j].locator_params(axis='x', nbins=6, tight=True)


# In[17]:


plt.figure(figsize=(7,7))
corr = stock_rets.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            cmap="Blues")


# In[18]:


# Box plot
stock_rets.plot(kind='box',figsize=(12,8))


# In[19]:


rets = stock_rets.dropna()

plt.figure(figsize=(12,8))
plt.scatter(rets.mean(), rets.std(),alpha = 0.5)

plt.title('Stocks Risk & Returns')
plt.xlabel('Expected returns')
plt.ylabel('Risk')
plt.grid(which='major')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# In[20]:


rets = stock_rets.dropna()
area = np.pi*20.0

sns.set(style='darkgrid')
plt.figure(figsize=(12,8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel("Expected Return", fontsize=15)
plt.ylabel("Risk", fontsize=15)
plt.title("Return vs. Risk for Stocks", fontsize=20)

for label, x, y in zip(rets.columns, rets.mean(), rets.std()) : 
    plt.annotate(label, xy=(x,y), xytext=(50, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle='-', connectionstyle='bar,angle=180,fraction=-0.2'),
                bbox=dict(boxstyle="round", fc="w"))


# In[21]:


rest_rets = rets.corr()
pair_value = rest_rets.abs().unstack()
pair_value.sort_values(ascending = False)


# In[22]:


# Normalized Returns Data
Normalized_Value = ((rets[:] - rets[:].min()) /(rets[:].max() - rets[:].min()))
Normalized_Value.head()


# In[23]:


Normalized_Value.corr()


# In[24]:


normalized_rets = Normalized_Value.corr()
normalized_pair_value = normalized_rets.abs().unstack()
normalized_pair_value.sort_values(ascending = False)


# In[25]:


print("Stock returns: ")
print(rets.mean())
print('-' * 50)
print("Stock risks:")
print(rets.std())


# In[26]:


table = pd.DataFrame()
table['Returns'] = rets.mean()
table['Risk'] = rets.std()
table.sort_values(by='Returns')


# In[27]:


table.sort_values(by='Risk')


# In[28]:


rf = 0.01
table['Sharpe Ratio'] = (table['Returns'] - rf) / table['Risk']
table


# In[29]:


table['Max Returns'] = rets.max()


# In[30]:


table['Min Returns'] = rets.min()


# In[31]:


table['Median Returns'] = rets.median()


# In[32]:


total_return = stock_rets[-1:].transpose()
table['Total Return'] = 100 * total_return
table


# In[33]:


table['Average Return Yearly'] = (1 + total_return)**(1 / number_of_years) - 1
table


# In[34]:


initial_value = df.iloc[0]
ending_value = df.iloc[-1]
table['CAGR'] = ((ending_value / initial_value) ** (252.0 / days)) -1
table


# In[35]:


table.sort_values(by='Average Return Yearly')

