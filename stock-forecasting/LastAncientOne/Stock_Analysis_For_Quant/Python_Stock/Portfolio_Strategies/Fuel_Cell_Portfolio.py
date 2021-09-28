#!/usr/bin/env python
# coding: utf-8

# # Fuel Cell Portfolio Risk and Returns 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# yahoo finance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[3]:


# input
# 4 Fuel Cell
symbols = ['FCEL','PLUG','BLDP','BE']
start = '2019-01-01'
end = '2020-12-22'


# In[4]:


df = pd.DataFrame()
for s in symbols:
    df[s] = yf.download(s,start,end)['Adj Close']


# In[5]:


from datetime import datetime
from dateutil import relativedelta

d1 = datetime.strptime(start, "%Y-%m-%d")
d2 = datetime.strptime(end, "%Y-%m-%d")
delta = relativedelta.relativedelta(d2,d1)
print('How many years of investing?')
print('%s years' % delta.years)


# In[6]:


number_of_years = delta.years


# In[7]:


months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
months


# In[8]:


days = (df.index[-1] - df.index[0]).days
days


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


plt.figure(figsize=(12,8))
plt.plot(df)
plt.title('Fuel Cell Stocks Closing Price')
plt.legend(labels=df.columns)


# In[12]:


# Normalize the data
normalize = (df - df.min())/ (df.max() - df.min())


# In[13]:


plt.figure(figsize=(18,12))
plt.plot(normalize)
plt.title('Fuel Cell Stocks Normalize')
plt.legend(labels=normalize.columns)


# In[14]:


stock_rets = df.pct_change().dropna()


# In[15]:


plt.figure(figsize=(12,8))
plt.plot(stock_rets)
plt.title('Fuel Cell Stocks Returns')
plt.legend(labels=stock_rets.columns)


# In[16]:


plt.figure(figsize=(12,8))
plt.plot(stock_rets.cumsum())
plt.title('Fuel Cell Stocks Returns Cumulative Sum')
plt.legend(labels=stock_rets.columns)


# In[17]:


sns.set(style='ticks')
ax = sns.pairplot(stock_rets, diag_kind='hist')

nplot = len(stock_rets.columns)
for i in range(nplot) :
    for j in range(nplot) :
        ax.axes[i, j].locator_params(axis='x', nbins=6, tight=True)


# In[18]:


ax = sns.PairGrid(stock_rets)
ax.map_upper(plt.scatter, color='purple')
ax.map_lower(sns.kdeplot, color='blue')
ax.map_diag(plt.hist, bins=30)
for i in range(nplot) :
    for j in range(nplot) :
        ax.axes[i, j].locator_params(axis='x', nbins=6, tight=True)


# In[19]:


plt.figure(figsize=(7,7))
corr = stock_rets.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            cmap="Reds")


# In[20]:


# Box plot
stock_rets.plot(kind='box',figsize=(12,8))


# In[21]:


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


# In[22]:


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


# In[23]:


rest_rets = rets.corr()
pair_value = rest_rets.abs().unstack()
pair_value.sort_values(ascending = False)


# In[24]:


# Normalized Returns Data
Normalized_Value = ((rets[:] - rets[:].min()) /(rets[:].max() - rets[:].min()))
Normalized_Value.head()


# In[25]:


Normalized_Value.corr()


# In[26]:


normalized_rets = Normalized_Value.corr()
normalized_pair_value = normalized_rets.abs().unstack()
normalized_pair_value.sort_values(ascending = False)


# In[27]:


print("Stock returns: ")
print(rets.mean())
print('-' * 50)
print("Stock risks:")
print(rets.std())


# In[28]:


table = pd.DataFrame()
table['Returns'] = rets.mean()
table['Risk'] = rets.std()
table.sort_values(by='Returns')


# In[29]:


table.sort_values(by='Risk')


# In[30]:


rf = 0.001
table['Sharpe Ratio'] = (table['Returns'] - rf) / table['Risk']
table


# In[31]:


table['Max Returns'] = rets.max()


# In[32]:


table['Min Returns'] = rets.min()


# In[33]:


table['Median Returns'] = rets.median()


# In[34]:


total_return = stock_rets[-1:].transpose()
table['Total Return'] = 100 * total_return
table


# In[35]:


table['Average Return Days'] = (1 + total_return)**(1 / days) - 1
table


# In[36]:


initial_value = df.iloc[0]
ending_value = df.iloc[-1]
table['CAGR'] = ((ending_value / initial_value) ** (252.0 / days)) -1
table


# In[37]:


table.sort_values(by='Average Return Days')

