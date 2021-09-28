#!/usr/bin/env python
# coding: utf-8

# # Profit and Loss in Trading

# https://www.investopedia.com/ask/answers/how-do-you-calculate-percentage-gain-or-loss-investment/
# 
# https://www.investopedia.com/ask/answer/07/portfolio_calculations.asp

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
symbol = 'MSFT'
start = '2016-01-01'
end = '2019-09-11'

# Read data 
dataset = yf.download(symbol,start,end)

# View Columns
dataset.head()


# In[34]:


dataset.tail()


# In[3]:


Start = 5000 # How much to invest


# In[4]:


dataset['Shares'] = 0
dataset['PnL'] = 0
dataset['End'] = Start


# In[5]:


dataset['Shares'] = dataset['End'].shift(1) / dataset['Adj Close'].shift(1)


# In[6]:


dataset['PnL'] = dataset['Shares'] * (dataset['Adj Close'] - dataset['Adj Close'].shift(1))


# In[7]:


dataset['End'] = dataset['End'].shift(1) + dataset['PnL']


# In[8]:


dataset.head()


# In[9]:


dataset.tail()


# In[10]:


plt.figure(figsize=(16,8))
plt.plot(dataset['PnL'])
plt.title('Profit and Loss for Daily')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[11]:


plt.figure(figsize=(16,8))
plt.plot(dataset['End'])
plt.title('Ending Value for Daily')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[12]:


# How many shares can get with the current money?
Shares = round(int(float(Start) / dataset['Adj Close'][0]),1)
Purchase_Price = dataset['Adj Close'][0] # Invest in the Beginning Price
Current_Value = dataset['Adj Close'][-1] # Value of stock of Ending Price
Purchase_Cost = Shares * Purchase_Price
Current_Value = Shares * Current_Value
Profit_or_Loss = Current_Value - Purchase_Cost 


# In[13]:


print(symbol + ' profit or loss of $%.2f' % (Profit_or_Loss))


# In[31]:


percentage_gain_or_loss = (Profit_or_Loss/Current_Value) * 100
print('%s %%' % round(percentage_gain_or_loss,2))


# In[32]:


percentage_returns = (Current_Value - Purchase_Cost)/ Purchase_Cost 
print('%s %%' % round(percentage_returns,2))


# In[37]:


net_gains_or_losses = (dataset['Adj Close'][-1] - dataset['Adj Close'][0]) / dataset['Adj Close'][0]
print('%s %%' % round(net_gains_or_losses,2))


# In[39]:


total_return = ((Current_Value/Purchase_Cost)-1) * 100
print('%s %%' % round(total_return,2))


# In[41]:


print("Financial Analysis")
print('-' * 50)
print(symbol + ' profit or loss of $%.2f' % (Profit_or_Loss))
print('Percentage gain or loss: %s %%' % round(percentage_gain_or_loss,2))
print('Percentage of returns: %s %%' % round(percentage_returns,2))
print('Net gains or losses: %s %%' % round(net_gains_or_losses,2))
print('Total Returns: %s %%' % round(total_return,2))

