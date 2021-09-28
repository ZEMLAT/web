#!/usr/bin/env python
# coding: utf-8

# # Tactical Asset Allocation

# https://www.investopedia.com/terms/t/tacticalassetallocation.asp

# ### Tactical Asset Allocation (TAA) Basics 
# Cash = 10%  
# Bonds = 35%  
# Stocks = 45%  
# Commodities = 10%  

# Cash = 5%  
# Bonds = 35%  
# Stocks = 45%  
# Commodities = 15%  

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# yfinance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


Cash = 100000.00
Cash_5 = Cash * 0.05


# In[3]:


print('How much cash left to invest?')
cash_to_invest = Cash - Cash_5
print(round(cash_to_invest, 2))


# In[4]:


# input
symbols = ['ZROZ','AAPL','SB']
start = '2012-01-01'
end = '2019-01-01'
title = "Tactical Asset Allocation"


# In[5]:


df = pd.DataFrame()
for s in symbols:
    df[s] = yf.download(s,start,end)['Adj Close']


# In[6]:


from datetime import datetime
from dateutil import relativedelta

d1 = datetime.strptime(start, "%Y-%m-%d")
d2 = datetime.strptime(end, "%Y-%m-%d")
delta = relativedelta.relativedelta(d2,d1)
print('How many years of investing?')
print('%s years' % delta.years)


# In[7]:


df.head()


# In[8]:


df.tail()


# In[9]:


print('Percentage of invest:')
percent_invest = [0.35, 0.45, 0.15]
names = ['Bonds', 'Stocks', 'Commodities']
for i, x in zip(names, percent_invest):
    cost = x * cash_to_invest
    print('{}: {}'.format(i, cost))


# In[10]:


df.iloc[0]


# In[11]:


df.ix[-1]


# In[12]:


print('Number of Shares:')
percent_invest = [0.35, 0.45, 0.15]
for i, x, y in zip(df.columns, percent_invest, df.iloc[0]):
    cost = x * cash_to_invest
    shares = int(cost/y)
    print('{}: {}'.format(i, shares))


# In[13]:


print('Beginning Value:')
percent_invest = [0.35, 0.45, 0.15]
for i, x, y in zip(df.columns, percent_invest, df.iloc[0]):
    cost = x * cash_to_invest
    shares = int(cost/y)
    Begin_Value = round(shares * y, 2)
    print('{}: ${}'.format(i, Begin_Value))


# In[14]:


print('Current Value:')
percent_invest = [0.35, 0.45, 0.15]
for i, x, y, z in zip(df.columns, percent_invest, df.iloc[0], df.iloc[-1]):
    shares = int(cost/x)
    Begin_Value = round(shares * y, 2)
    Current_Value = round(shares * z, 2)
    print('{}: ${}'.format(i, Current_Value))


# In[15]:


result = []
percent_invest = [0.35, 0.45, 0.15]
for i, x, y, z in zip(df.columns, percent_invest, df.iloc[0], df.iloc[-1]):
    shares = int(cost/x)
    Begin_Value = round(shares * y, 2)
    Current_Value = round(shares * z, 2)
    result.append(Current_Value)
print('Total Value: $%s' % round(sum(result),2))


# In[16]:


# Calculate Daily Returns
returns = df.pct_change()
returns = returns.dropna()


# In[17]:


# Calculate mean returns
meanDailyReturns = returns.mean()
print(meanDailyReturns)


# In[18]:


# Calculate std returns
stdDailyReturns = returns.std()
print(stdDailyReturns)


# In[19]:


# Define weights for the portfolio
weights = np.array([0.35, 0.45, 0.15])


# In[20]:


# Calculate the covariance matrix on daily returns
cov_matrix = (returns.cov())*250
print (cov_matrix)


# In[21]:


# Calculate expected portfolio performance
portReturn = np.sum(meanDailyReturns*weights)


# In[22]:


# Print the portfolio return
print(portReturn)


# In[23]:


# Create portfolio returns column
returns['Portfolio'] = returns.dot(weights)


# In[24]:


returns.head()


# In[25]:


returns.tail()


# In[26]:


# Calculate cumulative returns
daily_cum_ret=(1+returns).cumprod()
print(daily_cum_ret.tail())


# In[27]:


returns['Portfolio'].hist()
plt.show()


# In[28]:


import matplotlib.dates

# Plot the portfolio cumulative returns only
fig, ax = plt.subplots()
ax.plot(daily_cum_ret.index, daily_cum_ret.Portfolio, color='purple', label="portfolio")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
plt.title(title)
plt.legend()
plt.show()


# In[29]:


# Print the mean
print("mean : ", returns['Portfolio'].mean()*100)

# Print the standard deviation
print("Std. dev: ", returns['Portfolio'].std()*100)

# Print the skewness
print("skew: ", returns['Portfolio'].skew())

# Print the kurtosis
print("kurt: ", returns['Portfolio'].kurtosis())


# In[30]:


# Calculate the standard deviation by taking the square root
port_standard_dev = np.sqrt(np.dot(weights.T, np.dot(weights, cov_matrix)))

# Print the results 
print(str(np.round(port_standard_dev, 4) * 100) + '%')


# In[31]:


# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Print the result
print(str(np.round(port_variance, 4) * 100) + '%')


# In[32]:


# Calculate total return and annualized return from price data 
total_return = (returns['Portfolio'][-1] - returns['Portfolio'][0]) / returns['Portfolio'][0]

# Annualize the total return over 5 year 
annualized_return = ((total_return + 1)**(1/5))-1


# In[33]:


# Calculate annualized volatility from the standard deviation
vol_port = returns['Portfolio'].std() * np.sqrt(250)


# In[34]:


# Calculate the Sharpe ratio 
rf = 0.01
sharpe_ratio = ((annualized_return - rf) / vol_port)
print(sharpe_ratio)


# In[35]:


# Create a downside return column with the negative returns only
target = 0
downside_returns = returns.loc[returns['Portfolio'] < target]

# Calculate expected return and std dev of downside
expected_return = returns['Portfolio'].mean()
down_stdev = downside_returns.std()

# Calculate the sortino ratio
rf = 0.01
sortino_ratio = (expected_return - rf)/down_stdev

# Print the results
print("Expected return: ", expected_return*100)
print('-' * 50)
print("Downside risk:")
print(down_stdev*100)
print('-' * 50)
print("Sortino ratio:")
print(sortino_ratio)


# In[36]:


# Calculate the max value 
roll_max = returns['Portfolio'].rolling(center=False,min_periods=1,window=252).max()

# Calculate the daily draw-down relative to the max
daily_draw_down = returns['Portfolio']/roll_max - 1.0

# Calculate the minimum (negative) daily draw-down
max_daily_draw_down = daily_draw_down.rolling(center=False,min_periods=1,window=252).min()

# Plot the results
plt.figure(figsize=(15,15))
plt.plot(returns.index, daily_draw_down, label='Daily drawdown')
plt.plot(returns.index, max_daily_draw_down, label='Maximum daily drawdown in time-window')
plt.legend()
plt.show()


# In[37]:


plt.figure(figsize=(7,7))
corr = returns.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            cmap="Reds")


# In[38]:


# Box plot
returns.plot(kind='box')


# In[39]:


rets = returns.dropna()

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


# In[40]:


area = np.pi*20.0

sns.set(style='darkgrid')
plt.figure(figsize=(12,8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel("Expected Return", fontsize=15)
plt.ylabel("Risk", fontsize=15)
plt.title("Return vs. Risk for " + title, fontsize=20)

for label, x, y in zip(rets.columns, rets.mean(), rets.std()) : 
    plt.annotate(label, xy=(x,y), xytext=(50, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle='-', connectionstyle='bar,angle=180,fraction=-0.2'),
                bbox=dict(boxstyle="round", fc="w"))


# In[41]:


print("Stock returns: ")
print(rets.mean())
print('-' * 50)
print("Stock risk:")
print(rets.std())


# In[42]:


table = pd.DataFrame()
table['Returns'] = rets.mean()
table['Risk'] = rets.std()
table.sort_values(by='Returns')


# In[43]:


table.sort_values(by='Risk')


# In[44]:


rf = 0.01
table['Sharpe_Ratio'] = (table['Returns'] - rf) / table['Risk']
table

