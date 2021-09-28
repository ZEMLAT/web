#!/usr/bin/env python
# coding: utf-8

# # The Dave Ramsey Portfolio

# https://www.daveramsey.com/blog/daves-investing-philosophy

# Step 1: Set goals for your investments.  
# Step 2: Save 15% of your income for retirement.  
# Step 3: Choose good growth stock mutual funds.  
# Step 4: Invest with a long-term perspective.  
# Step 5: Get help from an investing professional.  

# Age: 38 Years Old  
# 
# Reitrement: Have 1 million dollar by age 60
# 
# College fund: Save $100,000 in ten years for daughter's tuition
# 
# Buy a home: Buy a $500,000  

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


income_for_retirement = 5000 # Monthly
save_15_percent = 5000 * 0.15


# In[3]:


print('Save 15% of your income for retirement: ', save_15_percent)
print('Save in a year: $', save_15_percent*12)
print('Save in 2 years: $', save_15_percent*24)
print('Save in 5 years: $', save_15_percent*60)
print('Save in 10 years: $', save_15_percent*120)
print('Save in 20 years: $', save_15_percent*240)
print('Save in age of 60: $', save_15_percent*264)
print('Save in age of 65: $', save_15_percent*324)


# In[4]:


# input
symbols = ['VTSAX','SPY','VGSLX','VSIAX']
start = '2014-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbols,start,end)['Adj Close']

# View Columns
dataset.head()


# In[5]:


dataset.tail()


# In[6]:


# Calculate Daily Returns
returns = dataset.pct_change()
returns = returns.dropna()


# In[7]:


returns.head()


# In[8]:


# Calculate mean returns
meanDailyReturns = returns.mean()
print(meanDailyReturns)


# In[9]:


# Calculate std returns
stdDailyReturns = returns.std()
print(stdDailyReturns)


# In[10]:


# Define weights for the portfolio
weights = np.array([0.25, 0.25, 0.25, 0.25])


# In[11]:


# Calculate the covariance matrix on daily returns
cov_matrix = (returns.cov())*250
print (cov_matrix)


# In[12]:


# Calculate expected portfolio performance
portReturn = np.sum(meanDailyReturns*weights)


# In[13]:


# Print the portfolio return
print(portReturn)


# In[14]:


# Create portfolio returns column
returns['Portfolio'] = returns.dot(weights)


# In[15]:


returns.head()


# In[16]:


returns.tail()


# In[17]:


# Calculate cumulative returns
daily_cum_ret=(1+returns).cumprod()
print(daily_cum_ret.tail())


# In[18]:


returns['Portfolio'].hist()
plt.show()


# In[19]:


import matplotlib.dates

# Plot the portfolio cumulative returns only
fig, ax = plt.subplots()
ax.plot(daily_cum_ret.index, daily_cum_ret.Portfolio, color='purple', label="portfolio")
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
plt.legend()
plt.show()


# In[20]:


# Print the mean
print("mean : ", returns['Portfolio'].mean()*100)

# Print the standard deviation
print("Std. dev: ", returns['Portfolio'].std()*100)

# Print the skewness
print("skew: ", returns['Portfolio'].skew())

# Print the kurtosis
print("kurt: ", returns['Portfolio'].kurtosis())


# In[21]:


# Calculate the standard deviation by taking the square root
port_standard_dev = np.sqrt(np.dot(weights.T, np.dot(weights, cov_matrix)))

# Print the results 
print(str(np.round(port_standard_dev, 4) * 100) + '%')


# In[22]:


# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Print the result
print(str(np.round(port_variance, 4) * 100) + '%')


# In[23]:


# Calculate total return and annualized return from price data 
total_return = (returns['Portfolio'][-1] - returns['Portfolio'][0]) / returns['Portfolio'][0]

# Annualize the total return over 5 year 
annualized_return = ((total_return + 1)**(1/5))-1


# In[24]:


# Calculate annualized volatility from the standard deviation
vol_port = returns['Portfolio'].std() * np.sqrt(250)


# In[25]:


# Calculate the Sharpe ratio 
rf = 0.01
sharpe_ratio = ((annualized_return - rf) / vol_port)
print(sharpe_ratio)


# In[26]:


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


# In[27]:


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


# In[28]:


plt.figure(figsize=(7,7))
corr = returns.corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
            cmap="Blues")


# In[29]:


# Box plot
returns.plot(kind='box')


# In[30]:


rets = returns.dropna()

colors=['red','green','blue','yellow','purple']
plt.scatter(rets.mean(), rets.std(), c=colors,alpha = 0.5)

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


# In[31]:


area = np.pi*10.0

sns.set(style='darkgrid')
plt.figure(figsize=(12,8))
colors=['red','green','blue','yellow','purple']

plt.scatter(rets.mean(), rets.std(), s=area, c=colors)
plt.xlabel("Expected Return", fontsize=15)
plt.ylabel("Risk", fontsize=15)
plt.title("Return vs. Risk for Dave Ramsey Portfolio", fontsize=20)

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.2', color = 'black'))


# In[32]:


print("Stock returns: ")
print(rets.mean())
print('-' * 50)
print("Stock risk:")
print(rets.std())


# In[33]:


table = pd.DataFrame()
table['Returns'] = rets.mean()
table['Risk'] = rets.std()
table.sort_values(by='Returns')


# In[34]:


table.sort_values(by='Risk')


# In[35]:


rf = 0.01
table['Sharpe_Ratio'] = (table['Returns'] - rf) / table['Risk']
table


# Dave Ramsey Portfolio is the lowest risk with lowest returns. Compare other portfolio strategies, Dave Ramsey Portfolio is the safest strategies.
