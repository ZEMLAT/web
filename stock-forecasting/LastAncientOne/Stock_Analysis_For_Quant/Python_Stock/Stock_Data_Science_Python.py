#!/usr/bin/env python
# coding: utf-8

# # Data Science 101 for Python

# ## Statistical Data Analysis

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AMD'
start = '2014-01-01'
end = '2019-01-01'

# Read data 
dataset = yf.download(symbol,start,end)

# View Columns
dataset.head()


# In[3]:


# Create more data
dataset['Increase_Decrease'] = np.where(dataset['Volume'].shift(-1) > dataset['Volume'],1,-1)
dataset['Buy_Sell_on_Open'] = np.where(dataset['Open'].shift(-1) > dataset['Open'],1,-1)
dataset['Buy_Sell'] = np.where(dataset['Adj Close'].shift(-1) > dataset['Adj Close'],1,-1)
dataset['Returns'] = dataset['Adj Close'].pct_change()
dataset = dataset.dropna()
dataset.head()


# ## Measuring Central Tendency

# In[4]:


print("Mean Values in the Distribution")
print("-"*35)
print(dataset.mean())
print("***********************************")
print("Median Values in the Distribution")
print("-"*35)
print(dataset.median())


# In[5]:


print("Mode Value")
print(dataset.mode())


# ## Measuring Variance

# In[6]:


# Measuring Standard Deviation
print("Measuring Standard Deviation")
print(dataset.std())


# In[7]:


# Measuring Skewness
print("Measuring Skewness")
print(dataset.skew())


# ## Normal Distribution

# In[8]:


import math
import matplotlib.mlab as mlab

# Define Variables
mu = dataset['Returns'].mean() # Mean Returns
sigma = dataset['Returns'].std() # Volatility


# In[9]:


[n,bins,patches] = plt.hist(dataset['Returns'], 100)
# Daily returns using normal distribution
s = mlab.normpdf(bins, mu, sigma)
# Create the bins and histogram
plt.plot(bins, s, color='y', lw=2)
plt.title("Stock Returns on Normal Distribution")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.show()


# In[10]:


from scipy.stats import norm

mu = dataset['Returns'].mean()
sigma = dataset['Returns'].std()

x_min = dataset['Returns'].min()
x_max = dataset['Returns'].max()



def plot_normal(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    x = x_range
    if cdf:
        y = norm.cdf(x, mu, sigma)
    else:
        y = norm.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
    
x = np.linspace(x_min, x_max, 100)
plot_normal(x)
plot_normal(x, cdf=True)


# In[11]:


plot_normal(x, -2, 1, color='red', lw=2, ls='-', alpha=0.5)
plot_normal(x, 2, 1.2, color='blue', lw=2, ls='-', alpha=0.5)
plot_normal(x, 0, 0.8, color='green', lw=2, ls='-', alpha=0.5)


# In[12]:


mu, std = norm.fit(dataset['Returns'])

# Plot the histogram.
plt.hist(dataset['Returns'], bins=25, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


# # Gamma Distribution

# In[26]:


from scipy import stats

# input
symbol = 'AAPL'
market = 'SPY'
start = '2014-01-01'
end = '2018-01-01'

# Read data 
dataset1 = yf.download(symbol,start,end)
dataset2 = yf.download(market,start,end)

stock_ret = dataset1['Adj Close'].pct_change().dropna()
mkt_ret = dataset2['Adj Close'].pct_change().dropna()

beta, alpha, r_value, p_value, std_err = stats.linregress(mkt_ret, stock_ret)
print(beta, alpha)


# In[27]:


from scipy.stats import gamma

mu, std = gamma.stats(dataset['Returns'])

# Plot the histogram.
plt.hist(dataset['Returns'], bins=25, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1171)
p = gamma.pdf(x, alpha, scale=1/beta)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Gamma Distribution for Stock")

plt.show()


# ## Binomial Distribution

# In[15]:


from scipy.stats import binom

n = 10 # number of trials
p = 0.5 # probaility of success and failure
k = np.arange(0,21) # * number of repeat the trial
binomial = binom.pmf(k, n, p)
binomial 


# In[32]:


data_binom = binom.rvs(n=len(dataset['Adj Close']),p=0.5,size=1000)

plt.figure(figsize=(16,10))
ax = sns.distplot(data_binom,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')


# ## Poisson Distribution

# In[16]:


from scipy.stats import poisson

mu = dataset['Returns'].mean()
dist = poisson.rvs(mu=mu, loc=0, size=1000)
print("Mean: %g" % np.mean(dataset['Returns']))
print("SD: %g" % np.std(dataset['Returns'], ddof=1))

plt.hist(dist, bins=10, normed=True)
#plt.xlabel()
plt.title('Poisson Distribution Curve')
plt.show()


# ## Bernoulli Distribution

# In[17]:


from scipy.stats import bernoulli

countIncrease = dataset[dataset.Increase_Decrease == 1].Increase_Decrease.count()
countAll = dataset.Increase_Decrease.count()

Increase_dist = bernoulli(countIncrease / countAll)
# the given value is the probability of outcome 1 (increase) (let's call it p). # The probability of the opposite outcome (0 - decrease) is 1 - p.

_, ax = plt.subplots(1, 1)
ax.vlines(0, 0, Increase_dist.pmf(0), colors='r', linestyles='-', lw=5, label="probability of decrease")
ax.vlines(1, 0, Increase_dist.pmf(1), colors='b', linestyles='-', lw=5, label="probability of increase")
ax.legend(loc='best', frameon=False)
plt.title("Bernoulli distribution of increase variable")
plt.show()


# ## Exponential Distribution

# In[18]:


from scipy.stats import expon

mu = dataset['Returns'].mean()
sigma = dataset['Returns'].std()

x_m = dataset['Returns'].max()

def plot_exponential(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    if cdf:
        y = expon.cdf(x, mu, sigma)
    else:
        y = expon.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
    

x = np.linspace(0, x_m, 5000)
plot_exponential(x, 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
plot_exponential(x, 0, 1, cdf=True, color='blue', lw=2, ls='-', alpha=0.5, label='cdf')
plt.xlabel('Adj Close')
plt.ylabel('Probability')
plt.legend(loc='best')
plt.show()


# # Discrete random variable

# In[19]:


from scipy.stats import rv_discrete

increase_probability = pd.DataFrame({'probability': dataset.groupby(by = "Increase_Decrease", as_index=False).size() / dataset.Increase_Decrease.count()}).reset_index()

values = increase_probability.Increase_Decrease
probabilities = increase_probability.probability
custom_discrete_dist = rv_discrete(values=(values, probabilities))

x = dataset['Returns']

_, ax = plt.subplots(1, 1)
ax.plot(x, custom_discrete_dist.pmf(x), 'ro', lw=2)
plt.title('Custom discrete distribution of values between 0 and 4')
plt.show()


# ## P-Value

# In[20]:


from scipy import stats

# input
symbol = 'AAPL'
market = 'SPY'
start = '2014-01-01'
end = '2018-01-01'

# Read data 
dataset1 = yf.download(symbol,start,end)
dataset2 = yf.download(market,start,end)

stock_ret = dataset1['Adj Close'].pct_change().dropna()
mkt_ret = dataset2['Adj Close'].pct_change().dropna()

beta, alpha, r_value, p_value, std_err = stats.linregress(mkt_ret, stock_ret)
print(beta, alpha)
print("R-squared=", r_value**2)
print("p-value =", p_value)


# In[21]:


if p_value < alpha:  # null hypothesis: x comes from a normal distribution
    print("The null hypothesis can be rejected")
else:
    print("The null hypothesis cannot be rejected")


# ## Correlation

# In[22]:


sns.pairplot(dataset, kind="scatter")
plt.show()


# # Chi-square Test

# In[23]:


from scipy import stats

x = dataset['Adj Close']
fig,ax = plt.subplots(1,1)

linestyles = [':', '--', '-.', '-']
deg_of_freedom = [1, 4, 7, 6]
for df, ls in zip(deg_of_freedom, linestyles):
  ax.plot(x, stats.chi2.pdf(x, df), linestyle=ls)

plt.xlim(0, 10)
plt.ylim(0, 0.4)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Chi-Square Distribution')

plt.legend()
plt.show()


# ## Linear Regression

# In[24]:


sns.regplot(x = "Adj Close", y = "Open", data = dataset)
plt.show()

