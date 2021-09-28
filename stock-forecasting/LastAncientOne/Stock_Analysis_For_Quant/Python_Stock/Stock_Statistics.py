#!/usr/bin/env python
# coding: utf-8

# # Stock Statistics

# Statistics is a branch of applied mathematics concerned with collecting, organizing, and interpreting data. Statistics is also the mathematical study of the likelihood and probability of events occurring based on known quantitative data or a collection of data.
# 
# http://www.icoachmath.com/math_dictionary/Statistics

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbol = 'AAPL'
start = '2014-01-01'
end = '2019-01-01'

# Read data 
df = yf.download(symbol,start,end)

# View Columns
df.head()


# In[3]:


returns = df['Adj Close'].pct_change()[1:].dropna()


# ### Mean is the average number, sum of the values divided by the number of values.
# ### Median is the middle value in the list of numbers.
# ### Mode is the value that occurs often.

# In[4]:


import statistics as st

print('Mean of returns:', st.mean(returns))
print('Median of returns:', st.median(returns))
print('Median Low of returns:', st.median_low(returns))
print('Median High of returns:', st.median_high(returns))
print('Median Grouped of returns:', st.median_grouped(returns))
print('Mode of returns:', st.mode(returns))


# In[5]:


from statistics import mode

print('Mode of returns:', mode(returns))
# Since all of the returns are distinct, we use a frequency distribution to get an alternative mode.
# np.histogram returns the frequency distribution over the bins as well as the endpoints of the bins
hist, bins = np.histogram(returns, 20) # Break data up into 20 bins
maxfreq = max(hist)
# Find all of the bins that are hit with frequency maxfreq, then print the intervals corresponding to them
print('Mode of bins:', [(bins[i], bins[i+1]) for i, j in enumerate(hist) if j == maxfreq])


# ### Arithmetic Average Returns is average return on the the stock or investment

# In[6]:


print('Arithmetic average of returns:\n')
print(returns.mean())


# ### Geometric mean is the average of a set of products, the calculation of which is commonly used to determine the performance results of an investment or portfolio. It is technically defined as "the nth root product of n numbers." The geometric mean must be used when working with percentages, which are derived from values, while the standard arithmetic mean works with the values themselves.  
# 
# https://www.investopedia.com/terms/h/harmonicaverage.asp

# In[7]:


# Geometric mean
from scipy.stats.mstats import gmean
print('Geometric mean of stock:', gmean(returns))


# In[8]:


ratios = returns + np.ones(len(returns))
R_G = gmean(ratios) - 1
print('Geometric mean of returns:', R_G)


# ### Standard deviation of returns is the risk of returns

# In[9]:


print('Standard deviation of returns')
print(returns.std())


# In[10]:


T = len(returns)
init_price = df['Adj Close'][0]
final_price = df['Adj Close'][T]
print('Initial price:', init_price)
print('Final price:', final_price)
print('Final price as computed with R_G:', init_price*(1 + R_G)**T)


# ### Harmonic Mean is numerical average. 
# 
# Formula: A set of n numbers, add the reciprocals of the numbers in the set, divide the sum by n, then take the reciprocal of the result.

# In[11]:


# Harmonic mean

print('Harmonic mean of returns:', len(returns)/np.sum(1.0/returns))


# In[12]:


print('Skew:', stats.skew(returns))
print('Mean:', np.mean(returns))
print('Median:', np.median(returns))

plt.hist(returns, 30); 


# In[13]:


# Plot some example distributions stock's returns
xs = np.linspace(-6,6, 1257)
normal = stats.norm.pdf(xs)
plt.plot(returns,stats.laplace.pdf(returns), label='Leptokurtic')
print('Excess kurtosis of leptokurtic distribution:', (stats.laplace.stats(returns)))
plt.plot(returns, normal, label='Mesokurtic (normal)')
print('Excess kurtosis of mesokurtic distribution:', (stats.norm.stats(returns)))
plt.plot(returns,stats.cosine.pdf(returns), label='Platykurtic')
print('Excess kurtosis of platykurtic distribution:', (stats.cosine.stats(returns)))
plt.legend()


# In[14]:


print("Excess kurtosis of returns: ", stats.kurtosis(returns))


# In[15]:


from statsmodels.stats.stattools import jarque_bera

_, pvalue, _, _ = jarque_bera(returns)

if pvalue > 0.05:
    print('The returns are likely normal.')
else:
    print('The returns are likely not normal.')

