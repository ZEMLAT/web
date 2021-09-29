#!/usr/bin/env python
# coding: utf-8

# # PCA for Algorithmic Trading

# PCA is useful for algorithmic trading in several respects. These include the data-driven derivation of risk factors by applying PCA to asset returns, and the construction of uncorrelated portfolios based on the principal components of the correlation matrix of asset returns.

# In [Chapter 07 - Linear Models](../../07_linear_models), we explored risk factor models used in quantitative finance to capture the main drivers of returns. These models explain differences in returns on assets based on their exposure to systematic risk factors and the rewards associated with these factors. 
# 
# In particular, we explored the Fama-French approach that specifies factors based on prior knowledge about the empirical behavior of average returns, treats these factors as observable, and then estimates risk model coefficients using linear regression. An alternative approach treats risk factors as latent variables and uses factor analytic techniques like PCA to simultaneously estimate the factors and how the drive returns from historical returns.
# 
# In this section, we will review how this method derives factors in a purely statistical or data-driven way with the advantage of not requiring ex-ante knowledge of the behavior of asset returns.

# ## Imports & Settings

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
import os
from pathlib import Path
import quandl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.preprocessing import scale


# In[2]:


warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
np.random.seed(42)


# ## Get returns for equities with highest market cap

# We will use the Quandl stock price data and select the daily adjusted close prices the 500 stocks with the largest market capitalization and data for the 2010-2018 period. We then compute the daily returns as follows:

# In[19]:


idx = pd.IndexSlice
with pd.HDFStore('../../data/assets.h5') as store:
    stocks = store['us_equities/stocks'].marketcap.nlargest(500)
    returns = (store['quandl/wiki/prices']
               .loc[idx['2010': '2018', stocks.index], 'adj_close']
               .unstack('ticker')
               .pct_change())


# We obtain 215 stocks and returns for over 2,000 trading days:

# In[20]:


returns.info()


# ### Winsorize & standardize returns

# PCA is sensitive to outliers so we winsorize the data at the 2.5% and 97.5% quantiles, respectively:

# In[5]:


returns = returns.clip(lower=returns.quantile(q=.025),
                       upper=returns.quantile(q=.975),
                       axis=1)


# ### Impute missing values

# PCA does not permit missing data, so we will remove stocks that do not have data for at least 95% of the time period, and in a second step remove trading days that do not have observations on at least 95% of the remaining stocks.

# In[6]:


returns = returns.dropna(thresh=int(returns.shape[0] * .95), axis=1)
returns = returns.dropna(thresh=int(returns.shape[1] * .95))
returns.info()


# We are left with 173 equity return series covering a similar period.

# We impute any remaining missing values using the average return for any given trading day:

# In[7]:


daily_avg = returns.mean(1)
returns = returns.apply(lambda x: x.fillna(daily_avg))


# ## Fit PCA

# Now we are ready to fit the principal components model to the asset returns using default parameters to compute all components using the full SVD algorithm:

# In[8]:


cov = np.cov(returns, rowvar=False) 


# In[9]:


pca = PCA(n_components='mle')
pca.fit(returns)


# ### Visualize Explained Variance

# We find that the most important factor explains around 30% of the daily return variation. The dominant factor is usually interpreted as ‘the market’, whereas the remaining factors can be interpreted as industry or style factors in line with our discussion in chapters 5 and 7, depending on the results of closer inspection (see next example). 
# 
# The plot on the right shows the cumulative explained variance and indicates that around 10 factors explain 60% of the returns of this large cross-section of stocks.  

# The cumulative plot shows a typical 'elbow' pattern that can help identify a suitable target dimensionality because it indicates that additional components add less explanatory value.

# In[10]:


fig, axes = plt.subplots(ncols=2, figsize=(14,4))
pd.Series(pca.explained_variance_ratio_).iloc[:15].sort_values().plot.barh(title='Explained Variance Ratio by Top Factors',ax=axes[0]);
pd.Series(pca.explained_variance_ratio_).cumsum().plot(ylim=(0,1),ax=axes[1], title='Cumulative Explained Variance');


# In[11]:


risk_factors = pd.DataFrame(pca.transform(returns)[:, :2], 
                            columns=['Principal Component 1', 'Principal Component 2'], 
                            index=returns.index)
risk_factors.info()


# We can select the top two principal components to verify that they are indeed uncorrelated:

# In[12]:


risk_factors['Principal Component 1'].corr(risk_factors['Principal Component 2'])


# Moreover, we can plot the time series to highlight how each factor captures different volatility patterns.

# In[13]:


risk_factors.plot(subplots=True, figsize=(14,6));


# A risk factor model would employ a subset of the principal components as features to predict future returns, similar to our approach in chapter 7.

# ## Simulation for larger number of stocks

# In[14]:


idx = pd.IndexSlice
with pd.HDFStore('../../data/assets.h5') as store:
    returns = (store['quandl/wiki/prices']
              .loc[idx['2000': '2018', :], 'adj_close']
              .unstack('ticker')
              .pct_change())


# In[15]:


pca = PCA()
n_trials, n_samples = 100, 500
explained = np.empty(shape=(n_trials, n_samples))
for trial in range(n_trials):
    returns_sample = returns.sample(n=n_samples)
    returns_sample = returns_sample.dropna(thresh=int(returns_sample.shape[0] * .95), axis=1)
    returns_sample = returns_sample.dropna(thresh=int(returns_sample.shape[1] * .95))
    daily_avg = returns_sample.mean(1)
    returns_sample = returns_sample.apply(lambda x: x.fillna(daily_avg))
    pca.fit(returns_sample)
    explained[trial, :len(pca.components_)] = pca.explained_variance_ratio_


# In[16]:


explained = pd.DataFrame(explained, columns=list(range(1, explained.shape[1] + 1)))
explained.info()


# In[17]:


fig, axes =plt.subplots(ncols=2, figsize=(14, 4.5))
pc10 = explained.iloc[:, :10].stack().reset_index()
pc10.columns = ['Trial','Principal Component', 'Value']

pc10['Cumulative'] = pc10.groupby('Trial').Value.transform(np.cumsum)
sns.barplot(x='Principal Component', y='Value', data=pc10, ax=axes[0])
sns.lineplot(x='Principal Component', y='Cumulative', data=pc10, ax=axes[1])
fig.suptitle('Explained Variance of Top 10 Principal Components | 100 Trials')
fig.tight_layout()
fig.subplots_adjust(top=.90);


# ## Eigenportfolios

# Another application of PCA involves the covariance matrix of the normalized returns. The principal components of the correlation matrix capture most of the covariation among assets in descending order and are mutually uncorrelated. Moreover, we can use standardized the principal components as portfolio weights.
# 
# Let’s use the 30 largest stocks with data for the 2010-2018 period to facilitate the exposition:

# In[22]:


idx = pd.IndexSlice
with pd.HDFStore('../../data/assets.h5') as store:
    stocks = store['us_equities/stocks'].marketcap.nlargest(30)
    returns = (store['quandl/wiki/prices']
               .loc[idx['2010': '2018', stocks.index], 'adj_close']
               .unstack('ticker')
               .pct_change())


# We again winsorize and also normalize the returns:

# In[23]:


normed_returns = scale(returns
                       .clip(lower=returns.quantile(q=.025), 
                             upper=returns.quantile(q=.975), 
                             axis=1)
                      .apply(lambda x: x.sub(x.mean()).div(x.std())))


# In[24]:


returns = returns.dropna(thresh=int(returns.shape[0] * .95), axis=1)
returns = returns.dropna(thresh=int(returns.shape[1] * .95))
returns.info()


# In[25]:


cov = returns.cov()


# In[26]:


sns.clustermap(cov);


# After dropping assets and trading days as in the previous example, we are left with 23 assets and over 2,000 trading days. We estimate all principal components and find that the two largest explain 57.6% and 12.4% of the covariation, respectively:

# In[27]:


pca = PCA()
pca.fit(cov)
pd.Series(pca.explained_variance_ratio_).to_frame('Explained Variance').head().style.format('{:,.2%}'.format)


# ### Create PF weights from principal components

# Next, we select and normalize the four largest components so that they sum to 1 and we can use them as weights for portfolios that we can compare to an equal-weighted portfolio formed from all stocks::

# In[28]:


top4 = pd.DataFrame(pca.components_[:4], columns=cov.columns)
eigen_portfolios = top4.div(top4.sum(1), axis=0)
eigen_portfolios.index = [f'Portfolio {i}' for i in range(1, 5)]


# ### Eigenportfolio Weights

# The weights show distinct emphasis, e.g., portfolio 3 puts large weights on Mastercard and Visa, the two payment processors in the sampel whereas potfolio 2 has more exposure to some technology companies:

# In[29]:


eigen_portfolios.T.plot.bar(subplots=True, layout=(2,2), figsize=(14,6), legend=False, sharey=True);


# ### Eigenportfolio Performance

# When comparing the performance of each portfolio over the sample period to ‘the market’ consisting of our small sample, we find that portfolio 1 performs very similarly, whereas the other portfolios capture different return patterns.

# In[30]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,8), sharex=True)
axes = axes.flatten()
returns.mean(1).add(1).cumprod().sub(1).plot(title='The Market', ax=axes[0]);
for i in range(3):
    returns.mul(eigen_portfolios.iloc[i]).sum(1).add(1).cumprod().sub(1).plot(title=f'Portfolio {i+1}', ax=axes[i+1]);


# In[ ]:




