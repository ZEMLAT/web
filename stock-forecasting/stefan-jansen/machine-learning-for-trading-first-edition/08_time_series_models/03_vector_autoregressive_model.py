#!/usr/bin/env python
# coding: utf-8

# # How to use the VAR model for macro fundamentals forecasts

# The vector autoregressive VAR(p) model extends the AR(p) model to k series by creating a system of k equations where each contains p lagged values of all k series. The coefficients on the own lags provide information about the dynamics of the series itself, whereas the cross-variable coefficients offer some insight into the interactions across the series.

# ## Imports and Settings

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys
import warnings
from datetime import date
import pandas as pd
import pandas_datareader.data as web
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import seaborn as sns

import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
sns.set(style='darkgrid', context='notebook', color_codes=True)


# ## Helper Functions

# ### Correlogram Plot

# In[3]:


def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)


# ### Unit Root Test

# In[4]:


def test_unit_root(df):
    return df.apply(lambda x: f'{pd.Series(adfuller(x)).iloc[1]:.2%}').to_frame('p-value')


# ## Load Data

# We will extend the univariate example of a single time series of monthly data on industrial production and add a monthly time series on consumer sentiment, both provided by the Federal Reserve's data service. We will use the familiar pandas-datareader library to retrieve data from 1970 through 2017:

# In[5]:


sent = 'UMCSENT'
df = web.DataReader(['UMCSENT', 'IPGMFN'], 'fred', '1970', '2017-12').dropna()
df.columns = ['sentiment', 'ip']


# In[6]:


df.info()


# In[7]:


df.plot(subplots=True, figsize=(14,8));


# In[8]:


plot_correlogram(df.sentiment, lags=24)


# In[9]:


plot_correlogram(df.ip, lags=24)


# ## Stationarity Transform

# Log-transforming the industrial production series and seasonal differencing using lag 12 of both series yields stationary results:

# In[10]:


df_transformed = pd.DataFrame({'ip': np.log(df.ip).diff(12),
                              'sentiment': df.sentiment.diff(12)}).dropna()


# ## Inspect Correlograms

# In[11]:


plot_correlogram(df_transformed.sentiment, lags=24)


# In[12]:


plot_correlogram(df_transformed.ip, lags=24)


# In[13]:


test_unit_root(df_transformed)


# In[14]:


df_transformed.plot(subplots=True, figsize=(14,8));


# ## VAR Model

# To limit the size of the output, we will just estimate a VAR(1) model using the statsmodels VARMAX implementation (which allows for optional exogenous variables) with a constant trend using the first 480 observations. The output contains the coefficients for both time series equations.

# In[31]:


model = VARMAX(df_transformed.iloc[:468], order=(1,1), trend='c').fit(maxiter=1000)
print(model.summary())


# ### Plot Diagnostics

# `statsmodels` provides diagnostic plots to check whether the residuals meet the white noise assumptions, which are not exactly met in this simple case:

# #### Industrial Production

# In[17]:


model.plot_diagnostics(variable=0, figsize=(14,8), lags=24)
plt.gcf().suptitle('Industrial Production - Diagnostics', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=.9);


# #### Sentiment

# In[18]:


model.plot_diagnostics(variable=1, figsize=(14,8), lags=24)
plt.title('Sentiment - Diagnostics');


# ### Impulse-Response Function

# In[19]:


median_change = df_transformed.diff().quantile(.5).tolist()
model.impulse_responses(steps=12, impulse=median_change).plot.bar(subplots=True);


# ### Generate Predictions

# Out-of-sample predictions can be generated as follows:

# In[20]:


start = 430
preds = model.predict(start=480, end=len(df_transformed)-1)
preds.index = df_transformed.index[480:]

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

df_transformed.ip.iloc[start:].plot(ax=axes[0], label='actual', title='Industrial Production')
preds.ip.plot(label='predicted', ax=axes[0])
trans = mtransforms.blended_transform_factory(axes[0].transData, axes[0].transAxes)
axes[0].legend()
axes[0].fill_between(x=df_transformed.index[481:], y1=0, y2=1, transform=trans, color='grey', alpha=.5)

trans = mtransforms.blended_transform_factory(axes[0].transData, axes[1].transAxes)
df_transformed.sentiment.iloc[start:].plot(ax=axes[1], label='actual', title='Sentiment')
preds.sentiment.plot(label='predicted', ax=axes[1])
axes[1].fill_between(x=df_transformed.index[481:], y1=0, y2=1, transform=trans, color='grey', alpha=.5)
fig.tight_layout();


# ### Out-of-sample forecasts

# A visualization of actual and predicted values shows how the prediction lags the actual values and does not capture non-linear out-of-sample patterns well:

# In[32]:


forecast = model.forecast(steps=24)

fig, axes = plt.subplots(nrows=2, figsize=(12, 8))

df_transformed.ip.plot(ax=axes[0], label='actual', title='Liquor')
preds.ip.plot(label='predicted', ax=axes[0])
axes[0].legend()

df_transformed.sentiment.plot(ax=axes[1], label='actual', title='Sentiment')
preds.sentiment.plot(label='predicted', ax=axes[1])
axes[1]
fig.tight_layout();


# In[35]:


mean_absolute_error(forecast, df_transformed.iloc[468:])


# In[ ]:




