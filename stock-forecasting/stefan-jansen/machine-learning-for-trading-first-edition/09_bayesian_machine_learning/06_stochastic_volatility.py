#!/usr/bin/env python
# coding: utf-8

# # Stochastic Volatility model

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk

from scipy import optimize


# Asset prices have time-varying volatility (variance of day over day `returns`). In some periods, returns are highly variable, while in others very stable. Stochastic volatility models model this with a latent volatility variable, modeled as a stochastic process. The following model is similar to the one described in the No-U-Turn Sampler paper, Hoffman (2011) p21.
# 
# $$ \sigma \sim Exponential(50) $$
# 
# $$ \nu \sim Exponential(.1) $$
# 
# $$ s_i \sim Normal(s_{i-1}, \sigma^{-2}) $$
# 
# $$ log(r_i) \sim t(\nu, 0, exp(-2 s_i)) $$
# 
# Here, $r$ is the daily return series and $s$ is the latent log volatility process.

# ## Build Model

# First we load some daily returns of the S&P 500.

# In[2]:


n = 400
returns = pd.read_hdf('../data/assets.h5', key='sp500/prices').loc['2000':, 'close'].pct_change().dropna()
returns[:5]


# As you can see, the volatility seems to change over time quite a bit but cluster around certain time-periods. Around time-points 2500-3000 you can see the 2009 financial crash.

# In[3]:


returns.plot(figsize=(15,4))


# Specifying the model in `PyMC3` mirrors its statistical specification. 

# In[4]:


with pm.Model() as model:
    step_size = pm.Exponential('sigma', 50.)
    s = GaussianRandomWalk('s', sd=step_size, shape=len(returns))
    
    nu = pm.Exponential('nu', .1)
    r = pm.StudentT('r', nu=nu, lam=pm.math.exp(-2*s), 
                    observed=returns)


# ## Fit Model

# For this model, the full maximum a posteriori (MAP) point is degenerate and has infinite density. NUTS, however, gives the correct posterior.

# In[5]:


with model:
    trace = pm.sample(tune=2000, nuts_kwargs=dict(target_accept=.9))


# In[ ]:


pm.traceplot(trace, varnames=['sigma', 'nu']);


# In[ ]:


fig, ax = plt.subplots()

plt.plot(trace['s'].T, 'b', alpha=.03);
ax.set(title=str(s), xlabel='time', ylabel='log volatility');


# Looking at the returns over time and overlaying the estimated standard deviation we can see how the model tracks the volatility over time.

# In[ ]:


pm.trace_to_dataframe(trace).info()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(returns.values)
ax.plot(np.exp(trace[s]).T, 'r', alpha=.03);
ax.set(xlabel='time', ylabel='returns')
ax.legend(['S&P500', 'stoch vol']);

