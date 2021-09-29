#!/usr/bin/env python
# coding: utf-8

# # Analysis of An $AR(1)$ Model in pyMC3

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


plt.style.use('seaborn-darkgrid')
np.random.seed(seed=42)


# Consider the following AR(1) process, initialized in the
# infinite past:
# $$
#    y_t = \theta y_{t-1} + \epsilon_t,
# $$
# where $\epsilon_t \sim iid{\cal N}(0,1)$.  Suppose you'd like to learn about $\theta$ from a a sample of observations $Y^T = \{ y_0, y_1,\ldots, y_T \}$.
# 
# First, let's generate our sample.

# In[3]:


T = 100
y = np.zeros((T,))

for i in range(1,T):
    y[i] = 0.95 * y[i-1] + np.random.normal()

plt.plot(y);


# Consider the following prior for $\theta$: $\theta \sim {\cal N}(0,\tau^2)$.
# We can show that the posterior distribution of $\theta$ is of the form
# 
# $$
#  \theta |Y^T \sim {\cal N}( \tilde{\theta}_T, \tilde{V}_T),
# $$
# 
# where
# 
# $$
# \begin{eqnarray}
#         \tilde{\theta}_T &=& \left( \sum_{t=1}^T y_{t-1}^2 + \tau^{-2} \right)^{-1} \sum_{t=1}^T y_{t}y_{t-1} \\
#         \tilde{V}_T      &=& \left( \sum_{t=1}^T y_{t-1}^2 + \tau^{-2} \right)^{-1}
# \end{eqnarray}
# $$

# In[4]:


tau = 1.0
with pm.Model() as ar1:
    beta = pm.Normal('beta', mu=0, sd=tau)
    data = pm.AR('y', beta, sd=1.0, observed=y)
    trace = pm.sample(1000, cores=4)
    
pm.traceplot(trace);


# In[5]:


mup = ((y[:-1]**2).sum() + tau**-2)**-1 * np.dot(y[:-1],y[1:])
Vp =  ((y[:-1]**2).sum() + tau**-2)**-1
print('Mean: {:5.3f} (exact = {:5.3f})'.format(trace['beta'].mean(), mup))
print('Std: {:5.3f} (exact = {:5.3f})'.format(trace['beta'].std(), np.sqrt(Vp)))


# In[6]:


import pandas as p
from scipy.stats import norm
ax=p.Series(trace['beta']).plot(kind='kde')
xgrid = np.linspace(0.4, 1.2, 1000)
fgrid = norm(loc=mup, scale=np.sqrt(Vp)).pdf(xgrid)
ax.plot(xgrid,fgrid);


# ## Extension to AR(p)
# We can instead estimate an AR(2) model using pyMC3.
# $$
#  y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \epsilon_t.
# $$
# The `AR` distribution infers the order of the process by size the of `rho` argmument passed to `AR`. 

# In[7]:


with pm.Model() as ar2:
    beta = pm.Normal('beta', mu=0, sd=tau, shape=2)
    data = pm.AR('y', beta, sd=1.0, observed=y)
    trace = pm.sample(1000, cores=4)
    
pm.traceplot(trace);


# You can also pass the set of AR parameters as a list. 

# In[8]:


with pm.Model() as ar2:
    beta = pm.Normal('beta', mu=0, sd=tau)
    beta2 = pm.Uniform('beta2')
    data = pm.AR('y', [beta, beta2], sd=1.0, observed=y)
    trace = pm.sample(1000, tune=1000, cores=4)

pm.traceplot(trace);

