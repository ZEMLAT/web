#!/usr/bin/env python
# coding: utf-8

# # BTC - Facebook Prophet

# ### Importing libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
plt.style.use('ggplot')
import _pickle as pickle

from fbprophet import Prophet as proph


# ### Loading in the Data

# In[2]:


with open("curr_bitcoin.pickle",'rb') as fp:
    ts = pickle.load(fp)


# ### Formatting the data for Facebook Prophet

# In[3]:


# Resetting the index back so Dates are no longer indexed
ts.reset_index(inplace=True)

# Renaming the columns for use in FB prophet
ts.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

ts.head()


# #### Plotting

# In[4]:


# Plotting the price 
pd.plotting.register_matplotlib_converters()

ax = ts.set_index('ds').plot(figsize=(16,8))
ax.set_ylabel('Bitcoin Price')
ax.set_xlabel('Date')

plt.show()


# ### Modeling

# In[5]:


# Fitting and training
mod = proph(interval_width=0.95)
mod.fit(ts)


# #### Creating future dates to forecast

# In[6]:


# Setting up predictions to be made
future = mod.make_future_dataframe(periods=30, freq='D')
future.tail()


# #### Forecasting future values

# In[7]:


# Making predictions
forecast = mod.predict(future)
forecast.tail()


# ### Plotting Values
# * Blue line = forecasted values
# * Black dots = observed values
# * Uncertainty intervals = blue shaded region

# In[17]:


mod.plot(forecast, uncertainty=True)
plt.title('Facebook Prophet Forecast and Fitting')
plt.savefig('fb_fc_fit.png')
plt.show()


# #### Graph above zoomed in

# In[13]:


mod.plot(forecast, uncertainty=True)

plt.xlim(['2019-04-01', '2019-10-10'])
plt.ylim([5000, 15000])
plt.savefig('fb_zoom.png')
plt.show()


# #### Plotted components of the forecasts

# In[10]:


mod.plot_components(forecast)
plt.show()


# In[ ]:




