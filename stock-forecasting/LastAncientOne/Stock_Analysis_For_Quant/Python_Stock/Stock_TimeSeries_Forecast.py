#!/usr/bin/env python
# coding: utf-8

# # Time Series Stock Forecast

# In[1]:


# load required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import fix_yahoo_finance as yf
from pandas_datareader import data as pdr


# In[2]:


stock = 'RAD'
start = '2015-01-01' 
end = '2017-12-08'
df = pdr.get_data_yahoo(stock, start, end)


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df['Adj Close'].plot(figsize=(12,8))
plt.show()


# In[6]:


df = df.reset_index().rename(columns={'Date':'ds', 'Adj Close':'y'})
#df['ds'] = pd.to_datetime(df.index)
#df['y'] = pd.DataFrame(df['Adj Close'])


# In[7]:


# Log Transform Data
df['y'] = pd.DataFrame(np.log(df['y']))

# plot data
ax = df['y'].plot(color='#006699');
ax.set_ylabel('Price');
ax.set_xlabel('Date');
plt.show()


# In[8]:


# train test split
df_train = df[:740]
df_test = df[740:]


# In[9]:


# Model Fitting
# instantiate the Prophet class
mdl = Prophet(interval_width=0.95, daily_seasonality=True)
 
# fit the model on the training data
mdl.fit(df_train)
 
# define future time frame
future = mdl.make_future_dataframe(periods=24, freq='MS')


# In[10]:


# instantiate the Prophet class
mdl = Prophet(interval_width=0.95, daily_seasonality=True)
 
# fit the model on the training data
mdl.fit(df_train)
 
# define future time frame
future = mdl.make_future_dataframe(periods=24, freq='MS')


# In[11]:


# generate the forecast
forecast = mdl.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[22]:


forecast['yhat_lower'].head()


# In[12]:


mdl.plot(forecast);
plt.show()


# In[13]:


# plot time series components
mdl.plot_components(forecast)
plt.show()


# In[14]:


import math
# retransform using e
y_hat = np.exp(forecast['yhat'][740:])
y_true = np.exp(df['y'])
 
# compute the mean square error
mse = ((y_hat - y_true) ** 2).mean()
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))


# In[26]:


plt.plot(y_true, label='Original', color='#006699');
plt.plot(y_hat, color='#ff0066', label='Forecast');
plt.ylabel('Price');
plt.xlabel('Date');
plt.show()

