#!/usr/bin/env python
# coding: utf-8

# # Modeling BTC with Technical Indicators

# ## Importing Necessary Libraries

# In[52]:


import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from tqdm import tqdm_notebook as tqdm
import _pickle as pickle
import datetime
plt.style.use('bmh')


# ## Loading the Data

# In[53]:


with open("df_indicators.pkl",'rb') as fp:
    df = pickle.load(fp)
    
df


# ### Shifting the Data
# Done so that the technical indicators would be seen in the "past"

# In[54]:


df = df[['Close']].join(df[df.columns[1:]].shift(-1)).dropna()
df


# ### Removing Rows
# Removing data points so the data reflects only the current market volatility from 2017 onwards

# In[55]:


df = df['2017':]


# ## Plotting Historical Prices

# In[56]:


df.Close.plot(figsize=(16,5))

plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.title('Price')
plt.show()


# ## Detrending

# ### Taking the Log then Plotting

# In[58]:


df['Close'] = df.Close.apply(np.log)

df.Close.plot(figsize=(16,5))

plt.xlabel('Date')
plt.ylabel('Log Price in USD')
plt.title('Log Price')
plt.show()


# ### Differencing the Data

# In[59]:


# Differencing the price
df_diff = df.Close.diff(1).dropna()

# Plotting the differences daily
df_diff.plot(figsize=(12,8))
plt.title('Plot of the Daily Changes in Price')
plt.ylabel('Change in USD')
plt.show()


# ### Testing for Stationarity

# In[60]:


results = adfuller(df_diff)
print(f"P-value: {results[1]}")


# ## PACF and ACF

# #### ACF and PACF for the Differencing

# In[61]:


fig, (ax1, ax2) = plt.subplots(2,1,figsize=(16,8))
plot_acf(df_diff, ax=ax1, lags=40)
plot_pacf(df_diff, ax=ax2, lags=40)
plt.show()


# Appears to be some correlation at day 5 and 10 mostly.

# ## Modeling

# ### Finding the Best Parameters for ARIMA

# In[62]:


def best_param(model, data, pdq, pdqs, exog=None):
    """
    Loops through each possible combo for pdq and pdqs
    Runs the model for each combo
    Retrieves the model with lowest AIC score
    """
    # Instantiating an empty list to append the combinations and AIC score
    ans = []
    
    # Iterating through all the different possible combinations
    for comb in tqdm(pdq):
        for combs in pdqs:
            
            # Running a model with different combinations
            try:
                mod = model(data,
                            order=comb,
                            exog=exog,
                            seasonal_order=combs,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            freq='D')

                output = mod.fit()
                
                # Appending the results to the empty list
                ans.append([comb, combs, output.aic])
            except:
                continue

    # Creating a DataFrame with the different combinations and respective scores            
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    
    # Returning the parameters with the lowest AIC score
    return ans_df.loc[ans_df.aic.idxmin()]


# In[81]:


# Assigning variables for p, d, q.
p = d = q = range(0,2)

# Creating a list of all possible combinations of p, d, and q.
pdq = list(itertools.product(p, d, q))

# Seasonality
s = [12]
pdqs = list(itertools.product(p, d, q, s))


# In[82]:


# Finding the best parameters
best_param(SARIMAX, df.Close, pdq, pdqs, exog=df[df.columns[1:]])


# #### Best Parameters according to the function

# In[83]:


pdq    =     (0, 1, 1)
pdqs  =  (0, 0, 0, 12)
# aic          -7380.95


# ### Fitting and Training SARIMAX

# #### Train, test, split
# To get a more accurate depiction and to be fair to the predictive power of the model, we will be using a smaller size than the usual 80/20 split

# In[84]:


# Splitting 95/5
index = round(len(df)*.95)

train = df.iloc[:index]
test  = df.iloc[index:]


# In[85]:


# Fitting the model to the training set
model = SARIMAX(train.Close, 
                order=pdq, 
                exog=train[train.columns[1:]],
                seasonal_order=pdqs, 
                freq='D', 
                enforce_stationarity=False, 
                enforce_invertibility=False)
output = model.fit()


# ### Summary and Diagnostics from fitting the model

# In[86]:


print(output.summary())
output.plot_diagnostics(figsize=(15,8))
plt.show()


# ### Creating Future Exogenous Variables
# Using the same model and same parameters as before

# In[87]:


def create_exog(df, start, end, pdq=(0,0,0), pdqs=(0,0,0,0)):
    """
    Creates future exogenous variables using parameter from the same model used 
    """
    print('Creating future exogenous variables...')
    
    # Instantiating a new DF containing the forecasted technical indicators
    future_indicators = pd.DataFrame(columns=df.columns[1:])

    # Iterating through each tech indicator and running a time series model    
    for i in tqdm(df.columns):
        tech_models = SARIMAX(df[i], 
                              order=pdq, 
                              seasonal_order=pdqs, 
                              freq='D', 
                              enforce_stationarity=False, 
                              enforce_invertibility=False)
        tech_output = tech_models.fit()

        # Forecasting the future exogenous variables
        tech_future = tech_output.predict(start=start, end=end)

        # Assigning the values to the respective columns
        future_indicators[i] = tech_future
        
    return future_indicators


# ### Predictions with SARIMAX

# ### Assigning Variables for Plotting

# In[88]:


# Creating test exog variables for the test set
test_exog = create_exog(train[train.columns[1:]], 
                        pdq=pdq, 
                        pdqs=pdqs, 
                        start=test.index[0], 
                        end=test.index[-1])


# In[89]:


# Values to test against the test set
fc   = output.get_prediction(start=test.index[0], end=test.index[-1], exog=test_exog)
conf = fc.conf_int()

# Assigning the values as a series
fc_series    = np.exp(pd.Series(fc.predicted_mean, index=test.index))
lower_series = np.exp(pd.Series(conf.iloc[:, 0], index=test.index))
upper_series = np.exp(pd.Series(conf.iloc[:, 1], index=test.index))

train = np.exp(train)
test  = np.exp(test)

# Values to test against the train set, see how the model fits
predictions = output.get_prediction(start=pd.to_datetime('2018'), end=train.index[-1], dynamic=False)
pred        = np.exp(predictions.predicted_mean)

# Confidence interval for the training set
conf_int   = predictions.conf_int()
low_conf   = pd.Series(conf_int.iloc[:,0], index=train.index)
upper_conf = pd.Series(conf_int.iloc[:,1], index=train.index)


# In[91]:


fc_series


# ### Plotting the Fitted Model and Testing against the Test set

# In[90]:


rcParams['figure.figsize'] = 16, 8

# Plotting the training set, test set,forecast, and confidence interval.
plt.plot(train.Close, label='train')
plt.plot(test.Close, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

# Plotting against the training data
pred.plot(label='Fit to Training', color='orange')

# Confidence interval for the fitted data
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color='g',alpha=.5)

# Limiting the viewing size
plt.xlim(['2018-01', '2020-05'])
plt.ylim([0, 20000])

plt.title('Fit to Train Data and \nForecasting vs Actual Test Values')
plt.legend()
plt.show()


# ### Calculating the RMSE for SARIMA

# In[92]:


forecast = pred
actual_val = train.Close

# Calculating our errors
rmse = np.sqrt(((forecast - actual_val) ** 2).mean())

print("The Root Mean Squared Error: ", rmse)


# ### Forecasting Future Values

# #### Fitting the model to the entire dataset

# In[93]:


model = SARIMAX(df.Close, 
                order=pdq, 
                exog=df[df.columns[1:]],
                seasonal_order=pdqs, 
                freq='D', 
                enforce_stationarity=False, 
                enforce_invertibility=False)
output = model.fit()


# In[94]:


# Creating future exog variables
future_exog = create_exog(df[df.columns[1:]], start=df.index[-1], end=datetime.timedelta(29)+df.index[-1], pdq=pdq, pdqs=pdqs)


# In[95]:


# Getting the forecast of future values
future = output.get_prediction(start=df.index[-1], 
                               end=datetime.timedelta(30)+df.index[-1], 
                               exog=future_exog)

# Transforming values back
pred_fut = future.predicted_mean

# Confidence interval for our forecasted values
pred_conf = future.conf_int()

# Transforming value back
pred_conf = pred_conf


# ### Plotting the forecasted values

# In[96]:


# Plotting the prices up to the most recent
ax = df.Close.plot(label='Actual', figsize=(16,8))

# Plottting the forecast
pred_fut.plot(ax=ax, label='Future Vals')

# Shading in the confidence interval
ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Price')
ax.set_xlim(['2018-01', '2020-05'])

plt.title('Forecasted values')
plt.legend()
plt.show()


# ### Zooming in on the Graph above

# In[97]:


ax = df.Close.plot(label='Actual', figsize=(16,8))
pred_fut.plot(ax=ax, label='Future Vals')

ax.fill_between(pred_conf.index,
                pred_conf.iloc[:, 0],
                pred_conf.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Price')
ax.set_xlim(['2019-11','2020-05'])

plt.title('Forecasted values \nZoomed in')
plt.legend()
plt.show()


# In[ ]:




