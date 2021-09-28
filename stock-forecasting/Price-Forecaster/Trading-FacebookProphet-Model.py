#!/usr/bin/env python
# coding: utf-8

# # Using and Backtesting Facebook Prophet

# In[189]:


# Importing libraries
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import itertools
import pandas as pd
import numpy as np
import plotly.express as px
from statistics import mean, median
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import yfinance as yf


# ## Getting the Data

# In[35]:


# Getting the date five years ago to download the current timeframe
years = (date.today() - timedelta(weeks=260)).strftime("%Y-%m-%d")

# Stocks to analyze
stocks = ['GE', 'GPRO', 'FIT', 'F']

# Getting the data for multiple stocks
df = yf.download(stocks, start=years).dropna()

print("Rows in DataFrame: ", df.shape[0])


# In[36]:


# Storing the dataframes in a dictionary
stock_df = {}

for col in set(df.columns.get_level_values(0)):
    
    # Assigning the information (High, Low, etc.) for each stock in the dictionary
    stock_df[col] = df[col]


# # Preprocessing the Data

# In[37]:


# Finding the log returns
stock_df['LogReturns'] = stock_df['Adj Close'].apply(np.log).diff().dropna()

# Trying out Moving average
stock_df['MovAvg'] = stock_df['Adj Close'].rolling(10).mean().dropna()

# Logarithmic scaling of the data and rounding the result
stock_df['Log'] = stock_df['MovAvg'].apply(np.log).apply(lambda x: round(x, 2))


# # Visualizing the Data

# In[38]:


px.line(stock_df['MovAvg'], 
        x=stock_df['MovAvg'].index, 
        y=stock_df['MovAvg'].columns,
        labels={'variable': 'Stock',
                'value': 'Price'},
        title='Moving Average')


# In[39]:


px.line(stock_df['Log'], 
        x=stock_df['Log'].index, 
        y=stock_df['Log'].columns,
        labels={'variable': 'Stock',
                'value': 'Log Scale'},
        title='Log of Moving Averages')


# # Using FBProphet

# In[253]:


def fb_opt_param(data, cv_len=5):
    """
    Finds the best parameters for FBProphet
    
    Warning: Running this function will take a large amount of time
    """
    param_grid = {  
    'changepoint_prior_scale': [0.001, 0.05],
    'seasonality_prior_scale': [0.01, 10],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in tqdm(all_params):
        m = Prophet(**params,
                    daily_seasonality=True,
                    yearly_seasonality=False).fit(data)  # Fit model with given params
        df_cv = cross_validation(m, 
                                 initial=f'{len(data)} days',
                                 horizon=f'{cv_len} days', 
                                 parallel='processes')
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses

    return all_params[np.argmin(rmses)]


# Formatting the Data to fit to FBProphet's specifications

# In[254]:


proph_df = {}

for stock in stocks:
    
    # Creating a quick dictionary for the datafram
    d = {'ds': stock_df['MovAvg'][stock].index,
         'y': stock_df['MovAvg'][stock].values}
    
    # Creating the dataframe
    proph_df[stock] = pd.DataFrame(d)


# In[260]:


# Training amount of days
train_days = 50

# Forecasting amount
pred_ahead = 5

# Creating a new DF for the predictions
stock_df['Predictions'] = pd.DataFrame(index=stock_df['MovAvg'].index,
                                       columns=stock_df['MovAvg'].columns)


for stock in tqdm(stocks):
    
    # Current predicted value
    pred_val = 0
    
    # Training the model in a predetermined date range
    for day in tqdm(range(1100, 
                          stock_df['MovAvg'].shape[0]-pred_ahead)):        
        
        # Data to use, containing a specific amount of days
        training = proph_df[stock].iloc[day-train_days:day+1].dropna()
            
        # Determining if the actual value crossed the predicted value
        cross = ((training['y'].iloc[-1] >= pred_val >= training['y'].iloc[-2]) or 
                 (training['y'].iloc[-1] <= pred_val <= training['y'].iloc[-2]))
        
        # Running the model when the latest training value crosses the predicted value or every other day 
        if cross or day % 2 == 0:
            
            # Finding the optimum parameters
            #params = fb_opt_param(training, cv_len=pred_ahead)
            
            # Instantiating FBprophet
            m = Prophet(interval_width=.95,
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=False)

            # Fitting the model
            m.fit(training)
            
            # Forecasting prices and getting predictions
            forecast = m.make_future_dataframe(periods=pred_ahead)
                        
            predictions = m.predict(forecast)
            
            preds = predictions['yhat'].tail(pred_ahead)
            
            #display(preds)
                                    
            # Inserting the predicted values into our own DF
            stock_df['Predictions'][stock].iloc[day:day+pred_ahead] = mean(preds.values)
            
            # Updating the current predicted value
            pred_val = mean(preds.values)


# # Predictions vs Actual Values

# In[261]:


# Shift ahead by 1 to compare the actual values to the predictions
pred_df = stock_df['Predictions'].shift(1).astype(float).dropna()

pred_df


# ## Plotting the Predictions

# In[262]:


for stock in stocks:
    
    fig = go.Figure()
    
    # Plotting the actual values
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=stock_df['MovAvg'][stock].loc[pred_df.index],
                             name='Actual Moving Average',
                             mode='lines'))
    
    # Plotting the predicted values
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=pred_df[stock],
                             name='Predicted Moving Average',
                             mode='lines'))
    
    # Setting the labels
    fig.update_layout(title=f'Predicting the Moving Average for the Next {pred_ahead} days for {stock}',
                      xaxis_title='Date',
                      yaxis_title='Prices')
    
    fig.show()


# ## Evaluation Metric

# In[171]:


for stock in stocks:
    
    # Finding the root mean squared error
    rmse = mean_squared_error(stock_df['MovAvg'][stock].loc[pred_df.index],
                              pred_df[stock],
                              squared=False)

    print(f"On average, the model is off by {rmse} for {stock}\n")


# # Trading Signal

# In[172]:


def get_positions(difference, thres=3, short=True):
    """
    Compares the percentage difference between actual values and the respective predictions.
    
    Returns the decision or positions to long or short based on the difference.
    
    Optional: shorting in addition to buying
    """
    
    if difference > thres/100:
        
        return 1
    
    
    elif short and difference < -thres/100:
        
        return -1
    
    
    else:
        
        return 0


# ## Creating a Trading DF

# In[181]:


# Creating a DF for trading the model
trade_df = {}

# Getting the percentage difference between the predictions and the actual values
trade_df['PercentDiff'] = (stock_df['Predictions'].dropna() / 
                           stock_df['MovAvg'].loc[stock_df['Predictions'].dropna().index]) - 1

# Getting positions
trade_df['Positions'] = trade_df['PercentDiff'].applymap(lambda x: get_positions(x, 
                                                                                 thres=.5, 
                                                                                 short=True) / len(stocks))

# Preventing lookahead bias by shifting the positions
trade_df['Positions'] = trade_df['Positions'].shift(2).dropna()

# Getting Log Returns
trade_df['LogReturns'] = stock_df['LogReturns'].loc[trade_df['Positions'].index]                                    
    
display(trade_df['PercentDiff'].tail(20))
display(trade_df['Positions'].tail(20))


# ## Plotting Positions

# In[182]:


# Getting the number of positions
pos = trade_df['Positions'].apply(pd.value_counts)

# Plotting total positions
fig = px.bar(pos, 
             x=pos.index, 
             y=pos.columns,
             title='Total Positions',
             labels={'variable':'Stocks',
                      'value':'Count of Positions',
                      'index':'Type of Position'})

fig.show()


# # Calculating and Plotting Potential Returns

# ## Returns on Each Individual Stock

# In[183]:


# Calculating Returns by multiplying the positions by the log returns
returns = trade_df['Positions'] * trade_df['LogReturns']

# Calculating the performance as we take the cumulative sum of the returns and transform the values back to normal
performance = returns.cumsum().apply(np.exp)

# Plotting the performance per stock
px.line(performance,
        x=performance.index,
        y=performance.columns,
        title='Returns Per Stock Using ARIMA Forecast',
        labels={'variable':'Stocks',
                'value':'Returns'})


# ## Returns on Overall Portfolio

# In[184]:


# Returns for the portfolio
returns = (trade_df['Positions'] * trade_df['LogReturns']).sum(axis=1)

# Returns for SPY
spy = yf.download('SPY', start=returns.index[0]).loc[returns.index]

spy = spy['Adj Close'].apply(np.log).diff().dropna().cumsum().apply(np.exp)

# Calculating the performance as we take the cumulative sum of the returns and transform the values back to normal
performance = returns.cumsum().apply(np.exp)

# Plotting the comparison between SPY returns and ARIMA returns
fig = go.Figure()

fig.add_trace(go.Scatter(x=spy.index,
                         y=spy,
                         name='SPY Returns',
                         mode='lines'))

fig.add_trace(go.Scatter(x=performance.index,
                         y=performance.values,
                         name='Portfolio Returns',
                         mode='lines'))

fig.update_layout(title='SPY vs ARIMA Overall Portfolio Returns',
                  xaxis_title='Date',
                  yaxis_title='Returns')

fig.show()


# In[ ]:





# In[ ]:




