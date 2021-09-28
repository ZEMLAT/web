#!/usr/bin/env python
# coding: utf-8

# # AutoARIMA on Stock Prices

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
from pmdarima.arima import AutoARIMA
import plotly.express as px
from statistics import mean
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import yfinance as yf


# Choosing Stocks that have significantly lost value in the past few years

# In[2]:


# Getting the date five years ago to download the current timeframe
years = (date.today() - timedelta(weeks=260)).strftime("%Y-%m-%d")

# Stocks to analyze
stocks = ['GE', 'GPRO', 'FIT', 'F']

# Getting the data for multiple stocks
df = yf.download(stocks, start=years).dropna()

print("Rows in DataFrame: ", df.shape[0])


# In[3]:


# Storing the dataframes in a dictionary
stock_df = {}

for col in set(df.columns.get_level_values(0)):
    
    # Assigning the information (High, Low, etc.) for each stock in the dictionary
    stock_df[col] = df[col]


# # Preprocessing Data

# Scale the data using a logarithmic scale.  Also rounding the log result by 2 decimal points in order to reduce any unnecessary noise.

# In[4]:


# Finding the log returns
stock_df['LogReturns'] = stock_df['Adj Close'].apply(np.log).diff().dropna()

# Trying out Moving average
stock_df['MovAvg'] = stock_df['Adj Close'].rolling(10).mean().dropna()

# Logarithmic scaling of the data and rounding the result
stock_df['Log'] = stock_df['MovAvg'].apply(np.log).apply(lambda x: round(x, 2))


# # Visualizing the Data

# In[5]:


px.line(stock_df['MovAvg'], 
        x=stock_df['MovAvg'].index, 
        y=stock_df['MovAvg'].columns,
        labels={'variable': 'Stock',
                'value': 'Price'},
        title='Moving Average')


# In[6]:


px.line(stock_df['Log'], 
        x=stock_df['Log'].index, 
        y=stock_df['Log'].columns,
        labels={'variable': 'Stock',
                'value': 'Log Scale'},
        title='Log of Moving Averages')


# ## Optimum Parameter Search Function

# In[7]:


opt_param = AutoARIMA(start_p=0, start_q=0,
                      start_P=0, start_Q=0,
                      max_p=8, max_q=8,
                      max_P=5, max_Q=5,
                      error_action='ignore',
                      information_criterion='bic',
                      suppress_warnings=True)

for stock in tqdm(stocks):

    opt_param.fit(stock_df['Log'][stock])

    print(f'Summary for {stock}', '--'*20)
    display(opt_param.summary())


# # Using the ARIMA Model
# Using the price history from the past N days to make predictions

# In[76]:


# Days in the past to train on
days_to_train = 180 

# Days in the future to predict
days_to_predict = 5

# Establishing a new DF for predictions
stock_df['Predictions'] = pd.DataFrame(index=stock_df['Log'].index,
                                       columns=stock_df['Log'].columns)

# Iterate through each stock
for stock in tqdm(stocks):
    
    # Current predicted value
    pred_val = 0
    
    # Training the model in a predetermined date range
    for day in tqdm(range(1000, 
                          stock_df['Log'].shape[0]-days_to_predict)):        

        # Data to use, containing a specific amount of days
        training = stock_df['Log'][stock].iloc[day-days_to_train:day+1].dropna()
        
        # Determining if the actual value crossed the predicted value
        cross = ((training[-1] >= pred_val >= training[-2]) or 
                 (training[-1] <= pred_val <= training[-2]))
        
        # Running the model when the latest training value crosses the predicted value or every other day 
        if cross or day % 2 == 0:

            # Finding the best parameters
            model    = AutoARIMA(start_p=0, start_q=0,
                                 start_P=0, start_Q=0,
                                 max_p=8, max_q=8,
                                 max_P=5, max_Q=5,
                                 error_action='ignore',
                                 information_criterion='bic',
                                 suppress_warnings=True)

            # Getting predictions for the optimum parameters by fitting to the training set            
            forecast = model.fit_predict(training,
                                         n_periods=days_to_predict)

            # Getting the last predicted value from the next N days
            stock_df['Predictions'][stock].iloc[day:day+days_to_predict] = np.exp(forecast[-1])


            # Updating the current predicted value
            pred_val = forecast[-1]


# # Predictions vs Actual Values

# In[77]:


# Shift ahead by 1 to compare the actual values to the predictions
pred_df = stock_df['Predictions'].shift(1).astype(float).dropna()

pred_df


# ## Plotting the Predictions
# Comparing the actual values with the predictions

# In[78]:


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
    fig.update_layout(title=f'Predicting the Moving Average for the Next {days_to_predict} days for {stock}',
                      xaxis_title='Date',
                      yaxis_title='Prices')
    
    fig.show()


# ## Evaluation Metric

# In[79]:


for stock in stocks:
    
    # Finding the root mean squared error
    rmse = mean_squared_error(stock_df['MovAvg'][stock].loc[pred_df.index],
                              pred_df[stock],
                              squared=False)

    print(f"On average, the model is off by {rmse} for {stock}\n")


# # Trading Signal
# Turning the model into a Trading Signal

# In[80]:


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


# ### Creating a Trading DF
# __Note:__ _On Preventing Lookahead Bias_
# 
# For example, if the model is ran after hours and a position is established on the next day's opening, then a shift ahead of 1 is ok.  But if a position is established on the next day, near the close, then it needs to be shifted ahead by 2, because the newly established position missed any gains or losses that day.  These are due to the fact that gains or losses in the day are determined when a trade is entered.
# 
# (This can also determine how long the predicted forecast remains valid.)

# In[81]:


# Creating a DF for trading the model
trade_df = {}

# Getting the percentage difference between the predictions and the actual values
trade_df['PercentDiff'] = (stock_df['Predictions'].dropna() / 
                           stock_df['MovAvg'].loc[stock_df['Predictions'].dropna().index]) - 1

# Getting positions
trade_df['Positions'] = trade_df['PercentDiff'].applymap(lambda x: get_positions(x, 
                                                                                 thres=1, 
                                                                                 short=True) / len(stocks))

# Preventing lookahead bias by shifting the positions
trade_df['Positions'] = trade_df['Positions'].shift(2).dropna()

# Getting Log Returns
trade_df['LogReturns'] = stock_df['LogReturns'].loc[trade_df['Positions'].index]                                    
    
display(trade_df['PercentDiff'].tail(20))
display(trade_df['Positions'].tail(20))


# ## Plotting the Positions

# In[87]:


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


# # Calculating and Plotting the Potential Returns

# ## Returns on Each Individual Stock

# In[83]:


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


# ## Returns on the Overall Portfolio

# In[86]:


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




