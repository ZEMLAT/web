#!/usr/bin/env python
# coding: utf-8

# # Using and Backtesting the GARCH Model

# In[3]:


# Importing libraries
from arch import arch_model
import pandas as pd
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from statistics import mean
from datetime import date, timedelta
import yfinance as yf


# In[4]:


# Getting the date five years ago to download the current timeframe
years = (date.today() - timedelta(weeks=104)).strftime("%Y-%m-%d")

# Stocks to analyze
stocks = ['GE', 'GPRO', 'FIT', 'F']

# Getting the data for multiple stocks
df = yf.download(stocks, start=years)

print('Number of Rows: ', df.shape[0])


# In[5]:


# Storing the dataframes in a dictionary
stock_df = {}

for col in set(df.columns.get_level_values(0)):
    
    # Assigning the information (High, Low, etc.) for each stock in the dictionary
    stock_df[col] = df[col]


# # Differencing the Data
# The closing prices only

# In[6]:


stock_df['Diff'] = stock_df['Close'].pct_change(1).dropna() * 100

stock_df['Diff']


# ## Plotting the Volatility

# In[7]:


for stock in stock_df['Diff'].columns:
    
    fig = px.line(stock_df['Diff'],
                  x=stock_df['Diff'].index,
                  y=stock_df['Diff'][stock],
                  title=f'Daily Return Percentage for {stock}')
    
    fig.show()


# # Optimum Model Parameters

# In[8]:


def param_search(model, data, order):
    """
    Loops through each iteration of the order combinations of the model and returns the best performing parameter
    with the lowest AIC score
    """
    
    # Empty list containing the combination and AIC score
    lst = []
    
    # Loop to find the best combination
    for comb in order:
        try:
            # Instantiating the model
            mod = model(data,
                        p=comb[0],
                        o=comb[1],
                        q=comb[2])
            
            # Fitting the model
            output = mod.fit()
            
            # Appending to the list
            lst.append([comb, output.aic])
        
        except:
            continue
            
    # Sorting the list
    lst = sorted(lst, key=lambda i: i[1])
    
    # Returning the combination with the lowest score
    return lst[0][0]


# In[9]:


## For the param_search function
# Assigning variables to test out
p = o = q = range(0,5)

# Finding all possible combinations of p and q
poq = list(itertools.product(p, o, q))


# # Testing out the Model

# In[10]:


# Train test split
# 80/20 Split
split = round(stock_df['Diff'].shape[0]*.95)

train = stock_df['Diff']['F'].iloc[:split]
test = stock_df['Diff']['F'].iloc[split:]

# Optimizing parameters
best_param = param_search(arch_model, train, poq)

# Using the best parameters
model = arch_model(train, 
                   p=best_param[0],
                   o=best_param[1],
                   q=best_param[2])

output = model.fit()

# Getting the predictions
predictions = output.forecast(horizon=test.shape[0])


# In[11]:


garch_df = pd.DataFrame(index=test.index)

garch_df['Actual'] = test

garch_df['Predicted'] = np.sqrt(predictions.variance.values[-1,:])

px.line(garch_df, x=garch_df.index, y=['Actual', 'Predicted'])


# # Using the Model

# In[49]:


pred_ahead = 5

days_to_train = 50 

# Establishing new DF for predictions
stock_df['Predictions'] = pd.DataFrame(index=stock_df['Diff'].index, 
                                       columns=stock_df['Diff'].columns)

for stock in tqdm(stocks):
    
    for day in tqdm(range(250, 
                          stock_df['Diff'].shape[0]-pred_ahead, 
                          pred_ahead)):
        
        # The training data to use
        training = stock_df['Diff'][stock].iloc[day-days_to_train:day+1]
        
        # Optimizing parameters
        best_param = param_search(arch_model, training, poq)
        
        # Using the best parameters
        model = arch_model(training, 
                           p=best_param[0],
                           o=best_param[1],
                           q=best_param[2])
        
        output = model.fit()
        
        # Getting the predictions
        predictions = output.forecast(horizon=pred_ahead)
        
        # Getting the average volatility for the next N days
        stock_df['Predictions'][stock].iloc[day:day+pred_ahead] = np.sqrt(predictions.variance.values[-1,:])[-1]
        


# In[50]:


stock_df['Predictions'] = stock_df['Predictions'].astype(float)


# # Predicted Average Values vs Actual Values

# In[51]:


# Dropping the Nans
pred_df = stock_df['Predictions'].astype(float).shift(1).dropna()

# Plotting for each stock
for stock in stocks:
    
    # Plotting the volatility and comparing the results
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=pred_df[stock],
                             name='Average Prediction',
                             mode='lines'))
    
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=stock_df['Diff'][stock].tail(len(pred_df)),
                             name='Actual',
                             mode='lines'))
    
    fig.update_layout(title=f'{pred_ahead} Day Average Volatility Comparison for {stock}',
                      xaxis_title='Date',
                      yaxis_title='% Change')
    
    fig.show()


# # Trading Using Predicted Volatility

# In[52]:


def position_decision(num, thres=2, short=True):
    """
    If the forecasted volatility is low, then we buy.
    Optional: if it's is high, then we short.
    Otherwise, we don't do anything.
    """
    
    if short and (num > thres or num < -thres):
        return -1
    
    elif num < thres and num > -thres:
        return 1
    
    else:
        return 0
    


# ## DF for Trading

# In[85]:


trade_df = {}

# Establishing the DF for the volatility predictions 
vol_pred = stock_df['Predictions'].astype(float).dropna()

# Scaling the data in the predictions
scaler = StandardScaler()

vol_pred = pd.DataFrame(scaler.fit_transform(vol_pred), 
                        index=vol_pred.index, 
                        columns=vol_pred.columns)

# Getting positions
trade_df['Positions'] = vol_pred.applymap(lambda x: position_decision(x, 
                                                                      thres=.5, 
                                                                      short=True) / len(stocks))

# Preventing lookahead bias
trade_df['Positions'] = trade_df['Positions'].shift(2).dropna()
    
# Getting the log returns
trade_df['LogReturns'] = stock_df['Close'].loc[trade_df['Positions'].index].apply(np.log).diff().dropna()
    
display(vol_pred)

display(trade_df['LogReturns'])

display(trade_df['Positions'])


# ## Plotting the Positions

# In[86]:


pos = trade_df['Positions'].apply(pd.value_counts)

# Plotting total positions
fig = px.bar(pos, 
             x=pos.index, 
             y=pos.columns,
             title='Total Positions',
             labels={'variable':'Stocks',
                      'value':'Count of Positions',
                      'index':'Position'})

fig.show()


# # Calculating and Plotting Returns

# ## Returns for each Stock

# In[87]:


# Returns
returns = trade_df['Positions'] * trade_df['LogReturns']

# Calculating the performance as we take the cumulative sum of the returns and transform the values back to normal
performance = returns.cumsum().apply(np.exp)

# Plotting the performance per stock
px.line(performance,
        x=performance.index,
        y=performance.columns,
        title='Returns Using GARCH',
        labels={'variable':'Stocks',
                'value':'Returns'})


# ## Returns for the Overall Portfolio vs SPY

# In[88]:


# Returns for the portfolio
returns = (trade_df['Positions'] * trade_df['LogReturns']).sum(axis=1)

# Returns for SPY
spy = yf.download('SPY', start=returns.index[0])

spy = spy['Close'].apply(np.log).diff().dropna().cumsum().apply(np.exp)

# Calculating the performance as we take the cumulative sum of the returns and transform the values back to normal
performance = returns.cumsum().apply(np.exp)


# Plotting the comparison between SPY returns and GARCH returns
fig = go.Figure()

fig.add_trace(go.Scatter(x=spy.index,
                         y=spy,
                         name='SPY Returns',
                         mode='lines'))

fig.add_trace(go.Scatter(x=performance.index,
                         y=performance.values,
                         name='GARCH Returns on Portfolio',
                         mode='lines'))

fig.update_layout(title='SPY vs GARCH Overall Portfolio Returns',
                  xaxis_title='Date',
                  yaxis_title='Returns')

fig.show()


# In[ ]:





# In[ ]:




