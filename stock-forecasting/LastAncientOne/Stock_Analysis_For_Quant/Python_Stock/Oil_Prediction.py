#!/usr/bin/env python
# coding: utf-8

# # Oil Price Prediction Using Machine Learning

# In[1]:


# LinearRegression is a machine learning library for linear regression 
from sklearn.linear_model import LinearRegression 

# pandas and numpy are used for data manipulation 
import pandas as pd 
import numpy as np 

# matplotlib and seaborn are used for plotting graphs 
import matplotlib.pyplot as plt 
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


# Read data 
Df = yf.download('USO','2016-01-01','2018-09-10')

# Only keep close columns 
Df=Df[['Close']] 

# Drop rows with missing values 
Df= Df.dropna() 

# Plot the closing price of GLD 
Df.Close.plot(figsize=(10,5)) 
plt.ylabel("OIL ETF Prices")
plt.show()


# In[3]:


# Define explanatory variables
Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean() 
Df['S_9']= Df['Close'].shift(1).rolling(window=9).mean() 
Df= Df.dropna() 
X = Df[['S_3','S_9']] 
X.head()


# In[4]:


# Define dependent variable
y = Df['Close']
y.head()


# In[5]:


# Split the data into train and test dataset
t=.8 
t = int(t*len(Df)) 

# Train dataset 
X_train = X[:t] 
y_train = y[:t]  

# Test dataset 
X_test = X[t:] 
y_test = y[t:]


# In[6]:


# Create a linear regression model
# Y = m1 * X1 + m2 * X2 + C
# Gold ETF price = m1 * 3 days moving average + m2 * 15 days moving average + c
linear = LinearRegression().fit(X_train,y_train) 
print("MJ ETF Price =", round(linear.coef_[0],2), "* 3 Days Moving Average", round(linear.coef_[1],2), "* 9 Days Moving Average +", round(linear.intercept_,2))


# In[7]:


# Predicting the Oil ETF prices
predicted_price = linear.predict(X_test)  
predicted_price = pd.DataFrame(predicted_price,index=y_test.index,columns = ['price'])  
predicted_price.plot(figsize=(10,5))  
y_test.plot()  
plt.legend(['predicted_price','actual_price'])  
plt.ylabel("USO ETF Price")  
plt.show()


# In[8]:


r2_score = linear.score(X[t:],y[t:])*100  
float("{0:.2f}".format(r2_score))


# Oil Stock

# In[9]:


Oil_stock = ['PBR', 'VALE', 'RIG', 'WLL']


# In[10]:


start = '2016-01-01'
end = '2018-09-10'
df = yf.download(Oil_stock,start,end)


# In[11]:


stocks = pd.DataFrame(df['Adj Close'])
stocks.head()


# In[12]:


plt.figure(figsize=(16,8))
plt.plot(stocks)
plt.title('Oil Stock Adj Close')
plt.legend(stocks)
plt.grid()
plt.ylabel('Price')
plt.xlabel('Date')
plt.show()


# In[13]:


stocks.plot(grid = True, figsize=(12,8))


# In[14]:


stock_return = stocks.apply(lambda x: x / x[0])
stock_return.head()


# In[15]:


stock_return.plot(grid = True, figsize=(12,8)).axhline(y = 1, color = "black", lw = 2)
plt.axhline(y=4, color = 'red', lw=2)


# In[16]:


stock_return = stocks.pct_change(1).dropna()
stock_return.head()


# In[17]:


stock_return.tail()


# In[18]:


stock_return.plot(grid = True, figsize=(12,8)).axhline(y = 1, color = "black", lw = 2)


# In[19]:


stock_change = stocks.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()


# In[20]:


stock_change.plot(grid = True, figsize=(12,8)).axhline(y = 0, color = "black", lw = 2)


# In[21]:


sns.pairplot(stock_change[1:])


# In[22]:


stock_change.idxmin()


# In[23]:


stock_change.idxmax()


# In[24]:


stock_change.std()


# In[25]:


# Sharpe Ratio for Each Stocks
N = 252
returns = stocks.pct_change().dropna()
annualised_sharpe = np.sqrt(N) * returns.mean() / returns.std()
annualised_sharpe


# In[26]:


annualised_sharpe.index


# In[27]:


annualised_sharpe.sort_values()


# In[28]:


annualised_sharpe.sort_index()


# In[29]:


# Equity Sharpe - Buy and Hold
N = 252 # Number of trading in a year
risk = 0.01
excess_daily_ret =  returns - (risk * N)
equity_sharpe = np.sqrt(N) * excess_daily_ret.mean() / excess_daily_ret.std()
equity_sharpe.sort_values()


# In[30]:


# Market Neutral Sharpe
start = '2016-01-01'
end = '2018-09-10'
market = 'SPY'
ticker = ['PBR', 'VALE', 'RIG', 'WLL']
bench = yf.download(market,start,end)
stocks = yf.download(ticker,start,end)


# In[31]:


tick = pd.DataFrame(stocks['Adj Close'])
tick.head()


# In[32]:


daily_rets = tick.pct_change().dropna()
daily_rets.head()


# In[33]:


bench_rets = bench['Adj Close'].pct_change().dropna()
bench_rets.head()


# In[34]:


strat = (daily_rets.sub(bench_rets, axis=0))/2
strat.head()


# In[35]:


N = 252
market_neutral_sharpe = np.sqrt(N) * strat.mean() / strat.std()
market_neutral_sharpe.sort_values()

