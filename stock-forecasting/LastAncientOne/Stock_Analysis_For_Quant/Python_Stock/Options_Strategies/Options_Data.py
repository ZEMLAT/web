#!/usr/bin/env python
# coding: utf-8

# # Download Options Data

# ## yfinance

# In[ ]:


import yfinance as yf


# In[ ]:


aapl = yf.Ticker("aapl")


# In[ ]:


aapl.options


# In[ ]:


option_exp = aapl.option_chain('2020-03-26')


# In[ ]:


option_exp.calls


# In[ ]:


option_exp.puts


# # yahooquery

# In[ ]:


from yahooquery import Ticker


# In[ ]:


aapl = Ticker('aapl')


# In[ ]:


import pandas as pd
df = aapl.option_chain


# In[ ]:


df.index.names


# In[ ]:


df


# In[ ]:


df.loc['aapl']


# In[ ]:


df.loc['aapl', '2020-04-17']


# In[ ]:


df.loc['aapl', '2020-04-17', 'calls']


# In[ ]:


df.loc[df['inTheMoney'] == True]


# In[ ]:


df.loc[df['inTheMoney'] == True].xs('aapl')


# # wallstreet

# In[ ]:


from wallstreet import Stock, Call, Put


# In[ ]:


a = Call('AAPL', d=12, m=2, y=2020, strike=300)


# In[ ]:


a.price


# In[ ]:


a.implied_volatility()


# In[ ]:


a.delta()


# In[ ]:


a.vega()


# In[ ]:


a.underlying.price


# # yahoo_fin

# ## for python 3.6+

# In[ ]:


from yahoo_fin import options


# In[ ]:


aapl_dates = options.get_expiration_dates("aapl")


# In[ ]:


chain = options.get_options_chain("aapl")


# In[ ]:


chain["calls"]


# In[ ]:


chain["puts"]

