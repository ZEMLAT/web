#!/usr/bin/env python
# coding: utf-8

# # Plot Different type of Charts for Trading

# ### Library Created by DanielGoldfarb
# https://github.com/matplotlib/mplfinance

# https://www.stockcharts.com/freecharts/
# 
# 

# In[1]:


import mplfinance as mpl

import warnings
warnings.filterwarnings("ignore")

# yahoo finance is used to fetch data 
import yfinance as yf
yf.pdr_override()


# In[2]:


symbol = 'AMD'
start = '2020-09-01'
end = '2020-11-03'


# In[3]:


df = yf.download(symbol,start,end)


# In[4]:


df.head()


# ## Bar Charts 

# https://school.stockcharts.com/doku.php?id=chart_analysis:what_are_charts

# In[5]:


mpl.plot(df)


# ## Line Charts

# https://school.stockcharts.com/doku.php?id=chart_analysis:what_are_charts

# In[6]:


mpl.plot(df, type='line')


# ## Candlestick Charts

# https://school.stockcharts.com/doku.php?id=chart_analysis:introduction_to_candlesticks

# In[7]:


mpl.plot(df,type='candle')


# ## Renko Charts

# https://school.stockcharts.com/doku.php?id=chart_analysis:renko

# In[8]:


mpl.plot(df,type='renko')


# ## Point and Figure Charts

# https://school.stockcharts.com/doku.php?id=chart_analysis:pnf_charts

# In[9]:


mpl.plot(df,type='pnf')

