#!/usr/bin/env python
# coding: utf-8

# # Download and manipulate data

# This notebook contains information on downloading the Quandl Wiki stock prices and a few other sources that we use throughout the book. 

# ## Imports & Settings

# In[2]:


from pathlib import Path
import numpy as np
import pandas as pd
import pandas_datareader.data as web

pd.set_option('display.expand_frame_repr', False)


# ## Set Data Store path

# Modify path if you would like to store the data elsewhere and change the notebooks accordingly

# In[4]:


DATA_STORE = Path('assets.h5')


# ## Quandl Wiki Prices

# [Quandl](https://www.quandl.com/) makes available a [dataset](https://www.quandl.com/databases/WIKIP/documentation) with stock prices, dividends and splits for 3000 US publicly-traded companies. Quandl decided to discontinue support in favor of its commercial offerings but the historical data are still useful to demonstrate the application of the machine learning solutions in the book, just ensure you implement your own algorithms on current data.
# 
# > As of April 11, 2018 this data feed is no longer actively supported by the Quandl community. We will continue to host this data feed on Quandl, but we do not recommend using it for investment or analysis.

# 1. Follow the instructions to create a free [Quandl]([Quandl](https://www.quandl.com/)) account
# 2. [Download](https://www.quandl.com/databases/WIKIP/usage/export) the entire WIKI/PRICES data
# 3. Extract the .zip file,
# 4. Move to this directory and rename to wiki_prices.csv
# 5. Run the below code to store in fast HDF format (see [Chapter 02 on Market & Fundamental Data](../02_market_and_fundamental_data) for details).

# In[ ]:


df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)


# ### Wiki Prices Metadata

# > Quandl no longer makes the metadata available. I've added the `wiki_stocks.csv` file to this directory so you can proceed directly with the next code cell and load the file.

# 1. Follow the instructions to create a free [Quandl]([Quandl](https://www.quandl.com/)) account if you haven't done so yet
# 2. Find link to download wiki metadata under Companies](https://www.quandl.com/databases/WIKIP/documentation) or use the download link with your API_KEY: https://www.quandl.com/api/v3/databases/WIKI/metadata?api_key=<API_KEY>
# 3. Extract the .zip file,
# 4. Move to this directory and rename to wiki_stocks.csv
# 5. Run the following code to store in fast HDF format

# In[7]:


meta_data = pd.read_csv('wiki_stocks.csv')
meta_data = pd.concat([meta_data.loc[:, 'code'].str.strip(),
                meta_data.loc[:, 'name'].str.split('(', expand=True)[0].str.strip().to_frame('name')], axis=1)
meta_data.info()


# In[8]:


meta_data.head()


# In[ ]:


with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/stocks', meta_data)


# ## S&P 500 Prices

# The following code downloads historical S&P 500 prices from FRED (only last 10 years of daily data is freely available)

# In[ ]:


df = web.DataReader(name='SP500', data_source='fred', start=2008)
print(df.info())
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/prices', df)


# ### S&P 500 Constituents

# The following code downloads the current S&P 500 constituents from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).

# In[5]:


url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url, header=0)[0]
df.columns = ['name', 'ticker', 'sec_filings', 'gics_sector', 'gics_sub_industry',
              'location', 'first_added', 'cik', 'founded']
df = df.drop('sec_filings', axis=1).set_index('ticker')
print(df.info())
with pd.HDFStore(DATA_STORE) as store:
    store.put('sp500/stocks', df)


# ## Metadata on US-traded companies

# The following downloads several attributes for [companies](https://www.nasdaq.com/screening/companies-by-name.aspx) traded on NASDAQ, AMEX and NYSE

# In[44]:


url = 'https://www.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange={}&render=download'
exchanges = ['NASDAQ', 'AMEX', 'NYSE']
df = pd.concat([pd.read_csv(url.format(ex)) for ex in exchanges]).dropna(how='all', axis=1)
df = df.rename(columns=str.lower).set_index('symbol').drop('summary quote', axis=1)
df = df[~df.index.duplicated()]
print(df.info()) 


# In[45]:


df.head()


# ### Convert market cap information to numerical format

# Market cap is provided as strings so we need to convert it to numerical format.

# In[46]:


mcap = df[['marketcap']].dropna()
mcap['suffix'] = mcap.marketcap.str[-1]
mcap.suffix.value_counts()


# Keep only values with value units:

# In[47]:


mcap = mcap[mcap.suffix.str.endswith(('B', 'M'))]
mcap.marketcap = pd.to_numeric(mcap.marketcap.str[1:-1])
mcaps = {'M': 1e6, 'B': 1e9}
for symbol, factor in mcaps.items():
    mcap.loc[mcap.suffix == symbol, 'marketcap'] *= factor
mcap.info()


# In[49]:


df['marketcap'] = mcap.marketcap
df.marketcap.describe(percentiles=np.arange(.1, 1, .1).round(1)).apply(lambda x: f'{int(x):,d}')


# ### Store result

# In[28]:


with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df)


# ## Bond Price Indexes

# The following code downloads several bond indexes from the Federal Reserve Economic Data service ([FRED](https://fred.stlouisfed.org/))

# In[ ]:


securities = {'BAMLCC0A0CMTRIV'   : 'US Corp Master TRI',
              'BAMLHYH0A0HYM2TRIV': 'US High Yield TRI',
              'BAMLEMCBPITRIV'    : 'Emerging Markets Corporate Plus TRI',
              'GOLDAMGBD228NLBM'  : 'Gold (London, USD)',
              'DGS10'             : '10-Year Treasury CMR',
              }

df = web.DataReader(name=list(securities.keys()), data_source='fred', start=2000)
df = df.rename(columns=securities).dropna(how='all').resample('B').mean()

with pd.HDFStore(DATA_STORE) as store:
    store.put('fred/assets', df)

