#!/usr/bin/env python
# coding: utf-8

# # Factor Analysis Portfolio

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore")

# fix_yahoo_finance is used to fetch data 
import fix_yahoo_finance as yf
yf.pdr_override()


# In[2]:


# input
symbols = ['AAPL','MSFT','AMD','NVDA']
start = '2012-01-01'
end = '2019-09-11'


# In[3]:


df = pd.DataFrame()
for s in symbols:
    df[s] = yf.download(s,start,end)['Adj Close']


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


from factor_analyzer import FactorAnalyzer


# In[7]:


fa = FactorAnalyzer(rotation=None)


# In[8]:


fa.fit(df)


# In[9]:


fa.get_communalities()


# In[20]:


ev, v = fa.get_eigenvalues()
ev


# In[21]:


plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Factor Analysis')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[10]:


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
chi_square_value, p_value


# In[11]:


from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)


# In[12]:


kmo_model


# In[13]:


from factor_analyzer import (ConfirmatoryFactorAnalyzer,              ModelSpecificationParser)


# In[14]:


model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df)


# In[15]:


cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)


# In[16]:


cfa.fit(df.values)


# In[17]:


cfa.loadings_


# In[18]:


cfa.factor_varcovs_


# In[19]:


cfa.transform(df.values)

