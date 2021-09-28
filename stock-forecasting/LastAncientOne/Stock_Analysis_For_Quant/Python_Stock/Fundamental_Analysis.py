#!/usr/bin/env python
# coding: utf-8

# # Fundamental Analysis in Python

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[3]:


symbol = 'MSFT'
stock = yf.Ticker(symbol)


# In[4]:


stock.info


# In[5]:


stock.history(period='1mo')


# In[10]:


stock.history(period="max")


# In[11]:


df = stock.history(period="max")


# In[12]:


df.head()


# In[13]:


df.tail()


# In[6]:


stock.financials


# In[17]:


IC = stock.financials[stock.financials.columns[::-1]] # Reverse Column


# In[18]:


IC


# In[28]:


IC.loc['Net Income Applicable To Common Shares']


# In[39]:


# Change Data object to float
NI = IC.loc['Net Income Applicable To Common Shares'].astype(float)


# In[41]:


plt.plot(NI)
plt.xlabel('Date')
plt.ylabel('Price in millions')
plt.title('Net Income in Yearly')


# In[7]:


stock.balancesheet


# In[47]:


BS = stock.balancesheet.iloc[:,::-1] # Reverse Column


# In[59]:


BS


# In[60]:


BS.loc['Retained Earnings']


# In[61]:


CR = BS.loc['Total Current Assets'].astype(int) / BS.loc['Total Current Liabilities'].astype(int)
CR


# In[8]:


stock.cashflow


# In[22]:


CF = stock.cashflow.loc[::,::-1] # Reverse Column


# In[23]:


CF


# ## Ratio Analysis

# ### Short term solvency ratios

# In[76]:


print('Short term solvency ratios')
print('-'*40)
print('Current Ratio')
CR = BS.loc['Total Current Assets'].astype(int) / BS.loc['Total Current Liabilities'].astype(int) # Total Current Assets/Total Current Liabilities
print(CR,"\n")
print('Quick Ratio')
QR = (BS.loc['Total Current Assets'].astype(int) - BS.loc['Inventory'].astype(int)) / BS.loc['Total Current Liabilities'].astype(int) # (Total Current Assets-Inventory)/Total Current Liabilities
print(QR,"\n")
print('Cash Ratio')
CashR = (BS.loc['Cash And Cash Equivalents'].astype(int) - BS.loc['Short Term Investments'].astype(int)) / BS.loc['Total Current Liabilities'].astype(int) # (Cash And Cash Equivalents + Short Term Investments + Intangible Assets)/ Total Current Liabilities)
print(CashR,"\n")
print('Networking Capital to Current Liabilities')
NCCL = (BS.loc['Total Current Assets'].astype(int) - BS.loc['Total Current Liabilities'].astype(int)) / BS.loc['Total Current Liabilities'].astype(int) # (Total Current Assets - Total Current Liabilities)/Total Current Liabilities)
print(NCCL,"\n")


# ### Asset Utilization or Turnover ratios

# In[75]:


print('Asset Utilization or Turnover ratios')
print('-'*40)
print('Average Collection Period')
ACP = BS.loc['Net Receivables'].astype(int) / (IC.loc['Total Revenue'].astype(int)/360) # Net Receivables/(Total Revenue/360)
print(ACP,"\n")
print('Inventory Turnover Ratios')
ITR = IC.loc['Total Revenue'].astype(int) / BS.loc['Inventory'].astype(int)# Total Revenue / Inventory
print(ITR,"\n")                                             
print('Receivable Turnover')
RT = IC.loc['Total Revenue'].astype(int)/BS.loc['Net Receivables'].astype(int) # Total Revenue / Net Receivables
print(RT,"\n")                                                     
print('Fixed Asset Turnover')
FAT = IC.loc['Total Revenue'].astype(int)/BS.loc['Net Receivables'].astype(int) # Total Revenue / Property, plant and equipment
print(FAT,"\n")   
print('Total Asset Turnover')                                             
TAT = IC.loc['Total Revenue'].astype(int)/BS.loc['Total Assets'].astype(int)# Total Revenue / Total Assets
print(TAT,"\n")                                                


# ### Financial Leverage ratios

# In[86]:


print('Financial Leverage ratios')
print('-'*40)
print('Total Debt Ratio')
TDR = BS.loc['Total Liabilities'].astype(int) / BS.loc['Total Assets'].astype(int) # Total Liabilities / Total Assets
print(TDR,"\n")
print('Debt/Equity')
DE = BS.loc['Total Liabilities'].astype(int) / BS.loc["Total stockholders' equity"].astype(int) # Total Liabilities / Total stockholders' equity
print(DE,"\n")
print('Equity Ratio')
ER = BS.loc["Total stockholders\' equity"].astype(int) / BS.loc['Total Assets'].astype(int) # Total stockholders' equity / Total Assets
print(ER,"\n")
print('Long-term Debt Ratio')
LTDR = BS.loc['Long Term Debt'].astype(int) / BS.loc['Total Assets'].astype(int) # Long Term Debt / Total Assets
print(LTDR,"\n") 
print('Times Interest Earned Ratio')
TIER = IC.loc['Earnings Before Interest and Taxes'].astype(int) / IC.loc['Interest Expense'].astype(int) # Earnings Before Interest and Taxes / Interest Expense
print(TIER,"\n") 


# ### Profitability ratios

# In[92]:


print('Profitability ratios')
print('-'*40)
print('Gross Profit Margin')
GPM = IC.loc['Gross Profit'].astype(int) / IC.loc['Total Revenue'].astype(int) # Gross Profit / Total Revenue
print(GPM,"\n") 
print('Net Profit Margin')
NPM = IC.loc['Net Income Applicable To Common Shares'].astype(int) / IC.loc['Total Revenue'].astype(int) # Net Income / Total Revenue
print(NPM,"\n") 
print('Return on Assets (ROA)')
ROA = IC.loc['Net Income Applicable To Common Shares'].astype(int) / BS.loc['Total Assets'].astype(int) # Net Income / Total Assets
print(ROA,"\n") 
print('Return on Equity (ROE)')
ROE = IC.loc['Net Income Applicable To Common Shares'].astype(int) / BS.loc['Total stockholders\' equity'].astype(int) # Net Income / Total Equity
print(ROE,"\n") 
print('Earning Per Share (EPS)')
EPS = IC.loc['Net Income Applicable To Common Shares'].astype(int) / (BS.loc['Common Stock'].astype(int) - BS.loc['Treasury Stock'].astype(int)) # Net Income / (Common Stock - Treasury Stock)
print(EPS,"\n") 

