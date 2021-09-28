#!/usr/bin/env python
# coding: utf-8

# # Black Scholes Formula 

# In[1]:


import numpy as np
import scipy.stats as ss


# In[2]:


def d1(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    return d1


# In[3]:


def d2(S0, K, r, sigma, T):
    d2 = (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return d2


# In[4]:


def BlackScholesCall(S0, K, r, sigma, T):
    BSC = S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    return BSC
       


# In[5]:


def BlackScholesPut(S0, K, r, sigma, T):
    BSP = K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T)) 
    return BSP


# In[6]:


# Input
S0 = 100.0
K = 100.0
r = 0.1
sigma = 0.30
T = 3


# In[7]:


print("S0\tCurrent Stock Price:", S0)
print("K\tStrike Price:", K)
print("r\tContinuously compounded risk-free rate:", r)
print("sigma\tVolatility of the stock price per year:", sigma)
print("T\tTime to maturity in trading years:", T)


# In[8]:


Call_BS = BlackScholesCall(S0, K, r, sigma, T)
Call_BS


# In[9]:


Put_BS = BlackScholesPut(S0, K, r, sigma, T)
Put_BS

