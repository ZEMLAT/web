#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[1]:


import re
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.performance import *
from alphalens.plotting import *
from alphalens.tears import *


# In[2]:


warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# ### Version conflict

# At the time of writing, `zipline` required `pandas<=0.22` so you may need to run `pip install -U zipline` to temporarily downgrade `pandas` or set up a separate environment.

# In[ ]:


get_ipython().system('pip install -U zipline')


# ## Zipline AlphaFactor Test

# We are first going to illustrate the zipline alpha factor research workflow in an offline environment. In particular, we will develop and test a simple mean-reversion factor that measures how much recent performance has deviated from the historical average. 
# 
# Short-term reversal is a common strategy that takes advantage of the weakly predictive pattern that stock price increases are likely to mean-revert back down over horizons from less than a minute to one month.

# To this end, the factor computes the z-score for the last monthly return relative to the rolling monthly returns over the last year. At this point, we will not place any orders to simply illustrate the implementation of a CustomFactor and record the results during the simulation.
# 
# After some basic settings, `MeanReversion` subclasses `CustomFactor` and defines a `compute()` method. It creates default inputs of monthly returns over an also default year-long window so that the monthly_return variable will have 252 rows and one column for each security in the Quandl dataset on a given day.
# 
# The `compute_factors()` method creates a `MeanReversion` factor instance and creates long, short, and ranking pipeline columns. The former two contain Boolean values that could be used to place orders, and the latter reflects that overall ranking to evaluate the overall factor performance. Furthermore, it uses the built-in `AverageDollarVolume` factor to limit the computation to more liquid stocks

# The result would allow us to place long and short orders. We will see in the next chapter how to build a portfolio by choosing a rebalancing period and adjusting portfolio holdings as new signals arrive.

# - The `initialize()` method registers the compute_factors() pipeline, and the before_trading_start() method ensures the pipeline runs on a daily basis. 
# - The `record()` function adds the pipeline's ranking column as well as the current asset prices to the performance DataFrame returned by the `run_algorithm()` function

# We will use the factor and pricing data stored in the performance DataFrame to evaluate the factor performance for various holding periods in the next section, but first, we'll take a look at how to create more complex signals by combining several alpha factors from a diverse set of data sources on the Quantopian platform.

# Run using jupyter notebook extension

# In[3]:


get_ipython().run_line_magic('load_ext', 'zipline')


# In[4]:


get_ipython().run_cell_magic('zipline', '--start 2015-1-1 --end 2018-1-1 --output single_factor.pickle', '\nfrom zipline.api import (\n    attach_pipeline,\n    date_rules,\n    time_rules,\n    order_target_percent,\n    pipeline_output,\n    record,\n    schedule_function,\n    get_open_orders,\n    calendars\n)\nfrom zipline.finance import commission, slippage\nfrom zipline.pipeline import Pipeline, CustomFactor\nfrom zipline.pipeline.factors import Returns, AverageDollarVolume\nimport numpy as np\nimport pandas as pd\n\nMONTH = 21\nYEAR = 12 * MONTH\nN_LONGS = N_SHORTS = 25\nVOL_SCREEN = 1000\n\n\nclass MeanReversion(CustomFactor):\n    """Compute ratio of latest monthly return to 12m average,\n       normalized by std dev of monthly returns"""\n    inputs = [Returns(window_length=MONTH)]\n    window_length = YEAR\n\n    def compute(self, today, assets, out, monthly_returns):\n        df = pd.DataFrame(monthly_returns)\n        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())\n\n\ndef compute_factors():\n    """Create factor pipeline incl. mean reversion,\n        filtered by 30d Dollar Volume; capture factor ranks"""\n    mean_reversion = MeanReversion()\n    dollar_volume = AverageDollarVolume(window_length=30)\n    return Pipeline(columns={\'longs\': mean_reversion.bottom(N_LONGS),\n                             \'shorts\': mean_reversion.top(N_SHORTS),\n                             \'ranking\': mean_reversion.rank(ascending=False)},\n                    screen=dollar_volume.top(VOL_SCREEN))\n\n\ndef exec_trades(data, assets, target_percent):\n    """Place orders for assets using target portfolio percentage"""\n    for asset in assets:\n        if data.can_trade(asset) and not get_open_orders(asset):\n            order_target_percent(asset, target_percent)\n\n\ndef rebalance(context, data):\n    """Compute long, short and obsolete holdings; place trade orders"""\n    factor_data = context.factor_data\n    record(factor_data=factor_data.ranking)\n\n    assets = factor_data.index\n    record(prices=data.current(assets, \'price\'))\n\n    longs = assets[factor_data.longs]\n    shorts = assets[factor_data.shorts]\n    divest = set(context.portfolio.positions.keys()) - set(longs.union(shorts))\n\n    exec_trades(data, assets=divest, target_percent=0)\n    exec_trades(data, assets=longs, target_percent=1 / N_LONGS)\n    exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS)\n\n\ndef initialize(context):\n    """Setup: register pipeline, schedule rebalancing,\n        and set trading params"""\n    attach_pipeline(compute_factors(), \'factor_pipeline\')\n    schedule_function(rebalance,\n                      date_rules.week_start(),\n                      time_rules.market_open(),\n                      calendar=calendars.US_EQUITIES)\n    context.set_commission(commission.PerShare(cost=.01, min_trade_cost=0))\n    context.set_slippage(slippage.VolumeShareSlippage())\n\n\ndef before_trading_start(context, data):\n    """Run factor pipeline"""\n    context.factor_data = pipeline_output(\'factor_pipeline\')')

