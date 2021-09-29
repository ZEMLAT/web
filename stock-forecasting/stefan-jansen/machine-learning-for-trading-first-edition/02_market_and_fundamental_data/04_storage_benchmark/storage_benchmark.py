#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
import random
import string


# In[2]:


results = {}


# ## Generate Test Data

# The test `DataFrame` that can be configured to contain numerical or text data, or both. For the HDF5 library, we test both the fixed and table format. 

# In[3]:


def generate_test_data(nrows=100000, numerical_cols=2000, text_cols=0, text_length=10):
    ncols = numerical_cols + text_cols
    s = "".join([random.choice(string.ascii_letters)
                 for _ in range(text_length)])
    data = pd.concat([pd.DataFrame(np.random.random(size=(nrows, numerical_cols))),
                      pd.DataFrame(np.full(shape=(nrows, text_cols), fill_value=s))],
                     axis=1, ignore_index=True)
    data.columns = [str(i) for i in data.columns]
    return data


# In[4]:


df = generate_test_data()
df.info()


# ## Parquet

# ### Size

# In[7]:


parquet_file = Path('test.parquet')


# In[8]:


df.to_parquet(parquet_file)
size = parquet_file.stat().st_size


# ### Read

# In[9]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_parquet(parquet_file)')


# In[10]:


read = _


# In[11]:


parquet_file.unlink()


# ### Write

# In[12]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_parquet(parquet_file)\nparquet_file.unlink()')


# In[13]:


write = _


# ### Results

# In[14]:


results['parquet'] = {'read': read.all_runs, 'write': write.all_runs, 'size': size}


# ## HDF5

# In[15]:


test_store = Path('index.h5')


# ### Fixed Format

# #### Size

# In[16]:


with pd.HDFStore(test_store) as store:
    store.put('file', df)
size = test_store.stat().st_size


# #### Read

# In[17]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.get('file')")


# In[18]:


read = _


# In[19]:


test_store.unlink()


# #### Write

# In[20]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.put('file', df)\ntest_store.unlink()")


# In[21]:


write = _


# #### Results

# In[22]:


results['hdf_fixed'] = {'read': read.all_runs, 'write': write.all_runs, 'size': size}


# ### Table Format

# #### Size

# In[23]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t')
size = test_store.stat().st_size    


# #### Read

# In[24]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    df = store.get('file')")


# In[25]:


read = _


# In[26]:


test_store.unlink()


# #### Write

# Note that `write` in table format does not work with text data.

# In[27]:


get_ipython().run_cell_magic('timeit', '-o', "with pd.HDFStore(test_store) as store:\n    store.append('file', df, format='t')\ntest_store.unlink()    ")


# In[28]:


write = _


# #### Results

# In[29]:


results['hdf_table'] = {'read': read.all_runs, 'write': write.all_runs, 'size': size}


# ### Table Select

# #### Size

# In[30]:


with pd.HDFStore(test_store) as store:
    store.append('file', df, format='t', data_columns=['company', 'form'])
size = test_store.stat().st_size 


# #### Read

# In[31]:


company = 'APPLE INC'


# In[32]:


# %%timeit
# with pd.HDFStore(test_store) as store:
#     s = store.select('file', 'company = company')


# In[33]:


# read = _


# In[34]:


# test_store.unlink()


# #### Write

# In[35]:


# %%timeit
# with pd.HDFStore(test_store) as store:
#     store.append('file', df, format='t', data_columns=['company', 'form'])
# test_store.unlink() 


# In[36]:


# write = _


# #### Results

# In[37]:


# results['hdf_select'] = {'read': read.all_runs, 'write': write.all_runs, 'size': size}


# ## CSV

# In[38]:


test_csv = Path('test.csv')


# ### Size

# In[39]:


df.to_csv(test_csv)
test_csv.stat().st_size


# ### Read

# In[40]:


get_ipython().run_cell_magic('timeit', '-o', 'df = pd.read_csv(test_csv)')


# In[41]:


read = _


# In[42]:


test_csv.unlink()  


# ### Write

# In[43]:


get_ipython().run_cell_magic('timeit', '-o', 'df.to_csv(test_csv)\ntest_csv.unlink()')


# In[44]:


write = _


# ### Results

# In[45]:


results['csv'] = {'read': read.all_runs, 'write': write.all_runs, 'size': size}


# ## Store Results

# In[47]:


text_num = pd.concat([pd.DataFrame(data).mean().to_frame(f) for f, data in results.items()], axis=1).T
text_num[['read', 'write']].plot.barh();


# In[48]:


df = pd.concat([pd.DataFrame(data).mean().to_frame(f) for f, data in results.items()], axis=1).T
# df.to_csv('num_only.csv')
df[['read', 'write']].plot.barh();


# In[ ]:


# for f, data in results.items():
#     pd.DataFrame(data).to_csv('{}.csv'.format(f))

