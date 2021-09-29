#!/usr/bin/env python
# coding: utf-8

# # Visualize Image Filter

# ## Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D


# ## Import the Image

# In[2]:


# Feel free to try out your own images
img_path = 'images/building.png'


# In[3]:


# load color image 
bgr_img = cv2.imread(img_path)


# In[4]:


### convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


# In[5]:


# resize to smaller
small_img = scipy.misc.imresize(gray_img, 0.3)


# In[6]:


# rescale entries to lie in [0,1]
small_img = small_img.astype("float32")/255


# In[7]:


# plot image
plt.imshow(small_img, cmap='gray')
plt.axis('off');


# ## Specify the Filters

# In[8]:


# Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], 
                        [-1, -1, 1, 1], 
                        [-1, -1, 1, 1], 
                        [-1, -1, 1, 1]])


# In[9]:


# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = [filter_1, filter_2, filter_3, filter_4]


# In[10]:


# visualize all filters
fig, axes = plt.subplots(ncols=4, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(filters[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Filter {i}')
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')


# ## Visualize the Activation Maps for Each Filter

# In[11]:


plt.imshow(small_img, cmap='gray')
plt.axis('off')


# ### Define single-layer CNN

# In[12]:


cnn = Sequential([
    Convolution2D(1, (4, 4), 
                  activation='relu', 
                  input_shape=(small_img.shape[0], small_img.shape[1], 1))
])


# In[13]:


cnn.summary()


# ### Apply convolutional filter and return output

# In[14]:


def apply_filter(img, index, filter_list, ax):
    # set the weights of the filter in the convolutional layer to filter_list[i]
    cnn.layers[0].set_weights([np.reshape(filter_list[i], (4,4,1,1)), np.array([0])])
    # plot the corresponding activation map
    ax.imshow(np.squeeze(cnn.predict(np.reshape(img, (1, img.shape[0], img.shape[1], 1)))), cmap='gray')


# In[15]:


# visualize all filters
fig, axes = plt.subplots(ncols=4, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(filters[i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Filter {i}')
fig.tight_layout()
fig.savefig('images/filters', dpi=300)

# visualize all activation maps
fig, axes = plt.subplots(ncols=4, figsize=(20, 20))
for i, ax in enumerate(axes):
    apply_filter(small_img, i, filters, ax)
    ax.axis('off')
    ax.set_title(f'Activation Map for Filter {i}')
fig.tight_layout()


# In[ ]:




