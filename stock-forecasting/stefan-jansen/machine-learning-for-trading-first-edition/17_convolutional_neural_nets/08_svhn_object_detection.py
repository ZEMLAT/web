#!/usr/bin/env python
# coding: utf-8

# # Object Detection with Street View House Numbers

# This notebook illustrates how to build a deep CNN using Keras’ functional API to generate multiple outputs: one to predict how many digits are present, and five for the value of each in the order they appear.

# ## Imports & Settings

# In[1]:


import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, Activation


# ## Best Architecture

# [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/abs/1312.6082), Goodfellow, et al, 2014

# - eight convolutional hidden layers, 
# - one locally connected hidden layer
# - two densely connected hidden layers. 
# - the first hidden layer contains maxout units with three filters per unit
# - the others contain rectifier units 
# - the number of units is [48, 64, 128, 160] for the first four layers 
# - 192 for all other locally connected layers
# - the fully connected layers contain 3,072 units each. 
# - Each convolutional layer includes max pooling and subtractive normalization
# - The max pooling window size is 2 × 2. 
# - The stride alternates between 2 and 1 at each layer, so that half of the layers don’t reduce the spatial size of the representation
# - All convolutions use zero padding on the input to preserve representation size. 
# - The subtractive normalization operates on 3x3 windows and preserves representation size. 
# - All convolution kernels were of size 5 × 5. 
# - We trained with dropout applied to all hidden layers but not the input.

# The best-performing architecture on the original dataset has eight convolutional layers and two final fully-connected layers. The convolutional layers are similar so that we can define a function to simplify their creation:

# In[2]:


def svhn_layer(model, filters, strides, n, input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters, kernel_size=5, padding='same', name='CONV{}'.format(n), input_shape=input_shape))
    else:
        model.add(Conv2D(filters, kernel_size=5, padding='same', activation='relu', name='CONV{}'.format(n)))
    model.add(BatchNormalization(name='NORM{}'.format(n)))
    model.add(MaxPooling2D(pool_size=2, strides=strides, name='POOL{}'.format(n)))
    model.add(Dropout(0.2, name='DROP{}'.format(n)))
    return model


# The entire model combines the Sequential and functional API as follows:

# In[3]:


model = Sequential()

svhn_layer(model, 48, 1, n=1, input_shape=(32,32,1))

for i, kernel in enumerate([48, 64, 128, 160] + 3 * [192], 2):
    svhn_layer(model, kernel, strides=2 if i % 2 == 0 else 1, n=i)

model.add(Flatten())
model.add(Dense(3072, name='FC1'))
model.add(Dense(3072, name='FC2'))
y = model.output

n_digits = (Dense(units=6, activation='softmax'))(y)
digit1 = (Dense(units=10, activation='softmax'))(y)
digit2 = (Dense(units=11, activation='softmax'))(y)
digit3 = (Dense(units=11, activation='softmax'))(y)
digit4 = (Dense(units=11, activation='softmax'))(y)
digit5 = (Dense(units=11, activation='softmax'))(y)

svhn_model = Model(inputs=model.input, outputs=[n_digits, digit1, digit2, digit3, digit4, digit5])


# As a result, the model produces six distinct outputs that we can evaluate.

# In[4]:


svhn_model.summary()


# ### Get Data 

# In[5]:


svhn_model.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=["accuracy"])


# In[6]:


with pd.HDFStore('images/svhn/data.h5') as store:
    X_train = store['train/data'].values.reshape(-1, 32, 32, 1)
    y_train = store['train/labels']
    X_test = store['test/data'].values.reshape(-1, 32, 32, 1)
    y_test = store['test/labels']   


# In[7]:


train_digits = [to_categorical(d) for d in y_train.values.T]
test_digits = [to_categorical(d) for d in y_test.values.T]


# In[8]:


svhn_path = 'models/svhn.cnn.weights.best.hdf5'


# In[9]:


checkpointer = ModelCheckpoint(filepath=svhn_path, 
                               verbose=1, 
                               save_best_only=True)


# In[10]:


epochs = 25
result = svhn_model.fit(x=X_train,
                        y=train_digits,
                        batch_size=32,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_test, test_digits),
                        callbacks=[checkpointer])


# In[11]:


n_digits, digit1, digit2, digit3, digit4, digit5 = svhn_model.predict(X_test, verbose=1)


# In[12]:


(y_test[0] == np.argmax(n_digits, axis=1)).sum()/len(n_digits)


# In[22]:


pd.DataFrame(result.history)[['val_dense_{}_acc'.format(i) for i in range(1, 7)]].plot();


# In[23]:


confusion_matrix(y_true=y_test[0], y_pred=np.argmax(n_digits, axis=1))


# In[24]:


confusion_matrix(y_true=y_test[1], y_pred=np.argmax(digit1, axis=1))


# In[23]:


pd.Series(np.argmax(digit1, axis=1)).value_counts()


# In[18]:


y_test[0].value_counts(normalize=True)


# In[ ]:




