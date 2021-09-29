#!/usr/bin/env python
# coding: utf-8

# # Basic Image Classification with Feedforward NN and LetNet5

# All libraries we introduced in the last chapter provide support for convolutional layers. We are going to illustrate the LeNet5 architecture using the most basic MNIST handwritten digit dataset, and then use AlexNet on CIFAR10, a simplified version of the original ImageNet to demonstrate the use of data augmentation.
# LeNet5 and MNIST using Keras.

# ## Imports

# In[45]:


get_ipython().run_line_magic('matplotlib', 'inline')
from random import randint
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras import models, layers
from keras.datasets import mnist
from keras.utils import np_utils
import keras.backend as K
from keras.callbacks import ModelCheckpoint   
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ## Load MNIST Database

# The original MNIST dataset contains 60,000 images in 28x28 pixel resolution with a single grayscale containing handwritten digits from 0 to 9. A good alternative is the more challenging but structurally similar Fashion MNIST dataset that we encountered in Chapter 12 on Unsupervised Learning.

# We can load it in keras out of the box:

# In[2]:


# use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))


# In[3]:


X_train.shape, X_test.shape


# ## Visualize Data

# ### Visualize First 10 Training Images

# The below figure shows the first ten images in the dataset and highlights significant variation among instances of the same digit. On the right, it shows how the pixel values for an indivual image range from 0 to 255.

# In[4]:


fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(20, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray')
    ax.axis('off')
    ax.set_title('Digit: {}'.format(y_train[i]), fontsize=16)
fig.suptitle('First 10 Digits', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=.9)


# ### Show random image in detail

# In[5]:


fig, ax = plt.subplots(figsize = (14, 14)) 

i = randint(0, len(X_train))
img = X_train[i]

ax.imshow(img, cmap='gray')
ax.set_title('Digit: {}'.format(y_train[i]), fontsize=16)

width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        ax.annotate('{:2}'.format(img[x][y]), 
                    xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')


# ## Prepare Data

# ### Rescale pixel values

# We rescale the pixel values to the range [0, 1] to normalize the training data and faciliate the backpropagation process and convert the data to 32 bit floats that reduce memory requirements and computational cost while providing sufficient precision for our use case:

# In[4]:


# rescale [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255 


# ### One-Hot Label Encoding using Keras

# Print first ten labels

# In[5]:


print('Integer-valued labels:')
print(y_train[:10])


# We also need to convert the one-dimensional label to 10-dimensional one-hot encoding to make it compatible with the cross-entropy loss that receives a 10-class softmax output from the network:

# In[6]:


# one-hot encode the labels
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[7]:


# print first ten (one-hot) training labels
y_train[:10]


# ## Feed-Forward NN

# ### Model Architecture

# In[35]:


model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


# In[36]:


model.summary()


# ### Compile the Model

# In[37]:


model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])


# ### Calculate Baseline Classification Accuracy

# In[8]:


# evaluate test accuracy
score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

# print test accuracy
print('Test accuracy: %.4f%%' % accuracy)


# ### Callback for model persistence

# In[ ]:


mnist_path = 'models/mnist.ffn.best.hdf5'


# In[39]:


checkpointer = ModelCheckpoint(filepath=mnist_path, 
                               verbose=1, 
                               save_best_only=True)


# ### Train the Model

# In[40]:


hist = model.fit(X_train, 
                 y_train, 
                 batch_size=128, 
                 epochs=10,
                 validation_split=0.2, 
                 callbacks=[checkpointer],
                 verbose=1, 
                 shuffle=True)


# ### Load the Best Model

# In[41]:


# load the weights that yielded the best validation accuracy
model.load_weights(mnist_path)


# ### Test Classification Accuracy

# In[44]:


# evaluate test accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

print(f'Test accuracy: {accuracy:.2%}')


# ## LeNet5

# In[33]:


K.clear_session()


# We can define a simplified version of LeNet5 that omits the original final layer containing radial basis functions as follows, using the default ‘valid’ padding and single step strides unless defined otherwise:

# In[34]:


lenet5 = Sequential([
    Conv2D(filters=6, kernel_size=5, activation='relu', input_shape=(28, 28, 1), name='CONV1'),
    AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid', name='POOL1'),
    Conv2D(filters=16, kernel_size=(5, 5), activation='tanh', name='CONV2'),
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='POOL2'),
    Conv2D(filters=120, kernel_size=(5, 5), activation='tanh', name='CONV3'),
    Flatten(name='FLAT'),
    Dense(units=84, activation='tanh', name='FC6'),
    Dense(units=10, activation='softmax', name='FC7')
])


# The summary indicates that the model thus defined has over 300,000 parameters:

# In[35]:


lenet5.summary()


# We compile using crossentropy loss and the original stochastic gradient optimizer:

# In[36]:


lenet5.compile(loss=categorical_crossentropy,
               optimizer='SGD',
               metrics=['accuracy'])


# In[37]:


lenet_path = 'models/mnist.lenet.best.hdf5'


# In[38]:


checkpointer = ModelCheckpoint(filepath=lenet_path,
                               verbose=1,
                               save_best_only=True)


# Now we are ready to train the model. The model expects 4D input so we reshape accordingly. We use the standard batch size of 32, 80-20 train-validation split, use checkpointing to store the model weights if the validation error improves, and make sure the dataset is randomly shuffled:

# In[42]:


training = lenet5.fit(X_train.reshape(-1, 28, 28, 1),
                      y_train,
                      batch_size=32,
                       epochs=50,
                       validation_split=0.2, # use 0 to train on all data
                       callbacks=[checkpointer],
                       verbose=1,
                       shuffle=True)


# On a single GPU, 50 epochs take around 2.5 minutes, resulting in a test accuracy of 99.19%, almost exactly the same result as for the original LeNet5:

# In[48]:


pd.DataFrame(training.history)[['acc','val_acc']].plot();


# In[49]:


# evaluate test accuracy
accuracy = lenet5.evaluate(X_test.reshape(-1, 28, 28, 1), y_test, verbose=0)[1]
print('Test accuracy: {:.2%}'.format(accuracy))


# ## Summary

# For comparison, a simple two-layer feedforward network achieves only 37.36% test accuracy. 
# 
# The LeNet5 improvement on MNIST is, in fact, modest. Non-neural methods have also achieved classification accuracies greater than or equal to 99%, including K-Nearest Neighbours or Support Vector Machines. CNNs really shine with more challenging datasets as we will see next.

# In[ ]:




