#!/usr/bin/env python
# coding: utf-8

# # How to further train a pre-trained model

# We will demonstrate how to freeze some or all of the layers of a pre-trained model and continue training using a new fully-connected set of layers and data with a different format.

# ## Imports & Settings

# In[116]:


from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from pathlib import Path

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential, Model 
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Dog Dataset

# Before running the code cell below, download the dataset of dog images [here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).

# In[68]:


(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# In[69]:


cifar10_labels = {0: 'airplane',
                  1: 'automobile',
                  2: 'bird',
                  3: 'cat',
                  4: 'deer',
                  5: 'dog',
                  6: 'frog',
                  7: 'horse',
                  8: 'ship',
                  9: 'truck'}


# In[70]:


num_classes = len(cifar10_labels)


# In[71]:


y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[72]:


# X_train, X_valid = X_train[5000:], X_train[:5000]
# y_train, y_valid = y_train[5000:], y_train[:5000]


# ## Obtain the VGG-16 Bottleneck Features

# We use the VGG16 weights, pre-trained on ImageNet with the much smaller 32 x 32 CIFAR10 data. Note that we indicate the new input size upon import and set all layers to not trainable:

# In[118]:


vgg16 = VGG16(include_top=False, input_shape =X_train.shape[1:])
vgg16.summary()


# ## Freeze model layers

# ### Selectively freeze layers

# In[120]:


for layer in vgg16.layers:
    layer.trainable = False


# In[98]:


vgg16.summary()


# ### Add new layers to model

# We use Keras’ functional API to define the vgg16 output as input into a new set of fully-connected layers like so:

# In[99]:


#Adding custom Layers 
x = vgg16.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)


# We define a new model in terms of inputs and output, and proceed from there on as before:

# In[100]:


transfer_model = Model(inputs = vgg16.input, 
                       outputs = predictions)


# In[101]:


transfer_model.compile(loss = 'categorical_crossentropy', 
                       optimizer = 'Adam', 
                       metrics=["accuracy"])


# In[102]:


validation_split = .1


# We use a more elaborate ImageDataGenerator that also defines a validation_split:

# In[103]:


datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=30,
        validation_split=validation_split)


# In[104]:


batch_size =32
epochs = 10


# We define both train- and validation generators for the fit method:

# In[105]:


train_generator = datagen.flow(X_train, 
                               y_train, 
                               subset='training')
val_generator = datagen.flow(X_train, 
                             y_train, 
                             subset='validation')


# In[108]:


vgg16_path = 'models/cifar10.transfer.vgg16.weights.best.hdf5'
checkpointer = ModelCheckpoint(filepath=vgg16_path, 
                               verbose=1, 
                               save_best_only=True)


# And now we proceed to train the model:

# In[109]:


transfer_model.fit_generator(train_generator,
                             steps_per_epoch=X_train.shape[0] // batch_size,
                             epochs=epochs,
                             validation_data=val_generator,
                             validation_steps=(X_train.shape[0] * .2) // batch_size,
                             callbacks=[checkpointer],
                             verbose=1)


# In[111]:


# load the weights that yielded the best validation accuracy
transfer_model.load_weights(vgg16_path)


# In[112]:


transfer_model.evaluate(X_test, y_test)[1]


# ### Test Set Classification Accuracy

# 10 epochs lead to a mediocre test accuracy of 35.87% because the assumption that image features translate to so much smaller images is somewhat questionable but it serves to illustrate the workflow.

# In[114]:


# get index of predicted dog breed for each image in test set
vgg16_predictions = np.argmax(transfer_model.predict(X_test), axis=1)


# In[115]:


test_accuracy = np.sum(vgg16_predictions==np.argmax(y_test, axis=1))/len(vgg16_predictions)
print('\nTest accuracy: %.4f%%' % test_accuracy)

