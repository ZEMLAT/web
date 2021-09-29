#!/usr/bin/env python
# coding: utf-8

# # LSTM & Word Embeddings for Sentiment Classification

# RNNs are commonly applied to various natural language processing tasks. We've already encountered sentiment analysis using text data in part three of [this book](https://www.amazon.com/Hands-Machine-Learning-Algorithmic-Trading-ebook/dp/B07JLFH7C5/ref=sr_1_2?ie=UTF8&qid=1548455634&sr=8-2&keywords=machine+learning+algorithmic+trading).
# 
# We are now going to illustrate how to apply an RNN model to text data to detect positive or negative sentiment (which can easily be extended to a finer-grained sentiment scale). We are going to use word embeddings to represent the tokens in the documents. We covered word embeddings in Chapter 15, Word Embeddings. They are an excellent technique to convert text into a continuous vector representation such that the relative location of words in the latent space encodes useful semantic aspects based on the words' usage in context.
# 
# We saw in the previous RNN example that Keras has a built-in embedding layer that allows us to train vector representations specific to the task at hand. Alternatively, we can use pretrained vectors.

# ## Imports & Settings

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import minmax_scale
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Input, concatenate, Embedding, Reshape
from keras.preprocessing.sequence import pad_sequences
import keras
import keras.backend as K
import tensorflow as tf


# In[2]:


sns.set_style('whitegrid')
np.random.seed(42)
K.clear_session()


# ## Load Reviews

# To keep the data manageable, we will illustrate this use case with the IMDB reviews dataset, which contains 50,000 positive and negative movie reviews evenly split into a train and a test set, and with balanced labels in each dataset. The vocabulary consists of 88,586 tokens.
# 
# The dataset is bundled into Keras and can be loaded so that each review is represented as an integer-encoded sequence. We can limit the vocabulary to num_words while filtering out frequent and likely less informative words using skip_top, as well as sentences longer than maxlen. We can also choose oov_char, which represents tokens we chose to exclude from the vocabulary on frequency grounds, as follows:

# In[3]:


vocab_size = 20000


# In[4]:


(X_train, y_train), (X_test, y_test) = imdb.load_data(seed=42, 
                                                      skip_top=0,
                                                      maxlen=None, 
                                                      oov_char=2, 
                                                      index_from=3
                                                      num_words=vocab_size)


# In[5]:


ax = sns.distplot([len(review) for review in X_train])
ax.set(xscale='log');


# ## Prepare Data

# In the second step, convert the lists of integers into fixed-size arrays that we can stack and provide as input to our RNN. The pad_sequence function produces arrays of equal length, truncated, and padded to conform to maxlen, as follows:

# In[6]:


maxlen = 100


# In[7]:


X_train_padded = pad_sequences(X_train, 
                        truncating='pre', 
                        padding='pre', 
                        maxlen=maxlen)

X_test_padded = pad_sequences(X_test, 
                       truncating='pre', 
                       padding='pre', 
                       maxlen=maxlen)


# In[8]:


X_train_padded.shape, X_test_padded.shape


# ## Define Model Architecture

# Now we can define our RNN architecture. The first layer learns the word embeddings. We define the embedding dimension as previously using the input_dim keyword to set the number of tokens that we need to embed, the output_dim keyword, which defines the size of each embedding, and how long each input sequence is going to be.

# ### Custom Loss Metric

# In[9]:


# source: https://github.com/keras-team/keras/issues/3230
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """prob false alert for binary classifier"""
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / (N + 1)


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    """prob true alerts for binary classifier"""
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP / (P + 1)


# In[10]:


embedding_size = 100


# Note that we are using GRUs this time, which train faster and perform better on smaller data. We are also using dropout for regularization, as follows:

# In[11]:


rnn = Sequential([
    Embedding(input_dim=vocab_size, output_dim= embedding_size, input_length=maxlen),
    GRU(units=32,  dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
rnn.summary()


# The resulting model has over 2 million parameters.

# We compile the model to use our custom AUC metric, which we introduced previously, and train with early stopping:

# In[16]:


rnn.compile(loss='binary_crossentropy', 
            optimizer='RMSProp', 
            metrics=['accuracy', auc])


# In[17]:


rnn_path = 'models/imdb.gru.weights.best.hdf5'
checkpointer = ModelCheckpoint(filepath=rnn_path,
                              monitor='val_auc',
                              save_best_only=True,
                              save_weights_only=True,
                              period=5)


# In[18]:


early_stopping = EarlyStopping(monitor='val_auc', 
                               mode='max',
                              patience=5,
                              restore_best_weights=True)


# Training stops after eight epochs and we recover the weights for the best models to find a high test AUC of 0.9346:

# In[19]:


rnn.fit(X_train_padded, 
        y_train, 
        batch_size=32, 
        epochs=25, 
        validation_data=(X_test_padded, y_test), 
        callbacks=[checkpointer, early_stopping],
        verbose=1)


# ## Evaluate Results

# In[20]:


rnn.load_weights(rnn_path)


# In[22]:


y_score = rnn.predict(X_test_padded)
y_score.shape


# In[23]:


roc_auc_score(y_score=y_score.squeeze(), y_true=y_test)


# In[ ]:




