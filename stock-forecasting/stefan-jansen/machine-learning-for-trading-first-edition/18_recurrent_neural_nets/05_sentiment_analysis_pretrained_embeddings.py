#!/usr/bin/env python
# coding: utf-8

# # Sentiment analysis with pretrained word vectors

# In Chapter 15, Word Embeddings, we discussed how to learn domain-specific word embeddings. Word2vec, and related learning algorithms, produce high-quality word vectors, but require large datasets. Hence, it is common that research groups share word vectors trained on large datasets, similar to the weights for pretrained deep learning models that we encountered in the section on transfer learning in the previous chapter.
# 
# We are now going to illustrate how to use pretrained Global Vectors for Word Representation (GloVe) provided by the Stanford NLP group with the IMDB review dataset.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
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
from keras.preprocessing.text import Tokenizer
import keras
import keras.backend as K
import tensorflow as tf


# In[2]:


sns.set_style('whitegrid')
np.random.seed(42)
K.clear_session()


# ## Load Reviews

# We are going to load the IMDB dataset from the source for manual preprocessing.

# Data source: [Stanford IMDB Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

# In[3]:


path = Path('data/aclImdb/')


# In[4]:


files = path.glob('**/*.txt')
len(list(files))


# In[5]:


files = path.glob('*/**/*.txt')
data = []
for f in files:
    _, _, data_set, outcome = str(f.parent).split('/')
    data.append([data_set, int(outcome=='pos'), f.read_text(encoding='latin1')])


# In[6]:


data = pd.DataFrame(data, columns=['dataset', 'label', 'review']).sample(frac=1.0)


# In[7]:


train_data = data.loc[data.dataset=='train', ['label', 'review']]
test_data = data.loc[data.dataset=='test', ['label', 'review']]


# In[8]:


train_data.label.value_counts()


# In[9]:


test_data.label.value_counts()


# ## Prepare Data

# ### Tokenizer

# Keras provides a tokenizer that we use to convert the text documents to integer-encoded sequences, as shown here:

# In[10]:


num_words = 10000
t = Tokenizer(num_words=num_words, 
              lower=True, 
              oov_token=2)
t.fit_on_texts(train_data.review)


# In[11]:


vocab_size = len(t.word_index) + 1
vocab_size


# In[12]:


train_data_encoded = t.texts_to_sequences(train_data.review)
test_data_encoded = t.texts_to_sequences(test_data.review)


# In[13]:


max_length = 100


# ### Pad Sequences

# We also use the pad_sequences function to convert the list of lists (of unequal length) to stacked sets of padded and truncated arrays for both the train and test datasets:

# In[14]:


X_train_padded = pad_sequences(train_data_encoded, 
                            maxlen=max_length, 
                            padding='post',
                           truncating='post')
y_train = train_data['label']
X_train_padded.shape


# In[15]:


X_test_padded = pad_sequences(test_data_encoded, 
                            maxlen=max_length, 
                            padding='post',
                           truncating='post')
y_test = test_data['label']
X_test_padded.shape


# ## Load Embeddings

# Assuming we have downloaded and unzipped the GloVe data to the location indicated in the code, we now create a dictionary that maps GloVe tokens to 100-dimensional real-valued vectors, as follows:

# In[16]:


# load the whole embedding into memory
glove_path = Path('data/glove/glove.6B.100d.txt')
embeddings_index = dict()

for line in glove_path.open(encoding='latin1'):
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
    except:
        continue
    embeddings_index[word] = coefs


# In[17]:


print('Loaded {:,d} word vectors.'.format(len(embeddings_index)))


# There are around 340,000 word vectors that we use to create an embedding matrix that matches the vocabulary so that the RNN model can access embeddings by the token index:

# In[18]:


embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[19]:


embedding_matrix.shape


# ## Define Model Architecture

# The difference between this and the RNN setup in the previous example is that we are going to pass the embedding matrix to the embedding layer and set it to non-trainable, so that the weights remain fixed during training:

# In[20]:


embedding_size = 100


# In[21]:


rnn = Sequential([
    Embedding(input_dim=vocab_size, 
              output_dim= embedding_size, 
              input_length=max_length,
              weights=[embedding_matrix], 
              trainable=False),
    GRU(units=32,  dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
rnn.summary()


# In[23]:


rnn.compile(loss='binary_crossentropy', 
            optimizer='RMSProp', 
            metrics=['accuracy'])


# In[24]:


rnn_path = 'models/imdb.gru_pretrained.weights.best.hdf5'
checkpointer = ModelCheckpoint(filepath=rnn_path,
                              monitor='val_loss',
                              save_best_only=True,
                              save_weights_only=True,
                              period=5)


# In[25]:


early_stopping = EarlyStopping(monitor='val_loss', 
                              patience=5,
                              restore_best_weights=True)


# In[26]:


rnn.fit(X_train_padded, 
        y_train, 
        batch_size=32, 
        epochs=25, 
        validation_data=(X_test_padded, y_test), 
        callbacks=[checkpointer, early_stopping],
        verbose=1)


# In[29]:


rnn.load_weights(rnn_path)


# In[30]:


y_score = rnn.predict(X_test_padded)
roc_auc_score(y_score=y_score.squeeze(), y_true=y_test)


# In[ ]:




