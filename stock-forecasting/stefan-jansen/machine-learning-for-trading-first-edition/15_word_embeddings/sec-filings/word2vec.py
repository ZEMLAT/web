#!/usr/bin/env python
# coding: utf-8

# ## Imports & Settings

# In[1]:


from pathlib import Path
from time import time
import warnings
from collections import Counter
import logging
from ast import literal_eval as make_tuple
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import word2vec


# In[2]:


pd.set_option('display.expand_frame_repr', False)
warnings.filterwarnings('ignore')
np.random.seed(42)


# In[3]:


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


# ### Logging Setup

# In[4]:


logging.basicConfig(
        filename='logs/word2vec.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


# ## word2vec

# In[6]:


analogies_path = Path().cwd().parent / 'data' / 'analogies' / 'analogies-en.txt'


# ### Set up Sentence Generator

# In[8]:


NGRAMS = 2


# To facilitate memory-efficient text ingestion, the LineSentence class creates a generator from individual sentences contained in the provided text file:

# In[9]:


sentence_path = Path('data', 'ngrams', f'ngrams_{NGRAMS}.txt')
sentences = LineSentence(sentence_path)


# ### Train word2vec Model

# The [gensim.models.word2vec](https://radimrehurek.com/gensim/models/word2vec.html) class implements the skipgram and CBOW architectures introduced above. The notebook [word2vec](../03_word2vec.ipynb) contains additional implementation detail.

# In[10]:


start = time()
model = Word2Vec(sentences,
                 sg=1,          # 1 for skip-gram; otherwise CBOW
                 hs=0,          # hierarchical softmax if 1, negative sampling if 0
                 size=300,      # Vector dimensionality
                 window=3,      # Max distance betw. current and predicted word
                 min_count=50,  # Ignore words with lower frequency
                 negative=10,    # noise word count for negative sampling
                 workers=8,     # no threads 
                 iter=1,        # no epochs = iterations over corpus
                 alpha=0.025,   # initial learning rate
                 min_alpha=0.0001 # final learning rate
                ) 
print('Duration:', format_time(time() - start))


# ### Persist model & vectors

# In[11]:


model.save('models/baseline/word2vec.model')
model.wv.save('models/baseline/word_vectors.bin')


# ### Load model and vectors

# In[40]:


model = Word2Vec.load('models/archive/word2vec.model')


# In[8]:


wv = KeyedVectors.load('models/baseline/word_vectors.bin')


# ### Get vocabulary

# In[12]:


vocab = []
for k, _ in model.wv.vocab.items():
    v_ = model.wv.vocab[k]
    vocab.append([k, v_.index, v_.count])


# In[13]:


vocab = (pd.DataFrame(vocab, 
                     columns=['token', 'idx', 'count'])
         .sort_values('count', ascending=False))


# In[14]:


vocab.info()


# In[15]:


vocab.head(10)


# In[16]:


vocab['count'].describe(percentiles=np.arange(.1, 1, .1)).astype(int)


# ### Evaluate Analogies

# In[110]:


def eval_analogies(w2v, max_vocab=15000):
    accuracy = w2v.wv.accuracy(ANALOGIES_PATH,
                               restrict_vocab=15000,
                               case_insensitive=True)
    return (pd.DataFrame([[c['section'],
                        len(c['correct']),
                        len(c['incorrect'])] for c in accuracy],
                      columns=['category', 'correct', 'incorrect'])
          .assign(average=lambda x: 
                  x.correct.div(x.correct.add(x.incorrect))))  


# In[52]:


def total_accuracy(w2v):
    df = eval_analogies(w2v)
    return df.loc[df.category == 'total', ['correct', 'incorrect', 'average']].squeeze().tolist()


# In[42]:


accuracy = eval_analogies(model)
accuracy


# ### Validate Vector Arithmetic

# In[105]:


pd.read_csv(ANALOGIES_PATH, header=None, sep=' ').head()


# In[112]:


sims=model.wv.most_similar(positive=['iphone'], 
                           restrict_vocab=15000)
print(pd.DataFrame(sims, columns=['term', 'similarity']))


# In[113]:


analogy = model.wv.most_similar(positive=['france', 'london'], 
                                negative=['paris'], 
                                restrict_vocab=15000)
print(pd.DataFrame(analogy, columns=['term', 'similarity']))


# ### Check similarity for random words

# In[41]:


VALID_SET = 5  # Random set of words to get nearest neighbors for
VALID_WINDOW = 100  # Most frequent words to draw validation set from
valid_examples = np.random.choice(VALID_WINDOW, size=VALID_SET, replace=False)
similars = pd.DataFrame()

for id in sorted(valid_examples):
    word = vocab.loc[id, 'token']
    similars[word] = [s[0] for s in model.wv.most_similar(word)]
similars


# ## Continue Training

# In[ ]:


accuracies = (eval_analogies(model)
              .set_index('category')
              .average
              .to_frame('baseline'))


# In[76]:


for i in range(1, 11):
    start = time()
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    accuracy = eval_analogies(model).set_index('category').average
    accuracies = accuracies.join(accuracy.to_frame(f'{n}'))
    print(f'{i} | Duration: {format_time(time() - start)} | Accuracy: {accuracy.total:.2%}')
    model.save(f'word2vec/models/word2vec_{i}.model')


# In[ ]:


model.wv.save('word_vectors_final.bin')

