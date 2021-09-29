#!/usr/bin/env python
# coding: utf-8

# # How to train your own word vector embeddings

# Many tasks require embeddings or domain-specific vocabulary that pre-trained models based on a generic corpus may not represent well or at all. Standard word2vec models are not able to assign vectors to out-of-vocabulary words and instead use a default vector that reduces their predictive value.
# 
# E.g., when working with industry-specific documents, the vocabulary or its usage may change over time as new technologies or products emerge. As a result, the embeddings need to evolve as well. In addition, corporate earnings releases use nuanced language not fully reflected in Glove vectors pre-trained on Wikipedia articles.
# 
# We will illustrate the word2vec architecture using the keras library that we will introduce in more detail in the next chapter and the more performant gensim adaptation of the code provided by the word2vec authors. 

# To illustrate the word2vec network architecture, we use the TED talk dataset with aligned English and Spanish subtitles that we first introduced in chapter 13. 
# 
# This notebook contains the code to tokenize the documents and assign a unique id to each item in the vocabulary. We require at least five occurrences in the corpus and keep a vocabulary of 31,300 tokens.

# ## Imports

# In[1]:


from time import time
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist, cosine

import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Dot, Embedding
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.callbacks import Callback, TensorBoard

from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import IncrementalPCA


# ### Settings

# In[2]:


plt.style.use('ggplot')
pd.set_option('float_format', '{:,.2f}'.format)
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)


# In[3]:


PROJECT_DIR = Path().cwd()


# In[4]:


LANGUAGES = ['en', 'es']
language_dict = dict(zip(LANGUAGES, ['English', 'Spanish']))


# ## TED2013 Corpus Statistics

# The inputs are produced by the [preprocessing](01_preprocessing.ipynb) notebook that needs to run first.

# In[5]:


SOURCE = 'TED'
LANGUAGE = 'en'


# In[6]:


with pd.HDFStore(Path('vocab', SOURCE, 'vocab.h5')) as store:
    print(store.info())


# In[7]:


with pd.HDFStore(Path('vocab', SOURCE, 'vocab.h5')) as store:
    df = store['{}/vocab'.format(LANGUAGE)]

wc = df['count'].value_counts().sort_index(ascending=False).reset_index()
wc.columns = ['word_count', 'freq']
wc['n_words'] = wc.word_count.mul(wc.freq)

wc['corpus_share'] = wc.n_words.div(wc.n_words.sum())
wc['coverage'] = wc.corpus_share.cumsum()
wc['vocab_size'] = wc.freq.cumsum()

print('# words: {:,d}'.format(wc.n_words.sum()))
(wc
 .loc[:, ['word_count', 'freq', 'n_words', 'vocab_size', 'coverage']]
 .head()
 .append(wc
         .loc[:, ['word_count', 'freq', 'n_words', 'vocab_size', 'coverage']]
         .tail()))


# In[8]:


wc.word_count.div(wc.n_words.sum()).mul(100).describe(percentiles=[.5, .75, .95, .96, .97, .98, .99, .999])


# ## Evaluation: Analogies

# The dimensions of the word and phrase vectors do not have an explicit meaning. However, the embeddings encode similar usage as proximity in the latent space in a way that carries over to semantic relationships. This results in the interesting properties that analogies can be expressed by adding and subtracting word vectors.
# 
# Just as words can be used in different contexts, they can be related to other words in different ways, and these relationships correspond to different directions in the latent space. Accordingly, there are several types of analogies that the embeddings should reflect if the training data permits.
# 
# The word2vec authors provide a list of several thousand relationships spanning aspects of geography, grammar and syntax, and family relationships to evaluate the quality of embedding vectors (see directory [analogies](data/analogies)).

# In[9]:


df = pd.read_csv(Path('data', 'analogies', 'analogies-en.txt'), header=None, names=['category'], squeeze=True)
categories = df[df.str.startswith(':')]
analogies = df[~df.str.startswith(':')].str.split(expand=True)
analogies.columns = list('abcd')


# In[10]:


df = pd.concat([categories, analogies], axis=1)
df.category = df.category.ffill()
df = df[df['a'].notnull()]
df.head()


# In[11]:


analogy_cnt = df.groupby('category').size().sort_values(ascending=False).to_frame('n')
analogy_example = df.groupby('category').first()


# In[12]:


analogy_cnt.join(analogy_example)


# ## `word2vec` - skipgram Architecture using Keras

# ### Settings

# In[11]:


NGRAMS = 3                                # Longest ngram in text
FILE_NAME = 'ngrams_{}'.format(NGRAMS)    # Input to use
MIN_FREQ = 5
SAMPLING_FACTOR = 1e-4
WINDOW_SIZE = 5
EMBEDDING_SIZE = 200
EPOCHS = 1
BATCH_SIZE = 50

# Set up validation
VALID_SET = 10      # Random set of words to get nearest neighbors for
VALID_WINDOW = 150  # Most frequent words to draw validation set from
NN = 10             # Number of nearest neighbors for evaluation

valid_examples = np.random.choice(VALID_WINDOW, size=VALID_SET, replace=False)


# In[12]:


path = Path('keras', SOURCE, LANGUAGE, FILE_NAME).resolve()
tb_path = path / 'tensorboard'
if not tb_path.exists():
    tb_path.mkdir(parents=True, exist_ok=True)


# ### Build Data Set

# #### Tokens to ID
# 
# 1. Extract the top *n* most common words to learn embeddings
# 2. Index these *n* words with unique integers
# 3. Create an `{index: word}` dictionary
# 4. Replace the *n* words with their index, and a dummy value `UNK` elsewhere

# In[15]:


def build_data(language, ngrams=1):
    file_path = PROJECT_DIR / 'vocab' / SOURCE / language / 'ngrams_{}.txt'.format(ngrams)
    words = file_path.read_text().split()
    
    # Get (token, count) tuples for tokens meeting MIN_FREQ 
    token_counts = [t for t in Counter(words).most_common() if t[1] >= MIN_FREQ]
    tokens, counts = list(zip(*token_counts))
    
    # create id-token dicts & reverse dicts
    id_to_token = pd.Series(tokens, index=range(1, len(tokens) + 1)).to_dict()
    id_to_token.update({0: 'UNK'})
    token_to_id = {t:i for i, t in id_to_token.items()}
    data = [token_to_id.get(word, 0) for word in words]
    return data, token_to_id, id_to_token


# In[16]:


data, token_to_id, id_to_token = build_data(LANGUAGE, ngrams=NGRAMS)


# In[17]:


vocab_size = len(token_to_id)


# In[18]:


vocab_size


# In[19]:


min(data), max(data)


# In[20]:


s = pd.Series(data).value_counts().reset_index()
s.columns = ['id', 'count']
s['token'] = s.id.map(id_to_token)


# In[21]:


s.sort_values('count', ascending=False).head(10)


# In[22]:


s.sort_values('id').token.dropna().to_csv(tb_path / 'meta.tsv', index=False)


# #### Analogies to ID

# In[23]:


df = pd.read_csv(Path('data', 'analogies', 'analogies-{}.txt'.format(LANGUAGE)), 
                 header=None, squeeze=True)
categories = df[df.str.startswith(':')]
analogies = df[~df.str.startswith(':')].str.split(expand=True)
analogies.columns = list('abcd')


# In[24]:


analogies.head()


# In[25]:


analogies_id = analogies.apply(lambda x: x.map(token_to_id))
analogies_id.notnull().all(1).sum()/len(analogies_id)


# ### Generate Sampling Probabilities
# 
# There is an alternative, faster scheme than the traditional SoftMax loss function called [Noise Contrastive Estimation (NCE)](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).
# 
# Instead of getting the softmax probability for all possible context words, randomly sample 2-20 possible context words and evaluate the probability only for these.

# **SAMPLING_FACTOR**: used for generating the `sampling_table` argument for `skipgrams`. 
# 
# `sampling_table[i]` is the probability of sampling the word i-th most common word in a dataset
# 
# The sampling probabilities are generated according
# to the sampling distribution used in word2vec:
# 
# $p(\text{word}) = \min(1, \frac{\sqrt{\frac{\text{word frequency}}{\text{sampling factor}}}}{\frac{\text{word frequency}}{\text{sampling factor}}}$

# In[26]:


df = s['count'].to_frame('freq')
factors = [1e-2, 1e-3, 1e-4, 1e-5]
for f in factors:
    sf = make_sampling_table(vocab_size, sampling_factor=f)
    df[f] = df.freq.mul(sf)
df[factors].plot(logy=True);


# In[27]:


sampling_table = make_sampling_table(vocab_size, sampling_factor=SAMPLING_FACTOR)


# In[28]:


pd.Series(sampling_table).plot(title='Skip-Gram Sampling Probabilities')
plt.tight_layout();


# ### Generate target-context word pairs

# In[29]:


pairs, labels = skipgrams(sequence=data,
                            vocabulary_size=vocab_size,
                            window_size=WINDOW_SIZE,
                            sampling_table=sampling_table,
                            negative_samples=1.0,
                            shuffle=True)

print('{:,d} pairs created'.format(len(pairs)))


# In[30]:


pairs[:5]


# In[31]:


target_word, context_word = np.array(pairs, dtype=np.int32).T
labels = np.array(labels, dtype=np.int8)
del pairs


# In[32]:


target_word[:5]


# In[33]:


df = pd.DataFrame({'target': target_word[:5], 'context': context_word[:5], 'label': labels[:5]})
df


# In[34]:


pd.Series(labels).value_counts()


# In[35]:


with pd.HDFStore(path / 'data.h5') as store:
    store.put('id_to_token', pd.Series(id_to_token))


# ### Define Keras Model Components

# #### Scalar Input Variables

# In[36]:


input_target = Input((1,), name='target_input')
input_context = Input((1,), name='context_input')


# #### Shared Embedding Layer

# In[37]:


embedding = Embedding(input_dim=vocab_size,
                      output_dim=EMBEDDING_SIZE,
                      input_length=1,
                      name='embedding_layer')


# In[38]:


target = embedding(input_target)
target = Reshape((EMBEDDING_SIZE, 1), name='target_embedding')(target)

context = embedding(input_context)
context = Reshape((EMBEDDING_SIZE, 1), name='context_embedding')(context)


# #### Create Similarity Measure

# In[39]:


dot_product = Dot(axes=1)([target, context])
dot_product = Reshape((1,), name='similarity')(dot_product)


# #### Sigmoid Output Layer

# In[40]:


output = Dense(units=1, activation='sigmoid', name='output')(dot_product)


# #### Compile Training Model

# In[41]:


model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')


# #### Display Architecture

# In[42]:


model.summary()


# #### Validation Model

# In[43]:


similarity = Dot(normalize=True, 
                 axes=1, 
                 name='cosine_similarity')([target, context])


# In[44]:


# create a secondary validation model to run our similarity checks during training
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)


# In[45]:


validation_model.summary()


# ![Keras Graph](https://s3.amazonaws.com/applied-ai/images/keras_graph_tensorboard.png)

# ### Create Keras Callbacks

# ####  Nearest Neighors & Analogies

# In[46]:


test_set = analogies_id.dropna().astype(int)
a, b, c, actual = test_set.values.T
actual = actual.reshape(-1, 1)
n_analogies = len(actual)


# In[47]:


class EvalCallback(Callback):
        
    def on_train_begin(self, logs={}):
        self.eval_nn()
        self.test_analogies()

    def on_train_end(self, logs={}):
        self.eval_nn()

    def on_epoch_end(self, batch, logs={}):
        self.test_analogies()

    @staticmethod
    def test_analogies():
        print('\nAnalogy Accuracy:\n\t', end='')
        embeddings = embedding.get_weights()[0]
        target = embeddings[c] + embeddings[b] - embeddings[a]
        neighbors = np.argsort(cdist(target, embeddings, metric='cosine'))
        match_id = np.argwhere(neighbors == actual)[:, 1]
        print('\n\t'.join(['Top {}: {:.2%}'.format(i, (match_id < i).sum() / n_analogies) for i in [1, 5, 10]]))

    def eval_nn(self):
        print('\n{} Nearest Neighbors:'.format(NN))
        for i in range(VALID_SET):
            valid_id = valid_examples[i]
            valid_word = id_to_token[valid_id]
            similarity = self._get_similiarity(valid_id).reshape(-1)
            nearest = (-similarity).argsort()[1:NN + 1]
            neighbors = [id_to_token[nearest[n]] for n in range(NN)]
            print('{}:\t{}'.format(valid_word, ', '.join(neighbors)))            
        
    @staticmethod
    def _get_similiarity(valid_word_idx):
        target = np.full(shape=vocab_size, fill_value=valid_word_idx)
        context = np.arange(vocab_size)
        return validation_model.predict([target, context])


evaluation = EvalCallback()


# #### Tensorboard Callback

# In[48]:


tensorboard = TensorBoard(log_dir=str(tb_path),
                          write_graph=True,
                          embeddings_freq=1,
                          embeddings_metadata=str(tb_path / 'meta.tsv'))


# ### Train Model

# In[ ]:


loss = model.fit(x=[target_word, context_word],
                 y=labels,
                 shuffle=True,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 callbacks=[evaluation, tensorboard])

model.save(str(path / 'skipgram_model.h5'))


# ## Optimized TensorFlow Model
# 
# Compile custom ops using `compile-ops.sh`.
# 
# Run from command line.

# In[46]:


get_ipython().system('ls tensorflow/')


# In[49]:


# %%bash
# python tensorflow/word2vec.py --language=en --source=Ted --file=ngrams_1 --embedding_size=300 --num_neg_samples=20 --starter_lr=.1 --target_lr=.05 --batch_size=10 --min_count=10 --window_size=10


# ## word2vec using Gensim

# ### Evaluation

# In[50]:


def accuracy_by_category(acc, detail=True):
    results = [[c['section'], len(c['correct']), len(c['incorrect'])] for c in acc]
    results = pd.DataFrame(results, columns=['category', 'correct', 'incorrect'])
    results['average'] = results.correct.div(results[['correct', 'incorrect']].sum(1))
    if detail:
        print(results.sort_values('average', ascending=False))
    return results.loc[results.category=='total', ['correct', 'incorrect', 'average']].squeeze().tolist()


# ### Settings

# In[51]:


ANALOGIES_PATH = PROJECT_DIR / 'data' / 'analogies' / 'analogies-{}.txt'.format(LANGUAGE)
gensim_path = PROJECT_DIR / 'gensim' / SOURCE / LANGUAGE / FILE_NAME
if not gensim_path.exists():
    gensim_path.mkdir(parents=True, exist_ok=True)


# ### Sentence Generator

# In[52]:


sentence_path = PROJECT_DIR / 'vocab' / SOURCE / LANGUAGE / '{}.txt'.format(FILE_NAME)
sentences = LineSentence(str(sentence_path))


# ### Model

# In[53]:


start = time()

model = Word2Vec(sentences,
                 sg=1,
                 size=300,
                 window=5,
                 min_count=10,
                 negative=10,
                 workers=8,
                 iter=20,
                 alpha=0.05)

model.wv.save(str(gensim_path / 'word_vectors.bin'))
print('Duration: {:,.1f}s'.format(time() - start))

# gensim computes accuracy based on source text files
detailed_accuracy = model.wv.accuracy(str(ANALOGIES_PATH), case_insensitive=True)

# get accuracy per category
summary = accuracy_by_category(detailed_accuracy)
print('Base Accuracy: Correct {:,.0f} | Wrong {:,.0f} | Avg {:,.2%}\n'.format(*summary))


# In[54]:


most_sim = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
pd.DataFrame(most_sim, columns=['token', 'similarity'])


# In[55]:


similars = pd.DataFrame()
for id in valid_examples:
    word = id_to_token[id]
    similars[word] = [s[0] for s in model.wv.most_similar(id_to_token[id])]
    
similars.T


# #### Continue Training

# In[62]:


accuracies = [summary]
for i in range(1, 11):
    start = time()
    model.train(sentences, epochs=1, total_examples=model.corpus_count)
    detailed_accuracy = model.wv.accuracy(str(ANALOGIES_PATH))
    accuracies.append(accuracy_by_category(detailed_accuracy, detail=False))
    print('{} | Duration: {:,.1f} | Accuracy: {:.2%} '.format(i, time() - start, accuracies[-1][-1]))

pd.DataFrame(accuracies, columns=['correct', 'wrong', 'average']).to_csv(gensim_path / 'accuracies.csv', index=False)
model.wv.save(str(gensim_path / 'word_vectors_final.bin'))


# ## The `google` command-line Tool

# ### Run from Command Line

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'file_name=../data/wiki/en/wiki.txt\ntime ./word2vec -train "$file_name" -output vectors_en.bin - cbow 1 -size 300  -min-count 10  -window 10 -negative 10 -hs 0 -sample 1e-4 -threads 8 -binary 1 -iter 1')


# ### Load Trained Model & Word Vectors via `gensim`

# In[32]:


file_name = 'word2vec/word_vectors/vectors_en.bin'
model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors='ignore')


# In[33]:


vectors = model.vectors[:100000]
vectors /= norm(vectors, axis=1).reshape(-1, 1)
vectors.shape


# In[34]:


words = model.index2word[:100000]
word2id = {w:i for i, w in enumerate(words)}


# ### Compute Accuracy

# In[37]:


analogy_path = PROJECT_DIR / 'data/analogies/analogies-en.txt'
accuracy = model.accuracy(questions=str(analogy_path), restrict_vocab=100000)


# In[38]:


summary = accuracy_by_category(accuracy, detail=True)
print('\nOverall Accuracy: Correct {:,.0f} | Wrong {:,.0f} | Avg {:,.2%}\n'.format(*summary))


# ### Project Data using `tensorboard` Projector

# In[66]:


PROJECTION_LIMIT = 10000
proj_path = Path('word2vec', 'projector')
pd.Series(words).iloc[:PROJECTION_LIMIT].to_csv(proj_path / 'meta_data.tsv', index=False, header=None, sep='\t')
pd.DataFrame(vectors).iloc[:PROJECTION_LIMIT].to_csv(proj_path / 'embeddings.tsv', index=False, header=None, sep='\t')


# ### Project Analogies

# #### Incremental PCA

# In[35]:


pca = IncrementalPCA(n_components=2)

vectors2D = pca.fit_transform(vectors)
pd.Series(pca.explained_variance_ratio_).mul(100)


# #### Group Analogies by Category

# In[39]:


results = pd.DataFrame()
correct = incorrect = 0
for section in accuracy:
    correct += len(section['correct'])
    incorrect += len(section['incorrect'])
    df = pd.DataFrame(section['correct']).apply(lambda x: x.str.lower()).assign(section=section['section'])
    results = pd.concat([results, df])


# #### Identify Analogy most similar in 2D

# In[41]:


def find_most_similar_analogy(v):
    """Find analogy that most similar in 2D"""
    v1 = vectors2D[v[1]] - vectors2D[v[0]]
    v2 = vectors2D[v[3]] - vectors2D[v[2]]
    idx, most_similar = None, np.inf
    
    for i in range(len(v1)):
        similarity = cosine(v1[i], v2[i])
        if similarity < most_similar:
            idx = i
            most_similar = similarity
    return idx


# In[42]:


def get_plot_lims(coordinates):
    xlim, ylim = coordinates.agg(['min', 'max']).T.values
    xrange, yrange = (xlim[1] - xlim[0]) * .1, (ylim[1] - ylim[0]) * .1
    xlim[0], xlim[1] = xlim[0] - xrange, xlim[1] + xrange
    ylim[0], ylim[1] = ylim[0] - yrange, ylim[1] + yrange
    return xlim, ylim


# In[43]:


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 9))
axes = axes.flatten()
fc = ec = 'darkgrey'
for s, (section, result) in enumerate(results.groupby('section')):
    if s > 11:
        continue
        
    df = result.drop('section', axis=1).apply(lambda x: x.map(word2id))
    most_similar_idx = find_most_similar_analogy(df)
    
    best_analogy = result.iloc[most_similar_idx, :4].tolist()

    analogy_idx = [words.index(word) for word in best_analogy]
    best_analogy = [a.capitalize() for a in best_analogy]
    
    coords = pd.DataFrame(vectors2D[analogy_idx])  # xy array
    
    xlim, ylim = get_plot_lims(coords)
    axes[s].set_xlim(xlim)
    axes[s].set_ylim(ylim)

    for i in [0, 2]:
        axes[s].annotate(s=best_analogy[i], xy=coords.iloc[i+1], xytext=coords.iloc[i],
                         arrowprops=dict(width=1,headwidth=5, headlength=5,
                                         fc=fc, ec=ec, shrink=.1),
                         fontsize=12)
    
        axes[s].annotate(best_analogy[i+1], xy=coords.iloc[i+1],
                         xytext=coords.iloc[i+1],
                         va='center', ha='center',
                         fontsize=12, color='darkred' if i == 2 else 'k');

    axes[s].axis('off')
    title = ' '.join([s.capitalize()
                      for s in section.split('-') if not s.startswith('gram')])
    axes[s].set_title(title, fontsize=16)

fig.tight_layout();


# ## Resources
# 
# - [Distributed representations of words and phrases and their compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
# - [Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf?)
# - [Sebastian Ruder's Blog](http://ruder.io/word-embeddings-1/)
