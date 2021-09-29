#!/usr/bin/env python
# coding: utf-8

# # Topic Modeling: Yelp Business Reviews

# This notebook contains an example of LDA applied to six million business review on yelp.

# ## Imports & Settings

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact, FloatRangeSlider

# spacy for language processing
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# sklearn for feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# gensim for topic models
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus

# topic model viz
import pyLDAvis
from pyLDAvis.gensim import prepare


# In[3]:


plt.style.use('fivethirtyeight')
pyLDAvis.enable_notebook()
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format


# In[4]:


stop_words = set(pd.read_csv('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words',
                             header=None,
                             squeeze=True).tolist())


# In[5]:


experiment_path = Path('experiments')
data_path = Path('data')
clean_path = Path('data', 'clean_reviews.txt')


# ## Load Yelp Reviews

# In[6]:


data_path = Path('..', '..', 'data', 'yelp')


# In[7]:


reviews = pd.read_parquet(data_path / 'combined.parquet')
reviews.info(null_counts=True)


# ### Tokens per review

# In[8]:


sns.distplot(reviews.text.str.split().str.len(), kde=False);


# In[9]:


token_count = Counter()
for i, doc in enumerate(reviews.text.tolist(), 1):
    if i % 1e6 == 0:
        print(i, end=' ', flush=True)
    token_count.update(doc.split())


# In[10]:


(pd.DataFrame(token_count.most_common(), columns=['token', 'count'])
 .pipe(lambda x: x[~x.token.str.lower().isin(stop_words)])
 .set_index('token')
 .squeeze()
 .iloc[:50]
 .sort_values()
 .plot
 .barh(figsize=(8, 10)));


# ### Preprocessing

# In[12]:


def clean_doc(d):
    doc = []
    for t in d:
        if not any([t.is_stop, t.is_digit, not t.is_alpha, t.is_punct, t.is_space, t.lemma_ == '-PRON-']):        
            doc.append(t.lemma_)
    return ' '.join(doc)    


# In[23]:


nlp = spacy.load('en')
nlp.max_length = 6000000
nlp.disable_pipes('ner')


# In[24]:


nlp.pipe_names


# In[ ]:


iter_reviews = (review for review in reviews.text)
clean_reviews = []
for i, doc in enumerate(nlp.pipe(iter_reviews, batch_size=100, n_threads=8)):
    if i % 10000 == 0: 
        print(f'{i/len(reviews):.2%}', end=' ', flush=True)
    clean_reviews.append(clean_doc(doc))


# In[ ]:


clean_reviews = [clean_doc(doc) for doc in parsed_reviews]


# In[ ]:


clean_path.write_text('\n'.join(clean_reviews))


# ## Vectorize data

# In[6]:


docs = clean_path.read_text().split('\n')
len(docs)


# ### Explore cleaned data

# In[24]:


review_length, token_count = [], Counter()
for i, doc in enumerate(docs, 1):
    if i % 1e6 == 0:
        print(i, end=' ', flush=True)
    d = doc.split()
    review_length.append(len(d))
    token_count.update(d)


# In[43]:


fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
(pd.DataFrame(token_count.most_common(), columns=['token', 'count'])
 .pipe(lambda x: x[~x.token.str.lower().isin(stop_words)])
 .set_index('token')
 .squeeze()
 .iloc[:25]
 .sort_values()
 .plot
 .barh(ax=axes[0]))

sns.boxenplot(x=pd.Series(review_length), ax=axes[1]);


# In[44]:


pd.Series(review_length).describe(percentiles=np.arange(.1, 1.0, .1))


# In[7]:


docs[:2]


# In[10]:


reviews.text.head(2)


# ### Set vocab parameters

# In[ ]:


min_df = 1000
max_df = .2
ngram_range = (1, 1)
binary = False


# In[319]:


vectorizer = CountVectorizer(stop_words='english',
                             min_df=min_df,
                             max_df=max_df,
                             ngram_range=ngram_range,
                             binary=binary)
dtm = doc_vect.fit_transform(docs)
tokens = doc_vect.get_feature_names()
dtm.shape


# In[321]:


corpus = Sparse2Corpus(dtm, documents_columns=False)
id2word = pd.Series(tokens).to_dict()
dictionary = Dictionary.from_corpus(corpus, id2word)


# ## Train & Evaluate LDA Model

# In[45]:


def show_word_list(model, corpus, top=10, save=False):
    top_topics = model.top_topics(corpus=corpus, coherence='u_mass', topn=20)
    words, probs = [], []
    for top_topic, _ in top_topics:
        words.append([t[1] for t in top_topic[:top]])
        probs.append([t[0] for t in top_topic[:top]])

    fig, ax = plt.subplots(figsize=(model.num_topics*1.2, 5))
    sns.heatmap(pd.DataFrame(probs).T,
                annot=pd.DataFrame(words).T,
                fmt='',
                ax=ax,
                cmap='Blues',
                cbar=False)
    fig.tight_layout()
    if save:
        fig.savefig('yelp_wordlist', dpi=300)


# In[46]:


def show_coherence(model, corpus, tokens, top=10, cutoff=0.01):
    top_topics = model.top_topics(corpus=corpus, coherence='u_mass', topn=20)
    word_lists = pd.DataFrame(model.get_topics().T, index=tokens)
    order = []
    for w, word_list in word_lists.items():
        target = set(word_list.nlargest(top).index)
        for t, (top_topic, _) in enumerate(top_topics):
            if target == set([t[1] for t in top_topic[:top]]):
                order.append(t)

    fig, axes = plt.subplots(ncols=2, figsize=(15,5))
    title = f'# Words with Probability > {cutoff:.2%}'
    (word_lists.loc[:, order]>cutoff).sum().reset_index(drop=True).plot.bar(title=title, ax=axes[1]);

    umass = model.top_topics(corpus=corpus, coherence='u_mass', topn=20)
    pd.Series([c[1] for c in umass]).plot.bar(title='Topic Coherence', ax=axes[0])
    fig.tight_layout();


# In[47]:


def show_top_docs(model, corpus, docs):
    doc_topics = model.get_document_topics(corpus)
    df = pd.concat([pd.DataFrame(doc_topic, 
                                 columns=['topicid', 'weight']).assign(doc=i) 
                    for i, doc_topic in enumerate(doc_topics)])

    for topicid, data in df.groupby('topicid'):
        print(topicid, docs[int(data.sort_values('weight', ascending=False).iloc[0].doc)])
        print(pd.DataFrame(lda.show_topic(topicid=topicid)))


# In[322]:


num_topics=25
chunksize=2000
passes=10
update_every=None
alpha='auto'
eta='auto'
decay=0.5
offset=1.0
eval_every=None
iterations=50
gamma_threshold=0.001
minimum_probability=0.01
minimum_phi_value=0.01
per_word_topics=False


# In[323]:


lda_model = LdaModel(corpus=doc_corpus,
                     id2word=doc_id2word,
                     num_topics=num_topics,
                     chunksize=chunksize,
                     update_every=update_every,
                     alpha=alpha,
                     eta=eta,
                     decay=decay,
                     offset=offset,
                     eval_every=eval_every,
                     passes=passes,
                     iterations=iterations,
                     gamma_threshold=gamma_threshold,
                     minimum_probability=minimum_probability,
                     minimum_phi_value=minimum_phi_value,
                     random_state=42)


# In[ ]:


2 ** (-lda_model.log_perplexity(exp_test_corpus))


# We show results for one model using a vocabulary of 3,800 tokens based on min_df=0.1% and max_df=25% with a single pass to avoid length training time for 20 topics. We can use pyldavis topic_info attribute to compute relevance values for lambda=0.6 that produces the following word list 

# In[ ]:


show_word_list(model=lda_model, corpus=exp_corpus)


# In[ ]:


show_coherence(model=lda_model, corpus=exp_corpus, tokens=exp_tokens)


# In[325]:


vis = prepare(doc_model, doc_corpus, doc_dictionary, mds='tsne')
pyLDAvis.display(vis)


# In[294]:


vis = prepare(doc_model, doc_corpus, doc_dictionary, mds='tsne')
pyLDAvis.display(vis)


# ## Load Experiments

# ### Load Document-Term Matrix

# In[48]:


max_df = .25    # [.1, .25, .5, 1.0]
min_df = .005   # [.001, .005, .01]
binary= False  # [True, False]


# In[49]:


vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
exp_dtm = sparse.load_npz(vocab_path / f'dtm.npz')
exp_tokens = pd.read_csv(vocab_path / f'tokens.csv', header=None, squeeze=True)
exp_dtm.shape


# In[50]:


exp_id2word = exp_tokens.to_dict()
exp_corpus = Sparse2Corpus(exp_dtm, documents_columns=False)
exp_dictionary = Dictionary.from_corpus(exp_corpus, exp_id2word)


# In[51]:


exp_train_dtm, exp_test_dtm = train_test_split(exp_dtm, test_size=.1)
exp_test_dtm


# ### Set Model Parameters

# In[52]:


num_topics = 20 # [3, 5, 7, 10, 15, 20, 25, 50]
passes = 1    # [1]


# In[53]:


exp_model_path = vocab_path / str(num_topics) / str(passes)
exp_lda = LdaModel.load(str(exp_model_path / 'lda'))


# In[ ]:





# In[54]:


show_word_list(model=exp_lda, corpus=exp_corpus, save=True)


# In[55]:


show_coherence(model=exp_lda, corpus=exp_corpus, tokens=exp_tokens)


# In[58]:


exp_vis = prepare(exp_lda, exp_corpus, exp_dictionary, mds='tsne')


# In[168]:


pyLDAvis.save_html(exp_vis, 'yelp_ldavis.html')


# In[71]:


pyLDAvis.display(exp_vis)


# In[170]:


terms = exp_vis.topic_info
terms = terms[terms.Category != 'Default']
terms['relevance'] = terms.logprob * .6 + terms.loglift * .4


# In[108]:


top_by_relevance = (terms
                    .groupby('Category')
                    .apply(lambda x: x.nlargest(n=10, columns='relevance'))
                    .reset_index('term', drop=True)
                   .loc[:, ['Term', 'relevance']])
top_by_relevance.head()


# In[156]:


relevance, terms = pd.DataFrame(), pd.DataFrame()
for topic, data in top_by_relevance.groupby(level='Category'):
    t = topic[:5] + f' {int(topic[5:]):0>2}'
    terms[t] = data.Term.tolist()
    relevance[t] = data.relevance.tolist()


# In[159]:


fig, ax = plt.subplots(figsize=(num_topics*1.2, 5))
sns.heatmap(relevance.sort_index(1),
            annot=terms.sort_index(1),
            fmt='',
            ax=ax,
            cmap='Blues',
            cbar=False)
fig.tight_layout()
fig.savefig('yelp_review_wordlist', dpi=300)


# ## LDAMultiCore Timing

# In[160]:


df = pd.read_excel('timings/timings.xlsx')
df.head()


# In[167]:


df[df.num_topics==10].set_index('workers')[['duration', 'test_perplexity']].plot.bar(subplots=True, layout=(1,2), figsize=(14,5), legend=False)

