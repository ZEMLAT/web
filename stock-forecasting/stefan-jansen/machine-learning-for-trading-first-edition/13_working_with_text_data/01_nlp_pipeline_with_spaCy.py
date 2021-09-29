#!/usr/bin/env python
# coding: utf-8

# # NLP Pipeline with spaCy

# [spaCy](https://spacy.io/) is a widely used python library with a comprehensive feature set for fast text processing in multiple languages. 
# 
# The usage of the tokenization and annotation engines requires the installation of language models. The features we will use in this chapter only require the small models, the larger models also include word vectors that we will cover in chapter 15.

# ![spaCy](assets/spacy.jpg)

# ## Setup

# ### Imports

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacy
from spacy import displacy
import textacy

from IPython.display import SVG, display


# In[2]:


pd.set_option('float_format', '{:,.2f}'.format)


# ### SpaCy Language Model Installation
# 
# In addition to the `spaCy` library, we need [language models](https://spacy.io/usage/models).

# #### English

# In[3]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy download en_core_web_sm\n\n# more comprehensive models:\n# {sys.executable} -m spacy download en_core_web_md\n# {sys.executable} -m spacy download en_core_web_lg')


# #### Spanish

# [Spanish language models](https://spacy.io/models/es#es_core_news_sm) trained on [AnCora Corpus](http://clic.ub.edu/corpus/) and [WikiNER](http://schwa.org/projects/resources/wiki/Wikiner)

# In[4]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy download es_core_news_sm\n\n# more comprehensive model:\n# {sys.executable} -m spacy download es_core_news_md')


# Create shortcut names

# In[6]:


get_ipython().run_cell_magic('bash', '', 'python -m spacy link en_core_web_sm en --force;\npython -m spacy link es_core_news_sm es --force;')


# #### Validate Installation

# In[7]:


# validate installation
get_ipython().system('{sys.executable} -m spacy validate')


# ## Get Data

# Download and unzip into the folder `bbc` in your main data directory.

# In[3]:


DATA_DIR = Path('../data')


# - [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files (download [link](http://mlg.ucd.ie/files/datasets/bbcsport-fulltext.zip)
# - [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus of TED talk subtitles in 15 langugages (sample provided)

# ## SpaCy Pipeline & Architecture

# ### The Processing Pipeline
# 
# When you call a spaCy model on a text, spaCy 
# 
# 1) tokenizes the text to produce a `Doc` object. 
# 
# 2) passes the `Doc` object through the processing pipeline that may be customized, and for the default models consists of
# - a tagger, 
# - a parser and 
# - an entity recognizer. 
# 
# Each pipeline component returns the processed Doc, which is then passed on to the next component.
# 
# ![Architecture](assets/pipeline.svg)

# ### Key Data Structures
# 
# The central data structures in spaCy are the **Doc** and the **Vocab**. Text annotations are also designed to allow a single source of truth:
# 
# - The **`Doc`** object owns the sequence of tokens and all their annotations. `Span` and `Token` are views that point into it. It is constructed by the `Tokenizer`, and then modified in place by the components of the pipeline. 
# - The **`Vocab`** object owns a set of look-up tables that make common information available across documents. 
# - The **`Language`** object coordinates these components. It takes raw text and sends it through the pipeline, returning an annotated document. It also orchestrates training and serialization.
# 
# ![Architecture](assets/spaCy-architecture.svg)

# ## SpaCy in Action

# ### Create & Explore the Language Object

# Once installed and linked, we can instantiate a spaCy language model and then call it on a document. As a result, spaCy produces a Doc object that tokenizes the text and processes it according to configurable pipeline components that by default consist of a tagger, a parser, and a named-entity recognizer.

# In[8]:


nlp = spacy.load('en') 


# In[9]:


type(nlp)


# In[10]:


nlp.lang


# In[11]:


spacy.info('en')


# In[12]:


def get_attributes(f):
    print([a for a in dir(f) if not a.startswith('_')], end=' ')


# In[13]:


get_attributes(nlp)


# ### Explore the Pipeline

# Let’s illustrate the pipeline using a simple sentence:

# In[14]:


sample_text = 'Apple is looking at buying U.K. startup for $1 billion'
doc = nlp(sample_text)


# In[15]:


get_attributes(doc)


# In[16]:


doc.is_parsed


# In[17]:


doc.is_sentenced


# In[18]:


doc.is_tagged


# In[19]:


doc.text


# In[20]:


get_attributes(doc.vocab)


# In[21]:


doc.vocab.length


# #### Explore `Token` annotations

# The parsed document content is iterable and each element has numerous attributes produced by the processing pipeline. The below sample illustrates how to access the following attributes:

# In[22]:


pd.Series([token.text for token in doc])


# In[23]:


pd.DataFrame([[t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.shape_, t.is_alpha, t.is_stop]
              for t in doc],
             columns=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop'])


# #### Visualize POS Dependencies

# We can visualize the syntactic dependency in a browser or notebook

# In[25]:


options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Source Sans Pro', 'notebook': True}


# In[28]:


displacy.serve(doc, style='dep', options=options)


# In[26]:


displacy.render(doc, style='dep', options=options, jupyter=True)


# #### Visualize Named Entities

# In[27]:


displacy.render(doc, style='ent', jupyter=True)


# ### Read BBC Data

# We will now read a larger set of 2,225 BBC News articles (see GitHub for data source details) that belong to five categories and are stored in individual text files. We 
# - call the .glob() method of the pathlib’s Path object, 
# - iterate over the resulting list of paths, 
# - read all lines of the news article excluding the heading in the first line, and 
# - append the cleaned result to a list

# In[7]:


files = (DATA_DIR / 'bbc').glob('**/*.txt')
bbc_articles = []
for i, file in enumerate(files):
    with file.open(encoding='latin1') as f:
        lines = f.readlines()
        body = ' '.join([l.strip() for l in lines[1:]]).strip()
        bbc_articles.append(body)


# In[5]:


len(bbc_articles)


# In[6]:


bbc_articles[0]


# ### Parse first article through Pipeline

# In[31]:


nlp.pipe_names


# In[32]:


doc = nlp(bbc_articles[0])
type(doc)


# ### Detect sentence boundary
# Sentence boundaries are calculated from the syntactic parse tree, so features such as punctuation and capitalisation play an important but non-decisive role in determining the sentence boundaries. 
# 
# Usually this means that the sentence boundaries will at least coincide with clause boundaries, even given poorly punctuated text.

# spaCy computes sentence boundaries from the syntactic parse tree so that punctuation and capitalization play an important but not decisive role. As a result, boundaries will coincide with clause boundaries, even for poorly punctuated text.
# 
# We can access the parsed sentences using the .sents attribute:

# In[37]:


sentences = [s for s in doc.sents]
sentences[:3]


# In[38]:


get_attributes(sentences[0])


# In[39]:


pd.DataFrame([[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[0]], 
             columns=['Token', 'POS Tag', 'Meaning']).head(15)


# In[40]:


options = {'compact': True, 'bg': '#09a3d5',
           'color': 'white', 'font': 'Source Sans Pro'}
displacy.render(sentences[0].as_doc(), style='dep', jupyter=True, options=options)


# In[41]:


for t in sentences[0]:
    if t.ent_type_:
        print('{} | {} | {}'.format(t.text, t.ent_type_, spacy.explain(t.ent_type_)))


# In[42]:


displacy.render(sentences[0].as_doc(), style='ent', jupyter=True)


# ### Named Entity-Recognition with textacy

# spaCy enables named entity recognition using the .ent_type_ attribute:

# Textacy makes access to the named entities that appear in the first article easy:

# In[43]:


from textacy.extract import named_entities
entities = [e.text for e in named_entities(doc)]
pd.Series(entities).value_counts().head()


# ### N-Grams with textacy

# N-grams combine N consecutive tokens. This can be useful for the bag-of-words model because, depending on the textual context, treating, e.g, ‘data scientist’ as a single token may be more meaningful than the two distinct tokens ‘data’ and ‘scientist’.
# 
# Textacy makes it easy to view the ngrams of a given length n occurring with at least min_freq times:

# In[44]:


from textacy.extract import ngrams
pd.Series([n.text for n in ngrams(doc, n=2, min_freq=2)]).value_counts()


# ### The spaCy streaming Pipeline API

# To pass a larger number of documents through the processing pipeline, we can use spaCy’s streaming API as follows:

# In[45]:


iter_texts = (bbc_articles[i] for i in range(len(bbc_articles)))
for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_threads=8)):
    if i % 100 == 0:
        print(i, end = ' ')
    assert doc.is_parsed


# ### Multi-language Features

# spaCy includes trained language models for English, German, Spanish, Portuguese, French, Italian and Dutch, as well as a multi-language model for named-entity recognition. Cross-language usage is straightforward since the API does not change.
# 
# We will illustrate the Spanish language model using a parallel corpus of TED talk subtitles. For this purpose, we instantiate both language models

# #### Create a Spanish Language Object

# In[46]:


model = {}
for language in ['en', 'es']:
    model[language] = spacy.load(language) 


# #### Read bilingual TED2013 samples

# In[48]:


text = {}
path = Path('data/TED')
for language in ['en', 'es']:
    file_name = path /  'TED2013_sample.{}'.format(language)
    text[language] = file_name.read_text()


# #### Sentence Boundaries English vs Spanish

# In[49]:


parsed, sentences = {}, {}
for language in ['en', 'es']:
    parsed[language] = model[language](text[language])
    sentences[language] = list(parsed[language].sents)
    print('Sentences:', language, len(sentences[language]))


# In[50]:


for i, (en, es) in enumerate(zip(sentences['en'], sentences['es']), 1):
    print('\n', i)
    print('English:\t', en)
    print('Spanish:\t', es)
    if i > 5: 
        break


# #### POS Tagging English vs Spanish

# In[51]:


pos = {}
for language in ['en', 'es']:
    pos[language] = pd.DataFrame([[t.text, t.pos_, spacy.explain(t.pos_)] for t in sentences[language][0]],
                                 columns=['Token', 'POS Tag', 'Meaning'])


# In[52]:


bilingual_parsed = pd.concat([pos['en'], pos['es']], axis=1)
bilingual_parsed.head(5).to_csv('data/bilingual.csv', index=False)
bilingual_parsed.head(15)


# In[53]:


displacy.render(sentences['es'][0].as_doc(), style='dep', jupyter=True, options=options)

