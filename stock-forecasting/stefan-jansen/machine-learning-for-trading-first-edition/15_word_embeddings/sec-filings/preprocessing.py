#!/usr/bin/env python
# coding: utf-8

# # Word vectors from SEC filings using gensim

# In this section, we will learn word and phrase vectors from annual SEC filings using gensim to illustrate the potential value of word embeddings for algorithmic trading. In the following sections, we will combine these vectors as features with price returns to train neural networks to predict equity prices from the content of security filings.
# 
# In particular, we use a dataset containing over 22,000 10-K annual reports from the period 2013-2016 that are filed by listed companies and contain both financial information and management commentary (see chapter 3 on Alternative Data). For about half of 11K filings for companies that we have stock prices to label the data for predictive modeling

# ## Imports & Settings

# In[2]:


from pathlib import Path
import numpy as np
import pandas as pd
from time import time
from collections import Counter
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


# In[3]:


pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


# In[ ]:


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


# ### Logging Setup

# In[4]:


logging.basicConfig(
        filename='preprocessing.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


# ### Paths

# Each filing is a separate text file and a master index contains filing metadata. We extract the most informative sections, namely
# - Item 1 and 1A: Business and Risk Factors
# - Item 7 and 7A: Management's Discussion and Disclosures about Market Risks
# 
# The notebook preprocessing shows how to parse and tokenize the text using spaCy, similar to the approach in chapter 14. We do not lemmatize the tokens to preserve nuances of word usage.
# 
# We use gensim to detect phrases. The Phrases module scores the tokens and the Phraser class transforms the text data accordingly. The notebook shows how to repeat the process to create longer phrases.

# In[80]:


filing_path = Path('data/filings')


# In[ ]:


sections_path = Path('data/sections')
if not sections_path.exists():
    sections_path.mkdir(exist_ok=True)


# ## Identify Sections

# In[ ]:


for i, filing in enumerate(filing_path.glob('*.txt')):
    if i % 500 == 0:
        print(i, end=' ', flush=True)
    filing_id = int(filing.stem)
    items = {}
    for section in filing.read_text().lower().split('°'):
        if section.startswith('item '):
            if len(section.split()) > 1:
                item = section.split()[1].replace('.', '').replace(':', '').replace(',', '')
                text = ' '.join([t for t in section.split()[2:]])
                    if items.get(item) is None or len(items.get(item)) < len(text):
                        items[item] = text

    txt = pd.Series(items).reset_index()
    txt.columns = ['item', 'text']
    txt.to_csv(sections_path / (filing.stem + '.csv'), index=False)


# ## Parse Sections

# Select the following sections:

# In[81]:


sections = ['1', '1a', '7', '7a']


# In[ ]:


clean_path = Path('data/selected_sections')
if not clean_path.exists():
    clean_path.mkdir(exist_ok=True)


# In[ ]:


nlp = spacy.load('en', disable=['ner'])
nlp.max_length = 6000000


# In[ ]:


vocab = Counter()
t = total_tokens = 0
stats = []

start = time()
done = 1
for text_file in sections_path.glob('*.csv'):
    file_id = int(text_file.stem)
    clean_file = clean_path / f'{file_id}.csv'
    if clean_file.exists():
        continue
    items = pd.read_csv(text_file).dropna()
    items.item = items.item.astype(str)
    items = items[items.item.isin(sections)]
    if done % 100 == 0:
        duration = time() - start
        to_go = (to_do - done) * duration / done
        print(f'{done:>5}\t{format_time(duration)}\t{total_tokens / duration:,.0f}\t{format_time(to_go)}')
    
    clean_doc = []
    for _, (item, text) in items.iterrows():
        doc = nlp(text)
        for s, sentence in enumerate(doc.sents):
            clean_sentence = []
            if sentence is not None:
                for t, token in enumerate(sentence, 1):
                    if not any([token.is_stop,
                                token.is_digit,
                                not token.is_alpha,
                                token.is_punct,
                                token.is_space,
                                token.lemma_ == '-PRON-',
                                token.pos_ in ['PUNCT', 'SYM', 'X']]):
                        clean_sentence.append(token.text.lower())
                total_tokens += t
                if len(clean_sentence) > 0:
                    clean_doc.append([item, s, ' '.join(clean_sentence)])
    (pd.DataFrame(clean_doc,
                  columns=['item', 'sentence', 'text'])
     .dropna()
     .to_csv(clean_file, index=False))
    done += 1


# ## Create ngrams

# In[4]:


ngram_path = Path('data', 'ngrams')
stats_path = Path('corpus_stats')


# In[5]:


def create_unigrams(min_length=3):
    texts = []
    sentence_counter = Counter()
    unigrams = ngram_path / 'ngrams_1.txt'
    vocab = Counter()
    for f in path.glob('*.csv'):
        df = pd.read_csv(f)
        df.item = df.item.astype(str)
        df = df[df.item.isin(items)]
        sentence_counter.update(df.groupby('item').size().to_dict())
        for sentence in df.text.str.split().tolist():
            if len(sentence) >= min_length:
                vocab.update(sentence)
                texts.append(' '.join(sentence))
    (pd.DataFrame(sentence_counter.most_common(), 
                  columns=['item', 'sentences'])
     .to_csv(stats_path / 'selected_sentences.csv', index=False))
    (pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
     .to_csv(stats_path / 'sections_vocab.csv', index=False))
    unigrams.write_text('\n'.join(texts))
    return [l.split() for l in texts]


# In[ ]:


start = time()
if not unigrams.exists():
    texts = create_unigrams()
else:
    texts = [l.split() for l in unigrams.open()]
print('Reading: ', format_time(time() - start))


# In[ ]:


def create_ngrams(max_length=3):
    """Using gensim to create ngrams"""

    n_grams = pd.DataFrame()
    start = time()
    for n in range(2, max_length + 1):
        print(n, end=' ', flush=True)

        sentences = LineSentence(f'ngrams_{n - 1}.txt')
        phrases = Phrases(sentences=sentences,
                          min_count=25,  # ignore terms with a lower count
                          threshold=0.5,  # accept phrases with higher score
                          max_vocab_size=40000000,  # prune of less common words to limit memory use
                          delimiter=b'_',  # how to join ngram tokens
                          progress_per=50000,  # log progress every
                          scoring='npmi')

        s = pd.DataFrame([[k.decode('utf-8'), v]
                          for k, v in phrases.export_phrases(sentences)]
                         , columns=['phrase', 'score']).assign(length=n)

        n_grams = pd.concat([n_grams, s])
        grams = Phraser(phrases)
        sentences = grams[sentences]
        Path(f'ngrams_{n}.txt').write_text('\n'.join([' '.join(s) for s in sentences]))

    n_grams = n_grams.sort_values('score', ascending=False)
    n_grams.phrase = n_grams.phrase.str.replace('_', ' ')
    n_grams['ngram'] = n_grams.phrase.str.replace(' ', '_')

    with pd.HDFStore('vocab.h5') as store:
        store.put('ngrams', n_grams)

    print('\n\tDuration: ', format_time(time() - start))
    print('\tngrams: {:,d}\n'.format(len(n_grams)))
    print(n_grams.groupby('length').size())


# In[ ]:


create_ngrams()


# ## Inspect Corpus

# In[40]:


ngrams = pd.read_parquet('corpus_stats/ngrams.parquet')


# In[41]:


ngrams.info()


# In[46]:


percentiles=np.arange(.1, 1, .1).round(2)
ngrams.score.describe(percentiles=percentiles)


# In[72]:


ngrams[ngrams.score>.7].sort_values(['length', 'score']).head(10)


# In[51]:


vocab = pd.read_csv('corpus_stats/sections_vocab.csv').dropna()


# In[52]:


vocab.info()


# In[53]:


vocab.n.describe(percentiles).astype(int)


# In[57]:


tokens = Counter()
for l in Path('data', 'ngrams', 'ngrams_3.txt').open():
    tokens.update(l.split())


# In[58]:


tokens = pd.DataFrame(tokens.most_common(),
                     columns=['token', 'count'])


# In[59]:


tokens.info()


# In[60]:


tokens.loc[tokens.token.str.contains('_'), 'count'].describe(percentiles).astype(int)


# In[74]:


tokens[tokens.token.str.contains('_')].head(20).to_csv('ngram_examples.csv', index=False)


# ## Get returns

# In[ ]:


with pd.HDFStore('../data/assets.h5') as store:
    stocks = store['quandl/wiki/stocks']
    prices = store['quandl/wiki/prices'].adj_close


# In[ ]:


sec = pd.read_csv('data/report_index.csv').rename(columns=str.lower)
sec.date_filed = pd.to_datetime(sec.date_filed)


# In[ ]:


idx = pd.IndexSlice


# In[ ]:


first = sec.date_filed.min() + relativedelta(months=-1)
last = sec.date_filed.max() + relativedelta(months=1)
prices = (prices
          .loc[idx[first:last, :]]
          .unstack().resample('D')
          .ffill()
          .dropna(how='all', axis=1)
          .filter(sec.ticker.unique()))


# In[ ]:


sec = sec.loc[sec.ticker.isin(prices.columns), ['ticker', 'date_filed']]

price_data = []
for ticker, date in sec.values.tolist():
    target = date + relativedelta(months=1)
    s = prices.loc[date: target, ticker]
    price_data.append(s.iloc[-1] / s.iloc[0] - 1)

df = pd.DataFrame(price_data,
                  columns=['returns'],
                  index=sec.index)

print(df.returns.describe())
sec['returns'] = price_data
print(sec.info())
sec.dropna().to_csv('data/sec_returns.csv', index=False)

