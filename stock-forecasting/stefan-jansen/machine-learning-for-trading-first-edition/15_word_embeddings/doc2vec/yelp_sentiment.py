#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('stopwords')


# In[2]:


from pathlib import Path
import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import logging
import warnings
from random import shuffle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import class_weight
import umap


# In[3]:


warnings.filterwarnings('ignore')
pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)


# In[4]:


logging.basicConfig(
        filename='doc2vec.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')


# ## Load Data

# In[38]:


df = pd.read_parquet('combined.parquet', engine='fastparquet').loc[:, ['stars', 'text']]


# In[39]:


df.stars.value_counts()


# In[91]:


stars = range(1, 6)


# In[40]:


sample = pd.concat([df[df.stars==s].sample(n=100000) for s in stars])


# In[41]:


sample.info()


# In[66]:


sample.stars = (sample.stars == 5).astype(int)


# In[42]:


sample.stars.value_counts()


# In[43]:


sample.to_parquet('yelp_sample_5.parquet')


# In[12]:


sample = pd.read_parquet('yelp_sample.parquet').reset_index(drop=True)


# In[44]:


sample.head()


# In[17]:


sns.distplot(sample.text.str.split().str.len());


# ## Doc2Vec

# ### Basic text cleaning

# In[45]:


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def clean(review):
    tokens = tokenizer.tokenize(review)
    return ' '.join([t for t in tokens if t not in stopword_set])


# In[46]:


sample.text = sample.text.str.lower().apply(clean)


# In[47]:


sample.sample(n=10)


# In[48]:


sample = sample[sample.text.str.split().str.len()>10]
sample.info()


# ### Create sentence stream

# In[49]:


sentences = []
for i, (_, text) in enumerate(sample.values):
    sentences.append(TaggedDocument(words=text.split(), tags=[i]))


# ### Formulate the model

# In[50]:


size=300
window=5
min_count=0
epochs=5
negative=5
dm = 1
dm_concat=0
dbow_words=0
workers = 8


# In[ ]:


model = Doc2Vec(documents=sentences,
                dm=1,
                size=size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=epochs,
                negative=negative,
                dm_concat=dm_concat,
                dbow_words=dbow_words)


# In[51]:


model = Doc2Vec(documents=sentences,
                dm=dm,
                size=size,
                window=window,
                min_count=min_count,
                workers=workers,
                epochs=epochs,
                negative=negative,
                dm_concat=dm_concat,
                dbow_words=dbow_words)


# In[90]:


model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)


# In[52]:


pd.DataFrame(model.most_similar('good'), columns=['token', 'similarity'])


# ## Persist Model

# In[53]:


model.save('sample5.model')


# In[6]:


model = Doc2Vec.load('sample.model')


# ## Evaluate

# In[62]:


y = sample.stars.sub(1)


# In[55]:


X = np.zeros(shape=(len(y), size))
for i in range(len(sample)):
    X[i] = model.docvecs[i]


# In[56]:


X.shape


# ### Train-Test Split

# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[64]:


mode = pd.Series(y_train).mode().iloc[0]
baseline = accuracy_score(y_true=y_test, y_pred=np.full_like(y_test, fill_value=mode))
print(f'Baseline Score: {baseline:.2%}')


# In[26]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


# In[27]:


class_weights


# ## LightGBM

# In[65]:


train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = train_data.create_valid(X_test, label=y_test)


# In[66]:


params = {'objective': 'multiclass',
          'num_classes': 5}


# In[67]:


lgb_model = lgb.train(params=params,
                      train_set=train_data,
                      num_boost_round=250,
                      valid_sets=[train_data, test_data],
                      verbose_eval=25)


# In[72]:


y_pred = np.argmax(lgb_model.predict(X_test), axis=1)


# In[88]:


cm = confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[99]:


sns.heatmap(pd.DataFrame(cm/np.sum(cm), 
                         index=stars, 
                         columns=stars), 
            annot=True, 
            cmap='Blues', 
            fmt='.1%')


# In[81]:


accuracy_score(y_true=y_test, y_pred=y_pred)


# In[36]:


roc_auc_score(y_score=lgb_model.predict(X_test), y_true=y_test)


# In[55]:


pd.DataFrame(lgb_model.predict(X_test)).describe()


# ## Random Forest

# In[28]:


rf = RandomForestClassifier(n_jobs=-1,  
                            n_estimators=100,
                            class_weight='balanced_subsample')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred):.2%}')


# In[38]:


y_pred_prob = rf.predict_proba(X_test)


# In[39]:


pd.DataFrame(y_pred_prob).describe()


# In[36]:


pd.Series(y_pred).value_counts()


# In[32]:


pd.Series(y_train).value_counts()


# In[33]:


(y_test == 0).mean()


# In[29]:


confusion_matrix(y_true=y_test, y_pred=y_pred)


# ## Logistic Regression

# ### Binary Classification

# In[44]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f'Accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred):.2%}')


# ### Multinomial Classification

# In[100]:


lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f'Accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred):.2%}')


# In[101]:


confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[ ]:




