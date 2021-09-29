#!/usr/bin/env python
# coding: utf-8

# # t-SNE and UMAP

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from os.path import join
import pandas as pd
import numpy as np
from numpy.random import choice, randint, uniform, randn
import seaborn as sns
import matplotlib.pyplot as plt
import ipyvolume as ipv
from sklearn.datasets import fetch_openml, make_swiss_roll, make_blobs
from sklearn.manifold import TSNE
import umap
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import colorlover as cl
import warnings


# In[2]:


warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.options.display.float_format = '{:,.2f}'.format
init_notebook_mode(connected=True)
ipv_cmap = sns.color_palette("Paired", n_colors=10)


# ## T-Stochastic Neighbor Embedding (TSNE): Parameter Settings

# [t-SNE](https://lvdmaaten.github.io/tsne/) is an award-winning algorithm developed in 2010 by Laurens van der Maaten and Geoff Hinton to detect patterns in high-dimensional data. It takes a probabilistic, non-linear approach to locating data on several different, but related low-dimensional manifolds. The algorithm emphasizes keeping similar points together in low dimensions, as opposed to maintaining the distance between points that are apart in high dimensions, which results from algorithms like PCA that minimize squared distances. 
# 
# The algorithm proceeds by converting high-dimensional distances to (conditional) probabilities, where high probabilities imply low distance and reflect the likelihood of sampling two points based on similarity. It accomplishes this by positioning a normal distribution over each point and computing the density for a point and each neighbor, where the perplexity parameter controls the effective number of neighbors. In a second step, it arranges points in low dimensions and uses similarly computed low-dimensional probabilities to match the high-dimensional distribution. It measures the difference between the distributions using the Kullback-Leibler divergence that puts a high penalty on misplacing similar points in low dimensions. The low-dimensional probabilities use a Student-t distribution with one degree of freedom because it has fatter tails that reduce the penalty of misplacing points that are more distant in high dimensions to manage the crowding problem.

# t-SNE is currently the state-of-the-art in high-dimensional data visualization. Weaknesses include the computational complexity that scales quadratically in the number n of points because it evaluates all pairwise distances, but a subsequent tree-based implementation has reduced the cost to n log n. 
# 
# t-SNE does not facilitate the projection of new data points into the low-dimensional space. The compressed output is not a very useful input for distance- or density-based cluster algorithms because t-SNE treats small and large distances differently.

# ### Perplexity: emphasis on local vs global structure

# In[3]:


data, label = make_blobs(n_samples=200, n_features=2, centers=2, random_state=42)


# In[4]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, p in enumerate([2, 10, 20, 30, 50], 1):
    embedding = TSNE(perplexity=p, n_iter=5000).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Perplexity: {:.0f}'.format(p))
    axes[i].axis('off')
fig.tight_layout()


# ### Convergence with `n_iter`

# In[5]:


data, label = make_blobs(n_samples=200, n_features=2, centers=2, random_state=42)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, n in enumerate([250, 500, 1000, 2500, 5000], 1):
    embedding = TSNE(perplexity=30, n_iter=n).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Iterations: {:,.0f}'.format(n))
    axes[i].axis('off')
fig.tight_layout();


# ### Different Cluster Sizes

# In[6]:


data, label = make_blobs(n_samples=200, n_features=2, cluster_std=[10, 1], centers=2, random_state=42)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, p in enumerate([2,10, 20, 30, 50], 1):
    embedding = TSNE(perplexity=p, n_iter=5000).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Perplexity: {:.0f}'.format(p))
    axes[i].axis('off')
fig.tight_layout();


# ### Different Cluster Distances

# In[7]:


data, label = make_blobs(n_samples=150, n_features=2, centers=[[-10, 0], [-8, 0], [10, 0]], random_state=2)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, p in enumerate([2,10, 30, 50, 100], 1):
    embedding = TSNE(perplexity=p, n_iter=5000).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Perplexity: {:.0f}'.format(p))
    axes[i].axis('off')
fig.tight_layout();


# ### More points require higher perplexity

# In[8]:


data, label = make_blobs(n_samples=600, n_features=2, centers=[[-10, 0], [-8, 0], [10, 0]], random_state=2)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, p in enumerate([2,10, 30, 50, 100], 1): 
    embedding = TSNE(perplexity=p, n_iter=5000).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Perplexity: {:.0f}'.format(p))
    axes[i].axis('off')
fig.tight_layout();


# ## Uniform Manifold Approximation and Projection (UMAP): Parameter Settings

# [UMAP](https://github.com/lmcinnes/umap) is a more recent algorithm for visualization and general dimensionality reduction. It assumes the data is uniformly distributed on a locally connected manifold and looks for the closest low-dimensional equivalent using fuzzy topology. It uses a neighbors parameter that impacts the result similarly as perplexity above.
# 
# It is faster and hence scales better to large datasets than t-SNE, and sometimes preserves global structure than better than t-SNE. It can also work with different distance functions, including, e.g., cosine similarity that is used to measure the distance between word count vectors.

# ### Neighbors

# In[9]:


data, label = make_blobs(n_samples=600, n_features=2, centers=2, random_state=42)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()
axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, n in enumerate([2,10, 20, 30, 50], 1):
    embedding = umap.UMAP(n_neighbors=n, min_dist=0.1).fit_transform(data)    
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Neighbors: {:.0f}'.format(n))
    axes[i].axis('off')
fig.tight_layout();


# ### Minimum Distance

# In[10]:


data, label = make_blobs(n_samples=200, n_features=2, centers=2, random_state=42)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8))
axes = axes.flatten()

axes[0].scatter(data[:, 0], data[:, 1], s=10, c=label)
axes[0].set_title('Original Data')
axes[0].axis('off')
for i, d in enumerate([.001, .01, .1, .2, .5], 1):
    embedding = umap.UMAP(n_neighbors=30, min_dist=d).fit_transform(data)
    axes[i].scatter(embedding[:, 0], embedding[:, 1], s=10, c=label)
    axes[i].set_title('Min. Distance: {:.3f}'.format(d))
    axes[i].axis('off')
fig.tight_layout();


# ## Non-Linear Manifolds: Swiss Roll

# In[11]:


n_samples = 10000
palette = sns.color_palette('viridis', n_colors=n_samples)


# In[12]:


zeros = np.zeros(n_samples) + .5
swiss_3d, swiss_val = make_swiss_roll(
    n_samples=n_samples, noise=.1, random_state=42)

swiss_3d = swiss_3d[swiss_val.argsort()[::-1]]
x, y, z = swiss_3d.T 


# ### TSNE

# Using pre-computed T-SNE and UMAP results due to the long running times, esp. for T-SNE.

# In[13]:


# pre-computed manifold results for the various datasets and algorithms, as well as parameter settings:
with pd.HDFStore(Path('data', 'manifolds.h5')) as store:
    print(store.info())


# In[14]:


fig, axes = plt.subplots(nrows=7, ncols=5, figsize=(20, 28))
method = 'tsne'
with pd.HDFStore(join('data', 'manifolds.h5')) as store:
    labels = store['/'.join(['swiss', 'label'])]
    for row, perplexity in enumerate([2, 5, 10, 20 , 30, 50, 100]):
        for col, n_iter in enumerate([250, 500, 1000, 3000, 5000]):
            x, y = store.get('/'.join(['swiss',  method, str(perplexity), str(n_iter)])).T.values
            axes[row, col].scatter(x, y, c=palette, s=5)
            axes[row, col].set_title('Perplexity: {} | Iterations: {}'.format(perplexity, n_iter))
            axes[row, col].axis('off')
fig.tight_layout()
fig.suptitle('T-Stochastic Neighbor Embedding (TSNE)')
fig.subplots_adjust(top=.94)


# ### UMAP

# In[15]:


fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 20))
method = 'umap'
with pd.HDFStore(join('data', 'manifolds.h5')) as store:
    labels = store['swiss/label']
    for row, n_neighbors in enumerate([2, 5, 10, 25, 50]):
        for col, min_distance in enumerate([1, 10, 20, 50]):
            x, y = store.get('swiss/' + method + '/' + str(n_neighbors) + '/' + str(min_distance)).T.values
            axes[row, col].scatter(x, y, c=palette, s=5)
            axes[row, col].set_title('Neighbors: {} | Min. Distance {:.2f}'.format(n_neighbors, min_distance/100))
            axes[row, col].axis('off')
fig.tight_layout()
fig.suptitle('Uniform Manifold Approximation and Projection (UMAP)')
fig.subplots_adjust(top=.94)


# ## Handwritten Digits

# In[22]:


mnist = fetch_openml('mnist_784', data_home='.')
classes = sorted(np.unique(mnist.target).astype(int))
mnist.data.shape


# In[23]:


ipv_cmap = sns.color_palette("Paired", n_colors=10)
print(classes)
sns.palplot(ipv_cmap)


# ### Plot sample images

# In[24]:


image_size = int(np.sqrt(mnist.data.shape[1])) # 28 x 28 pixels
n_samples = 15


# In[35]:


fig, ax = plt.subplots()
mnist_sample = np.empty(
    shape=(image_size * len(classes), image_size * n_samples))
for row, label in enumerate(classes):
    label_data = np.squeeze(np.argwhere(mnist.target.astype(int) == label))
    samples = choice(label_data, size=n_samples, replace=False)
    i = row * image_size
    for col, sample in enumerate(samples):
        j = col * image_size
        mnist_sample[i:i+image_size, j:j +
                     image_size] = mnist.data[sample].reshape(image_size, -1)

ax.imshow(mnist_sample, cmap='Blues')
plt.title('Handwritten Digits')
plt.axis('off')
plt.tight_layout()


# In[36]:


plotly_cmap = cl.to_rgb( cl.scales['10']['qual']['Paired'])
def plotly_scatter(data, label, title, color, x='x', y='y'):
    fig = dict(
        data=[
            dict(
                type='scattergl',
                x=data[:, 0],
                y=data[:, 1],
                legendgroup="group",
                text=label.astype(int),
                mode='markers',
                marker=Marker(
                    size=5,
                    color=color,
                    autocolorscale=False,
                    showscale=False,
                    opacity=.9,
                    colorbar=ColorBar(
                        title='Class'
                    ),
                    line=dict(width=1))),
        ],
        layout=dict(title=title,
                    width=1200,
                    font=dict(color='white'),
                    xaxis=dict(
                        title=x, 
                        hoverformat='.1f', 
                        showgrid=False),
                    yaxis=dict(title=y, 
                               hoverformat='.1f', 
                               showgrid=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                    ))

    iplot(fig, show_link=False)


# ### t-SNE and UMAP Visualization

# In[37]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 12))
methods = ['tsne', 'umap']
params = {}
params['tsne'] = [5, 10, 20, 35]
params['umap'] = [5, 15, 25, 35]
param_labels = {'tsne': 'Perplexity', 'umap': 'Neighbors'}
with pd.HDFStore(join('data', 'manifolds.h5')) as store:
    labels = store['mnist/labels']
    color = [sns.color_palette('Paired', 10)[int(i)] for i in labels]
    for row, method in enumerate(methods):
        for col, param in enumerate(params[method]):
            x, y = store.get('mnist/' + method + '/2/' + str(param)).T.values
            axes[row, col].scatter(x, y, c=color, s=5)
            axes[row, col].set_title('{} | {}: {}'.format(method.upper(), param_labels[method], param))
            axes[row, col].axis('off')
fig.tight_layout();


# In[38]:


def get_result(source, method, params):
    key = '/'.join([source, method, '/'.join([str(p) for p in params])])
    with pd.HDFStore(join('data', 'manifolds.h5')) as store:
        data = store[key].values
        labels = store['/'.join([source, 'labels'])]
    return data, labels


# ## Load Fashion MNIST Data

# In[16]:


fashion_mnist = pd.read_csv(Path('data', 'fashion-mnist_train.csv.gz'))
fashion_label = fashion_mnist.label
fashion_data = fashion_mnist.drop('label', axis=1).values
classes = sorted(np.unique(fashion_label).astype(int))


# In[17]:


image_size = int(np.sqrt(fashion_data.shape[1])) # 28 x 28 pixels
n_samples = 15


# ### Plot sample images

# In[18]:


fig, ax = plt.subplots(figsize=(14,8))
fashion_sample = np.empty(shape=(image_size * len(classes),
                               image_size * n_samples))
for row, label in enumerate(classes):
    label_data = np.squeeze(np.argwhere(fashion_label == label))
    samples = choice(label_data, size=n_samples, replace=False)
    i = row * image_size
    for col, sample in enumerate(samples):  
        j = col * image_size
        fashion_sample[i:i+image_size,
                     j:j + image_size] = fashion_data[sample].reshape(image_size, -1)

ax.imshow(fashion_sample, cmap='Blues')
plt.title('Fashion Images')
plt.axis('off')
plt.tight_layout();


# ### t-SNE and UMAP: Parameter Settings

# The upper panels of the following chart show how t-SNE is able to differentiate between the image classes. A higher perplexity value increases the number of neighbors used to compute local structure and gradually results in more emphasis on global relationships.

# The below figure illustrates how UMAP does indeed move the different clusters further apart, whereas t-SNE provides more granular insight into the local structure.

# In[19]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(25, 12))
source = 'fashion'
methods = ['tsne', 'umap']
params = {}
params['tsne'] = [10, 20, 30, 50]
params['umap'] = [5, 15, 25, 35]
param_labels = {'tsne': 'Perplexity', 'umap': 'Neighbors'}
with pd.HDFStore(join('data', 'manifolds.h5')) as store:
    labels = store[source + '/labels']
    color = [sns.color_palette('Paired', 10)[int(i)] for i in labels]
    for row, method in enumerate(methods):
        for col, param in enumerate(params[method]):
            x, y = store.get(source + '/' + method + '/2/' + str(param)).T.values
            axes[row, col].scatter(x, y, c=color, s=5)
            axes[row, col].set_title('{} | {}: {}'.format(method.upper(), param_labels[method], param))
            axes[row, col].axis('off')
fig.tight_layout();


# In[21]:


plotly_cmap = cl.to_rgb( cl.scales['10']['qual']['Paired'])
def plotly_scatter(data, label, title, color, x='x', y='y'):
    fig = dict(
        data=[
            dict(
                type='scattergl',
                x=data[:, 0],
                y=data[:, 1],
                legendgroup="group",
                text=label.astype(int),
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    autocolorscale=True,
                    showscale=False,
                    opacity=.9,
                    colorbar=go.ColorBar(
                        title='Class'
                    ),
                    line=dict(width=1))),
        ],
        layout=dict(title=title,
                    width=1200,
                    font=dict(color='white'),
                    xaxis=dict(
                        title=x, 
                        hoverformat='.1f', 
                        showgrid=False),
                    yaxis=dict(title=y, 
                               hoverformat='.1f', 
                               showgrid=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                    ))

    iplot(fig, show_link=False)


# ### Plotly: t-SNE

# In[22]:


data, labels = get_result('fashion', 'tsne', [2, 25])
plotly_color = [plotly_cmap[int(i)] for i in labels]

plotly_scatter(data=data, 
               title='MNIST TSNE Projection',
               label=labels,
               color=plotly_color)


# ### Plotly UMAP

# In[23]:


data, labels = get_result('fashion', 'umap', [2, 15])
plotly_color = [plotly_cmap[int(i)] for i in labels]
plotly_scatter(data=data, 
               title='MNIST UMAP Projection',
               label=labels,
               color=plotly_color)


# ### t-SNE in 3D

# In[24]:


data, labels = get_result('fashion', 'tsne', [3, 25])
ipv_color = [ipv_cmap[int(t)] for t in labels]
ipv.quickscatter(*data.T, size=.5, color=ipv_color, marker='sphere')

