from pca import pca
import numpy as np
import pandas as pd
from matplotlib  import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import NMF


def get_pca_plots(data, list_of_clusters):
    pca_2 = pca(n_components=2)
    pca_2.fit_transform(data)
    pca_2.scatter(labels=list_of_clusters)
    plt.show()

def get_tsne_plots(data, list_of_clusters):
    tSNE = TSNE(random_state=420)
    X_proj = tSNE.fit_transform(data)
    scatter(X_proj, list_of_clusters)
    plt.show()

def get_nmf_plots(data, list_of_clusters):
    nmf = NMF(n_components=2, init='random', random_state=0)
    X_transformed = nmf.fit_transform(data)
    scatter(X_transformed,list_of_clusters)
    plt.show()

def scatter(x, colors):
    palette = np.array(sns.color_palette("hls", len(np.unique(colors))))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int64)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for i in range(len(np.unique(colors))):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txts.append(txt)

    return f, ax, sc, txts