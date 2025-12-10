# src/comparative/evaluation/latent_analysis.py
import umap
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np

def compute_umap(X, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=random_state)
    return reducer.fit_transform(X)

def compute_tsne(X, n_components=2, random_state=42):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(X)

def compute_silhouette(X, labels):
    return silhouette_score(X, labels)
