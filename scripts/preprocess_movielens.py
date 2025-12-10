import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/raw/movielens')
PROCESSED = Path('D:/COmparative_Study_of_Multimodal_Represenations/data/processed/movielens')
PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Load data ---
ratings = pd.read_csv(RAW / 'rating.csv')
movies = pd.read_csv(RAW / 'movie.csv')
tags = pd.read_csv(RAW / 'tag.csv')
genome_tags = pd.read_csv(RAW / 'genome_tags.csv')
genome_scores = pd.read_csv(RAW / 'genome_scores.csv')

# --- Remap user/movie to contiguous indices for embedding/graph ---
user2idx = {u: i for i, u in enumerate(sorted(ratings['userId'].unique()))}
movie2idx = {m: i for i, m in enumerate(sorted(ratings['movieId'].unique()))}
ratings['user_idx'] = ratings['userId'].map(user2idx)
ratings['movie_idx'] = ratings['movieId'].map(movie2idx)
movies['movie_idx'] = movies['movieId'].map(movie2idx)

# --- Merge movie features ---
movies = movies.dropna(subset=['movieId'])
movies['genres'] = movies['genres'].fillna('unknown')
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(map(str, x))).reset_index()
movies = movies.merge(movie_tags, on='movieId', how='left')
movies['tag'] = movies['tag'].fillna('')

# --- Movie genome feature: vector of genome tag relevance scores ---
movie_tag_matrix = genome_scores.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)

# -- Robust filter: only keep movies WITH genome info --
valid_movie_ids = set(movie_tag_matrix.index) & set(movies['movieId'])
movies = movies[movies['movieId'].isin(valid_movie_ids)].copy()
movie_tag_matrix = movie_tag_matrix.loc[movies['movieId']]

# -- Also filter ratings to only those movies --
ratings = ratings[ratings['movieId'].isin(valid_movie_ids)].copy()

# -- Rebuild movie2idx with filtered movie list --
movie2idx = {m: i for i, m in enumerate(sorted(movies['movieId'].unique()))}
movies['movie_idx'] = movies['movieId'].map(movie2idx)
ratings['movie_idx'] = ratings['movieId'].map(movie2idx)


# Optional: normalize each movie's tag vector
from sklearn.preprocessing import normalize
movie_tag_matrix = pd.DataFrame(
    normalize(movie_tag_matrix.values, axis=1),
    index=movie_tag_matrix.index,
    columns=movie_tag_matrix.columns
)
movie_tag_matrix.to_csv(PROCESSED / 'movie_genome_vectors.csv')

# --- Save user/movie index mappings for graph construction/model embedding ---
pd.DataFrame(list(user2idx.items()), columns=['userId','user_idx']).to_csv(PROCESSED/'user2idx.csv',index=False)
pd.DataFrame(list(movie2idx.items()), columns=['movieId','movie_idx']).to_csv(PROCESSED/'movie2idx.csv',index=False)

# --- Create and save bipartite edge list for graph (user ↔ movie) ---
edge_list = ratings[['user_idx', 'movie_idx']]
edge_list.to_csv(PROCESSED / 'edge_list.csv', index=False)

# --- (Optional) Movie–movie semantic similarity graph (cosine sim of genome tag vectors) ---
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(movie_tag_matrix.values)
sim_thresh = 0.5  # You can tune this
movie_indices = movie_tag_matrix.index.values
edges = []
for i in range(len(movie_indices)):
    for j in range(i+1, len(movie_indices)):
        sim = sim_matrix[i, j]
        if sim > sim_thresh:
            edges.append((movie2idx[movie_indices[i]], movie2idx[movie_indices[j]], sim))
movie_sim_graph = pd.DataFrame(edges, columns=['movie_idx_1', 'movie_idx_2', 'similarity'])
movie_sim_graph.to_csv(PROCESSED / 'movie_similarity_edges.csv', index=False)
print(f"Saved {len(movie_sim_graph)} strong movie-movie similarity edges.")

# --- Prepare main ML table for downstream experiments (ratings + movie text features) ---
ratings = ratings.merge(
    movies[['movieId', 'title', 'genres', 'tag', 'movie_idx']],
    on=['movieId', 'movie_idx'],
    how='left'
)
# Shuffle and split
ratings = ratings.sample(frac=1, random_state=42)
n_train = int(0.8 * len(ratings))
n_val = int(0.1 * len(ratings))
ratings.iloc[:n_train].to_csv(PROCESSED / 'train.csv', index=False)
ratings.iloc[n_train:n_train+n_val].to_csv(PROCESSED / 'val.csv', index=False)
ratings.iloc[n_train+n_val:].to_csv(PROCESSED / 'test.csv', index=False)

print("MovieLens advanced preprocessing done!")
