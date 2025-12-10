# Comparative Study of Multimodal Representations

## Amazon Reviews

- **Files:** `train.csv`, `test.csv`, `val.csv`
- **Columns:**
    - `full_text`: review title and review text (lowercased, cleaned).
    - `label`: sentiment class (0=negative, 1=positive).
- **Notes:** no missing values, total size: ~3.6M train, 400k test.


## Fashion

- **Files:** `train.csv`, `val.csv`, `test.cvs`
- **Columns:**
    - `image_path`: absolute path to product image.
    - `description`: cleaned product description.
    - `label`: product category (e.g. "Tshirts", "Casual Shoes", etc.).
- **Notes:** all rows have valid image files and labels.


## MovieLens 20M

- **Files:** `train.csv`, `val.csv`, `test.csv`
- **Columns:**
    - `userId`, `movieId`, `rating`, `timestamp`: original MovieLens fields.
    - `user_idx`, `movie_idx`: contiguous indices for embedding layers.
    - `title`: movie title.
    - `genres`: pipe-separated list of genres.
    - `tag`: concatenated user tags per movie.
- **Additional files:**
    - `user2idx.csv`, `movie2idx.csv`: mapping tables.
    - `movie_genome_vectors.csv`: 1129-dim tag embedding per movie.
    - `movie_similarity_edges.csv`: (movie1, movie2, similarity score) for semantic graph.
    - `edge_list.csv`: (user_idx, movie_idx) interactions.
- **Notes:** no missing values. User/movie indices align with embedding layers. Graph features for advanced modeling.


## Visual Genome

- **Files:** `visualgenome_dataset.jsonl`, `visualgenome_stats.json`
- **Columns:**
    - `image_id`
    – `image_path`: relative path to the JPEG,
    – `caption` : joined region phrases,
    – `num_regions`: number of region descriptions,
    – `object_labels`: list of object category strings,
    – `graph_features`: normalized predicate frequency vector of length graph_top_k.


#### Data Integrity: all splits are randomized with fixed seed (42) for reproducibility.

---

## commands to set up the environment path to run the models for training.
export PROJECT_ROOT="D:/COmparative_Study_of_Multimodal_Represenations"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"




