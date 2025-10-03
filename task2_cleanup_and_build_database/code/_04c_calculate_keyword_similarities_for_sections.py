#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "openai",
#   "pandas", 
#   "numpy",
#   "scipy",
#   "tiktoken",
#   "torch",
#   "sentence-transformers",
#   "tqdm",
#   "numba>=0.57.0",
#   "fastdist",
#   "openpyxl",
#   "polars",
#   "pyarrow"
# ]
# ///

from GLOBALS import *
import os
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from numba import jit
from multiprocessing import Pool, cpu_count

@jit(nopython=True)
def cosine_similarity_numba(u, v):
    uv, uu, vv = 0.0, 0.0, 0.0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] ** 2
        vv += v[i] ** 2
    return uv / np.sqrt(uu * vv) if uu and vv else 0

@jit(nopython=True)
def euclidean_distance_numba(u, v):
    return np.sqrt(np.sum((u - v) ** 2))

def calculate_average_similarities(args):
    section_embedding, category_keywords = args
    cosine_similarities = []
    euclidean_distances = []
    for _, keyword_row in category_keywords.iterrows():
        keyword_embedding = np.array(eval(keyword_row['embedding'])) if isinstance(keyword_row['embedding'], str) else np.array(keyword_row['embedding'])
        if section_embedding.shape[0] != keyword_embedding.shape[0]:
            raise ValueError("Mismatched dimensions between section and keyword embeddings.")
        cosine_similarity = cosine_similarity_numba(section_embedding, keyword_embedding)
        euclidean_distance = euclidean_distance_numba(section_embedding, keyword_embedding)
        cosine_similarities.append(cosine_similarity)
        euclidean_distances.append(euclidean_distance)
    average_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else np.nan
    average_euclidean_distance = np.mean(euclidean_distances) if euclidean_distances else np.nan
    return (average_cosine_similarity, average_euclidean_distance)

def add_category_similarity_columns(df: pd.DataFrame, category: str, categories_keywords_embeddings: str) -> pd.DataFrame:
    keywords_df = pd.read_csv(categories_keywords_embeddings)
    category_keywords = keywords_df[keywords_df['category'] == category]
    
    section_embeddings = [(np.array(eval(row['embedding'])) if isinstance(row['embedding'], str) else np.array(row['embedding']), category_keywords) for index, row in df.iterrows()]

    with Pool(cpu_count()) as pool:
        similarities = list(tqdm(pool.imap(calculate_average_similarities, section_embeddings), total=len(section_embeddings), desc=f"Processing {category} - sections"))
    
    df[f'sim_cosine_{category}'] = [sim[0] for sim in similarities]
    df[f'sim_euclidean_{category}'] = [sim[1] for sim in similarities]
    
    return df

def main():
    # Check if section embeddings exist
    section_embeddings_file = f'{OUTPUT_FOLDER}/section2_embeddings.csv'
    if not os.path.exists(section_embeddings_file):
        print(f"Section embeddings file not found: {section_embeddings_file}")
        print("Please run section embedding generation first.")
        return
    
    print("Loading section embeddings...")
    section_embeddings = pl.read_csv(section_embeddings_file).to_pandas()
    
    categories_keywords_embeddings = "task2_cleanup_and_build_database/out/category_keywords_embeddings.csv"
    categories = pl.read_csv(categories_keywords_embeddings)['category'].unique()
    
    print(f"Found {len(categories)} categories to process")
    
    for category in tqdm(categories, desc="Processing categories"):
        print(f"Processing category: {category}")
        section_embeddings = add_category_similarity_columns(section_embeddings, category, categories_keywords_embeddings)

    output_file = f'{OUTPUT_FOLDER}/section2_section_embeddings_with_similarity_to_keywords_cosine_and_euclidean.csv'
    section_embeddings.to_csv(output_file, index=False)
    print(f"Saved section keyword similarities to: {output_file}")

if __name__ == "__main__":
    main()