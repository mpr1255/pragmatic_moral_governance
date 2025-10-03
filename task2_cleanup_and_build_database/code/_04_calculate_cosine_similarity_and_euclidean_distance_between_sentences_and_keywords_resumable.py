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

#%% 
from GLOBALS import *
import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from numba import jit
from multiprocessing import Pool, cpu_count
import logging
import json
from pathlib import Path
import pickle

# Set up logging
log_filename = "../logs/calculate_similarity.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Checkpoint file
CHECKPOINT_FILE = Path(OUTPUT_FOLDER) / 'similarity_checkpoint.pkl'

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

def calculate_scores(args):
    keyword_embedding, row = args
    sentence_embedding = np.array(eval(row['embedding'])) if isinstance(row['embedding'], str) else np.array(row['embedding'])
    if keyword_embedding.shape[0] != sentence_embedding.shape[0]:
        raise ValueError("Mismatched dimensions between keyword and sentence embeddings.")
    cosine_score = cosine_similarity_numba(keyword_embedding, sentence_embedding)
    euclidean_score = euclidean_distance_numba(keyword_embedding, sentence_embedding)
    return (cosine_score, euclidean_score, row['sentence'])

def find_top_matches_multithread(keyword, sentence_embeddings, generate_embedding, top_n=50):
    keyword_embedding = np.array(generate_embedding(keyword))
    args = [(keyword_embedding, row) for index, row in sentence_embeddings.iterrows()]
    
    with Pool(cpu_count()) as pool:
        scores = list(tqdm(pool.imap(calculate_scores, args), total=len(args), desc="Calculating scores"))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    top_matches = scores[:top_n]
    for i, (cosine_score, euclidean_score, sentence) in enumerate(top_matches):
        print(f"Match {i+1}: {sentence}, Keyword: {keyword}, Cosine Similarity: {cosine_score}, Euclidean Distance: {euclidean_score}")
    return top_matches

def calculate_average_similarities(args):
    sentence_embedding, category_keywords = args
    cosine_similarities = []
    euclidean_distances = []
    for _, keyword_row in category_keywords.iterrows():
        keyword_embedding = np.array(eval(keyword_row['embedding'])) if isinstance(keyword_row['embedding'], str) else np.array(keyword_row['embedding'])
        if sentence_embedding.shape[0] != keyword_embedding.shape[0]:
            raise ValueError("Mismatched dimensions between sentence and keyword embeddings.")
        cosine_similarity = cosine_similarity_numba(sentence_embedding, keyword_embedding)
        euclidean_distance = euclidean_distance_numba(sentence_embedding, keyword_embedding)
        cosine_similarities.append(cosine_similarity)
        euclidean_distances.append(euclidean_distance)
    average_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else np.nan
    average_euclidean_distance = np.mean(euclidean_distances) if euclidean_distances else np.nan
    return (average_cosine_similarity, average_euclidean_distance)

def add_category_similarity_columns(df: pd.DataFrame, category: str, categories_keywords_embeddings: str) -> pd.DataFrame:
    keywords_df = pd.read_csv(categories_keywords_embeddings)
    category_keywords = keywords_df[keywords_df['category'] == category]
    
    sentence_embeddings = [(np.array(eval(row['embedding'])) if isinstance(row['embedding'], str) else np.array(row['embedding']), category_keywords) for index, row in df.iterrows()]

    with Pool(cpu_count()) as pool:
        similarities = list(tqdm(pool.imap(calculate_average_similarities, sentence_embeddings), total=len(sentence_embeddings), desc=f"Processing {category} - sentences"))
    
    df[f'sim_cosine_{category}'] = [sim[0] for sim in similarities]
    df[f'sim_euclidean_{category}'] = [sim[1] for sim in similarities]
    
    return df

def save_checkpoint(sentence_embeddings, completed_categories):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'dataframe': sentence_embeddings,
        'completed_categories': completed_categories
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logging.info(f"Checkpoint saved. Completed categories: {completed_categories}")

def load_checkpoint():
    """Load checkpoint if it exists"""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                checkpoint_data = pickle.load(f)
            logging.info(f"Checkpoint loaded. Completed categories: {checkpoint_data['completed_categories']}")
            return checkpoint_data['dataframe'], checkpoint_data['completed_categories']
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            return None, []
    return None, []

def cleanup_checkpoint():
    """Remove checkpoint file after successful completion"""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logging.info("Checkpoint file removed")

#%%
def main():
    logging.info("Starting the script")

    # Load categories
    categories_keywords_embeddings = "task2_cleanup_and_build_database/out/category_keywords_embeddings.csv"
    categories = pl.read_csv(categories_keywords_embeddings)['category'].unique()
    
    # Try to load from checkpoint
    sentence_embeddings, completed_categories = load_checkpoint()
    
    if sentence_embeddings is None:
        # Fresh start
        logging.info("Starting fresh - no checkpoint found")
        sentence_embeddings = pl.read_csv(f'{OUTPUT_FOLDER}/section2_sentences_embeddings.csv').to_pandas()
        completed_categories = []
    else:
        logging.info(f"Resuming from checkpoint. {len(completed_categories)} categories already completed")

    # Process remaining categories
    remaining_categories = [cat for cat in categories if cat not in completed_categories]
    
    if not remaining_categories:
        logging.info("All categories already processed!")
    else:
        logging.info(f"Processing {len(remaining_categories)} remaining categories: {remaining_categories}")
        
        for category in tqdm(remaining_categories, desc="Processing categories"):
            try:
                logging.info(f"Processing category: {category}")
                sentence_embeddings = add_category_similarity_columns(sentence_embeddings, category, categories_keywords_embeddings)
                completed_categories.append(category)
                
                # Save checkpoint after each category
                save_checkpoint(sentence_embeddings, completed_categories)
                
            except Exception as e:
                logging.error(f"Error processing category {category}: {e}")
                print(f"Error processing category {category}: {e}")
                print("Progress saved. You can resume by running the script again.")
                return

    # Save final result
    output_file = f'{OUTPUT_FOLDER}/section2_sentence_embeddings_with_similarity_to_keywords_cosine_and_euclidean.csv'
    logging.info(f"Saving final results to {output_file}")
    sentence_embeddings.to_csv(output_file, index=False)
    
    # Clean up checkpoint after successful completion
    cleanup_checkpoint()
    
    logging.info("Script finished successfully")
    print(f"Successfully created: {output_file}")
#%%

if __name__ == "__main__":
    main()