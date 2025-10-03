#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "pandas", 
#   "numpy",
#   "tqdm",
#   "numba>=0.57.0",
# ]
# ///

"""
Calculate similarities using individual keyword averaging approach.
Instead of creating a centroid first, this measures each keyword's similarity
to the text and then averages those similarities.
"""

import os
import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numba import jit
import sys

# Directories
OUTPUT_FOLDER = "task2_cleanup_and_build_database/out"
CACHE_DIR = Path(OUTPUT_FOLDER) / "similarity_cache_individual"
HASH_FILE = CACHE_DIR / "category_hashes.json"

@jit(nopython=True)
def cosine_similarity_numba(u, v):
    """Fast cosine similarity calculation."""
    uv, uu, vv = 0.0, 0.0, 0.0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] ** 2
        vv += v[i] ** 2
    return uv / np.sqrt(uu * vv) if uu and vv else 0

@jit(nopython=True)
def euclidean_distance_numba(u, v):
    """Fast euclidean distance calculation."""
    return np.sqrt(np.sum((u - v) ** 2))

def get_category_hash(keywords_df, category):
    """Generate hash for a category's keywords."""
    category_keywords = keywords_df[keywords_df['category'] == category]['keyword'].tolist()
    keywords_str = ','.join(sorted(category_keywords))
    return hashlib.md5(keywords_str.encode('utf-8')).hexdigest()

def load_category_hashes():
    """Load saved category hashes."""
    if HASH_FILE.exists():
        with open(HASH_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_category_hashes(hashes):
    """Save category hashes."""
    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_FILE, 'w') as f:
        json.dump(hashes, f, indent=2)

def has_category_changed(keywords_df, category, saved_hashes):
    """Check if a category's keywords have changed."""
    current_hash = get_category_hash(keywords_df, category)
    saved_hash = saved_hashes.get(category, '')
    return current_hash != saved_hash, current_hash

def calculate_individual_similarities(text_embedding, keyword_embeddings):
    """Calculate average similarity across individual keywords."""
    if len(keyword_embeddings) == 0:
        return 0.0, 0.0
    
    cosine_sims = []
    euclidean_dists = []
    
    for keyword_embedding in keyword_embeddings:
        cosine_sim = cosine_similarity_numba(text_embedding, keyword_embedding)
        euclidean_dist = euclidean_distance_numba(text_embedding, keyword_embedding)
        cosine_sims.append(cosine_sim)
        euclidean_dists.append(euclidean_dist)
    
    return np.mean(cosine_sims), np.mean(euclidean_dists)

def process_level(level_name, keywords_df, categories_to_process, all_categories):
    """Process similarities for a specific level."""
    
    # Define paths based on level
    if level_name == "sentences":
        input_file = f"{OUTPUT_FOLDER}/section2_sentences_embeddings.csv"
        cache_dir = CACHE_DIR / "sentences"
        output_file = f"{OUTPUT_FOLDER}/section2_sentence_embeddings_with_individual_keyword_similarities.csv"
    elif level_name == "articles":
        input_file = f"{OUTPUT_FOLDER}/section2_articles_embeddings.csv"
        cache_dir = CACHE_DIR / "articles"
        output_file = f"{OUTPUT_FOLDER}/section2_article_embeddings_with_individual_keyword_similarities.csv"
    else:  # sections
        input_file = f"{OUTPUT_FOLDER}/section2_embeddings.csv"
        cache_dir = CACHE_DIR / "sections"
        output_file = f"{OUTPUT_FOLDER}/section2_embeddings_with_individual_keyword_similarities.csv"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing {level_name}")
    print(f"{'='*60}")
    
    # Load base data
    print(f"Loading {input_file}...")
    base_df = pd.read_csv(input_file)
    print(f"Loaded {len(base_df)} {level_name}")
    
    # Get columns for output (exclude embedding column)
    output_columns = [col for col in base_df.columns if col != 'embedding']
    
    # Start with base data (without embeddings)
    result_df = base_df[output_columns].copy()
    
    # Process categories that need updating
    if categories_to_process:
        # Extract embeddings once
        print("Extracting embeddings...")
        embeddings = base_df['embedding'].apply(
            lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
        ).tolist()
        
        for category in categories_to_process:
            print(f"\nProcessing {category}...")
            
            # Get keyword embeddings
            category_keywords = keywords_df[keywords_df['category'] == category]
            keyword_embeddings = []
            for _, row in category_keywords.iterrows():
                embedding = row['embedding']
                if isinstance(embedding, str):
                    embedding = np.array(eval(embedding))
                else:
                    embedding = np.array(embedding)
                keyword_embeddings.append(embedding)
            
            print(f"  Calculating with {len(keyword_embeddings)} keywords...")
            
            # Calculate similarities
            cosine_sims = []
            euclidean_dists = []
            for embedding in tqdm(embeddings, desc=f"  {category}", leave=False):
                cos_sim, euc_dist = calculate_individual_similarities(embedding, keyword_embeddings)
                cosine_sims.append(cos_sim)
                euclidean_dists.append(euc_dist)
            
            # Save to cache
            cache_file = cache_dir / f"{category}_similarities.csv"
            cache_df = pd.DataFrame({
                'id': base_df['id'],
                f'sim_cosine_{category}': cosine_sims,
                f'sim_euclidean_{category}': euclidean_dists
            })
            cache_df.to_csv(cache_file, index=False)
            print(f"  Cached results")
    
    # Add all category similarities from cache
    print("\nCombining results from cache...")
    for category in all_categories:
        cache_file = cache_dir / f"{category}_similarities.csv"
        if cache_file.exists():
            cache_df = pd.read_csv(cache_file)
            # Add similarity columns to result
            for col in cache_df.columns:
                if col.startswith('sim_'):
                    result_df[col] = cache_df[col].values
            print(f"  Added {category}")
        else:
            print(f"  WARNING: No cache for {category}")
    
    # Save final result
    print(f"Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(result_df)} rows with {len(result_df.columns)} columns")
    
    return result_df

def main():
    print("=" * 80)
    print("Individual Keyword Similarity Calculation")
    print("=" * 80)
    
    # Load keywords
    print("\nLoading keyword embeddings...")
    keywords_df = pd.read_csv(f"{OUTPUT_FOLDER}/category_keywords_embeddings.csv")
    categories = keywords_df['category'].unique()
    print(f"Found {len(categories)} categories")
    
    # Check for changes
    saved_hashes = load_category_hashes()
    new_hashes = {}
    categories_to_process = []
    
    for category in categories:
        changed, new_hash = has_category_changed(keywords_df, category, saved_hashes)
        new_hashes[category] = new_hash
        
        if changed:
            categories_to_process.append(category)
            print(f"  ✓ {category}: Keywords changed")
        else:
            # Check if cache files exist for all levels
            all_exist = True
            for level in ['sentences', 'articles', 'sections']:
                cache_file = CACHE_DIR / level / f"{category}_similarities.csv"
                if not cache_file.exists():
                    all_exist = False
                    break
            
            if not all_exist:
                categories_to_process.append(category)
                print(f"  ✓ {category}: Cache incomplete")
            else:
                print(f"  - {category}: Using cache")
    
    # Save hashes
    save_category_hashes(new_hashes)
    
    # Process each level
    for level in ['sentences', 'articles', 'sections']:
        process_level(level, keywords_df, categories_to_process, categories)
    
    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)