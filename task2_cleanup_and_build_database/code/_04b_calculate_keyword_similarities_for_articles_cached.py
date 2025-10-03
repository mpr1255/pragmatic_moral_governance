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
Smart cached version of keyword similarity calculations.
Only recalculates similarities for categories whose keywords have changed.
"""

import os
import hashlib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from numba import jit

# Directories
OUTPUT_FOLDER = "task2_cleanup_and_build_database/out"
CACHE_DIR = Path(OUTPUT_FOLDER) / "similarity_cache"
ARTICLES_CACHE_DIR = CACHE_DIR / "articles"
COMBINED_DIR = CACHE_DIR / "combined"
HASH_FILE = CACHE_DIR / "category_hashes.json"

# Create directories
ARTICLES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

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
    with open(HASH_FILE, 'w') as f:
        json.dump(hashes, f, indent=2)

def has_category_changed(keywords_df, category, saved_hashes):
    """Check if a category's keywords have changed."""
    current_hash = get_category_hash(keywords_df, category)
    saved_hash = saved_hashes.get(category, '')
    return current_hash != saved_hash, current_hash

def calculate_category_similarities(articles_df, centroid, category_name):
    """Calculate similarities for a single category."""
    cosine_sims = []
    euclidean_dists = []
    
    # Convert embeddings to numpy arrays
    article_embeddings = articles_df['embedding'].apply(
        lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x)
    ).tolist()
    
    # Calculate similarities
    for article_embedding in tqdm(article_embeddings, 
                                desc=f"Processing {category_name} - articles",
                                leave=True):
        cosine_sim = cosine_similarity_numba(article_embedding, centroid)
        euclidean_dist = euclidean_distance_numba(article_embedding, centroid)
        cosine_sims.append(cosine_sim)
        euclidean_dists.append(euclidean_dist)
    
    # Create results dataframe
    results_df = articles_df[['id', 'article_index', 'article_text']].copy()
    results_df[f'sim_cosine_{category_name}'] = cosine_sims
    results_df[f'sim_euclidean_{category_name}'] = euclidean_dists
    
    return results_df

def main():
    print("Smart keyword similarity calculation with caching...")
    
    # Load data
    print("Loading article embeddings...")
    articles_df = pd.read_csv(f"{OUTPUT_FOLDER}/section2_articles_embeddings.csv")
    
    # Load keywords and centroids
    keywords_df = pd.read_csv(f"{OUTPUT_FOLDER}/category_keywords_embeddings.csv")
    with open(f"{OUTPUT_FOLDER}/category_keywords_embeddings_centroids.json", 'r') as f:
        centroids = json.load(f)
    
    # Load saved hashes
    saved_hashes = load_category_hashes()
    new_hashes = {}
    
    # Get all categories
    categories = keywords_df['category'].unique()
    print(f"Found {len(categories)} categories to process")
    
    # Track which categories to process
    categories_to_process = []
    unchanged_categories = []
    
    # Check each category for changes
    for category in categories:
        changed, new_hash = has_category_changed(keywords_df, category, saved_hashes)
        new_hashes[category] = new_hash
        
        if changed:
            categories_to_process.append(category)
            print(f"  ✓ {category}: Keywords changed, will recalculate")
        else:
            unchanged_categories.append(category)
            cache_file = ARTICLES_CACHE_DIR / f"{category}_similarities.csv"
            if not cache_file.exists():
                categories_to_process.append(category)
                print(f"  ✓ {category}: Cache missing, will calculate")
            else:
                print(f"  - {category}: Using cached results")
    
    # Process only changed categories
    if categories_to_process:
        print(f"\nProcessing {len(categories_to_process)} categories with changes...")
        
        for category in tqdm(categories_to_process, desc="Processing categories"):
            print(f"\nCalculating similarities for: {category}")
            
            # Get centroid
            centroid = np.array(centroids[category])
            
            # Calculate similarities
            results_df = calculate_category_similarities(articles_df, centroid, category)
            
            # Save to cache
            cache_file = ARTICLES_CACHE_DIR / f"{category}_similarities.csv"
            results_df.to_csv(cache_file, index=False)
            print(f"  Saved to cache: {cache_file.name}")
    else:
        print("\nNo categories need updating!")
    
    # Combine all results
    print("\nCombining all category results...")
    
    # Start with base article data
    combined_df = articles_df[['id', 'article_index', 'article_text', 'title', 
                              'office', 'publish', 'expiry', 'type', 'status', 
                              'url', 'embedding']].copy()
    
    # Add similarities from each category
    for category in categories:
        cache_file = ARTICLES_CACHE_DIR / f"{category}_similarities.csv"
        if cache_file.exists():
            cat_df = pd.read_csv(cache_file)
            # Add only the similarity columns
            sim_cols = [col for col in cat_df.columns if col.startswith('sim_')]
            for col in sim_cols:
                combined_df[col] = cat_df[col]
        else:
            print(f"  Warning: No cache found for {category}")
    
    # Save combined results
    output_file = f"{OUTPUT_FOLDER}/section2_article_embeddings_with_similarity_to_keywords_cosine_and_euclidean.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"\nSaved combined results to: {output_file}")
    
    # Save updated hashes
    save_category_hashes(new_hashes)
    print("Updated category hashes saved")
    
    # Summary
    print("\nSummary:")
    print(f"  Categories processed: {len(categories_to_process)}")
    print(f"  Categories cached: {len(unchanged_categories)}")
    print(f"  Total categories: {len(categories)}")

if __name__ == "__main__":
    main()