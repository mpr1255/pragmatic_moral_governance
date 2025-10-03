#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "openai",
#   "pandas", 
#   "numpy",
#   "scipy",
#   "tiktoken",
#   "tqdm",
#   "openpyxl",
#   "fastdist",
#   "numba>=0.57.0"
# ]
# ///

import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from openai import OpenAI
import GLOBALS

model = GLOBALS.MODEL
client = OpenAI()

# Cache directory for individual keyword embeddings
CACHE_DIR = Path("task2_cleanup_and_build_database/out/keyword_embeddings_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_keyword_hash(keyword: str) -> str:
    """Generate a hash for the keyword to use as a filename."""
    return hashlib.md5(keyword.encode('utf-8')).hexdigest()

def get_cached_embedding(keyword: str) -> np.ndarray | None:
    """Retrieve cached embedding for a keyword if it exists."""
    cache_file = CACHE_DIR / f"{get_keyword_hash(keyword)}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            data = json.load(f)
            if data['keyword'] == keyword:  # Verify it's the right keyword
                return np.array(data['embedding'])
    return None

def save_embedding_to_cache(keyword: str, embedding: np.ndarray) -> None:
    """Save an embedding to the cache."""
    cache_file = CACHE_DIR / f"{get_keyword_hash(keyword)}.json"
    with open(cache_file, 'w') as f:
        json.dump({
            'keyword': keyword,
            'embedding': embedding.tolist(),
            'model': model
        }, f)

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding using OpenAI API."""
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

def get_or_create_embedding(keyword: str) -> np.ndarray:
    """Get embedding from cache or create new one."""
    # Check cache first
    embedding = get_cached_embedding(keyword)
    if embedding is not None:
        print(f"  Using cached embedding for: {keyword}")
        return embedding
    
    # Generate new embedding
    print(f"  Generating new embedding for: {keyword}")
    embedding = generate_embedding(keyword)
    save_embedding_to_cache(keyword, embedding)
    return embedding

def calculate_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """Calculate the centroid of a list of embeddings."""
    return np.mean(embeddings, axis=0)

def get_keywords_hash(keywords_df):
    """Generate a hash of all keywords to detect changes."""
    # Sort by category and keyword to ensure consistent hashing
    sorted_df = keywords_df.sort_values(['category', 'keyword'])
    keywords_str = sorted_df.to_csv(index=False)
    return hashlib.md5(keywords_str.encode('utf-8')).hexdigest()

def process_keywords_with_caching(csv_path: str, output_csv_path: str) -> None:
    """
    Process keywords with intelligent caching.
    Creates embeddings only for new keywords and calculates centroids.
    """
    # Read input keywords
    keywords_df = pd.read_csv(csv_path)
    
    # Check if keywords have changed by comparing hash
    current_hash = get_keywords_hash(keywords_df)
    
    # Use a hash file specific to the input CSV to track changes
    csv_path_hash = hashlib.md5(csv_path.encode('utf-8')).hexdigest()[:8]
    hash_file = Path(output_csv_path).parent / f"keywords_hash_{csv_path_hash}.txt"
    
    needs_regeneration = True
    if hash_file.exists():
        with open(hash_file, 'r') as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            needs_regeneration = False
            print(f"  Keywords unchanged (hash: {current_hash[:8]}...)")
    
    # Check if output files exist
    centroids_path = output_csv_path.replace('.csv', '_centroids.json')
    if not needs_regeneration and Path(output_csv_path).exists() and Path(centroids_path).exists():
        print("✓ Keywords unchanged and output files exist - skipping regeneration")
        return
    
    print(f"  Processing {len(keywords_df)} keywords across {keywords_df['category'].nunique()} categories")
    
    # Group by category
    categories = keywords_df.groupby('category')
    
    # Process each category
    results = []
    category_centroids = {}
    
    for category_name, group in categories:
        print(f"\nProcessing category: {category_name}")
        embeddings = []
        
        # Get embeddings for all keywords in this category
        for keyword in group['keyword']:
            embedding = get_or_create_embedding(keyword)
            embeddings.append(embedding)
            
            # Store individual result
            results.append({
                'category': category_name,
                'keyword': keyword,
                'embedding': str(embedding.tolist())
            })
        
        # Calculate and store centroid
        if embeddings:
            centroid = calculate_centroid(embeddings)
            category_centroids[category_name] = centroid
            print(f"  Calculated centroid for {category_name} from {len(embeddings)} keywords")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved embeddings to: {output_csv_path}")
    
    # Save centroids separately
    centroids_dict = {cat: centroid.tolist() for cat, centroid in category_centroids.items()}
    with open(centroids_path, 'w') as f:
        json.dump(centroids_dict, f, indent=2)
    print(f"Saved centroids to: {centroids_path}")
    
    # Save current hash
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    # Print cache statistics
    cache_files = list(CACHE_DIR.glob("*.json"))
    print(f"\nCache statistics:")
    print(f"  Total cached embeddings: {len(cache_files)}")
    print(f"  Cache directory: {CACHE_DIR}")

if __name__ == "__main__":
    import sys
    
    # Simple parameter with default value
    categories_keywords = sys.argv[1] if len(sys.argv) > 1 else "task2_cleanup_and_build_database/ref/category_keywords.csv"
    categories_keywords_embeddings = "task2_cleanup_and_build_database/out/category_keywords_embeddings.csv"
    
    print(f"Loading keywords from: {categories_keywords}")
    process_keywords_with_caching(categories_keywords, categories_keywords_embeddings)
