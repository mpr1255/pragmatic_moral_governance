#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "openai",
#     "pandas",
#     "numpy",
#     "scipy",
#     "tiktoken",
#     "torch",
#     "sentence-transformers",
#     "tqdm",
#     "numba>=0.57.0",
#     "fastdist",
#     "openpyxl",
# ]
# ///

"""
Create embeddings for articles with resume capability.
This version can be interrupted and resumed.
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
import GLOBALS
import os
import json
import sys
from pathlib import Path

client = OpenAI()
model = GLOBALS.MODEL
OUTPUT_FOLDER = "task2_cleanup_and_build_database/out"

print("Loading libraries...")

def generate_embedding(text):
    """Generate embedding for a single text."""
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def save_checkpoint(df, embeddings, checkpoint_file):
    """Save intermediate results to checkpoint file."""
    checkpoint_data = {
        'processed_count': len(embeddings),
        'embeddings': embeddings
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f)
    print(f"Checkpoint saved: {len(embeddings)} embeddings processed")

def load_checkpoint(checkpoint_file):
    """Load checkpoint if it exists."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded: {checkpoint_data['processed_count']} embeddings already processed")
        return checkpoint_data['embeddings']
    return []

def main():
    print("Executing main in create article embeddings script (resumable version)")
    
    # Check if embeddings already exist in similarity file
    similarity_file = f'{OUTPUT_FOLDER}/section2_article_embeddings_with_similarity_to_keywords_cosine_and_euclidean.csv'
    output_path = f'{OUTPUT_FOLDER}/section2_articles_embeddings.csv'
    
    if Path(similarity_file).exists() and not Path(output_path).exists():
        print("Article embeddings not found, but similarity file exists. Extracting embeddings...")
        # Load similarity file and extract just the embedding columns we need
        similarity_df = pd.read_csv(similarity_file)
        # Select only the columns needed for embeddings file
        embedding_cols = ['id', 'article_index', 'article_text', 'title', 'office', 
                         'publish', 'expiry', 'type', 'status', 'url', 'embedding']
        article_embeddings = similarity_df[embedding_cols].copy()
        # Save as embeddings file for future use
        article_embeddings.to_csv(output_path, index=False)
        print(f"Extracted {len(article_embeddings)} article embeddings from similarity file")
        return
    
    # Load articles data
    df = pd.read_csv(f'{OUTPUT_FOLDER}/section2_articles.csv')
    print(f"Loaded {len(df)} articles")
    
    # Checkpoint file
    checkpoint_file = f'{OUTPUT_FOLDER}/article_embeddings_checkpoint.json'
    
    # Load checkpoint if exists
    embeddings = load_checkpoint(checkpoint_file)
    start_idx = len(embeddings)
    
    if start_idx >= len(df):
        print("All articles already processed!")
    else:
        print(f"Starting from article {start_idx}")
        
        # Process remaining articles
        checkpoint_interval = 50  # Save checkpoint every 50 articles
        
        for idx in tqdm(range(start_idx, len(df)), desc="Generating article embeddings", initial=start_idx, total=len(df)):
            row = df.iloc[idx]
            article_text = row['article_text']
            embedding = generate_embedding(article_text)
            
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0] * 1536)  # Default zero embedding
            
            # Save checkpoint periodically
            if (idx + 1) % checkpoint_interval == 0:
                save_checkpoint(df, embeddings, checkpoint_file)
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Save final results
    df.to_csv(output_path, index=False)
    print(f"\nSaved article embeddings to {output_path}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")

if __name__ == "__main__":
    main()