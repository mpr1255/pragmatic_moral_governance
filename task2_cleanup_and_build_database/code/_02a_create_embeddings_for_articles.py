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
Create embeddings for articles (from section2_articles.csv).
This is similar to _02_create_embeddings_for_sentences.py but operates at the article level.
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
import GLOBALS

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

def main():
    print("Executing main in create article embeddings script")
    
    # Load articles data
    df = pd.read_csv(f'{OUTPUT_FOLDER}/section2_articles.csv')
    print(f"Loaded {len(df)} articles")
    
    # Check if output file already exists with some embeddings
    output_path = f'{OUTPUT_FOLDER}/section2_articles_embeddings.csv'
    start_idx = 0
    
    # Generate embeddings for each article
    embeddings = []
    
    # Process in smaller batches to avoid timeout
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, len(df))
        print(f"\nProcessing batch {batch_num + 1}/{total_batches} (articles {start}-{end-1})")
        
        for idx in tqdm(range(start, end), desc=f"Batch {batch_num + 1}"):
            row = df.iloc[idx]
            article_text = row['article_text']
            embedding = generate_embedding(article_text)
            if embedding:
                embeddings.append(embedding)
            else:
                embeddings.append([0] * 1536)  # Default zero embedding
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved article embeddings to {output_path}")

if __name__ == "__main__":
    main()