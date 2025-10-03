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
Create embeddings for entire sections (from section2.csv).
This operates at the full section level (the entire Section 2 text).
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm.auto import tqdm
import tiktoken
import GLOBALS
import sys
from pathlib import Path

client = OpenAI()
model = GLOBALS.MODEL
OUTPUT_FOLDER = "task2_cleanup_and_build_database/out"

print("Loading libraries...")

def truncate_text_to_tokens(text, max_tokens=8000):
    """Truncate text to fit within token limit."""
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    return text

def generate_embedding(text):
    """Generate embedding for a single text."""
    try:
        # Truncate if needed
        text = truncate_text_to_tokens(text)
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def main():
    print("Executing main in create section embeddings script")
    
    # Check if embeddings already exist in similarity file
    similarity_file = f'{OUTPUT_FOLDER}/section2_section_embeddings_with_similarity_to_keywords_cosine_and_euclidean.csv'
    output_path = f'{OUTPUT_FOLDER}/section2_embeddings.csv'
    
    if Path(similarity_file).exists() and not Path(output_path).exists():
        print("Section embeddings not found, but similarity file exists. Extracting embeddings...")
        # Load similarity file and extract just the embedding columns we need
        similarity_df = pd.read_csv(similarity_file)
        # Select only the columns needed for embeddings file
        embedding_cols = ['id', 'title', 'office', 'publish', 'expiry', 'type', 
                         'status', 'url', 'content', 'embedding']
        section_embeddings = similarity_df[embedding_cols].copy()
        # Save as embeddings file for future use
        section_embeddings.to_csv(output_path, index=False)
        print(f"Extracted {len(section_embeddings)} section embeddings from similarity file")
        return
    
    # Load sections data
    df = pd.read_csv(f'{OUTPUT_FOLDER}/section2.csv')
    print(f"Loaded {len(df)} sections")
    
    # Generate embeddings for each section
    embeddings = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating section embeddings"):
        section_text = str(row['content'])
        embedding = generate_embedding(section_text)
        if embedding:
            embeddings.append(embedding)
        else:
            embeddings.append([0] * 1536)  # Default zero embedding
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved section embeddings to {output_path}")

if __name__ == "__main__":
    main()