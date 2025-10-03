#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "openai",
#     "pandas",
#     "numpy",
#     "scipy",
#     "toml",
#     "tqdm",
#     "requests",
#     "numba>=0.57.0",
#     "fastdist",
#     "openpyxl",
# ]
# ///

"""
Calculate cosine similarities between full sections and concept paragraphs.
This operates at the highest level - entire Section 2 texts.
"""

import pandas as pd
import numpy as np
import toml
from openai import OpenAI
import GLOBALS
from scipy.spatial.distance import cdist
from tqdm import tqdm
import ast

# --- Configuration ---
CONCEPTS_PATH = "task1a_create_concept_paragraphs/out/concepts.toml"
SECTIONS_EMBEDDINGS_PATH = f"{GLOBALS.OUTPUT_FOLDER}/section2_embeddings.csv"
OUTPUT_CSV_PATH = f"{GLOBALS.OUTPUT_FOLDER}/section_concept_similarities.csv"
MODEL = GLOBALS.MODEL
client = OpenAI()

# --- Helper Functions ---
def generate_embedding(text):
    """Generate embedding for given text using OpenAI API."""
    try:
        response = client.embeddings.create(input=text, model=MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
        return None

# --- Main Logic ---
def main():
    print("Loading concepts from TOML...")
    concepts_data = toml.load(CONCEPTS_PATH)
    concept_names = list(concepts_data.keys())
    concept_paragraphs = [concepts_data[name]['chinese'] for name in concept_names]
    print(f"Loaded {len(concept_paragraphs)} concepts.")

    print("Generating embeddings for concept paragraphs...")
    concept_embeddings = []
    for paragraph in tqdm(concept_paragraphs, desc="Embedding Concepts"):
        embedding = generate_embedding(paragraph)
        if embedding is not None:
            concept_embeddings.append(embedding)
        else:
            print(f"Could not generate embedding for a concept paragraph. Skipping.")

    if not concept_embeddings:
        print("No concept embeddings were generated. Exiting.")
        return

    concept_embeddings_matrix = np.array(concept_embeddings)
    print(f"Generated concept embeddings matrix with shape: {concept_embeddings_matrix.shape}")

    print(f"Loading sections and their embeddings from {SECTIONS_EMBEDDINGS_PATH}...")
    try:
        sections_df = pd.read_csv(SECTIONS_EMBEDDINGS_PATH)
        sections_df['embedding_list'] = sections_df['embedding'].apply(ast.literal_eval)
        section_embeddings_matrix = np.array(sections_df['embedding_list'].tolist())
        print(f"Loaded {len(sections_df)} sections.")
        print(f"Section embeddings matrix shape: {section_embeddings_matrix.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {SECTIONS_EMBEDDINGS_PATH}")
        return
    except Exception as e:
        print(f"Error loading or processing section embeddings: {e}")
        return

    # Calculate cosine similarities
    print("Calculating cosine similarities between sections and concepts...")
    similarities = 1 - cdist(section_embeddings_matrix, concept_embeddings_matrix, metric='cosine')
    print(f"Similarity calculation complete. Shape: {similarities.shape}")

    # Create output DataFrame
    print("Creating output DataFrame...")
    output_data = {
        'id': sections_df['id'],
        'content': sections_df['content']
    }

    # Add similarity columns
    for i, concept_name in enumerate(concept_names):
        col_name = f"sim_cosine_{i+1}_{concept_name}"
        output_data[col_name] = similarities[:, i]

    output_df = pd.DataFrame(output_data)

    # Save results
    print(f"Saving results to {OUTPUT_CSV_PATH}...")
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Script finished successfully.")

if __name__ == "__main__":
    main()