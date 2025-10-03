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

import pandas as pd
import numpy as np
import toml
from openai import OpenAI
import GLOBALS # To get MODEL and OUTPUT_FOLDER
from scipy.spatial.distance import cdist
from tqdm import tqdm
import ast # To safely evaluate string representation of lists

# --- Configuration ---
CONCEPTS_PATH = "task1a_create_concept_paragraphs/out/concepts.toml"
SENTENCES_EMBEDDINGS_PATH = f"{GLOBALS.OUTPUT_FOLDER}/section2_sentences_embeddings.csv"
OUTPUT_CSV_PATH = f"{GLOBALS.OUTPUT_FOLDER}/sentence_concept_similarities.csv"
MODEL = GLOBALS.MODEL
client = OpenAI()

# --- Embedding Function ---
def generate_embedding(text: str, model: str = MODEL):
    """Generates embedding for a given text using the specified model."""
    try:
        response = client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error generating embedding for text: {text[:50]}... Error: {e}")
        return None

# --- Main Logic ---
def main():
    print("Loading concepts from TOML...")
    concepts_data = toml.load(CONCEPTS_PATH)
    concept_names = list(concepts_data.keys())
    # Extract Chinese paragraphs
    concept_paragraphs = [concepts_data[name]['chinese'] for name in concept_names]
    print(f"Loaded {len(concept_paragraphs)} concepts.")

    print("Generating embeddings for concept paragraphs...")
    concept_embeddings = []
    for paragraph in tqdm(concept_paragraphs, desc="Embedding Concepts"):
        embedding = generate_embedding(paragraph)
        if embedding is not None:
            concept_embeddings.append(embedding)
        else:
            # Handle error - maybe skip this concept or raise an error
            print(f"Could not generate embedding for a concept paragraph. Skipping.")
            # Or potentially: raise ValueError("Failed to generate embedding for a concept paragraph")

    if not concept_embeddings:
        print("No concept embeddings were generated. Exiting.")
        return

    concept_embeddings_matrix = np.array(concept_embeddings)
    print(f"Generated concept embeddings matrix with shape: {concept_embeddings_matrix.shape}")

    print(f"Loading sentences and their embeddings from {SENTENCES_EMBEDDINGS_PATH}...")
    try:
        sentences_df = pd.read_csv(SENTENCES_EMBEDDINGS_PATH)
        # Safely evaluate the string representation of the embedding list into a numpy array
        sentences_df['embedding_list'] = sentences_df['embedding'].apply(ast.literal_eval)
        sentence_embeddings_matrix = np.array(sentences_df['embedding_list'].tolist())
        print(f"Loaded {len(sentences_df)} sentences.")
        print(f"Sentence embeddings matrix shape: {sentence_embeddings_matrix.shape}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {SENTENCES_EMBEDDINGS_PATH}")
        return
    except Exception as e:
        print(f"Error loading or processing sentence embeddings: {e}")
        return

    # Ensure embedding dimensions match
    if sentence_embeddings_matrix.shape[1] != concept_embeddings_matrix.shape[1]:
        print(f"Error: Mismatch in embedding dimensions between sentences ({sentence_embeddings_matrix.shape[1]}) and concepts ({concept_embeddings_matrix.shape[1]})")
        return

    print("Calculating cosine similarities between sentences and concepts...")
    # Calculate cosine distance (1 - similarity) using cdist
    # It's generally faster for large matrices
    cosine_distances = cdist(sentence_embeddings_matrix, concept_embeddings_matrix, metric='cosine')

    # Convert distances to similarities (1 - distance)
    cosine_similarities = 1 - cosine_distances
    print("Similarity calculation complete.")

    print("Creating output DataFrame...")
    # Create a DataFrame for the results
    results_df = sentences_df[['id', 'sentence']].copy()

    # Add similarity columns
    for i, concept_name in enumerate(concept_names):
        results_df[f'sim_cosine_{concept_name}'] = cosine_similarities[:, i]

    print(f"Saving results to {OUTPUT_CSV_PATH}...")
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    print("Script finished successfully.")

if __name__ == "__main__":
    main()