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

#%% IMPORTS. MODEL.
import json
import os
import sqlite3
import sys
from collections import OrderedDict
from typing import List, Optional, Tuple
from openai import OpenAI
import pandas as pd
import re
import numpy as np
from numba import jit
import numpy as np
import timeit
from fastdist import fastdist
import numpy as np
from scipy.spatial import distance
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import GLOBALS

client = OpenAI()
import tiktoken
import torch
from sentence_transformers import SentenceTransformer, util
# here = os.path.dirname(sys.prefix)
# os.chdir(here)

OUTPUT_FOLDER = "task2_cleanup_and_build_database/out"
model = GLOBALS.MODEL

print("loading libraries...")

#%%
# FUNCTIONS
# Define generate_embedding function
def generate_keyword_embedding(keyword, client, model):
    return np.array(client.embeddings.create(input=keyword, model=model).data[0].embedding)

# Function to generate embeddings for a sentence
def generate_embedding(sentence):
    response = client.embeddings.create(input=sentence, model=model)
    embedding = np.array(response.data[0].embedding)
    return embedding.tolist()  # Convert numpy array to list for easier handling in DataFrame

# Function to extract content between 第二章 and 第三章 if more than one line
def extract_section(content):
    # Handle NaN or non-string values
    if pd.isna(content) or not isinstance(content, str):
        print(f"Non-string or NaN value encountered: {content}")
        return None
    
    # Pattern to capture content between 第二章 and 第三章, allowing for characters before and after
    pattern = r'(第二章.*?)(?=第三章)'
    try:
        matches = re.findall(pattern, content, flags=re.DOTALL)
        for match in matches:
            if match.count('\n') > 1:  # Ensure there's more than one line
                return match
    except Exception as e: 
        print(f"Error: {e}")
        print(f"No match found for content starting with: {content[:50] if isinstance(content, str) else content}")
        return None


if __name__ == "__main__":
    print("executing main in create embeddings script")

    #%% GET DATA INTO SHAPE
    # Read the file into a DataFrame
    df = pd.read_csv(f'{OUTPUT_FOLDER}/output.csv')  # replace 'your_file_path.csv' with the actual file path
    # Apply the extraction function to the 'content' column
    df['extracted_section'] = df['content'].apply(extract_section)

    # Filter out rows where the 'extracted_section' is None
    extracted_df = df.dropna(subset=['extracted_section']).copy()

    # Keep only the 'extracted_section' and other specified columns
    columns_to_keep = ['id', 'title', 'office', 'publish', 'expiry', 'type', 'status', 'url', 'extracted_section']
    final_df = extracted_df[columns_to_keep].copy()

    # Rename 'extracted_section' to 'content' for clarity
    final_df.rename(columns={'extracted_section': 'content'}, inplace=True)

    # Save the new table to a CSV file in the './out' folder
    final_df.to_csv(f'{OUTPUT_FOLDER}/section2.csv', index=False)

    #%%
    # Create a new DataFrame where each sentence from the 'content' column is a new row
    # Explode the DataFrame based on the 'content' split by newlines
    sentences_df = final_df.assign(sentence=final_df['content'].str.split('\n')).explode('sentence').reset_index(drop=True)

    # Drop the original 'content' column as it's replaced by 'sentence'
    sentences_df.drop(columns=['content'], inplace=True)

    # Filter out any empty sentences that may result from consecutive newlines
    sentences_df = sentences_df[sentences_df['sentence'].str.strip() != '']

    # Save the new DataFrame to a CSV file
    sentences_df.to_csv(f'{OUTPUT_FOLDER}/section2_sentences.csv', index=False)

    # Check if embeddings file exists, if not check similarity file or generate
    embeddings_file_path = f'{OUTPUT_FOLDER}/section2_sentences_embeddings.csv'
    similarity_file_path = f'{OUTPUT_FOLDER}/section2_sentence_embeddings_with_cosine_similarity_to_keywords.csv'
    
    if os.path.exists(embeddings_file_path):
        print("Embeddings file exists, loading from disk...")
        sentence_embeddings = pd.read_csv(embeddings_file_path)
    elif os.path.exists(similarity_file_path):
        print("Embeddings not found, but similarity file exists. Extracting embeddings...")
        # Load similarity file and extract just the embedding columns we need
        similarity_df = pd.read_csv(similarity_file_path)
        # Select only the columns needed for embeddings file
        embedding_cols = ['id', 'title', 'office', 'publish', 'expiry', 'type', 'status', 'url', 'sentence', 'embedding']
        sentence_embeddings = similarity_df[embedding_cols].copy()
        # Save as embeddings file for future use
        sentence_embeddings.to_csv(embeddings_file_path, index=False)
        print(f"Extracted {len(sentence_embeddings)} embeddings from similarity file")
    else:
        print("No embeddings found, generating new embeddings...")
        # Apply the function to each sentence in the DataFrame
        sentences_df['embedding'] = sentences_df['sentence'].apply(lambda x: generate_embedding(x))
        sentences_df.to_csv(embeddings_file_path, index=False)
        sentence_embeddings = sentences_df



#%%





#%%
# Example usage
# a, b = "环保行为", "嫁出去"
# execution_time = timeit.timeit(lambda: find_closest_match(a, sentence_embeddings, generate_embedding), number=1)
# print(f"Execution Time: {execution_time} seconds")

# execution_time = timeit.timeit(lambda: find_closest_match(b, sentence_embeddings, generate_embedding), number=1)
# print(f"Execution Time: {execution_time} seconds")
# a, b = "环保行为", "嫁出去"
# execution_time = timeit.timeit(lambda: find_closest_match(a, sentence_embeddings, generate_embedding), number=1)
# print(f"Execution Time: {execution_time} seconds")

# execution_time = timeit.timeit(lambda: find_closest_match(b, sentence_embeddings, generate_embedding), number=1)
# print(f"Execution Time: {execution_time} seconds")

#%%
# a, b = "环保行为", "嫁出去"
# a = "公共卫生"
# execution_time = timeit.timeit(lambda: find_top_matches(a, sentence_embeddings, generate_embedding), number=1)
# print(f"Execution Time: {execution_time} seconds")

strings = ["公共秩序", "社會安全", "公共突发事件", "生态", "志愿", "慈善", "英雄", "烈士"]
# for string in strings:
#     find_top_matches(string, sentence_embeddings, generate_embedding)


# with ThreadPoolExecutor() as executor:
#     futures = [executor.submit(find_top_matches_multithread, string, sentence_embeddings, generate_embedding) for string in strings]
#     results = [future.result() for future in futures]

# execution_time = timeit.timeit(lambda: find_top_matches(b, sentence_embeddings, generate_embedding), number=1)
# 
# print(f"Execution Time: {execution_time} seconds")


#%% CREATE EMBEDDINGS FOR KEYWORDS


#%%
