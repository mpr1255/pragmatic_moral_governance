#%%
# Pragmatic Moral Governance Analysis Pipeline - Global Configuration
import pandas as pd
import numpy as np
import openai
import os
import sys
from fastdist import fastdist
from tqdm import tqdm
import openpyxl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project configuration
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.getcwd())
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, os.getenv('EMBEDDINGS_OUT', 'task2_cleanup_and_build_database/out'))

# OpenAI configuration  
MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
