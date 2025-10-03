#%%
import json
import pandas as pd
import os
import subprocess
import tqdm
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

data = []
DATA_DIR = os.getenv('DATA_DIR', '/Volumes/t7/data/wenming')
JSON_DIR = f"{DATA_DIR}/json"
DOCS_DIR = f"{DATA_DIR}/docs"
DB_PATH = os.getenv('DB_PATH', f"{DATA_DIR}/pragmatic_moral_governance.db")

import sys
here = os.path.dirname(sys.prefix)
os.chdir(here)

#%%
for filename in os.listdir(JSON_DIR):
    if filename.endswith('.json'):
        try:
            with open(os.path.join(JSON_DIR, filename), 'r') as f:
                json_data = json.load(f)
                if '文明' in json_data.get('title', ''):
                    data.append(json_data)
        except Exception as e:
            print(f"Error processing JSON file {filename}: {str(e)}")

df = pd.DataFrame(data)

#%%

def read_doc(file_path):
    try:
        process = subprocess.Popen(['soffice', '--headless', '--cat', file_path], stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
        return output
    except Exception as e:
        print(f"Error reading DOC file {file_path}: {str(e)}")
        return None

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Create a list of futures
    futures = []
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.doc', '.docx'))]
    for filename in tqdm(files, desc="Processing DOCs"):
        if filename.endswith(('.doc', '.docx')):
            id_ = filename.split('.')[0]
            index = df[df['id'] == id_].index
            if len(index) > 0:
                # Submit the task to the executor
                future = executor.submit(read_doc, os.path.join(DOCS_DIR, filename))
                futures.append((index, future))

    # Collect the results as they become available
    for index, future in futures:
        try:
            content = future.result()
            if content is not None:
                df.loc[index, 'content'] = content
            else:
                print(f"Failed to extract content from DOC file")
        except Exception as e:
            print(f"Error processing DOC file: {str(e)}")

            
#%% Save the df out as a simple csv file!
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.getcwd())
output_dir = os.path.join(PROJECT_ROOT, 'task2_cleanup_and_build_database/out')
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, 'output.csv'), index=False)

#%%
        
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Load the simple tokenizer extension
# conn.enable_load_extension(True)
# conn.load_extension(f'{here}/bin/libsimple-aarch64-linux-gnu-gcc-9/libsimple.so')  

# Create the FTS table
cursor.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS file_index USING fts5(
    id,
    title,
    office,
    publish,
    expiry,
    type,
    status,
    url,
    content);
""")

# Load data from the DataFrame into the FTS table
df.to_sql('file_index', conn, if_exists='append', index=False)

# Commit the transaction
conn.commit()












#%% SINGLE THREADED CODE 
            

# def read_doc(file_path):
#     try:
#         result = subprocess.run(['soffice', '--headless', '--cat', file_path], capture_output=True, text=True)
#         return result.stdout
#     except Exception as e:
#         print(f"Error reading DOC file {file_path}: {str(e)}")
#         return None

# files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.doc', '.docx'))]

# for filename in tqdm(files, desc="Processing DOCs"):
#     try:
#         id_ = filename.split('.')[0]
#         index = df[df['id'] == id_].index
#         if len(index) > 0:
#             content = read_doc(os.path.join(DOCS_DIR, filename))
#             if content is not None:
#                 df.loc[index, 'content'] = content
#             else:
#                 print(f"Failed to extract content from: {filename}")
#         else:
#             print(f"No matching id found in DataFrame for file: {filename}")
#     except Exception as e:
#         print(f"Error processing DOC file {filename}: {str(e)}")