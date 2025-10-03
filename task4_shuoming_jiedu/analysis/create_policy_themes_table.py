#!/usr/bin/env python3

import sqlite3
import json
import os
from pathlib import Path

# Resolve paths relative to this file
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ANALYSIS_DIR.parent
DB_PATH = Path(os.getenv('WENMING_SHUOMING_DB', PROJECT_DIR / 'shuoming_jiedu.db'))

# Connect to the database
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

# Query the documents table
cursor.execute("SELECT province, city, year, policy_themes FROM documents")
rows = cursor.fetchall()
conn.close()

# Process data and deduplicate
documents_dict = {}
policy_themes = ['P1', 'P2', 'B1', 'B2', 'L1', 'L2']

for row in rows:
    province, city, year, themes_json = row
    
    # Handle None city case (Tianjin)
    if city is None or city == 'None':
        city = '天津市'
    
    document_key = (city, province, year)
    
    # Parse themes
    themes_present = set()
    if themes_json and themes_json != 'NULL':
        try:
            themes = json.loads(themes_json)
            for theme in themes:
                if theme in policy_themes:
                    themes_present.add(theme)
        except json.JSONDecodeError:
            pass
    
    # Merge themes if document already exists (deduplication)
    if document_key in documents_dict:
        documents_dict[document_key].update(themes_present)
    else:
        documents_dict[document_key] = themes_present

# Convert to list format and only include documents with themes
documents = []
for (city, province, year), themes_set in documents_dict.items():
    # Skip documents with no themes
    if not themes_set:
        continue
        
    document_name = f"{city}, {province} ({year})"
    themes_dict = {theme: theme in themes_set for theme in policy_themes}
    documents.append((document_name, themes_dict))

# Sort documents
documents.sort(key=lambda x: x[0])

# Create CSV
csv_content = "Document," + ",".join(policy_themes) + "\n"
for doc_name, themes in documents:
    row = [doc_name] + ["✓" if themes[theme] else "" for theme in policy_themes]
    csv_content += ",".join(f'"{cell}"' for cell in row) + "\n"

csv_path = ANALYSIS_DIR / 'policy_themes_table.csv'
with open(csv_path, 'w') as f:
    f.write(csv_content)

# Create markdown table
markdown_content = "# Policy Themes by Document\n\n"
markdown_content += "| Document | " + " | ".join(policy_themes) + " |\n"
markdown_content += "|" + "---|" * (len(policy_themes) + 1) + "\n"

for doc_name, themes in documents:
    row = [doc_name] + ["✓" if themes[theme] else "" for theme in policy_themes]
    markdown_content += "| " + " | ".join(row) + " |\n"

markdown_path = ANALYSIS_DIR / 'policy_themes_table.md'
with open(markdown_path, 'w') as f:
    f.write(markdown_content)

print("Created both CSV and Markdown versions:")
print(f"- {csv_path.name}")
print(f"- {markdown_path.name}")
print(f"\nProcessed {len(documents)} documents")
print("Policy themes:", ", ".join(policy_themes))
