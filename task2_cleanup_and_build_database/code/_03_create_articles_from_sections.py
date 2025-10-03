#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pandas",
#     "loguru",
#     "rich",
#     "tqdm", # Added tqdm
# ]
# ///

import pandas as pd
import re
import sys
from pathlib import Path
from loguru import logger
from rich.console import Console
from tqdm import tqdm # Added tqdm import

# --- Configuration ---
INPUT_CSV = Path("task2_cleanup_and_build_database/out/section2.csv")
OUTPUT_CSV = Path("task2_cleanup_and_build_database/out/section2_articles.csv")
ARTICLE_PATTERN = r'第.*?条' # Pattern to identify the start of an article
MIN_ARTICLE_LENGTH = 10 # Minimum characters to consider a chunk an article

# Set level for logger
logger.remove()
logger.add(sys.stderr, level="INFO")
console = Console()
# --- End Configuration ---

def main():
    """
    Reads section2.csv, splits content into articles based on '第...条',
    and saves them to section2_articles.csv.
    """
    logger.info(f"Reading section data from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        logger.error(f"Input file not found: {INPUT_CSV}")
        sys.exit(1)

    try:
        df_sections = pd.read_csv(INPUT_CSV)
        # Ensure essential columns exist and drop rows where content might be missing
        if 'id' not in df_sections.columns or 'content' not in df_sections.columns:
             logger.error("Input CSV must contain 'id' and 'content' columns.")
             sys.exit(1)
        df_sections.dropna(subset=['id', 'content'], inplace=True)
        df_sections['content'] = df_sections['content'].astype(str)
        logger.success(f"Loaded {len(df_sections)} sections.")
    except Exception as e:
        logger.error(f"Error reading input CSV {INPUT_CSV}: {e}")
        sys.exit(1)

    all_articles_data = []
    logger.info("Splitting sections into articles...")

    # Use tqdm for progress bar
    for index, row in tqdm(df_sections.iterrows(), total=len(df_sections), desc="Processing Sections"):
        doc_id = row['id']
        content = row['content']
        other_metadata = {col: row[col] for col in df_sections.columns if col not in ['content']} # Keep other columns

        # Find all starting positions of the article pattern
        matches = list(re.finditer(ARTICLE_PATTERN, content))

        if not matches:
            # If no pattern found, treat the whole content as one article if long enough
            if len(content.strip()) >= MIN_ARTICLE_LENGTH:
                article_data = other_metadata.copy()
                article_data['article_index'] = 0
                article_data['article_text'] = content.strip()
                all_articles_data.append(article_data)
            continue # Skip if no matches and content too short

        start_index = 0
        # Handle potential text before the first match
        first_match_start = matches[0].start()
        if first_match_start > 0:
             pre_text = content[:first_match_start].strip()
             if len(pre_text) >= MIN_ARTICLE_LENGTH:
                  # Decide whether to prepend this to the first article or treat separately
                  # For now, let's prepend it to the first actual article found
                  # Alternatively, could give it article_index -1 or similar
                  logger.debug(f"Found text before first article marker in {doc_id}. Prepending.")
                  # This text will be picked up when processing the first match below if start_index remains 0

        # Iterate through matches to define articles
        for i, match in enumerate(matches):
            current_article_start = match.start()

            # Text is from the start of this marker to the start of the next, or end of content
            next_article_start = matches[i+1].start() if (i + 1) < len(matches) else len(content)

            article_text = content[current_article_start:next_article_start].strip()

            if len(article_text) >= MIN_ARTICLE_LENGTH:
                article_data = other_metadata.copy()
                article_data['article_index'] = i # 0-based index of the article within the section
                article_data['article_text'] = article_text
                all_articles_data.append(article_data)
            # else:
                 # logger.debug(f"Skipping short article chunk in {doc_id}, index {i}")

    if not all_articles_data:
         logger.warning("No articles could be extracted. Output file will be empty.")
         # Create empty df with expected columns to avoid errors downstream
         df_articles = pd.DataFrame(columns=list(df_sections.columns).remove('content') + ['article_index', 'article_text'])
    else:
         df_articles = pd.DataFrame(all_articles_data)
         # Reorder columns nicely if needed
         cols_order = ['id', 'article_index', 'article_text'] + [col for col in df_sections.columns if col not in ['id', 'content']]
         # Ensure all columns exist before reordering
         cols_order = [col for col in cols_order if col in df_articles.columns]
         df_articles = df_articles[cols_order]


    logger.info(f"Extracted {len(df_articles)} articles.")
    logger.info(f"Saving articles to: {OUTPUT_CSV}")
    try:
        df_articles.to_csv(OUTPUT_CSV, index=False)
        logger.success("Articles saved successfully.")
    except Exception as e:
        logger.error(f"Error saving output CSV {OUTPUT_CSV}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 