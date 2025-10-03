#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pandas",
#     "toml",
#     "openai>=1.0", # Need >=1.0 for async client
#     "sqlite-utils",
#     "loguru",
#     "typer",
#     "rich",
# ]
# ///

import typer
import pandas as pd
import toml
import sqlite_utils
import os
import xml.etree.ElementTree as ET
import asyncio
from loguru import logger
from openai import AsyncOpenAI, APIError, RateLimitError # Use AsyncOpenAI
from typing import Optional, Dict, List, Tuple, Any
from rich.progress import track
import time # For verbose logging timestamp
import enum # Make sure this import is present
import httpx  # Need this for timeout configuration object

# --- Constants ---
REF_DIR = "task2_cleanup_and_build_database/ref" # Define base ref directory
OUTPUT_DIR = "task2_cleanup_and_build_database/out" # Define output dir for CSVs
# Input file paths based on granularity
SENTENCE_CSV_PATH = os.path.join(OUTPUT_DIR, "section2_sentences.csv")
ARTICLE_CSV_PATH = os.path.join(OUTPUT_DIR, "section2_articles.csv")
SECTION_CSV_PATH = os.path.join(OUTPUT_DIR, "section2.csv")

CONCEPTS_TOML_PATH = "task1a_create_concept_paragraphs/out/concepts.toml"
SCORE_PROMPT_TEMPLATE_PATH = os.path.join(REF_DIR, "score_prompt.txt") # Path for scoring
VOTE_PROMPT_TEMPLATE_PATH = os.path.join(REF_DIR, "vote_prompt.txt")   # Path for voting
DB_PATH = "task2_cleanup_and_build_database/out/concept_scores.db"
# Removed DB_TABLE_NAME constant, will be determined dynamically

# Default list of models, can be overridden by CLI
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    # "qwen/qwen-2.5-7b-instruct", # Removed due to unreliable XML output
    "deepseek/deepseek-chat-v3-0324", # Added standard version
    # "deepseek/deepseek-r1:free", # Removed due to NoneType errors
    # "qwen/qwen3-32b:free", # Removed due to NoneType errors
    "anthropic/claude-3-haiku",
    "openai/gpt-4.1-nano",
    # "google/gemini-2.0-flash-exp:free", # Removed due to errors
    "google/gemini-2.0-flash-lite-001",
    # "google/gemini-2.0-flash-001", # Keep just lite for now?
    # "qwen/qwen-2.5-72b-instruct", # Very large, remove for now
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CONCURRENCY_LIMIT = 50 # Limit to 50 concurrent API calls
CLIENT_TIMEOUT_SECONDS = 60.0 # Timeout for individual API requests

# --- Logger Setup ---
# Configure logger sink based on verbosity later in the main function
logger.remove() # Remove default stderr logger
logger.add(lambda msg: typer.echo(msg, err=True), level="INFO") # Default INFO to stderr via typer
logger.add("task2_cleanup_and_build_database/out/scoring_voting.log", rotation="10 MB", level="DEBUG") # Unified log file

# --- Typer App ---
app = typer.Typer()

# --- Global State (for verbose flag) ---
# Using a mutable type like dict for global state is a common pattern
state = {"verbose": False}

# --- Enums ---
class PromptType(str, enum.Enum):
    score = "score"
    vote = "vote"

# --- NEW: Granularity Enum ---
class Granularity(str, enum.Enum):
    sentence = "sentence"
    article = "article"
    section = "section"

# --- Helper Functions ---

def load_concepts(filepath: str) -> Dict[int, Tuple[str, str]]:
    """
    Loads concepts from a TOML file with keys like 'N_concept_name'
    and extracts the English slug and Chinese description.
    Returns a dictionary mapping concept ID (int) to Tuple[slug(str), chinese_desc(str)].
    """
    try:
        data = toml.load(filepath)
        concepts = {}
        for key, value_dict in data.items():
            try:
                # Extract number and slug from the key (e.g., "1_business_professional")
                parts = key.split('_', 1)
                if len(parts) != 2:
                    logger.warning(f"Could not parse key '{key}' into ID and slug. Skipping.")
                    continue
                concept_id_str, slug = parts
                concept_id = int(concept_id_str)
                # Basic slug sanitization (replace potential bad chars if needed, underscores are usually ok)
                sanitized_slug = slug.replace('-', '_').replace(' ', '_') # Example

                if isinstance(value_dict, dict) and 'chinese' in value_dict:
                    chinese_description = value_dict['chinese']
                    if isinstance(chinese_description, str):
                         # Store ID -> (slug, description)
                         concepts[concept_id] = (sanitized_slug, chinese_description.strip())
                    else:
                        logger.warning(f"Expected string for 'chinese' value in key '{key}', got {type(chinese_description)}. Skipping.")
                else:
                     logger.warning(f"Key '{key}' in {filepath} does not contain a 'chinese' field within a dictionary. Skipping.")

            except (ValueError, IndexError):
                logger.warning(f"Could not parse concept ID number from key '{key}'. Skipping.")
            except Exception as e:
                 logger.error(f"Unexpected error processing key '{key}': {e}. Skipping.")

        if not concepts:
             raise ValueError("No valid concepts could be loaded.")

        logger.info(f"Loaded {len(concepts)} concepts (ID -> (slug, description)) from {filepath}")
        sorted_concepts = dict(sorted(concepts.items()))
        return sorted_concepts # Returns Dict[int, Tuple[str, str]]

    except FileNotFoundError:
        logger.error(f"Concept file not found: {filepath}")
        raise typer.Exit(1)
    except (toml.TomlDecodeError, ValueError) as e:
        logger.error(f"Error loading or parsing concepts: {e}")
        raise typer.Exit(1)

def load_prompt_template(filepath: str) -> str:
    """Loads the prompt template from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            template = f.read()
        logger.info(f"Loaded prompt template from {filepath}")
        return template
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {filepath}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error reading prompt template file {filepath}: {e}")
        raise typer.Exit(1)

def get_timestamp() -> str:
    """Returns current timestamp in ISO 8601 UTC format."""
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def get_db(granularity: Granularity, concepts: Dict[int, Tuple[str, str]]) -> sqlite_utils.Database:
    """
    Initializes database connection, enables WAL mode, sets busy timeout,
    and defines table schemas BASED ON GRANULARITY.
    Requires concepts Dict[int, Tuple[slug, desc]] to create score columns.
    """
    logger.debug(f"Initializing database connection to: {DB_PATH}")
    db = sqlite_utils.Database(DB_PATH)

    # --- Enable WAL Mode & Set Busy Timeout ---
    try:
        # Check current journal mode
        current_journal_mode = db.execute("PRAGMA journal_mode;").fetchone()[0]
        logger.debug(f"Current journal_mode: {current_journal_mode}")

        if current_journal_mode.lower() != "wal":
            logger.info("Setting journal_mode=WAL for improved concurrency.")
            db.execute("PRAGMA journal_mode=WAL;")
            # Verify it changed
            new_journal_mode = db.execute("PRAGMA journal_mode;").fetchone()[0]
            logger.debug(f"New journal_mode: {new_journal_mode}")
            if new_journal_mode.lower() != "wal":
                 logger.warning("Failed to set journal_mode=WAL. Concurrency issues might persist.")
        else:
            logger.debug("journal_mode is already WAL.")

        # Set a busy timeout (e.g., 5000ms = 5 seconds)
        # SQLite will wait this long if locked before raising an error
        busy_timeout_ms = 5000
        logger.debug(f"Setting busy_timeout to {busy_timeout_ms}ms.")
        db.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")

    except Exception as e:
        logger.error(f"Failed to set PRAGMAs (WAL/busy_timeout): {e}")
        # Continue, but log the error

    # --- Define Schemas based on Granularity ---
    num_concepts = len(concepts)

    # -- Score Tables (Wide Format) --
    score_table_base_cols = { # Define common columns once
        "document_id": str,
        "model": str,
        "timestamp": str,
    }
    for concept_id, (slug, _) in concepts.items():
        col_name = f"{slug}_score"
        score_table_base_cols[col_name] = int

    if granularity == Granularity.sentence:
        score_table_name = "scores_wide_sentences"
        score_pk = ("row_index", "model")
        score_cols = {"row_index": int, "sentence_text": str, **score_table_base_cols}
        score_index_cols = ["document_id", "model"]
    elif granularity == Granularity.article:
        score_table_name = "scores_wide_articles"
        score_pk = ("document_id", "article_index", "model")
        score_cols = {"article_index": int, "article_text": str, **score_table_base_cols}
        score_index_cols = ["document_id", "model", "article_index"]
    elif granularity == Granularity.section:
        score_table_name = "scores_wide_sections"
        score_pk = ("document_id", "model")
        score_cols = {"section_text": str, **score_table_base_cols} # Use section_text column name
        score_index_cols = ["document_id", "model"]
    else:
        raise ValueError(f"Invalid granularity for score table: {granularity}")

    db[score_table_name].create(score_cols, pk=score_pk, if_not_exists=True)
    db[score_table_name].create_index(score_index_cols, if_not_exists=True)
    logger.debug(f"Ensured table '{score_table_name}' exists.")

    # -- Votes Tables --
    vote_table_base_cols = { # Define common columns
        "document_id": str,
        "model": str,
        "timestamp": str,
        "winning_concept_id": int,
        "winning_concept_name": str,
        "winning_concept_slug": str
    }

    if granularity == Granularity.sentence:
        vote_table_name = "votes_sentences"
        vote_pk = ("row_index", "model")
        vote_cols = {"row_index": int, "sentence_text": str, **vote_table_base_cols}
        vote_index_cols = ["document_id", "model", "winning_concept_slug"]
    elif granularity == Granularity.article:
        vote_table_name = "votes_articles"
        vote_pk = ("document_id", "article_index", "model")
        vote_cols = {"article_index": int, "article_text": str, **vote_table_base_cols}
        vote_index_cols = ["document_id", "model", "article_index", "winning_concept_slug"]
    elif granularity == Granularity.section:
        vote_table_name = "votes_sections"
        vote_pk = ("document_id", "model")
        vote_cols = {"section_text": str, **vote_table_base_cols}
        vote_index_cols = ["document_id", "model", "winning_concept_slug"]
    else:
        raise ValueError(f"Invalid granularity for vote table: {granularity}")

    db[vote_table_name].create(vote_cols, pk=vote_pk, if_not_exists=True)
    db[vote_table_name].create_index(vote_index_cols, if_not_exists=True)
    logger.debug(f"Ensured table '{vote_table_name}' exists.")

    logger.debug("Database initialization complete.")
    return db

def sync_check_existing_record(db_path: str, table_name: str, pk_values: Dict[str, Any]) -> bool:
    """Generic check if a record exists using provided primary key values."""
    db = sqlite_utils.Database(db_path)
    where_clause = " AND ".join([f"{key} = ?" for key in pk_values.keys()])
    params = list(pk_values.values())
    count = db[table_name].count_where(where_clause, params)
    return count > 0

async def check_existing_record_async(db_path: str, table_name: str, pk_values: Dict[str, Any]) -> bool:
    """Runs the synchronous record check in a thread."""
    return await asyncio.to_thread(sync_check_existing_record, db_path, table_name, pk_values)

def build_prompt(template: str, sentence: str, concepts: Dict[int, Tuple[str, str]]) -> str:
    """Builds the prompt, including concept slugs and descriptions."""
    num_concepts = len(concepts)
    # Format list as "1. slug: description"
    concept_list_for_prompt = "\n".join(
        [f"{idx}. {slug}: {desc}" for idx, (slug, desc) in concepts.items()]
    )
    try:
        return template.format(
            concept_list_for_prompt=concept_list_for_prompt,
            num_concepts=num_concepts,
            sentence=sentence
        )
    except KeyError as e:
        logger.error(f"Missing placeholder in prompt template: {e}")
        raise typer.Exit(1)


def parse_xml_scores(xml_string: str, num_concepts: int) -> Optional[Dict[int, int]]:
    """
    Parses the XML score string expecting <concept id="N">SCORE</concept> format
    and returns a dictionary mapping concept ID (int) to score (int).
    """
    scores = {}
    try:
        if not xml_string:
            logger.warning("Received empty XML string for parsing scores.")
            return None

        # Wrap in a root element to ensure valid XML for parsing
        clean_xml = "<root>" + xml_string.strip() + "</root>"
        root = ET.fromstring(clean_xml)

        concept_elements = root.findall("concept") # Find all <concept> elements

        if len(concept_elements) != num_concepts:
            logger.warning(f"Expected {num_concepts} <concept> tags, but found {len(concept_elements)}.")
            return None # Incorrect number of concept tags

        for element in concept_elements:
            concept_id_str = element.get("id") # Get the 'id' attribute
            score_text = element.text

            if concept_id_str is None:
                logger.warning("Found <concept> tag missing 'id' attribute.")
                return None # Malformed tag

            if score_text is None:
                logger.warning(f"Found <concept id='{concept_id_str}'> tag with empty score.")
                return None # Empty score

            try:
                concept_id = int(concept_id_str)
                score = int(score_text.strip())

                if not (1 <= concept_id <= num_concepts):
                     logger.warning(f"Parsed concept ID '{concept_id}' out of expected range (1-{num_concepts}).")
                     return None # Invalid concept ID range

                if not (0 <= score <= 10):
                    logger.warning(f"Score {score} for concept {concept_id} out of range (0-10).")
                    return None # Invalid score value

                if concept_id in scores:
                     logger.warning(f"Duplicate concept ID '{concept_id}' found in response.")
                     return None # Duplicate ID

                scores[concept_id] = score

            except ValueError:
                logger.warning(f"Non-integer score '{score_text}' or invalid ID '{concept_id_str}' found.")
                return None # Non-integer score or ID

        # Final check if we got all expected IDs after processing tags
        if len(scores) != num_concepts:
            logger.warning(f"After processing, expected {num_concepts} unique concept scores, but got {len(scores)}.")
            return None

        return scores

    except ET.ParseError as e:
        logger.warning(f"Failed to parse score XML: {e}\nRaw response:\n{xml_string}")
        return None
    except Exception as e: # Catch other potential errors during parsing
        logger.error(f"Unexpected error parsing score XML: {e}\nRaw response:\n{xml_string}")
        return None

def parse_vote_result(response_text: str, concepts: Dict[int, Tuple[str, str]]) -> Optional[int]:
    """Parses the vote result (expecting just the winning concept ID number)."""
    try:
        # Basic cleaning: strip whitespace
        cleaned_text = response_text.strip()
        # Attempt to directly convert to integer
        winning_id = int(cleaned_text)
        # Validate the ID against the known concepts
        if winning_id in concepts:
            return winning_id
        else:
            logger.warning(f"Parsed winning vote ID '{winning_id}' not found in known concepts.")
            return None
    except ValueError:
        logger.warning(f"Could not parse vote result - expected integer ID, got: '{response_text[:50]}...'")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing vote result: {e}\nRaw response:\n{response_text}")
        return None


# --- Modify Saving Functions ---
def sync_save_scores_wide(db_path: str, granularity: Granularity, record_data: Dict[str, Any], concepts: Dict[int, Tuple[str, str]]):
    """Synchronous save scores to the appropriate wide table based on granularity."""
    db = sqlite_utils.Database(db_path)
    # ... (Apply PRAGMAs as before) ...
    try:
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA busy_timeout = 5000;")
        db.execute("PRAGMA synchronous = NORMAL;")
    except Exception as e:
        logger.warning(f"Failed to set PRAGMAs in thread for wide save: {e}")

    # Determine table name and primary key based on granularity
    if granularity == Granularity.sentence:
        table_name = "scores_wide_sentences"
        pk = ("row_index", "model")
    elif granularity == Granularity.article:
        table_name = "scores_wide_articles"
        pk = ("document_id", "article_index", "model")
    elif granularity == Granularity.section:
        table_name = "scores_wide_sections"
        pk = ("document_id", "model")
    else:
        raise ValueError(f"Invalid granularity for sync_save_scores_wide: {granularity}")

    # Construct the record (already mostly done in record_data)
    record_to_save = record_data.copy() # Avoid modifying original dict
    scores = record_to_save.pop("scores") # Extract scores dict

    # Add concept scores dynamically using slugs
    for concept_id, score in scores.items():
        if concept_id in concepts:
            slug, _ = concepts[concept_id]
            col_name = f"{slug}_score"
            record_to_save[col_name] = score
        else:
            logger.warning(f"Concept ID {concept_id} from scores not found in concepts map during save. Skipping.")

    try:
        db[table_name].upsert(record_to_save, pk=pk, alter=True)
    except Exception as e:
        # Identify the record using PK for better logging
        pk_string = ", ".join([f"{k}={record_to_save.get(k)}" for k in pk])
        raise RuntimeError(f"Failed to save wide scores ({granularity.value}) for PK [{pk_string}], model {record_to_save.get('model')}: {e}") from e
    finally:
        db.close() # Explicitly close connection

async def save_scores_wide_async(db_path: str, granularity: Granularity, record_data: Dict[str, Any], concepts: Dict[int, Tuple[str, str]]):
    """Runs the synchronous wide score save in a thread."""
    try:
        await asyncio.to_thread(sync_save_scores_wide, db_path, granularity, record_data, concepts)
        # Log using appropriate identifier
        if granularity == Granularity.sentence:
             log_id = f"row {record_data.get('row_index')}"
        elif granularity == Granularity.article:
             log_id = f"doc {record_data.get('document_id')} art {record_data.get('article_index')}"
        else: # Section
             log_id = f"doc {record_data.get('document_id')}"
        logger.debug(f"Saved wide scores ({granularity.value}) for {log_id}, model {record_data.get('model')}")
    except Exception as e:
        logger.error(f"Error saving wide scores async ({granularity.value}): {e}")


def sync_save_vote(db_path: str, granularity: Granularity, record_data: Dict[str, Any], concepts: Dict[int, Tuple[str, str]]):
    """Synchronous save vote result to the appropriate table."""
    db = sqlite_utils.Database(db_path)
    # ... (Apply PRAGMAs as before) ...
    try:
        db.execute("PRAGMA journal_mode=WAL;")
        db.execute("PRAGMA busy_timeout = 5000;")
        db.execute("PRAGMA synchronous = NORMAL;")
    except Exception as e:
        logger.warning(f"Failed to set PRAGMAs in thread for vote save: {e}")

    # Determine table name and primary key
    if granularity == Granularity.sentence:
        table_name = "votes_sentences"
        pk = ("row_index", "model")
    elif granularity == Granularity.article:
        table_name = "votes_articles"
        pk = ("document_id", "article_index", "model")
    elif granularity == Granularity.section:
        table_name = "votes_sections"
        pk = ("document_id", "model")
    else:
        raise ValueError(f"Invalid granularity for sync_save_vote: {granularity}")

    # Construct record
    record_to_save = record_data.copy()
    winning_concept_id = record_to_save.pop("winning_concept_id") # Extract winner ID
    winning_slug, winning_name = concepts.get(winning_concept_id, (f"Unknown_{winning_concept_id}", f"Unknown Concept {winning_concept_id}"))
    record_to_save["winning_concept_id"] = winning_concept_id
    record_to_save["winning_concept_name"] = winning_name
    record_to_save["winning_concept_slug"] = winning_slug

    try:
        db[table_name].upsert(record_to_save, pk=pk, alter=True)
    except Exception as e:
        pk_string = ", ".join([f"{k}={record_to_save.get(k)}" for k in pk])
        raise RuntimeError(f"Failed to save vote ({granularity.value}) for PK [{pk_string}], model {record_to_save.get('model')}: {e}") from e
    finally:
        db.close() # Explicitly close connection

async def save_vote_async(db_path: str, granularity: Granularity, record_data: Dict[str, Any], concepts: Dict[int, Tuple[str, str]]):
    """Runs the synchronous vote save in a thread."""
    try:
        await asyncio.to_thread(sync_save_vote, db_path, granularity, record_data, concepts)
         # Log using appropriate identifier
        if granularity == Granularity.sentence:
             log_id = f"row {record_data.get('row_index')}"
        elif granularity == Granularity.article:
             log_id = f"doc {record_data.get('document_id')} art {record_data.get('article_index')}"
        else: # Section
             log_id = f"doc {record_data.get('document_id')}"
        logger.debug(f"Saved vote ({granularity.value}) for {log_id}, model {record_data.get('model')} (Winner ID: {record_data.get('winning_concept_id')})")

    except Exception as e:
        logger.error(f"Error saving vote async ({granularity.value}): {e}")

# --- Modify Async Processing Function ---

async def process_units_async( # Renamed from process_sentences_async
    df_to_process: pd.DataFrame, # Accepts dataframe with units to process
    granularity: Granularity,   # Pass granularity
    models: List[str],
    concepts: Dict[int, Tuple[str, str]],
    prompt_template: str,
    db_path: str,
    overwrite: bool,
    prompt_type: PromptType,
):
    """Processes text units (sentences, articles, sections) asynchronously."""
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    logger.info(f"Concurrency limited to {CONCURRENCY_LIMIT} simultaneous tasks.")

    # --- Variable Init ---
    # ... (counts remain similar) ...
    skipped_due_to_existing = 0
    skipped_invalid = 0
    failed_fetch_count = 0
    saved_count = 0
    num_concepts = len(concepts)

    # --- Determine Target Table and Text Column ---
    if prompt_type == PromptType.score:
        action_desc = "scoring"
        if granularity == Granularity.sentence:
            target_table = "scores_wide_sentences"
            text_col = "sentence_text"
            id_col_map = {"row_index": "row_index", "document_id": "id", "text": "sentence"}
            pk_cols = ["row_index", "model"]
        elif granularity == Granularity.article:
            target_table = "scores_wide_articles"
            text_col = "article_text"
            id_col_map = {"article_index": "article_index", "document_id": "id", "text": "article_text"}
            pk_cols = ["document_id", "article_index", "model"]
        elif granularity == Granularity.section:
            target_table = "scores_wide_sections"
            text_col = "section_text" # Assuming the column is named 'content' in section2.csv
            id_col_map = {"document_id": "id", "text": "content"}
            pk_cols = ["document_id", "model"]
        else: raise ValueError(f"Invalid granularity: {granularity}")

    elif prompt_type == PromptType.vote:
        action_desc = "voting"
        if granularity == Granularity.sentence:
            target_table = "votes_sentences"
            text_col = "sentence_text"
            id_col_map = {"row_index": "row_index", "document_id": "id", "text": "sentence"}
            pk_cols = ["row_index", "model"]
        elif granularity == Granularity.article:
            target_table = "votes_articles"
            text_col = "article_text"
            id_col_map = {"article_index": "article_index", "document_id": "id", "text": "article_text"}
            pk_cols = ["document_id", "article_index", "model"]
        elif granularity == Granularity.section:
            target_table = "votes_sections"
            text_col = "section_text" # Assuming the column is named 'content' in section2.csv
            id_col_map = {"document_id": "id", "text": "content"}
            pk_cols = ["document_id", "model"]
        else: raise ValueError(f"Invalid granularity: {granularity}")
    else:
        logger.error(f"Invalid prompt type: {prompt_type}")
        return

    # --- Configure Client ---
    # ... (Client config remains the same) ...
    timeout_config = httpx.Timeout(CLIENT_TIMEOUT_SECONDS, connect=10.0)
    client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, max_retries=0, timeout=timeout_config)

    logger.info(f"Processing {len(df_to_process)} input units (Granularity: {granularity.value}) against {len(models)} models (Type: {prompt_type.value}, Target: {target_table})...")

    # --- Phase 0: Prepare valid units map (using appropriate identifiers) ---
    valid_units_map = {} # Store pk_values -> text_unit
    logger.info("Validating input units...")
    for idx, row in track(df_to_process.iterrows(), description="Validating units", total=len(df_to_process)):
        # Extract PK values and text based on granularity
        pk_values = {}
        text_unit = ""
        try:
            if granularity == Granularity.sentence:
                # Use the DataFrame index as row_index if not present as a column
                pk_values["row_index"] = int(row.get('row_index', idx)) # Handle potential missing row_index col
                pk_values["document_id"] = str(row[id_col_map["document_id"]])
                text_unit = str(row[id_col_map["text"]])
            elif granularity == Granularity.article:
                pk_values["document_id"] = str(row[id_col_map["document_id"]])
                pk_values["article_index"] = int(row[id_col_map["article_index"]])
                text_unit = str(row[id_col_map["text"]])
            elif granularity == Granularity.section:
                pk_values["document_id"] = str(row[id_col_map["document_id"]])
                text_unit = str(row[id_col_map["text"]])

            # Validate text
            if not text_unit or len(text_unit.strip()) < 5:
                 logger.warning(f"Skipping invalid unit (PK: {pk_values}): '{text_unit[:50]}...'")
                 skipped_invalid += 1
                 continue

            # Store PK tuple -> text (PKs need to be hashable, dicts aren't)
            pk_tuple = tuple(sorted(pk_values.items()))
            valid_units_map[pk_tuple] = text_unit

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing row {idx} for {granularity.value}: {e}. Skipping. Row data: {row.to_dict()}")
            skipped_invalid += 1
            continue

    logger.info(f"Prepared {len(valid_units_map)} valid units for processing.")

    # --- Phase 1: Check existing records ---
    logger.info(f"Checking for existing records in '{target_table}'...")
    check_tasks = {} # Store pk_tuple -> list of model tasks
    if not overwrite:
        for pk_tuple in track(valid_units_map.keys(), description=f"Checking existing {action_desc}", total=len(valid_units_map)):
            pk_values_dict = dict(pk_tuple) # Convert back to dict for checking
            model_tasks = []
            for model in models:
                # Combine unit PK with model for the check
                check_pk = {**pk_values_dict, "model": model}
                model_tasks.append(asyncio.create_task(check_existing_record_async(db_path, target_table, check_pk)))
            check_tasks[pk_tuple] = model_tasks # Store list of tasks for this unit
    else:
         logger.info("Overwrite enabled; skipping existing checks.")

    # Run checks concurrently per unit
    existing_map = {} # pk_tuple -> {model: bool}
    if check_tasks:
        num_check_errors = 0
        for pk_tuple, model_tasks in track(check_tasks.items(), description="Gathering checks", total=len(check_tasks)):
             results_or_errors = await asyncio.gather(*model_tasks, return_exceptions=True)
             model_existence = {}
             for i, res in enumerate(results_or_errors):
                  model = models[i] # Assumes order is preserved
                  if isinstance(res, Exception):
                       logger.warning(f"Error checking existence for PK {dict(pk_tuple)}, model {model}: {res}")
                       model_existence[model] = False # Assume not existing on error
                       num_check_errors += 1
                  else:
                       model_existence[model] = res
             existing_map[pk_tuple] = model_existence
        if num_check_errors > 0:
             logger.warning(f"Encountered {num_check_errors} errors during existing record checks.")
        logger.info(f"Finished checking existing records.")


    # --- Phase 2: Schedule LLM calls ---
    logger.info(f"Scheduling LLM API calls ({granularity.value} level)...")
    llm_tasks = []

    # --- Define worker function to return result AND metadata ---
    async def worker(pk_values_dict, text_to_process, model_name):
        async with semaphore:
            log_identifier = f"PK:{pk_values_dict} M:{model_name}" # Create consistent log ID
            try:
                # Pass appropriate identifier for logging within get_llm_result
                # Using a simple string representation of PK for now
                llm_result = await get_llm_result(
                    client, str(pk_values_dict), text_to_process, concepts, model_name, prompt_template, prompt_type
                )
                return (pk_values_dict, model_name, text_to_process, llm_result) # Return necessary info + result
            except Exception as e:
                 logger.debug(f"Worker caught exception for {log_identifier}: {e}")
                 return (pk_values_dict, model_name, text_to_process, e) # Return exception

    num_tasks_scheduled = 0
    for pk_tuple, text_unit in track(valid_units_map.items(), description="Scheduling calls", total=len(valid_units_map)):
        pk_values_dict = dict(pk_tuple) # Convert back for use
        models_to_run_for_unit = models # Assume all models unless overwrite=False
        if not overwrite and pk_tuple in existing_map:
             models_to_run_for_unit = [m for m in models if not existing_map[pk_tuple].get(m, False)]
             skipped_count = len(models) - len(models_to_run_for_unit)
             if skipped_count > 0:
                 skipped_due_to_existing += skipped_count
                 # logger.debug(f"Skipping {skipped_count} models for PK {pk_values_dict} due to existing records.")


        for model in models_to_run_for_unit:
            task = asyncio.create_task(worker(pk_values_dict, text_unit, model))
            llm_tasks.append(task)
            num_tasks_scheduled += 1

    logger.info(f"Scheduled {num_tasks_scheduled} LLM tasks ({granularity.value} level) to process.")
    if skipped_due_to_existing > 0:
         logger.info(f"Skipped {skipped_due_to_existing} task instances due to existing records.")


    # --- Phase 3: Execute LLM calls and Save Immediately ---
    logger.info(f"Executing scheduled LLM calls (max {CONCURRENCY_LIMIT} concurrent) and saving results...")
    if llm_tasks:
        for future in track(
            asyncio.as_completed(llm_tasks),
            description="Processing results",
            total=num_tasks_scheduled
        ):
            try:
                # Await the future to get the tuple: (pk_dict, model, text, result_or_exception)
                pk_values_dict, model, text_unit, result_or_exception = await future

                log_identifier = f"PK:{pk_values_dict} M:{model}" # Consistent log ID

                if isinstance(result_or_exception, Exception):
                    e = result_or_exception
                    if isinstance(e, httpx.TimeoutException):
                        logger.error(f"LLM Task TIMEOUT for {log_identifier}, type {prompt_type.value}: {e}")
                    elif isinstance(e, APIError):
                        logger.error(f"LLM Task API Error for {log_identifier}, type {prompt_type.value}: {e}")
                    else:
                        logger.error(f"LLM Task failed unexpectedly for {log_identifier}, type {prompt_type.value}: {e}")
                    failed_fetch_count += 1
                elif result_or_exception is None:
                    logger.error(f"LLM Task returned None (failed parse/retry) for {log_identifier}, type {prompt_type.value}")
                    failed_fetch_count += 1
                else:
                    # Prepare record data for saving
                    llm_result_data = result_or_exception
                    current_timestamp = get_timestamp()
                    record_data = {
                         **pk_values_dict, # Add PK columns
                         "model": model,
                         "timestamp": current_timestamp,
                         # Add text column with correct name based on granularity
                         text_col: text_unit
                    }

                    try:
                        if prompt_type == PromptType.score:
                            record_data["scores"] = llm_result_data # Add scores dict for sync func
                            await save_scores_wide_async(db_path, granularity, record_data, concepts)
                            saved_count += 1
                        elif prompt_type == PromptType.vote:
                            record_data["winning_concept_id"] = llm_result_data # Add winner id for sync func
                            await save_vote_async(db_path, granularity, record_data, concepts)
                            saved_count += 1
                    except Exception as save_e:
                         logger.error(f"Failed to save result for {log_identifier}: {save_e}")
                         failed_fetch_count += 1

            except Exception as outer_e:
                logger.error(f"Error processing future result: {outer_e} - Future: {future}")
                failed_fetch_count += 1


    logger.info("All LLM tasks completed or failed.")
    # --- Summary Logging ---
    logger.info(f"--- Async Processing Summary (Granularity: {granularity.value}, Type: {prompt_type.value}) ---")
    logger.info(f"Total units in input file considered: {len(df_to_process)}")
    logger.info(f"Units skipped (invalid text/structure): {skipped_invalid}")
    logger.info(f"Task instances skipped (existing & no overwrite): {skipped_due_to_existing}")
    logger.info(f"Task instances attempted LLM processing: {num_tasks_scheduled}") # Total tasks we tried to run
    logger.info(f"LLM calls failed (API/parsing/timeout/save/internal): {failed_fetch_count}")
    logger.info(f"Results successfully saved to '{target_table}': {saved_count}")
    logger.info("------------------------------------------")

# --- Update get_llm_result to accept generic identifier string ---
async def get_llm_result(
    client: AsyncOpenAI,
    unit_identifier_str: str, # Generic identifier string for logging
    text_unit: str,
    concepts: Dict[int, Tuple[str, str]],
    model: str,
    prompt_template: str,
    prompt_type: PromptType,
    max_retries: int = 10
) -> Optional[Any]:
    prompt = build_prompt(prompt_template, text_unit, concepts) # Use text_unit
    num_concepts = len(concepts)
    attempts = 0

    log_prefix = f"[UNIT: {unit_identifier_str}] [MODEL: {model}] [TYPE: {prompt_type.value}]"

    if state["verbose"]:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"[{timestamp}] {log_prefix} PROMPT:\n{prompt}\n")

    while attempts <= max_retries:
        attempts += 1
        response_text = None
        try:
            # Use the same completion call, but pass the specific text unit
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300 if prompt_type == PromptType.score else 50, # Adjust max tokens maybe?
                extra_headers={"X-Title": "Wenming Project Scoring/Voting"}
            )
            response_text = completion.choices[0].message.content

            if state["verbose"]:
                 timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                 logger.debug(f"[{timestamp}] {log_prefix} RAW RESPONSE:\n{response_text}\n")

            if response_text:
                if prompt_type == PromptType.score:
                    parsed_result = parse_xml_scores(response_text, num_concepts)
                elif prompt_type == PromptType.vote:
                    parsed_result = parse_vote_result(response_text, concepts)
                else:
                    logger.error(f"Unknown prompt type: {prompt_type}")
                    return None

                if parsed_result is not None:
                    return parsed_result
                else:
                    logger.warning(f"Malformed response (attempt {attempts}/{max_retries+1}) for {log_prefix}")
            else:
                 logger.warning(f"Empty response (attempt {attempts}/{max_retries+1}) for {log_prefix}")

        except RateLimitError as e:
            logger.error(f"Rate limit hit for {log_prefix}: {e}. Stopping.")
            return None # Return None on rate limit
        except APIError as e:
             logger.error(f"OpenRouter API error (attempt {attempts}/{max_retries+1}) for {log_prefix}: {e}")
        except httpx.TimeoutException as e: # Explicitly catch timeout
             logger.error(f"Timeout error (attempt {attempts}/{max_retries+1}) for {log_prefix}: {e}")
             # Do not return here, let it retry if attempts remain
        except Exception as e:
            logger.error(f"Unexpected error (attempt {attempts}/{max_retries+1}) calling LLM for {log_prefix}: {e}")
            if response_text and state["verbose"]:
                 timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                 logger.debug(f"[{timestamp}] {log_prefix} RAW RESPONSE (before error):\n{response_text}\n")

        if attempts <= max_retries:
            await asyncio.sleep(1)
        else:
             logger.error(f"Failed to get valid result for {log_prefix} after {attempts} attempts.")
             # Explicitly return None after max retries if caught exception prevented success
             return None

    return None # Should only be reached if loop finishes without success


# --- Main Typer Command ---
@app.command()
def run_models(
    # --- NEW Granularity Option ---
    granularity: Granularity = typer.Option(Granularity.sentence, "--granularity", "-g", help="Level of text unit to process."),
    # --- Mode Selection Flags ---
    score_mode: bool = typer.Option(False, "--score", help="Run in scoring mode. Mutually exclusive with --vote."),
    vote_mode: bool = typer.Option(False, "--vote", help="Run in voting mode. Mutually exclusive with --score."),
    # --- Other Options ---
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit processing to the first N units."),
    overwrite: bool = typer.Option(False, "--overwrite", "-o", help="Overwrite existing results."),
    models: List[str] = typer.Option(DEFAULT_MODELS, "--model", "-m", help="OpenRouter model identifier(s)."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed logging."),
):
    """
    Processes text units (sentences, articles, or sections) from CSV,
    gets concept SCORES or VOTES using LLMs via OpenRouter asynchronously,
    and saves results to granularity-specific tables in a SQLite database.
    Requires EITHER --score OR --vote flag.
    Requires OPENROUTER_API_KEY environment variable.
    """
    # --- Validate Mode Selection ---
    if score_mode and vote_mode:
        logger.error("Cannot specify both --score and --vote flags.")
        raise typer.Exit(code=1)
    if not score_mode and not vote_mode:
        logger.error("Must specify either --score or --vote flag.")
        raise typer.Exit(code=1)

    # --- Determine Prompt Type and Template Path ---
    if score_mode:
        prompt_type = PromptType.score
        prompt_template_path = SCORE_PROMPT_TEMPLATE_PATH
    else:
        prompt_type = PromptType.vote
        prompt_template_path = VOTE_PROMPT_TEMPLATE_PATH

    # --- Determine Input Path based on Granularity ---
    if granularity == Granularity.sentence:
        input_csv_path = SENTENCE_CSV_PATH
        text_col_name = "sentence" # Expected text column in the input CSV
    elif granularity == Granularity.article:
        input_csv_path = ARTICLE_CSV_PATH
        text_col_name = "article_text"
    elif granularity == Granularity.section:
        input_csv_path = SECTION_CSV_PATH
        text_col_name = "content" # Text column in section2.csv is 'content'
    else:
        # Should be caught by typer/enum validation, but defensive check
        logger.error(f"Invalid granularity selected: {granularity}")
        raise typer.Exit(code=1)

    # --- Setup (Logging, API Key Check) ---
    state["verbose"] = verbose
    if verbose:
        logger.add(lambda msg: typer.echo(msg, err=True), level="DEBUG")
        logger.info("Verbose logging enabled.")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        raise typer.Exit(1)

    # --- Load Concepts ---
    try:
        concepts = load_concepts(CONCEPTS_TOML_PATH)
        num_concepts = len(concepts)
        # if num_concepts != 9: # Keep check if relevant
        #     logger.warning(f"Expected 9 concepts, but loaded {num_concepts}.")
    except Exception as e:
        logger.error(f"Failed to load concepts: {e}")
        raise typer.Exit(1)

    # --- Log Settings ---
    logger.info(f"Starting processing. Granularity: {granularity.value}, Task Type: {prompt_type.value}")
    logger.info(f"Models: {models}")
    logger.info(f"Reading units from: {input_csv_path}")
    logger.info(f"Loaded concepts from: {CONCEPTS_TOML_PATH}")
    logger.info(f"Loading prompt template from: {prompt_template_path}")
    logger.info(f"Writing results to: {DB_PATH} (Tables like 'scores_wide_{granularity.value}', 'votes_{granularity.value}')")
    logger.info(f"Limit: {'None' if limit is None else limit}")
    logger.info(f"Overwrite: {overwrite}")

    # --- Load Input Units & Prompt Template ---
    try:
        df_full = pd.read_csv(input_csv_path)
        # Basic validation for expected columns based on granularity
        required_cols = ['id'] # Always need document ID
        if granularity == Granularity.sentence: required_cols.extend(['sentence']) # or row_index if used as PK
        elif granularity == Granularity.article: required_cols.extend(['article_index', 'article_text'])
        elif granularity == Granularity.section: required_cols.extend(['content'])

        missing_cols = [col for col in required_cols if col not in df_full.columns]
        if missing_cols:
             logger.error(f"Missing required columns in {input_csv_path} for {granularity.value} granularity: {missing_cols}")
             raise typer.Exit(1)

        # Drop rows with missing text unit
        df_full = df_full.dropna(subset=[text_col_name])
        df_full[text_col_name] = df_full[text_col_name].astype(str)
        # Ensure ID cols are correct type
        df_full['id'] = df_full['id'].astype(str)
        if granularity == Granularity.article:
             df_full['article_index'] = df_full['article_index'].astype(int)

    except FileNotFoundError:
        logger.error(f"Input file not found: {input_csv_path}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Error reading input CSV file {input_csv_path}: {e}")
        raise typer.Exit(1)

    prompt_template = load_prompt_template(prompt_template_path)

    # Apply limit
    units_to_process_df = df_full.head(limit) if limit else df_full

    # Initialize DB (pass granularity and concepts)
    get_db(granularity, concepts) # Call get_db to ensure tables exist

    # --- Run Async Logic ---
    asyncio.run(process_units_async( # Call renamed function
        df_to_process=units_to_process_df,
        granularity=granularity,
        models=models,
        concepts=concepts,
        prompt_template=prompt_template,
        db_path=DB_PATH,
        overwrite=overwrite,
        prompt_type=prompt_type,
    ))

    logger.info("Processing finished.")


if __name__ == "__main__":
    app()
