#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.14" # Adjusted python range slightly
# dependencies = [
#     "krippendorff",
#     "pandas",
#     "numpy",
#     "sqlite-utils", # Using this for easier DB interaction
#     "loguru",
#     "rich", # For nice printing
#     "typer", # Added typer dependency
# ]
# ///

import sqlite3
import pandas as pd
import numpy as np
import krippendorff
import sys
import enum # Added enum
import itertools # Added import
from pathlib import Path
from rich.console import Console
from rich.table import Table
from loguru import logger
import typer # Added typer

# --- Configuration ---
DB_PATH = Path("./task2_cleanup_and_build_database/out/concept_scores.db")
# TABLE_NAME = "scores_wide"
TABLE_NAME = "scores_wide_articles"
# Base list of models to always exclude due to technical issues etc.
BASE_EXCLUDE_MODELS_LIST = [
    "qwen/qwen-2.5-7b-instruct",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-32b:free",
    "google/gemini-2.0-flash-exp:free",
]
# Unique identifier columns mapping per granularity
# Format: {granularity: {pk_col_db: pk_col_df, ...}, text_col_name_db: text_col_name_df}
# Note: DF names might be same as DB names if read directly
GRANULARITY_CONFIG = {
    "sentence": {
        "table_name": "scores_wide",
        "pk_cols": {"row_index": "row_index", "document_id": "document_id"},
        "text_col": "sentence_text",
        "id_cols_for_melt": ["row_index", "document_id", "model"] # Include model for melting
    },
    "article": {
        "table_name": "scores_wide_articles",
        "pk_cols": {"document_id": "document_id", "article_index": "article_index"},
        "text_col": "article_text",
        "id_cols_for_melt": ["document_id", "article_index", "model"]
    },
    "section": {
        "table_name": "scores_wide_sections",
        "pk_cols": {"document_id": "document_id"},
        "text_col": "section_text", # Text col in DB table
        "id_cols_for_melt": ["document_id", "model"]
    }
}
# Set level for logger
logger.remove()
logger.add(sys.stderr, level="INFO")
# --- End Configuration ---

# --- Enums ---
class Granularity(str, enum.Enum):
    sentence = "sentence"
    article = "article"
    section = "section"

# --- Typer App ---
app = typer.Typer()

# --- Helper Function to Calculate Alpha for a Subset ---
# (This function remains the same as the previous "best triplet" version)
def calculate_alpha_for_subset(df_filtered_global, models_to_include, score_cols, granularity_value, pk_cols_list_df, id_cols_for_melt):
    """Calculates Krippendorff's Alpha for a specific subset of models."""
    # logger.debug(f"Calculating Alpha for models: {', '.join(models_to_include)}")
    df_subset = df_filtered_global[df_filtered_global["model"].isin(models_to_include)].copy()

    if len(df_subset['model'].unique()) < 2:
         # logger.warning(f"Subset {models_to_include} has < 2 models. Skipping.")
         return None, {} # Return None for mean_alpha, empty dict for results

    results = {}
    # --- Data Prep for Subset ---
    try:
        # Ensure all melt ID columns exist in the DataFrame
        missing_melt_ids = [col for col in id_cols_for_melt if col not in df_subset.columns]
        if missing_melt_ids:
             logger.error(f"Internal Error: Missing columns required for melting subset: {missing_melt_ids}")
             return None, {}

        df_long = pd.melt(df_subset, id_vars=id_cols_for_melt, value_vars=score_cols, var_name="concept", value_name="score")

        # Create unique unit id
        # logger.debug(f"Creating unique ID for subset using PK columns: {pk_cols_list_df}")
        missing_pk_cols = [col for col in pk_cols_list_df if col not in df_long.columns]
        if missing_pk_cols:
            logger.error(f"Internal Error: Missing PK columns for unique ID in subset: {missing_pk_cols}")
            return None, {}
        df_long['unique_unit_id'] = df_long[pk_cols_list_df].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    except KeyError as e:
        logger.error(f"Error melting subset DataFrame. Missing column: {e}.")
        return None, {}
    except Exception as e:
        logger.error(f"Error processing subset data: {e}")
        return None, {}

    # --- Alpha Calculation for Subset ---
    for concept in score_cols:
        concept_df = df_long[df_long["concept"] == concept]
        try:
            pivot_df = concept_df.pivot_table(index='unique_unit_id', columns='model', values='score', aggfunc='mean')
        except Exception as e:
            results[concept] = {'alpha': np.nan, 'error': str(e)}
            continue

        reliability_data = pivot_df.T.to_numpy()
        if reliability_data.shape[0] < 2 or reliability_data.shape[1] < 2:
            results[concept] = {'alpha': np.nan, 'error': 'Insufficient data'}
            continue

        reliability_data = np.where(pd.isna(reliability_data), np.nan, reliability_data)

        try:
            alpha_value = krippendorff.alpha(reliability_data, level_of_measurement='ordinal')
            results[concept] = {'alpha': alpha_value, 'error': None}
        except ZeroDivisionError:
            # logger.warning(f"  Skipping alpha for '{concept}' in subset: Zero division error.")
            results[concept] = {'alpha': np.nan, 'error': 'Zero division error'}
        except Exception as e:
            # logger.warning(f"  Error calculating alpha for concept '{concept}' in subset: {e}")
            results[concept] = {'alpha': np.nan, 'error': str(e)}

    # --- Calculate Mean Alpha for Subset ---
    total_alpha = 0
    valid_concepts_count = 0
    for concept, res in results.items():
         if res['error'] is None and not np.isnan(res['alpha']):
             total_alpha += res['alpha']
             valid_concepts_count += 1

    mean_alpha = total_alpha / valid_concepts_count if valid_concepts_count > 0 else np.nan
    # logger.info(f"Mean Alpha for subset {models_to_include}: {mean_alpha:.4f}")
    return mean_alpha, results
# --- End Helper Function ---


# --- Main Command ---
@app.command()
def calculate_alpha(
    granularity: Granularity = typer.Option(Granularity.sentence, "--granularity", "-g", help="Level of text unit to process."),
    subset_size: int = typer.Option(3, "--subset-size", "-s", help="Size of model subset to test (e.g., 2 for pairs, 3 for triplets)."),
):
    """
    Calculates Krippendorff's Alpha for concept scores stored in an SQLite DB
    at the specified granularity level, finding the best agreeing subset of models.
    """
    console = Console()

    # --- Validate Subset Size ---
    if subset_size < 2:
        logger.error("Subset size must be at least 2.")
        raise typer.Exit(code=1)

    # --- Determine Config ---
    if granularity.value not in GRANULARITY_CONFIG:
        logger.error(f"Invalid granularity specified: {granularity.value}")
        raise typer.Exit(code=1)

    config = GRANULARITY_CONFIG[granularity.value]
    table_name = config["table_name"]
    pk_cols_map = config["pk_cols"]
    id_cols_for_melt = config["id_cols_for_melt"] # Get melt IDs from config
    pk_cols_list_df = list(pk_cols_map.values()) # Get df PK col names

    logger.info(f"Selected Granularity: {granularity.value}")
    logger.info(f"Finding best subset of size: {subset_size}")
    logger.info(f"Reading from table: {table_name}")

    # --- Database Connection & Loading ---
    logger.info(f"Connecting to database: {DB_PATH}")
    if not DB_PATH.exists():
        logger.error(f"Database file not found: {DB_PATH}")
        sys.exit(1)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if cursor.fetchone() is None:
             logger.error(f"Table '{table_name}' not found in the database.")
             sys.exit(1)

        logger.info(f"Loading data from table '{table_name}'...")
        all_needed_cols = list(pk_cols_map.values()) + ['model']
        cursor.execute(f"PRAGMA table_info({table_name});")
        table_columns = [info[1] for info in cursor.fetchall()]
        score_cols_db = [col for col in table_columns if col.endswith("_score")]
        if not score_cols_db:
            logger.error(f"No columns ending with '_score' found in table '{table_name}'.")
            sys.exit(1)
        logger.info(f"Identified score columns: {', '.join(score_cols_db)}")
        all_needed_cols.extend(score_cols_db)
        select_cols_str = ", ".join([f'"{col}"' for col in all_needed_cols])

        query = f"SELECT {select_cols_str} FROM {table_name}"
        logger.debug(f"Executing query: {query}")
        df_original = pd.read_sql_query(query, conn)
        logger.success(f"Loaded {len(df_original)} rows with {len(df_original.columns)} columns from {table_name}.")
    except Exception as e:
        logger.error(f"Database error: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()

    if df_original.empty:
         logger.error(f"No data loaded from table '{table_name}'. Exiting.")
         sys.exit(1)

    # --- Initial Filtering ---
    logger.info("Applying base model exclusions...")
    if "model" not in df_original.columns:
        logger.error("Column 'model' not found in the loaded data.")
        sys.exit(1)
    initial_models_all = df_original["model"].unique()
    logger.info(f"All initial models loaded ({len(initial_models_all)}): {', '.join(initial_models_all)}")
    base_exclude_set = set(BASE_EXCLUDE_MODELS_LIST)
    df_filtered_global = df_original[~df_original["model"].isin(base_exclude_set)].copy() # Keep filtered data accessible
    candidate_models = df_filtered_global["model"].unique()

    if len(candidate_models) < subset_size:
        logger.error(f"Need at least {subset_size} models to form a subset, only {len(candidate_models)} remain after base filtering. Exiting.")
        sys.exit(1)

    logger.success(f"Base filtering complete. Candidate models ({len(candidate_models)}): {', '.join(candidate_models)}")

    # --- Iterate through Subsets ---
    best_mean_alpha = -np.inf
    best_subset = None
    best_results = {}

    logger.info(f"Testing all combinations of {subset_size} models from the {len(candidate_models)} candidates...")

    # Use the globally filtered df_filtered_global here
    for model_subset in itertools.combinations(candidate_models, subset_size):
         mean_alpha, results_detail = calculate_alpha_for_subset(
             df_filtered_global, # Pass the pre-filtered dataframe
             model_subset,
             score_cols_db,
             granularity.value,
             pk_cols_list_df,
             id_cols_for_melt
        )

         if mean_alpha is not None and not np.isnan(mean_alpha):
            logger.info(f"Mean Alpha for subset {model_subset}: {mean_alpha:.4f}") # Log mean alpha for each subset
            if mean_alpha > best_mean_alpha:
                 best_mean_alpha = mean_alpha
                 best_subset = model_subset
                 best_results = results_detail
         else:
             logger.warning(f"Could not calculate valid mean alpha for subset {model_subset}")


    # --- Display Best Result ---
    if best_subset is None:
         logger.error(f"Could not find any valid subset combination of size {subset_size} with calculable Alpha.")
         sys.exit(1)

    logger.success(f"Found best subset: {', '.join(best_subset)} with Mean Alpha: {best_mean_alpha:.4f}")

    console.print(f"\n[bold cyan]Best Krippendorff's Alpha Results (Granularity: {granularity.value}, Subset Size: {subset_size}, Models: {', '.join(best_subset)}):[/bold cyan]")
    table = Table(title=f"Inter-Model Reliability (Best Subset: {', '.join(best_subset)})")
    table.add_column("Concept", style="dim", no_wrap=True)
    table.add_column("Alpha (Ordinal)", justify="right", style="magenta")
    table.add_column("Interpretation", justify="left")
    table.add_column("Error", justify="left", style="red")

    sorted_concepts = sorted(best_results.keys(), key=lambda k: best_results[k]['alpha'] if best_results[k].get('error') is None and not np.isnan(best_results[k]['alpha']) else -1)

    for concept in sorted_concepts:
        res = best_results[concept]
        alpha = res['alpha']
        error = res['error']
        interpretation = ""
        alpha_str = ""

        if error:
            alpha_str = "Error"
        elif np.isnan(alpha):
            alpha_str = "NaN"
            error = "Could not calculate"
        else:
            alpha_str = f"{alpha:.4f}"
            # Interpretation logic (same as before)
            if alpha < 0.0: interpretation = "[red]Below Chance[/red]"
            elif alpha < 0.67: interpretation = "[yellow]Poor Agreement[/yellow]"
            elif alpha < 0.80: interpretation = "[green]Fair/Good Agreement[/green]"
            elif alpha < 0.90: interpretation = "[bold green]Strong Agreement[/bold green]"
            else: interpretation = "[bold cyan]Excellent Agreement[/bold cyan]"

        display_concept = concept.replace("_score", "").replace("_", " ").title()
        table.add_row(display_concept, alpha_str, interpretation, error or "")

    console.print(table)
    console.print(f"\n[bold]Overall Mean Alpha for best subset ({granularity.value}, size {subset_size}): {best_mean_alpha:.4f}[/bold]")


if __name__ == "__main__":
    app() 