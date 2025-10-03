# /**
# * #/@ Author: Assistant
# * #/@ Create Time: 2024-05-19 18:45:00
# * #/@ Description: Analyse and visualise concept scores from the database.
# */

# %% LOAD LIBS
librarian::shelf(data.table, ggplot2, RSQLite, scales, stringr, ggridges, gridExtra, GGally, fmsb, irr)

# %% DEFINE PATHS
db_path <- "./task2_cleanup_and_build_database/out/concept_scores.db"
output_dir <- "./task3_analyse_data/graphs_and_maps/concept_scores_analysis/"
# Ensure this list exactly matches the models to be excluded
exclude_models_list <- unique(c(
    "qwen/qwen-2.5-7b-instruct",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-32b:free",
    "qwen/qwen-2.5-7b-instruct",
    "google/gemini-2.0-flash-exp:free"
)) # Models to exclude from visualizations
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# %% LOAD DATA
print(paste("Loading data from SQLite database:", db_path))
# Check if DB exists
if (!file.exists(db_path)) {
    stop(paste("Database file not found:", db_path))
}
# Connect to the SQLite database
con <- dbConnect(RSQLite::SQLite(), dbname = db_path)
# List tables to check
print("Tables in the database:")
print(dbListTables(con))
# Load the correct table into a data.table
correct_table_name <- "scores_wide" # Use the actual table name
if (correct_table_name %in% dbListTables(con)) {
    scores_dt <- setDT(dbReadTable(con, correct_table_name))
    print(paste("Loaded", nrow(scores_dt), "rows from", correct_table_name, "table."))
    print("Columns loaded:")
    print(names(scores_dt))

    # Filter out excluded models BEFORE any processing
    if (length(exclude_models_list) > 0 && "model" %in% names(scores_dt)) {
        original_rows <- nrow(scores_dt)
        original_models <- unique(scores_dt$model)
        print(paste("Initial models loaded:", paste(original_models, collapse = ", ")))
        print(paste("Excluding models:", paste(exclude_models_list, collapse = ", ")))

        scores_dt <- scores_dt[!model %in% exclude_models_list] # Apply the filter

        remaining_models <- unique(scores_dt$model)
        print(paste("Removed", original_rows - nrow(scores_dt), "rows corresponding to excluded models."))
        print(paste("Remaining rows:", nrow(scores_dt)))
        if (nrow(scores_dt) > 0) {
            print(paste("Models remaining for analysis:", paste(remaining_models, collapse = ", ")))
            # Verify specific model is gone
            if ("qwen/qwen-2.5-7b-instruct" %in% remaining_models) {
                warning("Excluded model 'qwen/qwen-2.5-7b-instruct' is still present after filtering!")
            } else {
                print("'qwen/qwen-2.5-7b-instruct' successfully excluded.")
            }
        } else {
            print("No data remaining after filtering excluded models.")
        }
    } else if (!"model" %in% names(scores_dt)) {
        warning("Could not apply model exclusion: 'model' column not found.")
    } else {
        print("No models specified for exclusion, using all loaded models.")
        print(paste("Models available for analysis:", paste(unique(scores_dt$model), collapse = ", ")))
    }
} else {
    stop(paste("Table '", correct_table_name, "' not found in the database.", sep = ""))
}
# Disconnect from the database
dbDisconnect(con)

# %% PREPARE DATA FOR PLOTTING
print("Preparing data for plotting...")
# Identify score columns
score_cols <- names(scores_dt)[grepl("_score$", names(scores_dt))]
if (length(score_cols) == 0) {
    stop("No columns ending with '_score' found.")
}
print("Identified score columns:")
print(score_cols)

# Identify necessary ID/metadata columns to keep
id_cols <- c("row_index", "document_id", "sentence_text", "model", "timestamp")
cols_to_keep <- c(id_cols, score_cols)
# Ensure these columns exist
cols_to_keep <- intersect(cols_to_keep, names(scores_dt))

# Melt the data to long format
plot_data_long <- melt(scores_dt[, ..cols_to_keep],
    id.vars = setdiff(id_cols, "sentence_text"), # Keep sentence_text out of id.vars for now if not needed for grouping
    measure.vars = score_cols,
    variable.name = "category",
    value.name = "score",
    na.rm = TRUE # Remove rows with NA scores for plotting
)

# Clean category names
plot_data_long[, category_label := tools::toTitleCase(gsub("_", " ", gsub("_score$", "", category)))]
# Shorten model names (optional, improves plot readability)
plot_data_long[, model_short := gsub("openai/", "", model)] # Example: keep only part after 'openai/'

print(paste("Data reshaped. Have", nrow(plot_data_long), "score observations."))
print("Sample of long data:")
print(head(plot_data_long))

# %% INTER-MODEL AGREEMENT ANALYSIS
print("Starting Inter-Model Agreement Analysis...")

# Ensure we have multiple models to compare
remaining_models <- unique(scores_dt$model)
if (length(remaining_models) < 2) {
    print("Skipping agreement analysis: Fewer than 2 models remaining after initial filtering.")
    outlier_models <- character(0) # No outliers if we can't compare
    models_for_consensus <- remaining_models
} else {
    print("Reshaping data for correlation analysis...")
    # We need a unique identifier for each sentence across models. Assuming document_id + sentence_text is sufficient.
    # If sentence_text can be identical across different documents, add row_index (assuming it's original text index)
    # Let's try document_id + sentence_text first
    scores_long_for_pivot <- melt(scores_dt,
        id.vars = c("document_id", "sentence_text", "model"),
        measure.vars = score_cols,
        variable.name = "concept",
        value.name = "score",
        na.rm = FALSE
    ) # Keep NAs for now

    # Cast wider with models as columns
    # Handle potential duplicate identifiers if needed (e.g., aggregate scores first or add more id vars)
    scores_pivoted <- dcast(scores_long_for_pivot,
        document_id + sentence_text + concept ~ model,
        value.var = "score",
        fun.aggregate = mean, # Aggregate duplicate entries using mean
        na.rm = TRUE # Ensure NAs in score don't break mean if only one value exists
    )

    print(paste("Pivoted data created with", ncol(scores_pivoted) - 3, "model columns.")) # -3 for id vars

    model_colnames <- setdiff(names(scores_pivoted), c("document_id", "sentence_text", "concept"))
    concept_names <- unique(scores_pivoted$concept)
    all_cor_matrices <- list()
    valid_concepts_for_corr <- c()

    print("Calculating pairwise correlations per concept...")
    for (con_name in concept_names) {
        concept_data <- scores_pivoted[concept == con_name, ..model_colnames]
        # Remove columns (models) with zero variance or too few finite observations for correlation
        valid_cols <- sapply(concept_data, function(col) {
            finite_vals <- col[is.finite(col)]
            length(finite_vals) >= 3 && var(finite_vals, na.rm = TRUE) > 0
        })

        if (sum(valid_cols) < 2) {
            print(paste("Skipping concept:", con_name, "- Fewer than 2 models with sufficient data/variance."))
            next
        }

        concept_data_valid <- concept_data[, ..valid_cols, with = FALSE]

        # Calculate correlation matrix
        cor_matrix <- cor(concept_data_valid, use = "pairwise.complete.obs", method = "pearson") # Or spearman? Pearson is fine for scores.

        # Check if correlation matrix is valid (not all NA)
        if (!all(is.na(cor_matrix))) {
            all_cor_matrices[[con_name]] <- cor_matrix
            valid_concepts_for_corr <- c(valid_concepts_for_corr, con_name)
            # print(paste("Correlation calculated for concept:", con_name)) # Optional verbose output
        } else {
            print(paste("Skipping concept:", con_name, "- Correlation matrix calculation resulted in all NAs."))
        }
    }

    if (length(all_cor_matrices) == 0) {
        print("Could not calculate valid correlations for any concept. Skipping outlier detection.")
        outlier_models <- character(0)
        models_for_consensus <- remaining_models
    } else {
        print(paste("Correlation matrices calculated for", length(valid_concepts_for_corr), "concepts."))
        # Calculate Average Agreement Score per Model
        avg_agreement <- list()
        all_models_in_corr <- unique(unlist(lapply(all_cor_matrices, colnames)))

        for (mod in all_models_in_corr) {
            model_corrs <- c()
            for (con_name in names(all_cor_matrices)) {
                cor_mat <- all_cor_matrices[[con_name]]
                if (mod %in% colnames(cor_mat)) {
                    # Get correlations of this model with others (excluding self-correlation)
                    other_mods <- setdiff(colnames(cor_mat), mod)
                    if (length(other_mods) > 0) {
                        # Handle cases where a model might only correlate with itself if only 2 models remain for a concept
                        mod_row_corrs <- cor_mat[mod, other_mods, drop = FALSE] # Ensure it stays a matrix/vector
                        model_corrs <- c(model_corrs, mod_row_corrs)
                    }
                }
            }
            # Calculate mean correlation, ignoring NAs (which occur if pairs were insufficient)
            if (length(model_corrs) > 0) {
                avg_agreement[[mod]] <- mean(model_corrs, na.rm = TRUE)
            } else {
                avg_agreement[[mod]] <- NA # Cannot calculate if no correlations found
            }
        }

        agreement_scores_dt <- data.table(model = names(avg_agreement), avg_corr = unlist(avg_agreement))
        agreement_scores_dt <- agreement_scores_dt[!is.na(avg_corr)] # Remove models with no valid correlations
        setorder(agreement_scores_dt, avg_corr)

        print("Average Inter-Model Correlation (Agreement Score):")
        print(agreement_scores_dt)

        # Identify Outliers (Simple Example: models below mean - 1.5 * SD)
        mean_corr <- mean(agreement_scores_dt$avg_corr, na.rm = TRUE)
        sd_corr <- sd(agreement_scores_dt$avg_corr, na.rm = TRUE)
        # Ensure sd_corr is not NA or zero before calculating threshold
        corr_threshold <- if (!is.na(sd_corr) && sd_corr > 0) mean_corr - 1.5 * sd_corr else -Inf

        # Define a minimum sensible threshold as well, e.g., 0.1 or 0? Let's use the SD threshold for now.
        print(paste("Agreement threshold (mean - 1.5*SD):", round(corr_threshold, 3)))

        outlier_models <- agreement_scores_dt[avg_corr < corr_threshold, model]

        if (length(outlier_models) > 0) {
            print(paste("Identified outlier models based on low agreement:", paste(outlier_models, collapse = ", ")))
        } else {
            print("No outlier models identified based on agreement score.")
        }
        models_for_consensus <- setdiff(remaining_models, outlier_models)
    }
} # End of check for < 2 models

# %% CALCULATE INTER-RATER RELIABILITY (Krippendorff's Alpha)
print("Calculating Inter-Rater Reliability (Krippendorff's Alpha)...")

# Check if we have the necessary data and library
if (exists("scores_pivoted") && length(remaining_models) >= 2 && requireNamespace("irr", quietly = TRUE)) {
    alpha_results <- list()
    model_colnames_for_irr <- remaining_models # Use models remaining after initial exclusion

    print(paste("Calculating Alpha using models:", paste(model_colnames_for_irr, collapse = ", ")))

    # Iterate through each concept
    for (con_name in concept_names) {
        print(paste("Processing concept:", con_name))

        # Extract data for the current concept: rows=sentences, cols=models
        concept_data_irr <- scores_pivoted[concept == con_name, ..model_colnames_for_irr]

        # Krippendorff's alpha requires a matrix where rows are subjects (sentences) and columns are raters (models)
        # It also needs ratings to be numeric. Scores should already be numeric.
        # The function handles missing data (NA) internally.

        # Check if we have at least 2 models with some data for this concept
        non_na_counts_per_model <- colSums(!is.na(concept_data_irr))
        models_with_data <- sum(non_na_counts_per_model > 0)

        if (nrow(concept_data_irr) < 2 || models_with_data < 2) {
            print(paste("Skipping Alpha for concept:", con_name, "- Insufficient data (rows < 2 or models < 2)."))
            alpha_results[[con_name]] <- list(method = "Krippendorff's alpha", subjects = nrow(concept_data_irr), raters = models_with_data, value = NA, error = "Insufficient data")
            next
        }

        # Convert to matrix for the irr function
        ratings_matrix <- as.matrix(concept_data_irr)

        # Calculate Krippendorff's Alpha
        alpha_calc <- tryCatch(
            {
                # Specify 'ordinal' data type as scores are ordered categories (1-10)
                # Use default bootstrap confidence intervals (nboot=1000) for robustness, can be slow
                irr::kripp.alpha(ratings_matrix, method = "ordinal", confint = FALSE)
                # Alternative: use "interval" if treating scores as continuous, but ordinal is likely better fit
                # irr::kripp.alpha(ratings_matrix, method = "interval")
            },
            error = function(e) {
                print(paste("Error calculating Alpha for concept", con_name, ":", e$message))
                list(method = "Krippendorff's alpha", subjects = nrow(ratings_matrix), raters = ncol(ratings_matrix), value = NA, error = e$message) # Return NA on error
            }
        )

        alpha_results[[con_name]] <- alpha_calc
        print(paste("Alpha for", con_name, ":", round(alpha_calc$value, 3)))
    }

    # Summarize results
    alpha_summary_dt <- data.table(
        concept = names(alpha_results),
        alpha_value = sapply(alpha_results, function(x) ifelse(is.null(x$value), NA, x$value)),
        subjects = sapply(alpha_results, function(x) ifelse(is.null(x$subjects), NA, x$subjects)),
        raters = sapply(alpha_results, function(x) ifelse(is.null(x$raters), NA, x$raters)),
        error = sapply(alpha_results, function(x) ifelse(is.null(x$error), "", x$error))
    )
    setorder(alpha_summary_dt, alpha_value)

    print("Krippendorff's Alpha Results per Concept:")
    print(alpha_summary_dt)

    # Calculate overall average alpha (weighted by number of subjects?) - simple mean for now
    overall_mean_alpha <- mean(alpha_summary_dt$alpha_value, na.rm = TRUE)
    print(paste("Overall Mean Alpha across concepts:", round(overall_mean_alpha, 3)))
} else {
    if (!exists("scores_pivoted")) print("Skipping Krippendorff's Alpha: Pivoted data not available.")
    if (length(remaining_models) < 2) print("Skipping Krippendorff's Alpha: Fewer than 2 models available.")
    if (!requireNamespace("irr", quietly = TRUE)) print("Skipping Krippendorff's Alpha: 'irr' package not found. Please install it.")
}

# %% FILTER DATA FOR CONSENSUS
print("Filtering data for consensus models...")
print(paste("Models included in consensus analysis:", paste(models_for_consensus, collapse = ", ")))

# Create datasets using only the consensus models
consensus_scores_dt <- scores_dt[model %in% models_for_consensus]
consensus_plot_data_long <- plot_data_long[model %in% models_for_consensus] # Filter the long data too

if (nrow(consensus_scores_dt) == 0) {
    warning("No data remaining after filtering for consensus models. Subsequent plots might fail.")
} else {
    print(paste("Consensus data contains", nrow(consensus_scores_dt), "rows."))
    print(paste("Consensus long data contains", nrow(consensus_plot_data_long), "rows."))
}


# %% VISUALISE DISTRIBUTIONS (NOW USING CONSENSUS DATA)
print("Generating visualizations for CONSENSUS models...")
# IMPORTANT: Update subsequent plotting code to use 'consensus_plot_data_long'
#            and 'consensus_scores_dt' instead of the original ones.
#            Also update titles and filenames to reflect 'Consensus'.

# Example update for density plot (apply similar changes to boxplot, ridgeline, ggpairs):
print("Generating faceted density plot (Consensus)...")
# Check if data exists
if (nrow(consensus_plot_data_long) > 0) {
    plot_density_consensus <- ggplot(consensus_plot_data_long, aes(x = score, fill = model_short)) + # Use consensus data
        geom_density(alpha = 0.6) +
        facet_wrap(~category_label, scales = "free_y", ncol = 3) +
        theme_minimal(base_size = 11) +
        labs(
            title = "Distribution of Concept Scores by Category (Consensus Models)", # Updated title
            subtitle = "Density plot of scores assigned per sentence",
            x = "Score",
            y = "Density",
            fill = "Model"
        ) +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 5)) +
        theme(
            strip.text = element_text(face = "bold", size = 9),
            axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position = "bottom",
            plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(size = 9)
        )

    # Save the density plot (updated filename)
    ggsave(
        filename = file.path(output_dir, "consensus_concept_scores_density_by_category_model.png"), # Updated filename
        plot = plot_density_consensus,
        width = 12, height = max(8, 2.5 * ceiling(length(unique(consensus_plot_data_long$category_label)) / 3)),
        dpi = 300, bg = "white"
    )
} else {
    print("Skipping density plot: No consensus data available.")
}


# ... (YOU WILL NEED TO MANUALLY UPDATE the boxplot, ridgeline, and ggpairs sections
#      similarly to use 'consensus_plot_data_long' or 'consensus_scores_dt'
#      and update titles/filenames) ...


# %% CONSENSUS RADAR PLOT
print("Generating Consensus Radar Plot...")

# Ensure fmsb is loaded
librarian::shelf(fmsb)

# Check if data exists
if (nrow(consensus_plot_data_long) > 0) {
    # Calculate mean score per category for the consensus models
    radar_data_agg <- consensus_plot_data_long[, .(mean_score = mean(score, na.rm = TRUE)), by = category_label]
    setorder(radar_data_agg, category_label) # Ensure consistent order

    # Prepare data for radarchart function
    radar_df_prep <- dcast(radar_data_agg, . ~ category_label, value.var = "mean_score")
    radar_df_prep[, . := NULL] # Remove the dummy '.' column

    # Define max and min for score range
    score_range <- range(consensus_plot_data_long$score, na.rm = TRUE)
    max_score <- ceiling(score_range[2])
    min_score <- floor(score_range[1])
    if (min_score < 0) min_score <- 0

    # --- START: Modified rbind preparation ---
    # Convert radar_df_prep to a standard data frame first
    radar_df_prep_df <- as.data.frame(radar_df_prep)

    # Create max and min rows as data frames with matching column names
    max_row <- data.frame(t(rep(max_score, ncol(radar_df_prep_df))))
    colnames(max_row) <- colnames(radar_df_prep_df)

    min_row <- data.frame(t(rep(min_score, ncol(radar_df_prep_df))))
    colnames(min_row) <- colnames(radar_df_prep_df)

    # Combine using rbind for data frames
    radar_df <- rbind(max_row, min_row, radar_df_prep_df)
    rownames(radar_df) <- c("Max", "Min", "Consensus Mean")
    # --- END: Modified rbind preparation ---

    print("Radar chart data prepared:")
    print(radar_df)

    # Create the radar chart
    radar_filename <- file.path(output_dir, "consensus_concept_scores_radar.png")
    png(radar_filename, width = 800, height = 800, res = 100) # Save as PNG

    # Customize plot appearance (optional)
    radarchart(radar_df,
        axistype = 1, # Labels on axes
        # Polygons
        pcol = rgb(0.2, 0.5, 0.5, 0.9), # Polygon border color
        pfcol = rgb(0.2, 0.5, 0.5, 0.4), # Polygon fill color
        plwd = 2, # Polygon line width
        plty = 1, # Polygon line type
        # Grid
        cglcol = "grey", # Grid line color
        cglty = 1, # Grid line type
        axislabcol = "grey", # Axis label color
        cglwd = 0.8, # Grid line width
        # Labels
        vlcex = 0.8, # Vertex label size
        title = "Mean Concept Scores (Consensus Models)"
    )
    dev.off() # Close the PNG device
    print(paste("Radar chart saved to:", radar_filename))
} else {
    print("Skipping radar plot: No consensus data available.")
}

# %% SCATTER PLOT MATRIX (USING CONSENSUS DATA)
# IMPORTANT: Update this section to use 'consensus_scores_dt'
#            and adjust title/filename

print("Generating scatter plot matrix (Consensus)...")
# ... (Update the ggpairs code block here) ...


print("Script finished.")
# %%