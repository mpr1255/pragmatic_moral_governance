# /**
#* #/@ Author: Matthew Robertson
#* #/@ Create Time: 2024-05-15 17:29:40
#* #/@ Modified by: Assistant
#* #/@ Modified time: 2024-05-19 18:30:00
#* #/@ Description: Updated to analyse CONCEPT PARAGRAPH similarities.
#

# This script does all the visualisations of the data analysis and exports a bunch of files for qualitative annotation, analysis etc.
# This version uses the similarity scores calculated against CONCEPT PARAGRAPHS.

# The order of business is roughly:

# 1. Load in libs etc.;
# 2. Load CONCEPT PARAGRAPH similarity data and merge with metadata;
# 3. Basic cleaning and city/province mapping;
# 4. Exports and copy the doc files to the new dir structure;
# 5. Lots of visualisations based on CONCEPT PARAGRAPH similarities.

# Note that I moved out a huge number of visualisations to the 'scratch' file in case they are needed later.

# /**

# %% LOAD LIBS
librarian::shelf(data.table, tidyverse, jsonlite, ggplot2, sf, dplyr, purrr, ggpattern, scales, fs, kableExtra, knitr, readr, writexl) # Added writexl

# Define output directory for CONCEPT PARAGRAPH analysis graphs
output_dir <- "../graphs_and_maps/concept_paragraphs/"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}
# Define output directory for exported data files from CONCEPT PARAGRAPH analysis
data_output_dir <- "../out/concept_paragraphs/"
if (!dir.exists(data_output_dir)) {
    dir.create(data_output_dir, recursive = TRUE)
}

# %% LOAD CONCEPT PARAGRAPH DATA & METADATA
print("Loading concept paragraph similarities...")
concept_sim_dt <- fread("../../task2_cleanup_and_build_database/out/sentence_concept_similarities.csv")
print(paste("Loaded", nrow(concept_sim_dt), "rows from concept similarities."))
print("Columns in concept_sim_dt:")
print(names(concept_sim_dt))


print("Loading sentence metadata...")
# Assuming section2.csv contains the necessary metadata linked by 'id'
sentence_metadata_path <- "../../task2_cleanup_and_build_database/out/section2.csv"
if (!file.exists(sentence_metadata_path)) {
    stop(paste("Metadata file not found:", sentence_metadata_path))
}
sentence_metadata_dt <- fread(sentence_metadata_path)
print(paste("Loaded", nrow(sentence_metadata_dt), "rows from metadata."))
# Keep only necessary metadata columns to avoid duplication after merge and ensure 'id' is present
metadata_cols_to_keep <- c("id", "title", "office", "publish", "expiry", "type", "status", "url")
# Check which metadata columns actually exist in the loaded data
metadata_cols_to_keep <- intersect(metadata_cols_to_keep, names(sentence_metadata_dt))
if (!"id" %in% names(sentence_metadata_dt)) {
    stop("Metadata file section2.csv must contain an 'id' column for merging.")
}
sentence_metadata_dt <- unique(sentence_metadata_dt[, ..metadata_cols_to_keep], by = "id") # Ensure unique IDs
print("Columns kept from metadata:")
print(metadata_cols_to_keep)

print("Merging similarities with metadata...")
# Merge concept similarities with metadata based on 'id'
concept_dt <- merge(concept_sim_dt, sentence_metadata_dt, by = "id", all.x = TRUE)
print(paste("Rows after merging with metadata:", nrow(concept_dt)))

# Check for NAs after merge, which might indicate missing metadata
missing_metadata_count <- sum(is.na(concept_dt$title))
if (missing_metadata_count > 0) {
    warning(paste(missing_metadata_count, "IDs in similarity data did not have matching metadata."))
    # Optionally filter out rows with missing metadata or handle them otherwise
    # concept_dt <- concept_dt[!is.na(title)]
}

# %% CITY/PROVINCE MAPPING
print("Loading and processing city/province mapping...")
# Load the JSONL files for city/province mapping
unique_cities_original <- lapply(readLines("../../task2_cleanup_and_build_database/hand/unique_cities_original.json"), fromJSON)
unique_cities_fixed <- lapply(readLines("../../task2_cleanup_and_build_database/hand/unique_cities_fixed.json"), fromJSON)

# Convert lists to data.table
original_dt <- data.table(id_map = seq_along(unique_cities_original), title_map = unlist(unique_cities_original))
fixed_dt <- data.table(id_map = seq_along(unique_cities_fixed), city_province = unlist(unique_cities_fixed))

# Join the original with fixed based on id
mapping_dt <- merge(original_dt, fixed_dt, by = "id_map")

# Separate city and province
mapping_dt[, c("city", "province") := tstrsplit(city_province, ", ")]

# Select relevant columns for mapping
mapping_dt <- mapping_dt[, .(title_map, city, province)]
setnames(mapping_dt, "title_map", "title") # Rename to match column in concept_dt
mapping_dt <- unique(mapping_dt, by = "title") # Ensure unique titles in mapping

print("Merging concept data with city/province mapping...")
# Ensure 'title' column exists and is correct type in both data.tables
if (!"title" %in% names(concept_dt)) {
    stop("Merged data ('concept_dt') is missing the 'title' column required for city/province mapping.")
}
final_dt <- merge(concept_dt, mapping_dt, by = "title", all.x = TRUE)
print(paste("Rows after merging with city/province mapping:", nrow(final_dt)))


# Check for NAs after merge
missing_mapping_count <- sum(is.na(final_dt$province))
if (missing_mapping_count > 0) {
    warning(paste(missing_mapping_count, "titles did not have matching city/province information."))
    # Assign default values if needed
    final_dt[is.na(province), province := "Unknown"]
    final_dt[is.na(city), city := "Unknown"]
}

# %% EXPORTS
final_dt[, year := year(as.Date(publish))]
print("Added 'year' column derived from 'publish'.")

# Check distinct documents per year/province
print("Distinct documents per year/province (sample):")
print(head(final_dt %>%
    distinct(id, .keep_all = TRUE) %>% # Use 'id' for distinct documents
    .[, .N, by = c("year", "province")] %>%
    .[order(-N)]))

# Example export (Shenzhen) - update path
print("Exporting Shenzhen example...")
final_dt %>%
    distinct(id, .keep_all = TRUE) %>%
        filter(str_detect(city, "深圳")) %>% # Use str_detect for partial match
        writexl::write_xlsx(path = paste0(data_output_dir, "concept_paragraphs_shenzhen_docs_metadata.xlsx")) # Changed name to reflect content

# Export all sentences with concept paragraph similarities - update path
print("Exporting all sentences with concept paragraph similarities...")
final_dt %>% writexl::write_xlsx(path = paste0(data_output_dir, "concept_paragraphs_final_all_sentences.xlsx"))

# %% prep for analysis (aggregation)
print("Preparing data for aggregation...")
final_dt_for_export <- copy(final_dt)

# Remove sentence column before aggregation if desired - KEEPING sentence for now
# final_dt_for_export[, sentence := NULL]

# Identify concept similarity columns
sim_cols_concepts <- grep("^sim_cosine_", names(final_dt_for_export), value = TRUE)
print("Similarity columns identified for aggregation:")
print(sim_cols_concepts)

if (length(sim_cols_concepts) == 0) {
    stop("No 'sim_cosine_' columns found for aggregation.")
}

# Define grouping columns
grouping_cols <- c("id", "title", "office", "publish", "expiry", "type", "status", "url", "city", "province", "year")
# Ensure grouping columns exist in the data
grouping_cols <- intersect(grouping_cols, names(final_dt_for_export))
print("Grouping columns for aggregation:")
print(grouping_cols)

print("Aggregating similarity scores by document...")
result <- final_dt_for_export[,
    lapply(.SD, mean, na.rm = TRUE), # Apply mean to all sim_cosine columns
    by = grouping_cols,
    .SDcols = sim_cols_concepts
]
# Rename columns to avg_sim_cosine_*
avg_col_names <- paste0("avg_", sim_cols_concepts)
setnames(result, sim_cols_concepts, avg_col_names)
print(paste("Aggregation complete. Result has", nrow(result), "rows (unique documents)."))

# Export aggregated results - update path
print("Exporting aggregated document scores...")
result %>% writexl::write_xlsx(path = paste0(data_output_dir, "concept_paragraphs_final_docs_with_averaged_sim_cosine_scores.xlsx"))


# %% COPY DOCUMENTS (This function definition remains the same, but call it with the new 'result' data)
copy_documents <- function(data, source_dir, target_dir) {
    # Ensure the target directory exists
    dir_create(target_dir)

    # Dynamically get column names for pwalk
    required_cols <- c("id", "title", "province", "year") # Minimum required
    present_cols <- names(data)
    cols_for_pwalk <- intersect(required_cols, present_cols)

    # Check if all required columns are present
    if (!all(required_cols %in% present_cols)) {
         missing_cols <- setdiff(required_cols, present_cols)
         warning(paste("Missing required columns (", paste(missing_cols, collapse=", "), ") in data for copy_documents. Skipping copy."))
         return() # Exit function if columns missing
    }

    # Use only necessary columns for pwalk to avoid errors if some are missing
    data_subset <- data[, ..required_cols] # Use only required cols

    print(paste("Attempting to copy documents for", nrow(data_subset), "entries..."))
    copied_count <- 0
    skipped_count <- 0
    error_count <- 0
    not_found_count <- 0

    pwalk(data_subset, function(id, title, province, year, ...) { # Use ... to ignore extra columns if present
        # Clean the title to make it file-system safe
        if (is.na(title) || is.na(province) || is.na(year)) {
            # skipped_count <<- skipped_count + 1 # Use <<- to modify global counter
            return() # Skip if essential info missing
        }
        clean_title <- gsub("[[:punct:][:space:]]+", "_", title)
        clean_title <- substr(clean_title, 1, 100) # Limit filename length

        # Construct source and target file paths
        doc_path <- file.path(source_dir, paste0(id, ".doc"))
        docx_path <- file.path(source_dir, paste0(id, ".docx"))
        target_subdir <- file.path(target_dir, province)
# Attempt to create subdir, handle errors gracefully
tryCatch(
    {
        if (!dir.exists(target_subdir)) dir_create(target_subdir)
    },
    error = function(e) {
        print(paste("Warning: Could not create directory", target_subdir, "-", e$message))
        # Decide whether to skip or proceed based on error
        return() # Skip this document if directory creation failed
    }
)

        target_path_doc <- file.path(target_subdir, paste0(year, "_", clean_title, ".doc"))
        target_path_docx <- file.path(target_subdir, paste0(year, "_", clean_title, ".docx"))

        # Copy the appropriate file if it doesn't already exist
        source_file_path <- NULL
        target_file_path <- NULL
        if (file_exists(doc_path)) {
            source_file_path <- doc_path
            target_file_path <- target_path_doc
        } else if (file_exists(docx_path)) {
            source_file_path <- docx_path
            target_file_path <- target_path_docx
        }

        if (!is.null(source_file_path) && !file_exists(target_file_path)) {
            tryCatch({
                file_copy(source_file_path, target_file_path, overwrite = FALSE) # Ensure overwrite is false
                copied_count <<- copied_count + 1
                # print(paste("Copied", basename(source_file_path), "to", target_file_path)) # Reduce verbosity
            }, error = function(e) {
                print(paste("Error copying", basename(source_file_path), "to", target_file_path, ":", e$message))
                error_count <<- error_count + 1
            })
        } else if (!is.null(target_file_path) && file_exists(target_file_path)) {
             skipped_count <<- skipped_count + 1
             # print(paste("Target file already exists:", target_file_path)) # Reduce verbosity
        } else {
             # print(paste("Source document not found for ID:", id)) # Reduce verbosity
             not_found_count <<- not_found_count + 1
        }
    })
     print(paste("Document copy summary: Copied", copied_count, "Skipped", skipped_count, "Not Found", not_found_count, "Errors", error_count))
}

# Example usage: Update target directory
source_dir <- "/volumes/8tb/data/wenming/docs"
target_copy_dir <- "./task3_analyse_data/docs_concept_paragraphs" # New target directory for concept analysis
print("Starting document copy process...")
copy_documents(result, source_dir, target_copy_dir) # Use the 'result' DT aggregated from concept similarities
print("Document copy process finished.")


# %% MAKE ALL THE KEY VISUALISATIONS using CONCEPT PARAGRAPH data
print("Starting visualization generation...")

# Reshape the data for plotting
plot_data <- melt(final_dt,
    measure.vars = patterns("^sim_cosine_"), # Use pattern matching
    variable.name = "category", value.name = "similarity",
    na.rm = TRUE # Remove rows where similarity is NA for plotting
)
# Clean category names (remove prefix with number)
plot_data[, category := gsub("sim_cosine_\\d+_", "", category)]
print(paste("Reshaped data for plotting. Have", nrow(plot_data), "rows."))

# Set a professional color palette (use more distinct colors if needed)
unique_categories_plot <- unique(plot_data$category)
color_palette <- scales::hue_pal()(length(unique_categories_plot)) # Dynamic palette
names(color_palette) <- unique_categories_plot

# Function to make pretty labels
pretty_category_names <- function(category) {
    tools::toTitleCase(gsub("_", " ", category))
}


# Create the first plot (Similarity Distribution)
print("Generating similarity distribution plot...")
plot1 <- ggplot(plot_data, aes(x = reorder(pretty_category_names(category), similarity, median), y = similarity, color = category)) + # Order by median
    geom_jitter(width = 0.2, alpha = 0.3, size = 0.3) + # Smaller points, less alpha
    geom_boxplot(alpha = 0.5, outlier.shape = NA, width = 0.5) + # Add boxplots, adjust width
    theme_minimal(base_size = 11) +
    labs(x = "Concept Category", y = "Cosine Similarity", title = "Cosine Similarities: Sentences vs Concept Paragraphs", subtitle = "Distribution of similarity scores for each sentence") +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9)
    ) +
    scale_color_manual(values = color_palette) +
    coord_flip() # Flip coordinates for better readability of labels

# Save the first plot - update path
ggsave(filename = paste0(output_dir, "concept_paragraphs_similarity_by_category.png"), plot = plot1, width = 8, height = 10, dpi = 300, bg = "white")

# Create the second plot (Similarity by Year)
print("Generating similarity by year plot...")
plot_data_year <- plot_data[!is.na(year)] # Filter NA years for this plot
plot_data_year[, year_factor := as.factor(year)]

plot2 <- ggplot(plot_data_year, aes(x = year_factor, y = similarity)) +
    geom_jitter(aes(color = category), width = 0.2, alpha = 0.2, size = 0.3) + # Color by category
    geom_boxplot(aes(group = year_factor), alpha = 0.5, outlier.shape = NA) + # Boxplot per year overall
    theme_minimal(base_size = 11) +
    labs(x = "Year", y = "Cosine Similarity", title = "Sentence vs Concept Paragraph Similarity by Year", subtitle = "Overall distribution per year, points colored by category") +
    scale_color_manual(values = color_palette, labels = pretty_category_names) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9),
        legend.title = element_blank()
    ) + # Adjust legend position and remove title
    guides(color = guide_legend(override.aes = list(alpha = 1, size = 3))) # Make legend points visible

# Save the second plot - update path
ggsave(filename = paste0(output_dir, "concept_paragraphs_similarity_by_year.png"), plot = plot2, width = 12, height = 7, dpi = 300, bg = "white")

# %% ANALYSIS BASED ON MAX SIMILARITY CATEGORY
print("Analyzing dominant categories...")

# Columns containing concept similarity scores
sim_cols <- grep("^sim_cosine_", names(final_dt), value = TRUE)

# Find the highest similarity category for each sentence
# Ensure there are similarity columns to work with
if (length(sim_cols) > 0) {
    # Create a temporary DT with only ID and sim_cols to avoid issues with NAs in other columns
    temp_sim_dt <- final_dt[, c("id", sim_cols), with = FALSE]
    # Handle potential NAs within similarity scores before max.col
    temp_sim_dt[, (sim_cols) := lapply(.SD, function(x) ifelse(is.na(x), -Inf, x)), .SDcols = sim_cols]
    # Use max.col to find the index of the max value in each row
    max_sim_indices <- max.col(temp_sim_dt[, ..sim_cols], ties.method = "first")
    # Assign the column name corresponding to the max index back to final_dt
    final_dt[, max_sim_col := sim_cols[max_sim_indices]]
    print("Determined dominant category for each sentence.")
} else {
    warning("No 'sim_cosine_' columns found to determine max similarity.")
    final_dt[, max_sim_col := NA_character_] # Assign NA if no sim columns
}

# Count the occurrences of each category being the max
category_counts <- final_dt[!is.na(max_sim_col) & !is.na(year) & !is.na(province), .N, by = .(max_sim_col, year, province)]
print(paste("Counted dominant categories. Result has", nrow(category_counts), "rows."))

# Prepare data for line graph and yearly aggregations (needs province_en later)
category_long <- melt(dcast(category_counts, year + province ~ max_sim_col, value.var = "N", fill = 0),
    id.vars = c("year", "province"), variable.name = "category", value.name = "count"
)
category_long[, category := gsub("sim_cosine_\\d+_", "", category)] # Clean category name, handle numbered format

# Province name mapping (ensure this is still correct and handles 'Unknown')
province_mapping <- list(
    "黑龙江省" = "Heilongjiang", "福建省" = "Fujian", "河南省" = "Henan", "广东省" = "Guangdong",
    "山东省" = "Shandong", "宁夏回族自治区" = "Ningxia", "山西省" = "Shanxi", "浙江省" = "Zhejiang",
    "云南省" = "Yunnan", "四川省" = "Sichuan", "江西省" = "Jiangxi", "安徽省" = "Anhui", # Corrected Sichuan
    "新疆维吾尔自治区" = "Xinjiang", "贵州省" = "Guizhou", "内蒙古自治区" = "Inner Mongolia",
    "北京市" = "Beijing", "湖北省" = "Hubei", "江苏省" = "Jiangsu", "吉林省" = "Jilin",
    "河北省" = "Hebei", "辽宁省" = "Liaoning", "天津市" = "Tianjin", "湖南省" = "Hunan",
    "西藏自治区" = "Tibet", "广西壮族自治区" = "Guangxi", "陕西省" = "Shaanxi",
    "甘肃省" = "Gansu", "海南省" = "Hainan", "青海省" = "Qinghai", "重庆市" = "Chongqing",
    "上海市" = "Shanghai", "Unknown" = "Unknown" # Handle unknown explicitly
)

# Apply province name mapping safely
map_province <- function(p) {
    mapped_name <- province_mapping[[p]]
    return(ifelse(is.null(mapped_name) | is.na(mapped_name), as.character(p), mapped_name)) # Return original string if not found or NA
}
category_long[, province_en := sapply(province, map_province)]


# Aggregate data by year and category (using category_long which now has province_en)
category_yearly_counts <- category_long[, .(count = sum(count, na.rm = TRUE)), by = .(year, category)]
# Ensure year is numeric and valid
category_yearly_counts <- category_yearly_counts[!is.na(year) & !is.na(count)]
category_yearly_counts[, year := as.numeric(as.character(year))]
category_yearly_counts <- category_yearly_counts[!is.na(year)] # Remove rows where year conversion failed
print(paste("Aggregated dominant category counts by year. Have", nrow(category_yearly_counts), "rows."))


# Dynamic color palette based on actual categories present
unique_categories_agg <- unique(category_yearly_counts$category)
color_palette_dynamic <- scales::hue_pal()(length(unique_categories_agg))
names(color_palette_dynamic) <- unique_categories_agg


# Line Graph: Counts of dominant category per year
print("Generating dominant category line plot...")
line_plot <- ggplot(category_yearly_counts, aes(x = year, y = count, color = category, group = category)) +
    geom_line(size = 1) +
    geom_point(size = 1.5) + # Add points
    theme_minimal(base_size = 11) +
    scale_color_manual(values = color_palette_dynamic, labels = pretty_category_names) +
    labs(
        title = "Dominant Concept Paragraph Category Counts by Year",
        subtitle = "Count of sentences where each category had the highest similarity score",
        x = "Year", y = "Count of Sentences", color = "Dominant Category"
    ) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9),
        legend.title = element_blank()
    ) +
        scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) # Adjust breaks if needed

    # Save line plot - update path
    ggsave(paste0(output_dir, "concept_paragraphs_category_counts_by_year_line.png"), plot = line_plot, width = 12, height = 7, dpi = 300, bg = "white")


# Stacked Bar Graph
print("Generating dominant category stacked bar plot...")
# Use category_long directly for stacked bar, filter out zero counts
category_long_plot <- category_long[!is.na(year) & count > 0]
category_long_plot[, year_factor := as.factor(year)]

bar_plot <- ggplot(category_long_plot, aes(x = year_factor, y = count, fill = category)) +
    geom_bar(stat = "identity", position = "stack") +
    theme_minimal(base_size = 11) +
    scale_fill_manual(values = color_palette_dynamic, labels = pretty_category_names) +
    labs(
        title = "Dominant Concept Paragraph Category Counts by Year",
        subtitle = "Count of sentences where each category had the highest similarity score",
        x = "Year", y = "Count of Sentences", fill = "Dominant Category"
    ) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9),
        legend.title = element_blank()
    )

    # Save stacked bar plot - update path
    ggsave(paste0(output_dir, "concept_paragraphs_category_counts_by_year_stacked_bar.png"), plot = bar_plot, width = 12, height = 7, dpi = 300, bg = "white")


# Aggregate counts directly by province_en and category for the heatmap
# We need total counts per province/category across all years
category_province_counts <- category_long[, .(count = sum(count, na.rm = TRUE)), by = .(province_en, category)]

# Now create the heatmap data using the aggregated counts
category_heatmap_long <- category_province_counts[province_en != "Unknown" & count > 0] # Filter unknowns and zero counts

# Heatmap: Counts by Province and Category
print("Generating dominant category heatmap...")
heatmap_plot <- ggplot(category_heatmap_long, aes(x = pretty_category_names(category), y = province_en, fill = count)) + # Use category_heatmap_long
    geom_tile(color = "white", size=0.2) + # Add white lines between tiles
    scale_fill_gradient(low = "lightblue", high = "darkblue", na.value = "grey90", name = "Sentence Count", labels = scales::label_number(scale = 1e-3, suffix = "k")) +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          axis.text.y = element_text(size=8),
          plot.title = element_text(face = "bold"),
          plot.subtitle = element_text(size = 9),
          axis.title = element_blank()) +
    labs(
        title = "Dominant Concept Paragraph Category Counts by Province",
        subtitle = "Total count of sentences (all years) where each category had the highest similarity score"
    )

# Save heatmap - update path
ggsave(paste0(output_dir, "concept_paragraphs_category_counts_by_province_heatmap.png"), plot = heatmap_plot, width = 10, height = 10, dpi = 300, bg = "white")


# Stacked Area Chart
print("Generating dominant category area plot...")
area_plot <- ggplot(category_yearly_counts, aes(x = year, y = count, fill = category, group = category)) +
    geom_area(position = "stack", alpha = 0.8) +
    theme_minimal(base_size = 11) +
    scale_fill_manual(values = color_palette_dynamic, labels = pretty_category_names) +
    labs(
        title = "Dominant Concept Paragraph Category Counts by Year",
        subtitle = "Count of sentences where each category had the highest similarity score",
        x = "Year", y = "Count of Sentences", fill = "Dominant Category"
    ) +
    scale_y_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) + # Format y-axis
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.text = element_text(face = "bold"),
        legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9),
        legend.title = element_blank()
    )

# Save area plot - update path
ggsave(paste0(output_dir, "concept_paragraphs_category_counts_by_year_area.png"), plot = area_plot, width = 12, height = 7, dpi = 300, bg = "white")


# Bar plot: Total counts per category
print("Generating total dominant category bar plot...")
category_total_counts <- category_long[, .(count = sum(count, na.rm = TRUE)), by = category]

bar_plot_total <- ggplot(category_total_counts, aes(x = reorder(pretty_category_names(category), -count), y = count, fill = category)) +
    geom_bar(stat = "identity") +
    theme_minimal(base_size = 11) +
    scale_fill_manual(values = color_palette_dynamic, guide = "none") + # No legend needed here
    labs(
        title = "Total Dominant Concept Paragraph Category Counts",
        subtitle = "Total count of sentences where each category had the highest similarity score (all years)",
        x = "Dominant Category", y = "Total Count of Sentences"
    ) +
    scale_y_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +
        theme(
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(face = "bold"),
            plot.subtitle = element_text(size = 9)
        )

    # Save total category counts bar plot - update path
    ggsave(paste0(output_dir, "concept_paragraphs_category_counts_total_bar.png"), plot = bar_plot_total, width = 10, height = 6, dpi = 300, bg = "white")


# Count the number of unique documents by year (using the new final_dt)
print("Generating document count by year plot...")
doc_counts <- final_dt %>%
    filter(!is.na(year)) %>%
    distinct(id, .keep_all = TRUE) %>% # Use 'id' as the unique document identifier
    .[, .N, by = year]

# Bar Graph: Document Counts per Year
doc_bar_plot <- ggplot(doc_counts, aes(x = as.factor(year), y = N)) +
    geom_bar(stat = "identity", fill = "grey50") +
    theme_minimal(base_size = 11) +
    labs(
        title = "Number of Unique Documents by Year (Concept Paragraph Analysis)",
        x = "Year", y = "Number of Documents"
    ) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold")
    )

# Save document counts plot - update path
ggsave(paste0(output_dir, "concept_paragraphs_documents_by_year_bar.png"), plot = doc_bar_plot, width = 10, height = 6, dpi = 300, bg = "white")

# %% Percentage Area Plot
print("Generating percentage area plot...")

# Calculate percentages by year based on dominant category counts
category_yearly_percentages <- category_yearly_counts[, .(
    percentage = count / sum(count), # Calculate proportion first
    category = category
), by = year][, percentage := percentage * 100] # Convert to percentage

# Get document counts per year again (using the potentially filtered final_dt)
docs_per_year <- final_dt[!is.na(year), .(n = uniqueN(id)), by = year][order(year)]

# Create the year labels with counts
year_labels <- docs_per_year[, paste0(year, "\n(n=", n, ")")]
names(year_labels) <- docs_per_year$year

# Reorder categories if needed (example: put one at the bottom)
# category_yearly_percentages[, category := forcats::fct_relevel(category, "public_order", after = Inf)] # Adjust category name if needed

# Create percentage area plot
percentage_area_plot <- ggplot(
    category_yearly_percentages,
    aes(x = year, y = percentage, fill = category)
) +
    geom_area(position = "fill", alpha = 0.8) + # position = "fill" automatically calculates percentages
    theme_minimal(base_size = 11) +
    scale_fill_manual(values = color_palette_dynamic, labels = pretty_category_names) +
    labs(
        title = "Concept Paragraph Category Distribution by Year",
        subtitle = "Percentage of sentences dominated by each category",
        x = "Year (Document Count)",
        y = "Percentage",
        fill <- "Dominant Category"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), expand = c(0, 0)) +
    scale_x_continuous(
        breaks = docs_per_year$year,
        labels = year_labels,
        expand = c(0, 0)
    ) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size=8), # Adjust size if needed
        legend.title = element_blank(),
        legend.text = element_text(face = "bold"),
        legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        plot.subtitle = element_text(size = 9)
    )

# Save percentage area plot - update path
ggsave(paste0(output_dir, "concept_paragraphs_category_percentages_by_year_area.png"),
    plot = percentage_area_plot,
    width = 12,
    height = 7,
    dpi = 300,
    bg = "white"
)

print(paste("All concept paragraph visualizations saved to:", output_dir))
print(paste("All concept paragraph data exports saved to:", data_output_dir))
print("Script finished.")
# %%
