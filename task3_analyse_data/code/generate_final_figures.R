#!/usr/bin/env Rscript
# Generate only the 3 final figures needed for publication
# figure1 = article_level dominant_category
# figure2 = sentence_level salience_scatter  
# figure3 = section_level ridgeplot_all_categories

librarian::shelf(data.table, tidyverse, ggplot2, ggridges, scales, purrr, jsonlite, quiet = TRUE)

# Set publication-quality theme
theme_set(theme_minimal(base_size = 12) +
    theme(
        text = element_text(size = 12, family = "Helvetica"),
        axis.text = element_text(size = 11, color = "black"),
        axis.title = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0),
        plot.subtitle = element_text(size = 11, face = "italic", color = "gray50", hjust = 0),
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12, face = "bold"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "gray90", size = 0.25),
        plot.margin = margin(10, 10, 10, 10),
        panel.background = element_rect(fill = "white", color = NA),
        plot.background = element_rect(fill = "white", color = NA),
        axis.line = element_line(color = "black", size = 0.5)
    )
)

# Master color palette
master_colors <- c(
    "public_order" = "#0072B2",
    "public_etiquette" = "#D55E00",
    "public_health" = "#009E73",
    "family_relations" = "#CC79A7",
    "business_professional" = "#E69F00",
    "revolutionary_culture" = "#56B4E9",
    "ecological_concerns" = "#F0E442",
    "voluntary_work" = "#999999"
)

# Helper function to convert snake_case to sentence case
to_sentence_case <- function(x) {
    x <- gsub("_", " ", x)
    x <- paste0(toupper(substring(x, 1, 1)), substring(x, 2))
    return(x)
}

# Configuration
if (file.exists("../../task2_cleanup_and_build_database/out/")) {
    base_data_path <- "../../task2_cleanup_and_build_database/out/"
    base_output_path <- "../graphs_and_maps/"
} else {
    base_data_path <- "task2_cleanup_and_build_database/out/"
    base_output_path <- "task3_analyse_data/graphs_and_maps/"
}

# Load metadata
print("Loading metadata...")
if (file.exists("../../task2_cleanup_and_build_database/out/section2.csv")) {
    metadata_path <- "../../task2_cleanup_and_build_database/out/section2.csv"
    cities_orig_path <- "../../task2_cleanup_and_build_database/hand/unique_cities_original.json"
    cities_fixed_path <- "../../task2_cleanup_and_build_database/hand/unique_cities_fixed.json"
} else {
    metadata_path <- "task2_cleanup_and_build_database/out/section2.csv"
    cities_orig_path <- "task2_cleanup_and_build_database/hand/unique_cities_original.json"
    cities_fixed_path <- "task2_cleanup_and_build_database/hand/unique_cities_fixed.json"
}

metadata_dt <- fread(metadata_path)
metadata_dt <- unique(metadata_dt[, .(id, title, office, publish, expiry, type, status, url)])

# Load city/province mapping
unique_cities_original <- lapply(readLines(cities_orig_path), fromJSON)
unique_cities_fixed <- lapply(readLines(cities_fixed_path), fromJSON)

original_dt <- data.table(id = seq_along(unique_cities_original), title_map = unlist(unique_cities_original))
fixed_dt <- data.table(id = seq_along(unique_cities_fixed), city_province = unlist(unique_cities_fixed))
mapping_dt <- merge(original_dt, fixed_dt, by = "id")
mapping_dt[, c("city", "province") := tstrsplit(city_province, ", ")]

# Function to load and prepare data
load_and_prep_data <- function(file_path, level_name) {
    print(paste("Loading", level_name, "data from:", file_path))
    
    if (!file.exists(file_path)) {
        stop(paste("File not found:", file_path))
    }
    
    dt <- fread(file_path)
    print(paste("Loaded", nrow(dt), "rows"))
    
    # Merge with metadata if needed
    if (!("title" %in% names(dt))) {
        if (level_name == "section") {
            dt <- merge(dt, metadata_dt, by = "id", all.x = TRUE)
        } else {
            dt[, doc_id := substr(id, 1, regexpr("_", id) - 1)]
            dt <- merge(dt, metadata_dt[, .(id, title, publish)], 
                       by.x = "doc_id", by.y = "id", all.x = TRUE)
        }
    }
    
    # Ensure we have publish column
    if (!("publish" %in% names(dt))) {
        dt <- merge(dt, metadata_dt[, .(title, publish)], by = "title", all.x = TRUE)
    }
    
    # Merge with city/province mapping
    dt <- merge(dt, mapping_dt[, .(title = title_map, city, province)], 
               by = "title", all.x = TRUE)
    dt[, year := as.integer(substr(publish, 1, 4))]
    
    # Filter for 2019 onwards
    dt <- dt[year >= 2019]
    print(paste("After filtering for 2019+:", nrow(dt), "rows"))
    
    return(dt)
}

# FIGURE 1: Article level dominant category plot
create_figure1 <- function() {
    print("\n=== CREATING FIGURE 1: Article level dominant category ===")
    
    file_path <- paste0(base_data_path, "section2_article_embeddings_with_individual_keyword_similarities.csv")
    dt <- load_and_prep_data(file_path, "article")
    
    # Get similarity columns
    sim_cols <- grep("^sim_cosine_", names(dt), value = TRUE)
    
    # Find dominant category for each item
    dt_dom <- dt[, c("id", sim_cols), with = FALSE]
    
    # Get dominant category
    dominant_cats <- apply(dt_dom[, -1], 1, function(x) {
        clean_names <- gsub("sim_cosine_", "", names(dt_dom)[-1])
        clean_names[which.max(x)]
    })
    
    dt_dom[, dominant_category := dominant_cats]
    
    # Count dominance
    dominance_counts <- dt_dom[, .N, by = dominant_category]
    dominance_counts <- dominance_counts[dominant_category %in% names(master_colors)]
    
    # Calculate percentages
    total_n <- nrow(dt_dom)
    dominance_counts[, percentage := N / total_n * 100]
    
    # Order by percentage (ascending so highest appears at top of horizontal bar chart)
    setorder(dominance_counts, percentage)
    dominance_counts[, dominant_category := factor(dominant_category, levels = dominant_category)]
    
    # Create horizontal bar plot
    p <- ggplot(dominance_counts, aes(x = percentage, y = dominant_category, fill = dominant_category)) +
        geom_col(width = 0.7) +
        scale_fill_manual(values = master_colors, guide = "none") +
        scale_x_continuous(limits = c(0, max(dominance_counts$percentage) * 1.15),
                          breaks = seq(0, 30, 10),
                          labels = function(x) paste0(x, "%"),
                          expand = c(0, 0)) +
        scale_y_discrete(labels = function(x) to_sentence_case(x)) +
        labs(
            title = paste0("Dominant category distribution (n = ", format(total_n, big.mark=","), " articles)"),
            subtitle = "Classification by highest similarity score | Documents from 2019 onwards",
            x = "Percentage",
            y = ""
        ) +
        theme(panel.grid.major.y = element_blank(),
              panel.grid.major.x = element_line(color = "gray85"),
              axis.ticks.y = element_blank()) +
        geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                 hjust = -0.1, size = 3.5, color = "black")
    
    # Save plot
    output_file <- paste0(base_output_path, "figure1.png")
    ggsave(output_file, p, width = 12, height = 8, dpi = 300, bg = "white")
    print(paste("Saved figure1.png:", output_file))
}

# FIGURE 2: Sentence level salience scatter plot
create_figure2 <- function() {
    print("\n=== CREATING FIGURE 2: Sentence level salience scatter ===")
    
    file_path <- paste0(base_data_path, "section2_sentence_embeddings_with_individual_keyword_similarities.csv")
    dt <- load_and_prep_data(file_path, "sentence")
    
    # Get similarity columns
    sim_cols <- grep("^sim_cosine_", names(dt), value = TRUE)
    
    # Calculate dominance and breadth for each category
    salience_data <- data.table()
    
    # Calculate median similarity across all categories as data-driven threshold
    all_sims <- unlist(dt[, sim_cols, with=FALSE])
    threshold <- median(all_sims, na.rm = TRUE)
    threshold <- round(threshold, 2)  # Round for cleaner display
    
    for(cat in names(master_colors)) {
        col_name <- paste0("sim_cosine_", cat)
        if(col_name %in% sim_cols) {
            # Dominance: percentage where this category is highest
            dominant_count <- sum(apply(dt[, sim_cols, with=FALSE], 1, function(x) {
                which.max(x) == which(sim_cols == col_name)
            }))
            dominance_rate <- dominant_count / nrow(dt) * 100
            
            # Breadth: percentage above median threshold
            breadth <- sum(dt[[col_name]] > threshold, na.rm = TRUE) / nrow(dt) * 100
            
            salience_data <- rbind(salience_data, 
                data.table(category = cat,
                          dominance_rate = dominance_rate,
                          breadth = breadth))
        }
    }
    
    # Create scatter plot
    p <- ggplot(salience_data, aes(x = breadth, y = dominance_rate, color = category)) +
        geom_point(size = 5, alpha = 0.8) +
        geom_text(aes(label = to_sentence_case(category)), 
                 vjust = -1, hjust = 0.5, size = 3.5) +
        scale_color_manual(values = master_colors, guide = "none") +
        scale_x_continuous(limits = c(-2, max(salience_data$breadth) * 1.2),
                          expand = c(0.02, 0)) +
        scale_y_continuous(limits = c(-1, max(salience_data$dominance_rate) * 1.2),
                          expand = c(0.02, 0)) +
        labs(
            title = paste0("Category salience patterns: Sentence level (n=", format(nrow(dt), big.mark=","), ")"),
            subtitle = paste0("Dominance vs breadth (similarity > ", threshold, ") | Documents from 2019 onwards"),
            x = "Breadth (% of sentences with similarity > median)",
            y = "Dominance Rate (% of sentences where category scores highest)"
        )
    
    # Save plot
    output_file <- paste0(base_output_path, "figure2.png")
    ggsave(output_file, p, width = 12, height = 8, dpi = 300, bg = "white")
    print(paste("Saved figure2.png:", output_file))
}

# FIGURE 3: Section level ridge plot
create_figure3 <- function() {
    print("\n=== CREATING FIGURE 3: Section level ridge plot ===")
    
    file_path <- paste0(base_data_path, "section2_embeddings_with_individual_keyword_similarities.csv")
    dt <- load_and_prep_data(file_path, "section")
    
    # Get similarity columns
    sim_cols <- grep("^sim_cosine_", names(dt), value = TRUE)
    
    # Reshape data to long format
    dt_long <- melt(dt, 
                    id.vars = c("id", "title", "city", "province", "year"),
                    measure.vars = sim_cols,
                    variable.name = "category",
                    value.name = "similarity")
    
    # Clean category names
    dt_long[, category := gsub("sim_cosine_", "", category)]
    
    # Filter to our categories
    dt_long <- dt_long[category %in% names(master_colors)]
    
    # Calculate medians for ordering
    medians <- dt_long[, .(median = median(similarity, na.rm = TRUE)), by = category]
    setorder(medians, median)  # Order ascending so highest appears at top
    
    # Create ordered factor
    dt_long[, category_ordered := factor(category, levels = medians$category)]
    
    # Create the plot with proper gridlines and baselines
    p <- ggplot(dt_long, aes(x = similarity, y = category_ordered, fill = category)) +
        # Add horizontal lines at each category position (baselines)
        geom_hline(yintercept = seq_along(levels(dt_long$category_ordered)), 
                   color = "gray40", linewidth = 0.3, alpha = 0.7) +
        geom_density_ridges(alpha = 0.85, scale = 1.3, rel_min_height = 0.01, 
                           quantile_lines = TRUE, quantiles = 2) +
        scale_fill_manual(values = master_colors, guide = "none") +
        scale_x_continuous(limits = c(-0.1, 0.7), 
                          breaks = seq(0, 0.6, 0.2),
                          labels = function(x) sprintf("%.1f", x)) +
        scale_y_discrete(labels = function(x) to_sentence_case(x)) +
        labs(
            title = paste0("Similarity distributions at section level (n=", format(nrow(dt), big.mark=","), ")"),
            subtitle = "Probability densities with median lines | Documents from 2019 onwards",
            x = "Cosine Similarity",
            y = ""
        ) +
        theme_bw(base_size = 12) +  # Use theme_bw which has gridlines by default
        theme(axis.text.y = element_text(size = 11),
              panel.grid.major.x = element_line(color = "gray70", linewidth = 0.5),  # Vertical lines
              panel.grid.minor.x = element_line(color = "gray85", linewidth = 0.25), # Minor vertical lines
              panel.grid.major.y = element_blank(),  # No horizontal lines across categories
              panel.grid.minor.y = element_blank(),
              axis.ticks.y = element_blank(),
              plot.title = element_text(size = 14, face = "bold"),
              plot.subtitle = element_text(size = 11, face = "italic", color = "gray50"),
              panel.background = element_rect(fill = "white", color = NA),
              plot.background = element_rect(fill = "white", color = NA))
    
    # Save plot
    output_file <- paste0(base_output_path, "figure3.png")
    ggsave(output_file, p, width = 12, height = 10, dpi = 300, bg = "white")
    print(paste("Saved figure3.png:", output_file))
}

# Main execution
main <- function() {
    print("\n===== GENERATING FINAL 3 FIGURES =====\n")
    
    # Create output directory
    dir.create(base_output_path, recursive = TRUE, showWarnings = FALSE)
    
    # Generate each figure
    create_figure1()  # Article level dominant category
    create_figure2()  # Sentence level salience scatter
    create_figure3()  # Section level ridge plot
    
    print("\n===== ALL FIGURES GENERATED =====")
    print(paste("Saved to:", base_output_path))
    print("figure1.png = Article level dominant category")
    print("figure2.png = Sentence level salience scatter") 
    print("figure3.png = Section level ridge plot")
}

# Run the analysis
main()