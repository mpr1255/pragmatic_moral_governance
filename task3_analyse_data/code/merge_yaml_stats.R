#!/usr/bin/env Rscript
# Merge the three level-specific YAML files into one combined file

library(yaml)

# Function to merge YAML files
merge_yaml_stats <- function() {
    base_path <- "../graphs_and_maps/keyword_centroid"
    
    all_stats <- list()
    
    # Load statistics for each level
    for (level in c('sentence', 'article', 'section')) {
        yaml_path <- file.path(base_path, paste0(level, "_level"), "keyword_statistics.yaml")
        
        if (file.exists(yaml_path)) {
            level_stats <- read_yaml(yaml_path)
            # Merge into main list
            all_stats <- c(all_stats, level_stats)
            cat(sprintf("Loaded %d statistics from %s level\n", length(level_stats), level))
        } else {
            cat(sprintf("Warning: %s not found\n", yaml_path))
        }
    }
    
    # Save combined YAML
    output_path <- file.path(base_path, "all_keyword_statistics.yaml")
    write_yaml(all_stats, output_path)
    cat(sprintf("\nCombined statistics saved to: %s\n", output_path))
    cat(sprintf("Total statistics: %d\n", length(all_stats)))
    
    # Also save to manuscripts directory for easier access
    manuscripts_path <- "../../manuscripts/keyword_statistics.yaml"
    write_yaml(all_stats, manuscripts_path)
    cat(sprintf("Also saved to: %s\n", manuscripts_path))
}

# Run the merge
merge_yaml_stats()