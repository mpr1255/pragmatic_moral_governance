#!/usr/bin/env python3
"""
Merge the three level-specific YAML files into one combined file
"""

import yaml
from pathlib import Path

def main():
    base_path = Path("../graphs_and_maps/keyword_centroid")
    
    all_stats = {}
    
    # Load statistics for each level
    for level in ['sentence', 'article', 'section']:
        yaml_path = base_path / f"{level}_level" / "keyword_statistics.yaml"
        
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                level_stats = yaml.safe_load(f)
                # Merge into main dict
                all_stats.update(level_stats)
            print(f"Loaded {len(level_stats)} statistics from {level} level")
        else:
            print(f"Warning: {yaml_path} not found")
    
    # Save combined YAML
    output_path = base_path / "all_keyword_statistics.yaml"
    with open(output_path, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False, sort_keys=True)
    
    print(f"\nCombined statistics saved to: {output_path}")
    print(f"Total statistics: {len(all_stats)}")
    
    # Also save to manuscripts directory for easier access
    manuscripts_path = Path("../../manuscripts/keyword_statistics.yaml")
    with open(manuscripts_path, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False, sort_keys=True)
    print(f"Also saved to: {manuscripts_path}")

if __name__ == "__main__":
    main()