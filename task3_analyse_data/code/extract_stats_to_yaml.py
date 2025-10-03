#!/usr/bin/env python3
"""
Extract statistics from JSON files to YAML for use with Pandoc
"""

import json
import yaml
import sys
from pathlib import Path

def flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    base_path = Path("../graphs_and_maps/keyword_centroid")
    
    all_stats = {}
    
    # Load statistics for each level
    for level in ['sentence', 'article', 'section']:
        json_path = base_path / f"{level}_level" / "keyword_statistics.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Add to main dict with level prefix
                for key, value in data.items():
                    if key != 'generated_at':  # Skip timestamp
                        all_stats[f"{level}.{key}"] = value
    
    # Flatten the nested structure
    flat_stats = flatten_dict(all_stats)
    
    # Output as YAML
    yaml_output = yaml.dump(flat_stats, default_flow_style=False, sort_keys=True)
    
    # Save to file
    output_path = Path("keyword_statistics.yaml")
    with open(output_path, 'w') as f:
        f.write(yaml_output)
    
    print(f"Statistics extracted to {output_path}")
    print(f"Use with: pandoc document.md --metadata-file={output_path} --lua-filter=substitute_stats_simple.lua")

if __name__ == "__main__":
    main()