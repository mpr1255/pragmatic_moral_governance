#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "tqdm>=4.66.0",
# ]
# ///

"""
Aggregate keyword suggestions from multiple LLMs to find consensus keywords.
Analyzes which keywords are most commonly suggested across models for each category.
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Configuration
TOP_N_KEYWORDS = 10  # Number of top keywords to select per category

# Expected categories
EXPECTED_CATEGORIES = [
    "public_health",
    "business_professional", 
    "revolutionary_culture",
    "voluntary_work",
    "family_relations",
    "ecological_concerns",
    "public_order",
    "public_etiquette"
]

def get_base_path() -> Path:
    """Get the base path for the project."""
    return Path(__file__).resolve().parents[2]

def load_model_keywords() -> Dict[str, Dict[str, List[str]]]:
    """Load keywords from all model output files."""
    keywords_dir = get_base_path() / "out" / "llm_keywords"
    model_keywords = {}
    
    print("\n📂 Loading keyword files from models...")
    
    # Find all CSV files
    csv_files = list(keywords_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No model output files found!")
        return {}
    
    print(f"   Found {len(csv_files)} model output files")
    
    for csv_file in tqdm(csv_files, desc="Loading files"):
        model_name = csv_file.stem
        
        # Initialize category dict for this model
        model_keywords[model_name] = defaultdict(list)
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    
                    parts = line.split(',', 1)
                    if len(parts) == 2:
                        category = parts[0].strip()
                        keyword = parts[1].strip()
                        
                        if category in EXPECTED_CATEGORIES and keyword:
                            model_keywords[model_name][category].append(keyword)
            
            # Report stats for this model
            total_keywords = sum(len(kws) for kws in model_keywords[model_name].values())
            print(f"   {model_name}: {total_keywords} keywords across {len(model_keywords[model_name])} categories")
            
        except Exception as e:
            print(f"   ⚠️  Error loading {csv_file}: {e}")
            continue
    
    return model_keywords

def analyze_keyword_consensus(model_keywords: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[Tuple[str, int, float]]]:
    """Analyze keyword consensus across models."""
    print("\n🔍 Analyzing keyword consensus across models...")
    
    consensus_keywords = {}
    total_models = len(model_keywords)
    
    for category in EXPECTED_CATEGORIES:
        # Collect all keywords for this category across all models
        keyword_counter = Counter()
        
        for model_name, categories in model_keywords.items():
            keywords = categories.get(category, [])
            # Add each unique keyword once per model (voting approach)
            keyword_counter.update(set(keywords))
        
        # Sort by frequency (number of models that suggested it)
        sorted_keywords = keyword_counter.most_common()
        
        # Add percentage of models that suggested each keyword
        keywords_with_stats = [
            (keyword, count, count / total_models * 100)
            for keyword, count in sorted_keywords
        ]
        
        consensus_keywords[category] = keywords_with_stats
        
        # Report top keywords
        print(f"\n   {category}:")
        top_5 = keywords_with_stats[:5]
        for keyword, count, percentage in top_5:
            print(f"      {keyword}: {count}/{total_models} models ({percentage:.1f}%)")
    
    return consensus_keywords

def select_consensus_keywords(
    consensus_keywords: Dict[str, List[Tuple[str, int, float]]], 
    top_n: int = TOP_N_KEYWORDS
) -> Dict[str, List[str]]:
    """Select top N most agreed-upon keywords per category."""
    print(f"\n✅ Selecting top {top_n} keywords per category...")
    
    selected_keywords = {}
    
    for category, keywords_with_stats in consensus_keywords.items():
        # Simply take the top N keywords (already sorted by agreement count)
        top_keywords = [keyword for keyword, count, percentage in keywords_with_stats[:top_n]]
        selected_keywords[category] = top_keywords
        
        # Show what was selected
        print(f"\n   {category} (top {top_n}):")
        for i, (keyword, count, percentage) in enumerate(keywords_with_stats[:top_n]):
            print(f"      {i+1}. {keyword}: {count} models ({percentage:.1f}%)")
    
    return selected_keywords

def save_consensus_keywords(selected_keywords: Dict[str, List[str]], output_filename: str = "category_keywords--models.csv"):
    """Save consensus keywords to CSV file."""
    output_path = get_base_path() / "ref" / output_filename
    
    print(f"\n💾 Saving consensus keywords to: {output_path}")
    
    # Create rows for CSV
    rows = []
    for category in EXPECTED_CATEGORIES:
        keywords = selected_keywords.get(category, [])
        for keyword in keywords:
            rows.append({'category': category, 'keyword': keyword})
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"   ✅ Saved {len(rows)} keywords across {len(selected_keywords)} categories")
    
    return output_path

def generate_analysis_report(
    model_keywords: Dict[str, Dict[str, List[str]]],
    consensus_keywords: Dict[str, List[Tuple[str, int, float]]],
    selected_keywords: Dict[str, List[str]]
):
    """Generate a detailed analysis report."""
    report_path = get_base_path() / "out" / "llm_keywords" / "aggregation_report.txt"
    
    print(f"\n📊 Generating analysis report...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("LLM KEYWORD AGGREGATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model participation
        f.write("MODELS ANALYZED:\n")
        f.write("-" * 40 + "\n")
        for i, model in enumerate(model_keywords.keys(), 1):
            total_kws = sum(len(kws) for kws in model_keywords[model].values())
            f.write(f"{i}. {model}: {total_kws} keywords\n")
        
        f.write(f"\nTotal models: {len(model_keywords)}\n")
        f.write("\n\n")
        
        # Top N consensus keywords per category
        f.write(f"TOP {TOP_N_KEYWORDS} KEYWORDS PER CATEGORY (by agreement count):\n")
        f.write("=" * 80 + "\n")
        
        for category in EXPECTED_CATEGORIES:
            f.write(f"\n{category.upper()}\n")
            f.write("-" * 40 + "\n")
            
            # Show top N with agreement stats
            top_keywords = consensus_keywords.get(category, [])[:TOP_N_KEYWORDS]
            for i, (keyword, count, percentage) in enumerate(top_keywords, 1):
                f.write(f"{i}. {keyword}: {count}/{len(model_keywords)} models ({percentage:.1f}%)\n")
            
            f.write("\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Method: Top {TOP_N_KEYWORDS} most agreed-upon keywords per category\n")
        f.write(f"Total keywords selected: {len(EXPECTED_CATEGORIES) * TOP_N_KEYWORDS} ({len(EXPECTED_CATEGORIES)} categories × {TOP_N_KEYWORDS} keywords)\n")
    
    print(f"   ✅ Report saved to: {report_path}")


def main():
    print("=" * 80)
    print("🔄 LLM KEYWORD AGGREGATION")
    print("=" * 80)
    
    # Load all model outputs
    model_keywords = load_model_keywords()
    
    if not model_keywords:
        print("\n❌ No model outputs to aggregate!")
        print("   Please run: python 01_generate_llm_keywords.py")
        sys.exit(1)
    
    # Analyze consensus
    consensus_keywords = analyze_keyword_consensus(model_keywords)
    
    # Select top N most agreed-upon keywords per category
    selected_keywords = select_consensus_keywords(
        consensus_keywords,
        top_n=TOP_N_KEYWORDS
    )
    
    # Save consensus keywords
    output_path = save_consensus_keywords(selected_keywords)
    
    # Generate detailed report
    generate_analysis_report(model_keywords, consensus_keywords, selected_keywords)
    
    # Calculate Krippendorff's alpha for intercoder reliability
    print("\n" + "=" * 80)
    print("📊 CALCULATING INTERCODER RELIABILITY")
    print("=" * 80)
    
    # Run the Krippendorff's alpha calculation script
    import subprocess
    krippendorff_script = Path(__file__).parent / "03_calculate_krippendorff_alpha.py"
    if krippendorff_script.exists():
        try:
            result = subprocess.run([str(krippendorff_script)], capture_output=True, text=True)
            # Extract just the alpha value from output
            for line in result.stdout.split('\n'):
                if 'KRIPPENDORFF\'S ALPHA:' in line or 'Krippendorff\'s α =' in line:
                    print(line.strip())
            if result.returncode != 0:
                print(f"⚠️  Error calculating Krippendorff's alpha: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not run Krippendorff's alpha calculation: {e}")
    else:
        print("⚠️  Krippendorff's alpha script not found")
    
    # Summary
    print("\n" + "=" * 80)
    print("✨ AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"📁 Consensus keywords saved to: {output_path}")
    print(f"📊 Analysis report: {get_base_path() / 'out' / 'llm_keywords' / 'aggregation_report.txt'}")
    print(f"📊 Krippendorff's alpha report: {get_base_path() / 'out' / 'llm_keywords' / 'krippendorff_alpha_report.txt'}")
    
    total_keywords = sum(len(kws) for kws in selected_keywords.values())
    print(f"✅ Total consensus keywords: {total_keywords}")
    print(f"📈 Method: Top {TOP_N_KEYWORDS} most agreed-upon keywords per category")

if __name__ == "__main__":
    main()
