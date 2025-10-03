#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "krippendorff>=0.6.0",
#     "tqdm>=4.66.0",
# ]
# ///

"""
Calculate Krippendorff's alpha for intercoder reliability of LLM keyword categorization.
This measures agreement between multiple LLMs (coders) on keyword-category assignments.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
import krippendorff
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

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

def load_all_model_keywords() -> Dict[str, Dict[str, Set[str]]]:
    """Load keywords from all model output files as sets for each category."""
    keywords_dir = get_base_path() / "out" / "llm_keywords"
    model_keywords = {}
    
    print("\n📂 Loading keyword files for Krippendorff's alpha calculation...")
    
    # Find all CSV files
    csv_files = list(keywords_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No model output files found!")
        return {}
    
    print(f"   Found {len(csv_files)} model output files")
    
    for csv_file in tqdm(csv_files, desc="Loading files"):
        model_name = csv_file.stem
        
        # Initialize category dict for this model with sets
        model_keywords[model_name] = defaultdict(set)
        
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
                            model_keywords[model_name][category].add(keyword)
            
            # Report stats for this model
            total_keywords = sum(len(kws) for kws in model_keywords[model_name].values())
            print(f"   {model_name}: {total_keywords} unique keywords")
            
        except Exception as e:
            print(f"   ⚠️  Error loading {csv_file}: {e}")
            continue
    
    return model_keywords

def create_reliability_matrix(model_keywords: Dict[str, Dict[str, Set[str]]]) -> Tuple[np.ndarray, List[str], List[str], Dict]:
    """
    Create a reliability matrix for Krippendorff's alpha calculation.
    Each row represents a coder (model), each column represents an item (keyword),
    and values are the assigned categories.
    """
    print("\n🔄 Creating reliability matrix...")
    
    # Get all unique keywords across all models
    all_keywords = set()
    for model_data in model_keywords.values():
        for category_keywords in model_data.values():
            all_keywords.update(category_keywords)
    
    all_keywords = sorted(list(all_keywords))
    model_names = sorted(list(model_keywords.keys()))
    
    print(f"   Total unique keywords: {len(all_keywords)}")
    print(f"   Total models (coders): {len(model_names)}")
    
    # Create category to numeric mapping
    category_to_num = {cat: i+1 for i, cat in enumerate(EXPECTED_CATEGORIES)}
    category_to_num[None] = 0  # 0 for keywords not assigned by a model
    
    # Create the reliability matrix
    # Rows = coders (models), Columns = items (keywords)
    matrix = np.zeros((len(model_names), len(all_keywords)))
    
    for i, model in enumerate(tqdm(model_names, desc="Building matrix")):
        for j, keyword in enumerate(all_keywords):
            # Find which category this model assigned to this keyword
            assigned_category = None
            for category, keywords_set in model_keywords[model].items():
                if keyword in keywords_set:
                    assigned_category = category
                    break
            
            if assigned_category:
                matrix[i, j] = category_to_num[assigned_category]
            else:
                matrix[i, j] = np.nan  # Missing data (model didn't assign this keyword)
    
    return matrix, model_names, all_keywords, category_to_num

def calculate_krippendorff_alpha(matrix: np.ndarray, level: str = 'nominal') -> float:
    """Calculate Krippendorff's alpha for the reliability matrix."""
    print(f"\n📊 Calculating Krippendorff's alpha (level={level})...")
    
    # Count non-NaN values
    total_values = matrix.size
    missing_values = np.isnan(matrix).sum()
    coded_values = total_values - missing_values
    
    print(f"   Matrix shape: {matrix.shape}")
    print(f"   Total values: {total_values:,}")
    print(f"   Coded values: {coded_values:,} ({coded_values/total_values*100:.1f}%)")
    print(f"   Missing values: {missing_values:,} ({missing_values/total_values*100:.1f}%)")
    
    # Calculate alpha
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    
    return alpha

def analyze_agreement_by_category(model_keywords: Dict[str, Dict[str, Set[str]]]) -> Dict[str, float]:
    """Analyze agreement levels for each category separately."""
    print("\n🔍 Analyzing agreement by category...")
    
    category_agreements = {}
    
    for category in EXPECTED_CATEGORIES:
        # Get keywords assigned to this category by each model
        model_assignments = {}
        all_category_keywords = set()
        
        for model, categories in model_keywords.items():
            keywords = categories.get(category, set())
            model_assignments[model] = keywords
            all_category_keywords.update(keywords)
        
        if not all_category_keywords:
            category_agreements[category] = 0.0
            continue
        
        # Calculate overlap percentage
        # For each pair of models, calculate Jaccard similarity
        similarities = []
        model_list = list(model_assignments.keys())
        
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                set1 = model_assignments[model_list[i]]
                set2 = model_assignments[model_list[j]]
                
                if set1 or set2:  # Avoid division by zero
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard = intersection / union if union > 0 else 0
                    similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        category_agreements[category] = avg_similarity
        
        print(f"   {category}: {avg_similarity:.3f} average Jaccard similarity")
    
    return category_agreements

def save_krippendorff_report(
    alpha: float, 
    category_agreements: Dict[str, float],
    matrix_shape: Tuple[int, int],
    model_keywords: Dict[str, Dict[str, Set[str]]]
):
    """Save Krippendorff's alpha and analysis to a report file."""
    report_path = get_base_path() / "out" / "llm_keywords" / "krippendorff_alpha_report.txt"
    
    print(f"\n💾 Saving Krippendorff's alpha report to: {report_path}")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("KRIPPENDORFF'S ALPHA - INTERCODER RELIABILITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Main result
        f.write("OVERALL KRIPPENDORFF'S ALPHA:\n")
        f.write("-" * 40 + "\n")
        f.write(f"α = {alpha:.4f}\n\n")
        
        # Interpretation
        f.write("INTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if alpha >= 0.8:
            interpretation = "Good reliability"
        elif alpha >= 0.667:
            interpretation = "Tentative reliability"
        else:
            interpretation = "Low reliability"
        f.write(f"{interpretation} (α = {alpha:.4f})\n")
        f.write("Reference: α ≥ 0.8 = good, α ≥ 0.667 = tentative, α < 0.667 = low\n\n")
        
        # Method details
        f.write("METHOD DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of coders (LLM models): {matrix_shape[0]}\n")
        f.write(f"Number of items (unique keywords): {matrix_shape[1]}\n")
        f.write(f"Number of categories: {len(EXPECTED_CATEGORIES)}\n")
        f.write(f"Measurement level: nominal\n\n")
        
        # Models analyzed
        f.write("MODELS ANALYZED:\n")
        f.write("-" * 40 + "\n")
        for i, model in enumerate(sorted(model_keywords.keys()), 1):
            total_kws = sum(len(kws) for kws in model_keywords[model].values())
            f.write(f"{i}. {model}: {total_kws} keywords\n")
        f.write("\n")
        
        # Category-specific agreement (Jaccard similarity)
        f.write("CATEGORY-SPECIFIC AGREEMENT (Average Jaccard Similarity):\n")
        f.write("-" * 40 + "\n")
        sorted_categories = sorted(category_agreements.items(), key=lambda x: x[1], reverse=True)
        for category, agreement in sorted_categories:
            f.write(f"{category:30s}: {agreement:.3f}\n")
        f.write("\n")
        
        # Summary statistics
        f.write("SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Krippendorff's α: {alpha:.4f}\n")
        f.write(f"Mean category agreement: {np.mean(list(category_agreements.values())):.3f}\n")
        f.write(f"Std category agreement: {np.std(list(category_agreements.values())):.3f}\n")
        
        # Add note about interpretation for academic use
        f.write("\n" + "=" * 80 + "\n")
        f.write("NOTE FOR ACADEMIC REPORTING:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Intercoder reliability was assessed using Krippendorff's alpha (α = {alpha:.3f}),\n")
        f.write(f"calculated across {matrix_shape[0]} large language models coding {matrix_shape[1]} unique keywords\n")
        f.write(f"into {len(EXPECTED_CATEGORIES)} categories. ")
        
        if alpha >= 0.8:
            f.write("This indicates good intercoder reliability.\n")
        elif alpha >= 0.667:
            f.write("This indicates tentative intercoder reliability.\n")
        else:
            f.write("This indicates low intercoder reliability.\n")
    
    print(f"   ✅ Report saved successfully")
    
    # Also save just the alpha value to a simple text file for easy reference
    alpha_file = get_base_path() / "out" / "llm_keywords" / "krippendorff_alpha.txt"
    with open(alpha_file, 'w') as f:
        f.write(f"{alpha:.4f}\n")
    print(f"   ✅ Alpha value saved to: {alpha_file}")

def main():
    print("=" * 80)
    print("📊 KRIPPENDORFF'S ALPHA CALCULATION")
    print("=" * 80)
    
    # Load all model outputs
    model_keywords = load_all_model_keywords()
    
    if not model_keywords:
        print("\n❌ No model outputs found!")
        sys.exit(1)
    
    # Create reliability matrix
    matrix, model_names, all_keywords, category_mapping = create_reliability_matrix(model_keywords)
    
    # Calculate Krippendorff's alpha
    alpha = calculate_krippendorff_alpha(matrix, level='nominal')
    
    print(f"\n✨ KRIPPENDORFF'S ALPHA: {alpha:.4f}")
    
    # Analyze agreement by category
    category_agreements = analyze_agreement_by_category(model_keywords)
    
    # Save report
    save_krippendorff_report(alpha, category_agreements, matrix.shape, model_keywords)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ CALCULATION COMPLETE")
    print("=" * 80)
    print(f"📊 Krippendorff's α = {alpha:.4f}")
    
    if alpha >= 0.8:
        print("✨ Good intercoder reliability")
    elif alpha >= 0.667:
        print("⚠️  Tentative intercoder reliability")
    else:
        print("❌ Low intercoder reliability")
    
    print(f"\n📁 Full report: {get_base_path() / 'out' / 'llm_keywords' / 'krippendorff_alpha_report.txt'}")

if __name__ == "__main__":
    main()
