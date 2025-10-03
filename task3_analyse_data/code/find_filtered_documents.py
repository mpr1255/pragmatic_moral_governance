#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "pandas",
#     "numpy",
# ]
# ///
"""
Script to identify which documents were filtered out during the 292 -> 250 reduction.
"""

import pandas as pd
import re
import os

def extract_section(content):
    """
    Same extraction function as used in _02_create_embeddings_for_sentences.py
    """
    # Handle NaN or non-string values
    if pd.isna(content) or not isinstance(content, str):
        return None
    
    # Pattern to capture content between 第二章 and 第三章, allowing for characters before and after
    pattern = r'(第二章.*?)(?=第三章)'
    try:
        matches = re.findall(pattern, content, flags=re.DOTALL)
        for match in matches:
            if match.count('\n') > 1:  # Ensure there's more than one line
                return match
        return None  # No match with sufficient content
    except Exception as e: 
        print(f"Error: {e}")
        return None

def main():
    # Read the original CSV with all 292 documents
    original_df = pd.read_csv('../../task2_cleanup_and_build_database/out/output.csv')
    print(f"Original documents: {len(original_df)}")
    
    # Apply the same extraction logic
    original_df['extracted_section'] = original_df['content'].apply(extract_section)
    
    # Find documents that were filtered out
    filtered_out = original_df[original_df['extracted_section'].isna()].copy()
    successful = original_df[original_df['extracted_section'].notna()].copy()
    
    print(f"Documents with successful Section 2 extraction: {len(successful)}")
    print(f"Documents filtered out: {len(filtered_out)}")
    
    # Analyze the filtered-out documents
    print("\n=== FILTERED OUT DOCUMENTS ===")
    print(f"Total filtered: {len(filtered_out)}")
    
    if len(filtered_out) > 0:
        # Show titles and reasons
        filtered_analysis = []
        
        for idx, row in filtered_out.iterrows():
            title = row['title']
            content = row['content']
            
            # Analyze why it failed
            reason = "Unknown"
            if pd.isna(content):
                reason = "Missing content"
            elif not isinstance(content, str):
                reason = "Non-string content"
            elif '第二章' not in content:
                reason = "No Chapter 2 (第二章)"
            elif '第三章' not in content:
                reason = "No Chapter 3 (第三章)"
            else:
                # Check if match exists but is too short
                pattern = r'(第二章.*?)(?=第三章)'
                matches = re.findall(pattern, content, flags=re.DOTALL)
                if matches:
                    if matches[0].count('\n') <= 1:
                        reason = "Chapter 2 content too short (≤1 line)"
                    else:
                        reason = "Extraction error"
                else:
                    reason = "Regex pattern no match"
            
            filtered_analysis.append({
                'title': title,
                'reason': reason,
                'publish': row.get('publish', 'Unknown'),
                'office': row.get('office', 'Unknown')
            })
        
        # Create summary report
        filtered_df = pd.DataFrame(filtered_analysis)
        
        # Count by reason
        reason_counts = filtered_df['reason'].value_counts()
        print("\nFiltering reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")
        
        # Show some examples
        print(f"\nFirst 10 filtered documents:")
        for i, doc in enumerate(filtered_analysis[:10]):
            print(f"{i+1:2d}. {doc['title'][:80]}...")
            print(f"    Reason: {doc['reason']}")
            print(f"    Office: {doc['office']}")
            print(f"    Published: {doc['publish']}")
            print()
        
        # Save full list to file
        output_file = "filtered_out_documents.csv"
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Full list saved to: {output_file}")
        
        # Also save successful documents for comparison
        successful_titles = successful[['title', 'publish', 'office']].copy()
        successful_titles.to_csv("successful_documents.csv", index=False, encoding='utf-8')
        print(f"Successful documents list saved to: successful_documents.csv")

if __name__ == "__main__":
    main()