#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.0.0",
#     "requests>=2.31.0",
#     "tqdm>=4.66.0",
# ]
# ///

"""
Generate keywords for categories using multiple LLMs via OpenRouter API.
Samples 20 random Section 2 documents and queries various models for keyword suggestions.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime

# Model configurations with pricing info and structured support
MODELS = [
    {"id": "openai/gpt-5", "name": "OpenAI GPT-5", "input_price": 5.00, "output_price": 15.00, "supports_structured": True},
    {"id": "openai/gpt-oss-120b", "name": "OpenAI GPT-OSS-120B", "input_price": 0.072, "output_price": 0.28, "supports_structured": False},
    {"id": "google/gemini-2.5-flash-lite", "name": "Google Gemini 2.5 Flash Lite", "input_price": 0.10, "output_price": 0.40, "supports_structured": True},
    {"id": "google/gemini-2.5-flash-lite-preview-06-17", "name": "Google Gemini 2.5 Flash Lite Preview", "input_price": 0.10, "output_price": 0.40, "supports_structured": True},
    {"id": "google/gemini-2.5-flash", "name": "Google Gemini 2.5 Flash", "input_price": 0.30, "output_price": 2.50, "supports_structured": True},
    {"id": "google/gemini-2.5-pro", "name": "Google Gemini 2.5 Pro", "input_price": 1.25, "output_price": 10.00, "supports_structured": True},
    {"id": "anthropic/claude-sonnet-4", "name": "Anthropic Claude Sonnet 4", "input_price": 3.00, "output_price": 15.00, "supports_structured": False},
    {"id": "openai/gpt-4.1-mini", "name": "OpenAI GPT-4.1 Mini", "input_price": 0.40, "output_price": 1.60, "supports_structured": True},
    {"id": "anthropic/claude-3.7-sonnet", "name": "Anthropic Claude 3.7 Sonnet", "input_price": 3.00, "output_price": 15.00, "supports_structured": False},
    {"id": "google/gemini-2.0-flash-001", "name": "Google Gemini 2.0 Flash", "input_price": 0.10, "output_price": 0.40, "supports_structured": True},
    {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "input_price": 0.40, "output_price": 2.00, "supports_structured": False},
]

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

def load_prompt_template() -> str:
    """Load the prompt template from prompt.md."""
    prompt_path = get_base_path() / "code" / "llm_keyword_generation" / "prompt.md"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def sample_documents(n: int = 20, seed: int = 42) -> List[Dict]:
    """Sample n random Section 2 documents from the CSV file."""
    print(f"\n📊 Sampling {n} random Section 2 documents...")
    
    csv_path = get_base_path() / "out" / "section2.csv"
    df = pd.read_csv(csv_path)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample n random rows
    sampled = df.sample(n=min(n, len(df)), random_state=seed)
    
    # Extract relevant fields
    documents = []
    for _, row in sampled.iterrows():
        documents.append({
            'id': row['id'],
            'title': row['title'],
            'content': row['content']
        })
    
    print(f"✅ Sampled {len(documents)} documents")
    print(f"   Document IDs: {[doc['id'] for doc in documents][:5]}..." if len(documents) > 5 else f"   Document IDs: {[doc['id'] for doc in documents]}")
    
    # Cache the sample for consistency
    cache_path = get_base_path() / "code" / "llm_keyword_generation" / "sample_documents.json"
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print(f"   Cached sample at: {cache_path}")
    
    return documents

def load_cached_sample() -> List[Dict]:
    """Load cached sample documents if they exist."""
    cache_path = get_base_path() / "code" / "llm_keyword_generation" / "sample_documents.json"
    if cache_path.exists():
        print(f"📂 Loading cached sample from: {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def prepare_prompt(documents: List[Dict]) -> str:
    """Prepare the full prompt with sample documents."""
    template = load_prompt_template()
    
    # Format documents for inclusion
    doc_text = "\n\n".join([
        f"Document {i+1} (ID: {doc['id']}):\n"
        f"Title: {doc['title']}\n"
        f"Content:\n{doc['content'][:3000]}..."  # Truncate very long documents
        if len(doc['content']) > 3000 else 
        f"Document {i+1} (ID: {doc['id']}):\n"
        f"Title: {doc['title']}\n"
        f"Content:\n{doc['content']}"
        for i, doc in enumerate(documents)
    ])
    
    # Replace placeholder in template
    prompt = template.replace("{SAMPLE_DOCUMENTS}", doc_text)
    
    return prompt

def create_keyword_schema():
    """Create JSON schema for structured keyword output."""
    # Create schema for exactly 20 keywords per category
    keyword_array_schema = {
        "type": "array",
        "minItems": 20,
        "maxItems": 20,
        "items": {
            "type": "string",
            "description": "A Chinese keyword or phrase representing the concept"
        },
        "description": "Exactly 20 Chinese keywords for this category"
    }
    
    return {
        "name": "keyword_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "public_health": keyword_array_schema,
                "business_professional": keyword_array_schema,
                "revolutionary_culture": keyword_array_schema,
                "voluntary_work": keyword_array_schema,
                "family_relations": keyword_array_schema,
                "ecological_concerns": keyword_array_schema,
                "public_order": keyword_array_schema,
                "public_etiquette": keyword_array_schema
            },
            "required": [
                "public_health",
                "business_professional",
                "revolutionary_culture",
                "voluntary_work",
                "family_relations",
                "ecological_concerns",
                "public_order",
                "public_etiquette"
            ],
            "additionalProperties": False
        }
    }

def query_model(model: Dict, prompt: str, api_key: str, use_structured: bool = True) -> Tuple[str, Dict]:
    """Query a single model via OpenRouter API."""
    print(f"\n🤖 Querying {model['name']}...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/wenming/analysis",
        "X-Title": "Wenming Keyword Analysis",
        "Content-Type": "application/json"
    }
    
    if use_structured:
        # Use structured output for guaranteed format
        system_prompt = (
            "You are a research assistant analyzing Chinese government documents. "
            "Generate exactly 20 conceptual keywords for each category based on the documents provided."
        )
        
        data = {
            "model": model['id'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": create_keyword_schema()
            },
            "temperature": 0.3,
            "max_tokens": 8000  # Increased for JSON output
        }
    else:
        # Fallback to CSV format
        system_prompt = (
            "You are a research assistant. You MUST return ONLY CSV data in the format: category,keyword\n"
            "Do not include any markdown formatting, explanations, or extra text. "
            "Start your response directly with the CSV data."
        )
        
        data = {
            "model": model['id'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000
        }
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Log token usage
        usage = result.get('usage', {})
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        input_cost = (input_tokens / 1_000_000) * model['input_price']
        output_cost = (output_tokens / 1_000_000) * model['output_price']
        total_cost = input_cost + output_cost
        
        print(f"   ✅ Success! Tokens: {input_tokens:,} in, {output_tokens:,} out")
        print(f"   💰 Cost: ${total_cost:.4f} (${input_cost:.4f} + ${output_cost:.4f})")
        
        metadata = {
            'model_id': model['id'],
            'model_name': model['name'],
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_cost': total_cost,
            'timestamp': datetime.now().isoformat()
        }
        
        return content, metadata
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error: {str(e)}")
        return None, {'error': str(e)}
    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing error: {str(e)}")
        return None, {'error': f'JSON error: {str(e)}'}

def json_to_csv(json_content: str, debug: bool = False) -> str:
    """Convert JSON response to CSV format."""
    try:
        # Parse JSON if it's a string
        if isinstance(json_content, str):
            # Debug: show first 500 chars of content
            if debug:
                print(f"   🔍 Raw JSON content (first 500 chars): {json_content[:500]}")
            
            # Try to find JSON object in the content
            json_start = json_content.find('{')
            json_end = json_content.rfind('}')
            
            if json_start != -1 and json_end != -1:
                json_content = json_content[json_start:json_end+1]
            
            data = json.loads(json_content)
        else:
            data = json_content
        
        if debug:
            print(f"   🔍 Parsed JSON keys: {list(data.keys())}")
            
        lines = []
        for category in EXPECTED_CATEGORIES:
            if category in data:
                keywords = data[category]
                # Ensure we have a list
                if isinstance(keywords, list):
                    for keyword in keywords:
                        if keyword:  # Skip empty keywords
                            lines.append(f"{category},{keyword}")
                else:
                    print(f"   ⚠️  Category {category} doesn't contain a list")
            else:
                print(f"   ⚠️  Missing category: {category}")
                
        return '\n'.join(lines)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"   ⚠️  Error converting JSON to CSV: {e}")
        print(f"   📝 Content appears to be: {type(json_content)}")
        # Fall back to treating it as CSV
        return clean_and_validate_csv(json_content, "unknown")

def clean_and_validate_csv(content: str, model_name: str) -> str:
    """Clean and validate CSV output from model."""
    lines = []
    
    # Check if this might be JSON
    if content.strip().startswith('{'):
        return json_to_csv(content, debug=True)
    
    # Remove any markdown formatting
    content = content.replace("```csv", "").replace("```", "")
    
    for line in content.strip().split('\n'):
        line = line.strip()
        
        # Skip empty lines or headers
        if not line or line.lower().startswith('category'):
            continue
            
        # Check if it's valid CSV format
        if ',' in line:
            parts = line.split(',', 1)
            if len(parts) == 2:
                category = parts[0].strip()
                remaining = parts[1].strip()
                
                # Check if this is DeepSeek format (all keywords on one line)
                if category in EXPECTED_CATEGORIES and ',' in remaining:
                    # Split multiple keywords on same line
                    keywords = [k.strip() for k in remaining.split(',')]
                    for keyword in keywords:
                        if keyword and not keyword.startswith('"'):
                            lines.append(f"{category},{keyword}")
                elif category in EXPECTED_CATEGORIES:
                    # Single keyword per line format
                    lines.append(f"{category},{remaining}")
                else:
                    print(f"   ⚠️  Skipping invalid category '{category}' in {model_name}")
    
    return '\n'.join(lines)

def save_model_output(model: Dict, content: str, metadata: Dict, debug: bool = False):
    """Save model output to CSV file."""
    model_slug = model['id'].replace('/', '_').replace('.', '-')
    output_path = get_base_path() / "out" / "llm_keywords" / f"{model_slug}.csv"
    
    print(f"   🔄 Processing output for {model['name']}...")
    
    # Save raw output for debugging
    if debug or "claude" in model['id'].lower():
        raw_path = output_path.with_suffix('.raw.txt')
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   🔍 Saved raw output to: {raw_path}")
    
    # Clean and validate the CSV content
    cleaned_content = clean_and_validate_csv(content, model['name'])
    
    if not cleaned_content:
        print(f"   ❌ No valid keywords extracted from {model['name']}")
        raise ValueError("No valid keywords extracted")
    
    # Save cleaned CSV
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"   💾 Saved to: {output_path}")
    
    # Count keywords per category
    category_counts = {}
    for line in cleaned_content.split('\n'):
        if ',' in line:
            category = line.split(',')[0]
            category_counts[category] = category_counts.get(category, 0) + 1
    
    print(f"   📊 Keywords per category: {category_counts}")
    
    # Check if all categories are present
    missing_categories = set(EXPECTED_CATEGORIES) - set(category_counts.keys())
    if missing_categories:
        print(f"   ⚠️  Missing categories: {missing_categories}")
    
    # Save metadata
    metadata_path = output_path.with_suffix('.json')
    metadata['category_counts'] = category_counts
    metadata['missing_categories'] = list(missing_categories)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def check_existing_output(model: Dict, overwrite: bool) -> bool:
    """Check if output already exists for a model."""
    model_slug = model['id'].replace('/', '_').replace('.', '-')
    output_path = get_base_path() / "out" / "llm_keywords" / f"{model_slug}.csv"
    
    if output_path.exists() and not overwrite:
        print(f"   ⏭️  Skipping {model['name']} - output already exists")
        print(f"      Use --overwrite to regenerate")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Generate keywords using multiple LLMs')
    parser.add_argument('--limit', type=int, help='Limit to first N models (for testing)')
    parser.add_argument('--model', type=str, help='Run only specified model ID')
    parser.add_argument('--use-cache', action='store_true', help='Use cached sample documents')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model outputs')
    parser.add_argument('--no-structured', action='store_true', help='Disable structured JSON output (use CSV)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 LLM KEYWORD GENERATION PIPELINE")
    print("=" * 80)
    
    # Check for API key
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENROUTER_API_KEY not found in environment variables")
        sys.exit(1)
    print("✅ OpenRouter API key found")
    
    # Sample or load documents
    if args.use_cache:
        documents = load_cached_sample()
        if not documents:
            print("⚠️  No cached sample found, sampling new documents...")
            documents = sample_documents(seed=args.seed)
    else:
        documents = sample_documents(seed=args.seed)
    
    # Prepare prompt
    print("\n📝 Preparing prompt with sample documents...")
    prompt = prepare_prompt(documents)
    prompt_length = len(prompt)
    print(f"   Prompt length: {prompt_length:,} characters")
    
    # Select models to run
    if args.model:
        models_to_run = [m for m in MODELS if m['id'] == args.model]
        if not models_to_run:
            print(f"❌ ERROR: Model '{args.model}' not found")
            sys.exit(1)
    elif args.limit:
        models_to_run = MODELS[:args.limit]
    else:
        models_to_run = MODELS
    
    print(f"\n🎯 Will query {len(models_to_run)} model(s)")
    if not args.overwrite:
        print("   ℹ️  Skipping models with existing outputs (use --overwrite to regenerate)")
    if not args.no_structured:
        print("   📋 Using structured JSON output (exactly 20 keywords per category)")
    else:
        print("   📝 Using CSV format (may have variable keywords per category)")
    
    # Query each model
    total_cost = 0
    successful = 0
    skipped = 0
    failed = []
    
    for model in tqdm(models_to_run, desc="Processing models"):
        # Check if output exists (unless overwrite is set)
        if check_existing_output(model, args.overwrite):
            skipped += 1
            continue
            
        try:
            # Check if model supports structured output
            model_supports_structured = model.get('supports_structured', True)
            use_structured = (not args.no_structured) and model_supports_structured
            
            if not model_supports_structured and not args.no_structured:
                print(f"   ⚠️  {model['name']} doesn't support structured output, falling back to CSV")
            
            content, metadata = query_model(model, prompt, api_key, use_structured=use_structured)
            
            if content:
                save_model_output(model, content, metadata, debug=args.debug)
                total_cost += metadata.get('total_cost', 0)
                successful += 1
            else:
                failed.append(model['name'])
                if 'error' in metadata:
                    print(f"   ❌ Error details: {metadata['error']}")
        except Exception as e:
            print(f"   ❌ Unexpected error for {model['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed.append(model['name'])
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Successful: {successful}/{len(models_to_run)}")
    if skipped:
        print(f"⏭️  Skipped (already exist): {skipped}/{len(models_to_run)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    print(f"💰 Total cost: ${total_cost:.4f}")
    print(f"📁 Output directory: {get_base_path() / 'out' / 'llm_keywords'}")
    
    if successful + skipped == len(models_to_run):
        print("\n✨ All models completed/exist! Ready for aggregation.")
        print("   Run: python 02_aggregate_llm_keywords.py")

if __name__ == "__main__":
    main()
