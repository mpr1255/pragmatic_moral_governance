# Task 4: 说明解读 (Explanatory Documents) Data Collection

This folder contains data collection and analysis for Chinese explanatory/interpretive documents (说明/解读) about "文明行为促进条例" (Civilized Behavior Promotion Regulations).

## Quick File-to-URL Mapping

To find the original URL for any `.md` file in this collection:

### Method 1: Direct Lookup
Check these URL mapping files in `data/new/`:
- `sample_websearch_results_urls.txt` - Sample URLs from initial searches
- `urls_clean.txt` - Cleaned URL list
- `all_urls_expanded.txt` - Complete expanded URL list

### Method 2: Filename-to-URL Pattern
MD filenames are derived from the last part of the URL path:

**Examples:**
- `t20220901_2806835.md` → `https://tamgw.beijing.gov.cn/xinwendongtai/dqdt/202209/t20220901_2806835.html`
- `t20201222_2180238.md` → `http://www.bjrd.gov.cn/zyfb/bg/202012/t20201222_2180238.html`
- `1863862.md` → `https://slj.xuancheng.gov.cn/OpennessContent/show/1863862.html`

**Pattern:** The filename (minus `.md`) is usually the final path component of the original URL (minus `.html`).

### Method 3: Search in JSON Results
Search these files for complete metadata:
```bash
# Find URL for a specific filename
grep -r "t20220901_2806835" task4_shuoming_jiedu/data/new/*.json
```

## Project Structure

```
task4_shuoming_jiedu/
├── README.md                    # This file
├── analysis/                    # Analysis outputs
│   ├── create_policy_themes_table.py
│   └── policy_themes_table.md
├── data/                        # Raw and processed data
│   ├── md/                      # Original markdown files (legacy)
│   ├── new/                     # Current data collection
│   │   ├── html/               # Downloaded HTML files
│   │   │   ├── *.html         # Raw HTML from websites
│   │   │   └── *.md           # HTML converted to markdown
│   │   ├── md/                # Final processed markdown files
│   │   ├── *.txt              # URL lists and mappings
│   │   ├── *.json             # Structured search results
│   │   ├── *.py               # Data collection scripts
│   │   └── *.sh               # Download automation scripts
│   └── docx/                   # Word documents (if any)
├── prompts/                    # LLM prompts for analysis
└── shuoming_jiedu.db          # SQLite database of processed data
```

## Data Derivation Workflow

### Phase 1: Web Search Collection
**Script:** `data/new/search_explanatory_docs.py`
- Uses Apify Google Search API to find documents
- Searches for specific patterns like "文明行为促进条例 说明" and "文明行为促进条例 解读"
- Filters results to only include explanatory/interpretive documents (not the regulations themselves)
- Must contain both "文明行为促进条例" AND ("说明" OR "解读")
- Prioritizes government domains (.gov.cn, .org.cn)

**Outputs:**
- `*_urls.txt` - URL lists by search pattern
- `*_results.json` - Complete search metadata
- `*_summary.txt` - Human-readable summaries

### Phase 2: Manual Collection Enhancement
**Script:** `data/new/simple_websearch_collector.py`
- Processes manual WebSearch tool results
- Applies same filtering criteria as automated search
- Sample results hardcoded for testing

**Key Search Patterns:**
```
"关于 文明行为促进条例（草案）的起草说明"
"关于 文明行为促进条例（草案） 说明"
"关于 文明行为促进条例（草案） 解读"
"文明行为促进条例 解读"
"文明行为促进条例 说明 解读"
"文明行为促进条例 起草说明"
"文明行为促进条例草案 说明"
"文明行为促进条例 政策解读"
```

### Phase 3: Content Download
**Script:** `data/new/download_urls.sh`
- Downloads HTML content from all collected URLs
- Extracts filename from URL path (e.g., `t20220901_2806835.html`)
- Saves to `data/new/html/`
- Includes 2-second delays to be respectful to servers

### Phase 4: Content Conversion
- HTML files converted to Markdown using automated tools
- Both versions preserved:
  - `data/new/html/*.html` - Original HTML
  - `data/new/html/*.md` - HTML-to-markdown conversion
  - `data/new/md/*.md` - Final processed markdown

## Document Types Collected

This collection focuses specifically on **explanatory/interpretive documents**, not the regulations themselves:

### Included Document Types:
- **起草说明** (Drafting explanations)
- **政策解读** (Policy interpretations) 
- **说明** (Explanations)
- **解读** (Interpretations)
- Legislative committee reports explaining proposed regulations
- Government department explanations of enacted regulations

### Excluded Document Types:
- The actual regulation texts
- News articles about regulations
- General policy discussions
- Documents lacking explanatory nature

## Key Files for Understanding the Collection

### URL Mapping Files:
- `data/new/all_urls_expanded.txt` - Master list of all URLs
- `data/new/urls_clean.txt` - Cleaned and deduplicated URLs
- `data/new/sample_websearch_results_urls.txt` - Sample URLs with known good results

### Search Results:
- `data/new/sample_websearch_results_results.json` - Structured metadata for sample results
- `data/new/sample_websearch_results_summary.txt` - Human-readable summary

### Collection Scripts:
- `data/new/search_explanatory_docs.py` - Main automated collection script
- `data/new/simple_websearch_collector.py` - Manual collection helper
- `data/new/download_urls.sh` - Bulk download script

## Usage Examples

### Find Original URL for a Markdown File:
```bash
# Method 1: Direct grep search
grep "t20220901_2806835" task4_shuoming_jiedu/data/new/*.txt

# Method 2: Pattern matching
# If file is "t20220901_2806835.md", likely URL ends with "t20220901_2806835.html"
```

### Verify Document Content:
```bash
# Check if HTML was downloaded
ls task4_shuoming_jiedu/data/new/html/t20220901_2806835.html

# Read converted markdown
cat task4_shuoming_jiedu/data/new/md/t20220901_2806835.md
```

### Find All Documents from a Specific Domain:
```bash
grep "beijing.gov.cn" task4_shuoming_jiedu/data/new/all_urls_expanded.txt
```

## Data Quality Notes

### Filtering Criteria Applied:
1. **Topic Relevance:** Must contain "文明行为促进条例"
2. **Document Type:** Must contain explanatory keywords ("说明", "解读", etc.)
3. **Source Quality:** Government domains preferred but not required
4. **Language:** Chinese language documents only
5. **Deduplication:** URLs deduplicated across search patterns

### Known Limitations:
- Some URLs may no longer be accessible
- HTML-to-markdown conversion may lose formatting
- Search was limited to Google results (not comprehensive coverage)
- Manual verification of document relevance may be needed

## Database Structure

The `shuoming_jiedu.db` SQLite database contains structured data from the collection. Key tables likely include:
- Documents metadata (title, URL, source, collection date)
- Content analysis results
- Classification/categorization data

## How I Found the URL Mapping

I discovered the URL-to-filename mapping by:

1. **Searching for the specific filenames** in the task4 directory using grep
2. **Finding multiple URL mapping files** in `data/new/` that contained both filenames and their source URLs
3. **Identifying the pattern** that filenames are derived from URL paths by removing `.html` and adding `.md`
4. **Verifying the pattern** across multiple examples in the collection

The systematic file organization and multiple cross-reference files make it straightforward to trace any document back to its original source.