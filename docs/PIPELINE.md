# Wenming Project - Technical Pipeline Documentation

This document provides detailed technical documentation for the streamlined publication pipeline.

## 📋 Pipeline Overview

The analysis pipeline transforms raw Chinese legal documents into publication-ready figures and reliability statistics through the following stages:

1. **Data Foundation**: Section 2 text extraction and preprocessing  
2. **Embedding Generation**: Create vector embeddings for all text units
3. **Similarity Analysis**: Calculate keyword-based similarity scores
4. **Visualization**: Generate publication figures
5. **Reliability Assessment**: LLM-based concept scoring and Krippendorff's alpha

## 🗂️ Directory Structure

```
wenming/
├── Makefile                    # Main pipeline orchestration
├── README.md                   # Quick start guide
├── docs/PIPELINE.md           # This technical documentation
├── task2_cleanup_and_build_database/
│   ├── code/                   # Core analysis scripts
│   │   ├── _02_*.py           # Embedding generation
│   │   ├── _03_*.py           # Article extraction & keyword processing  
│   │   ├── _04_*.py           # Similarity calculations
│   │   └── _05_*.py           # LLM scoring and voting
│   ├── out/                    # Generated data files
│   │   ├── section2.csv       # Input document data (250 docs)
│   │   ├── *_embeddings.csv   # Vector embeddings (~17K text units)
│   │   ├── *_similarities.csv # Keyword similarity scores
│   │   └── concept_scores.db  # LLM reliability scores
│   ├── ref/                    # Reference data
│   │   └── category_keywords.csv  # 9 behavioral categories
│   └── hand/                   # Manual mappings
│       ├── unique_cities_original.json
│       └── unique_cities_fixed.json
└── task3_analyse_data/
    ├── code/
    │   ├── generate_final_figures.R    # Creates 3 publication figures
    │   └── calculate_krippendorff.py   # Reliability analysis
    └── graphs_and_maps/
        ├── figure1.png         # Article-level dominant categories
        ├── figure2.png         # Sentence-level salience scatter  
        └── figure3.png         # Section-level ridge plot
```

## 🔧 Core Scripts

### Embedding Generation (task2/code/)

| Script | Purpose | Input | Output |
|--------|---------|-------|---------|
| `_02_create_embeddings_for_sentences.py` | Generate sentence embeddings | section2.csv | section2_sentences_embeddings.csv |
| `_02a_create_embeddings_for_articles_resumable.py` | Generate article embeddings | section2_articles.csv | section2_articles_embeddings.csv |
| `_02b_create_embeddings_for_sections.py` | Generate section embeddings | section2.csv | section2_embeddings.csv |

### Keyword Processing (task2/code/)

| Script | Purpose | Input | Output |
|--------|---------|-------|---------|
| `_03_create_articles_from_sections.py` | Extract articles from sections | section2.csv | section2_articles.csv |
| `_03_create_embeddings_for_keywords.py` | Generate keyword embeddings | category_keywords.csv | category_keywords_embeddings.csv |
| `_04_calculate_individual_keyword_similarities.py` | Calculate similarities | All embeddings | *_similarities.csv |

### LLM Reliability Assessment (task2/code/)

| Script | Purpose | Input | Output |
|--------|---------|-------|---------|
| `_05_model_voting_and_scoring.py` | Multi-model concept scoring | section2.csv | concept_scores.db |

### Visualization (task3/code/)

| Script | Purpose | Input | Output |
|--------|---------|-------|---------|
| `generate_final_figures.R` | Create 3 publication figures | *_similarities.csv | figure1.png, figure2.png, figure3.png |
| `calculate_krippendorff.py` | Inter-rater reliability | concept_scores.db | Alpha statistics |

## 📊 Data Flow

### 1. Foundation Data
- **Input**: `task2_cleanup_and_build_database/out/section2.csv`
  - 250 documents with metadata (title, office, publish date, etc.)
  - Section 2 text content (behavioral injunctions)
  - Pre-filtered for documents with substantial Section 2 content

### 2. Text Unit Generation
```
section2.csv → Sentence splitting → 14,450 sentences
           → Article extraction → 2,800 articles  
           → Section preservation → 250 sections
```

### 3. Embedding Generation
```
Text units → OpenAI text-embedding-3-small → 1536-dimensional vectors
```
- **Sentence level**: 14,450 embeddings
- **Article level**: 2,800 embeddings  
- **Section level**: 250 embeddings

### 4. Keyword Similarity Analysis
```
Keywords (9 categories) → Embeddings → Cosine similarity with text embeddings
```
- **Output**: Similarity scores for each text unit × category combination
- **Filtering**: Analysis focuses on documents from 2019+ (excludes early pilot documents)

### 5. Visualization Pipeline
```
Similarity scores → R analysis → 3 publication figures
```

## 🎯 Publication Figures

### Figure 1: Article-Level Dominant Categories
- **Script**: `generate_final_figures.R::create_figure1()`
- **Data**: `section2_article_embeddings_with_individual_keyword_similarities.csv`
- **Visualization**: Horizontal bar chart showing percentage of articles where each category scores highest
- **Insight**: Shows which behavioral categories dominate at the regulatory article level

### Figure 2: Sentence-Level Salience Scatter
- **Script**: `generate_final_figures.R::create_figure2()`
- **Data**: `section2_sentence_embeddings_with_individual_keyword_similarities.csv`
- **Visualization**: Scatter plot of dominance rate vs. breadth for each category
- **Insight**: Maps category prominence across two dimensions (frequency vs. prevalence)

### Figure 3: Section-Level Ridge Plot
- **Script**: `generate_final_figures.R::create_figure3()`
- **Data**: `section2_embeddings_with_individual_keyword_similarities.csv`
- **Visualization**: Ridge plot showing similarity distributions by category
- **Insight**: Reveals distributional patterns of category relevance across documents

## 🔬 Reliability Assessment

### Krippendorff's Alpha Analysis
- **Purpose**: Assess inter-model reliability in concept relevance scoring
- **Models**: Multiple LLMs score text units on 9-point relevance scales
- **Analysis**: Find best-agreeing model subsets, calculate alpha for each concept
- **Levels**: Sentence, article, and section granularity

### Database Schema (concept_scores.db)
```sql
-- scores_wide: Sentence-level scores
CREATE TABLE scores_wide (
    row_index INTEGER,
    document_id TEXT,
    model TEXT,
    business_professional_score REAL,
    revolutionary_culture_score REAL,
    -- ... other category scores
);

-- scores_wide_articles: Article-level scores
-- scores_wide_sections: Section-level scores
```

## ⚙️ Build System (Makefile)

### Target Categories

**Main Targets**
- `all` - Generate complete publication package (default)
- `final-figures` - Generate 3 publication figures only
- `publication-essentials` - Figures + reliability analysis  

**Data Preparation**
- `embeddings` - Generate all embeddings (expensive)
- `keyword-similarities` - Generate similarity calculations

**Reliability Analysis**
- `krippendorff-all` - Complete reliability assessment
- `krippendorff-sentence` - Sentence-level alpha only
- `krippendorff-article` - Article-level alpha only
- `krippendorff-section` - Section-level alpha only

**Maintenance**
- `verify` - Check pipeline status
- `clean-*` - Various cleaning options
- `help` - Show all available targets

### Dependency Management

The Makefile implements proper dependency tracking:
- Embeddings are expensive and cached (won't regenerate unnecessarily)
- Similarity calculations depend on both embeddings and keyword files
- Visualizations depend on similarity data
- LLM scoring is independent and creates its own database

### File Size Management

Large files (>100MB) are handled gracefully:
- Embeddings files are automatically cached
- Scripts check for existing files before regenerating
- Clear warnings for expensive operations

## 🔄 Extending the Analysis

### Adding New Behavioral Categories

1. **Update keywords**: Edit `task2_cleanup_and_build_database/ref/category_keywords.csv`
2. **Regenerate embeddings**: `make clean-similarities && make keyword-similarities`  
3. **Update visualizations**: Modify color palettes in `generate_final_figures.R`
4. **Regenerate figures**: `make final-figures`

### Adding New Analysis Levels

1. **Create text splitting script**: Follow pattern of `_03_create_articles_from_sections.py`
2. **Generate embeddings**: Follow pattern of `_02a_create_embeddings_for_articles_resumable.py`
3. **Calculate similarities**: Extend `_04_calculate_individual_keyword_similarities.py`
4. **Add visualization**: Extend `generate_final_figures.R`

### Performance Optimization

- **Embeddings**: Use caching, batch processing
- **Similarities**: Leverage numba/fastdist for speed
- **LLM calls**: Implement rate limiting, async processing
- **Visualizations**: Use data.table for large datasets

## 🐛 Common Issues and Solutions

### Missing Dependencies
```bash
# Check system status
make verify

# Install missing components
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv
# R packages auto-install via librarian::shelf()
```

### API Issues
```bash
# Set OpenAI API key
export OPENAI_API_KEY=your_key_here

# Test without API (figures only)
make final-figures
```

### File Path Issues
```bash
# Always run from project root
cd /path/to/wenming
make verify
```

### Memory Issues
- Large embeddings files may require 8GB+ RAM
- Consider processing in chunks for very large corpora
- Monitor disk space (embeddings can be several GB)

## 📈 Performance Metrics

### Typical Processing Times
- **Sentence embeddings**: ~30 minutes (14K sentences)
- **Keyword similarities**: ~10 minutes (cached embeddings)
- **Figure generation**: ~2 minutes
- **LLM reliability assessment**: ~2 hours (with API rate limits)

### Resource Requirements
- **RAM**: 8GB recommended for full pipeline
- **Storage**: 5GB for complete analysis outputs
- **API costs**: ~$5-10 for complete reliability analysis

## 🔮 Future Enhancements

### Potential Improvements
1. **Interactive visualizations** using Shiny or Observable
2. **Additional similarity metrics** (e.g., semantic clustering)
3. **Temporal analysis** of regulatory evolution
4. **Cross-provincial comparison** tools
5. **Multi-language support** for comparative analysis

### Research Extensions
1. **Causal analysis** of regulatory effectiveness
2. **Network analysis** of concept relationships
3. **Predictive modeling** of regulatory trends
4. **Comparative analysis** with other policy domains