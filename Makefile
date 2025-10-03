# Pragmatic Moral Governance Analysis Pipeline - Streamlined for Publication
# Keyword-based analysis pipeline for "Promotion of Civilised Behaviour" research

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tools
RSCRIPT = Rscript
UV = uv run

# Directories
TASK2_DIR = task2_cleanup_and_build_database
TASK3_DIR = task3_analyse_data
EMBEDDINGS_OUT = $(TASK2_DIR)/out
GRAPHS_OUT = $(TASK3_DIR)/graphs_and_maps

# Key input files
INPUT_DATA = $(EMBEDDINGS_OUT)/section2.csv
KEYWORDS_CSV ?= $(TASK2_DIR)/ref/category_keywords.csv
ATTRIBUTION_OUT = $(TASK3_DIR)/out/keyword_attribution

# ============================================================================
# SHARED DATA FILES
# ============================================================================

# Shared embeddings
SENTENCE_EMBEDDINGS = $(EMBEDDINGS_OUT)/section2_sentences_embeddings.csv
ARTICLE_EMBEDDINGS = $(EMBEDDINGS_OUT)/section2_articles_embeddings.csv
SECTION_EMBEDDINGS = $(EMBEDDINGS_OUT)/section2_embeddings.csv
ARTICLES_CSV = $(EMBEDDINGS_OUT)/section2_articles.csv

# ============================================================================
# 🟠 KEYWORD ANALYSIS (Individual Averaging Method)
# ============================================================================

# Keyword output files
K_KEYWORD_EMBEDDINGS = $(EMBEDDINGS_OUT)/category_keywords_embeddings.csv
K_SENTENCE_SIMILARITIES = $(EMBEDDINGS_OUT)/section2_sentence_embeddings_with_individual_keyword_similarities.csv
K_ARTICLE_SIMILARITIES = $(EMBEDDINGS_OUT)/section2_article_embeddings_with_individual_keyword_similarities.csv
K_SECTION_SIMILARITIES = $(EMBEDDINGS_OUT)/section2_embeddings_with_individual_keyword_similarities.csv

# K-1: Create keyword embeddings
$(K_KEYWORD_EMBEDDINGS): $(KEYWORDS_CSV) $(TASK2_DIR)/code/_03_create_embeddings_for_keywords.py
	$(TASK2_DIR)/code/_03_create_embeddings_for_keywords.py "$(KEYWORDS_CSV)"
	@echo "✅ K-1: Generated keyword embeddings using: $(KEYWORDS_CSV)"

# K-2: Calculate individual keyword similarities (all levels)
$(K_SENTENCE_SIMILARITIES) $(K_ARTICLE_SIMILARITIES) $(K_SECTION_SIMILARITIES): $(K_KEYWORD_EMBEDDINGS) $(SENTENCE_EMBEDDINGS) $(ARTICLE_EMBEDDINGS) $(SECTION_EMBEDDINGS)
	$(TASK2_DIR)/code/_04_calculate_individual_keyword_similarities.py
	@echo "✅ K-2: Calculated individual keyword similarities (all levels)"

# ============================================================================
# 🎯 FINAL FIGURES (Publication Ready)
# ============================================================================

# Final 3 figures for publication
FINAL_FIGURES = $(GRAPHS_OUT)/figure1.png $(GRAPHS_OUT)/figure2.png $(GRAPHS_OUT)/figure3.png

# Generate final figures (figure1, figure2, figure3)
$(FINAL_FIGURES): $(K_SENTENCE_SIMILARITIES) $(K_ARTICLE_SIMILARITIES) $(K_SECTION_SIMILARITIES)
	cd $(TASK3_DIR)/code && $(RSCRIPT) generate_final_figures.R
	@echo "✅ Generated final 3 figures for publication"

# ============================================================================
# 📊 RELIABILITY ANALYSIS (Krippendorff's Alpha)
# ============================================================================

# Concept scores database (prerequisite for krippendorff)
CONCEPT_SCORES_DB = $(EMBEDDINGS_OUT)/concept_scores.db

# Generate concept scores database (LLM voting and scoring)
$(CONCEPT_SCORES_DB): $(INPUT_DATA)
	$(TASK2_DIR)/code/_05_model_voting_and_scoring.py --granularity sentence
	$(TASK2_DIR)/code/_05_model_voting_and_scoring.py --granularity article
	$(TASK2_DIR)/code/_05_model_voting_and_scoring.py --granularity section
	@echo "✅ Generated concept scores database with LLM voting and scoring"

# Krippendorff's alpha analysis
krippendorff-sentence: $(CONCEPT_SCORES_DB)
	cd $(TASK3_DIR)/code && ./calculate_krippendorff.py --granularity sentence --subset-size 3
	@echo "✅ Calculated Krippendorff's alpha for sentence-level analysis"

krippendorff-article: $(CONCEPT_SCORES_DB)
	cd $(TASK3_DIR)/code && ./calculate_krippendorff.py --granularity article --subset-size 3
	@echo "✅ Calculated Krippendorff's alpha for article-level analysis"

krippendorff-section: $(CONCEPT_SCORES_DB)
	cd $(TASK3_DIR)/code && ./calculate_krippendorff.py --granularity section --subset-size 3
	@echo "✅ Calculated Krippendorff's alpha for section-level analysis"

# Run all krippendorff analyses
krippendorff-all: krippendorff-sentence krippendorff-article krippendorff-section

# ============================================================================
# SHARED PREREQUISITES
# ============================================================================

# SHARED-1: Create articles from sections
$(ARTICLES_CSV): $(INPUT_DATA) $(TASK2_DIR)/code/_03_create_articles_from_sections.py
	$(TASK2_DIR)/code/_03_create_articles_from_sections.py
	@echo "✅ SHARED-1: Created articles from sections"

# SHARED-2: Create sentence embeddings
$(SENTENCE_EMBEDDINGS):
	@if [ ! -f "$(SENTENCE_EMBEDDINGS)" ]; then \
		echo "Generating sentence embeddings..."; \
		$(TASK2_DIR)/code/_02_create_embeddings_for_sentences.py; \
		echo "✅ SHARED-2: Generated sentence embeddings"; \
	else \
		echo "✅ SHARED-2: Sentence embeddings already exist (skipping)"; \
	fi

# SHARED-3: Create article embeddings
$(ARTICLE_EMBEDDINGS): $(ARTICLES_CSV)
	@if [ ! -f "$(ARTICLE_EMBEDDINGS)" ]; then \
		echo "Generating article embeddings..."; \
		$(TASK2_DIR)/code/_02a_create_embeddings_for_articles_resumable.py; \
		echo "✅ SHARED-3: Generated article embeddings"; \
	else \
		echo "✅ SHARED-3: Article embeddings already exist (skipping)"; \
	fi

# SHARED-4: Create section embeddings
$(SECTION_EMBEDDINGS):
	@if [ ! -f "$(SECTION_EMBEDDINGS)" ]; then \
		echo "Generating section embeddings..."; \
		$(TASK2_DIR)/code/_02b_create_embeddings_for_sections.py; \
		echo "✅ SHARED-4: Generated section embeddings"; \
	else \
		echo "✅ SHARED-4: Section embeddings already exist (skipping)"; \
	fi

# ============================================================================
# MAIN TARGETS
# ============================================================================

# Default target: generate publication essentials
all: publication-essentials

# Generate final 3 figures only (streamlined for publication)
final-figures: $(FINAL_FIGURES)

# Publication essentials (figures + reliability analysis)
publication-essentials: final-figures krippendorff-all

# Generate all keyword similarities (prerequisite for figures)
keyword-similarities: $(K_SENTENCE_SIMILARITIES) $(K_ARTICLE_SIMILARITIES) $(K_SECTION_SIMILARITIES)

# Generate all embeddings (expensive but foundational)
embeddings: $(SENTENCE_EMBEDDINGS) $(ARTICLE_EMBEDDINGS) $(SECTION_EMBEDDINGS)

# ============================================================================
# VERIFICATION AND TESTING
# ============================================================================

verify:
	@echo "=== 📥 Input Dependencies ==="
	@[ -f "$(INPUT_DATA)" ] && echo "✅ Input data exists" || echo "❌ Input data missing: $(INPUT_DATA)"
	@[ -f "$(KEYWORDS_CSV)" ] && echo "✅ Keywords CSV exists" || echo "❌ Keywords CSV missing: $(KEYWORDS_CSV)"
	@echo ""
	@echo "=== 🗄️ Embeddings Status ==="
	@[ -f "$(SENTENCE_EMBEDDINGS)" ] && echo "✅ Sentence embeddings exist" || echo "❌ Sentence embeddings missing"
	@[ -f "$(ARTICLE_EMBEDDINGS)" ] && echo "✅ Article embeddings exist" || echo "❌ Article embeddings missing"
	@[ -f "$(SECTION_EMBEDDINGS)" ] && echo "✅ Section embeddings exist" || echo "❌ Section embeddings missing"
	@echo ""
	@echo "=== 🟠 Keyword Analysis Status ==="
	@[ -f "$(K_KEYWORD_EMBEDDINGS)" ] && echo "✅ Keyword embeddings exist" || echo "❌ Keyword embeddings missing"
	@[ -f "$(K_SENTENCE_SIMILARITIES)" ] && echo "✅ Sentence keyword similarities exist" || echo "❌ Sentence keyword similarities missing"
	@[ -f "$(K_ARTICLE_SIMILARITIES)" ] && echo "✅ Article keyword similarities exist" || echo "❌ Article keyword similarities missing"
	@[ -f "$(K_SECTION_SIMILARITIES)" ] && echo "✅ Section keyword similarities exist" || echo "❌ Section keyword similarities missing"
	@echo ""
	@echo "=== 🎯 Publication Outputs ==="
	@[ -f "$(GRAPHS_OUT)/figure1.png" ] && echo "✅ Figure 1 exists" || echo "❌ Figure 1 missing"
	@[ -f "$(GRAPHS_OUT)/figure2.png" ] && echo "✅ Figure 2 exists" || echo "❌ Figure 2 missing"
	@[ -f "$(GRAPHS_OUT)/figure3.png" ] && echo "✅ Figure 3 exists" || echo "❌ Figure 3 missing"
	@[ -f "$(CONCEPT_SCORES_DB)" ] && echo "✅ Concept scores database exists" || echo "❌ Concept scores database missing"

# ============================================================================
# CLEAN TARGETS
# ============================================================================

# Clean only visualizations (preserve expensive embeddings)
clean-graphs:
	rm -rf $(GRAPHS_OUT)/*.png
	@echo "🧹 Cleaned visualization outputs (preserved embeddings)"

# Clean keyword similarities (forces recalculation)
clean-similarities:
	rm -f $(EMBEDDINGS_OUT)/*similarities*.csv
	@echo "🧹 Cleaned similarity files"

# Clean concept scores database
clean-scores:
	rm -f $(CONCEPT_SCORES_DB)
	@echo "🧹 Cleaned concept scores database"

# Clean embeddings - requires confirmation (VERY expensive to regenerate)
clean-embeddings:
	@echo "⚠️  WARNING: This will delete all embeddings which are VERY expensive to regenerate!"
	@echo "This includes:"
	@echo "  - Sentence embeddings (~14,450 items)"
	@echo "  - Article embeddings (~2,800 items)"
	@echo "  - Section embeddings (~250 items)"
	@echo "  - Keyword embeddings"
	@echo ""
	@read -p "Are you sure you want to continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -f $(EMBEDDINGS_OUT)/*_embeddings*.csv; \
		rm -rf $(EMBEDDINGS_OUT)/keyword_embeddings_cache/; \
		echo "⚠️  Cleaned embeddings (expensive to regenerate!)"; \
	else \
		echo "❌ Clean cancelled"; \
	fi

# Clean everything - requires confirmation
clean-all:
	@echo "⚠️  WARNING: This will delete ALL generated files including expensive embeddings!"
	@echo ""
	@read -p "Are you sure you want to continue? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf $(GRAPHS_OUT)/*.png; \
		rm -f $(EMBEDDINGS_OUT)/*_embeddings*.csv; \
		rm -f $(EMBEDDINGS_OUT)/*similarities*.csv; \
		rm -f $(CONCEPT_SCORES_DB); \
		rm -rf $(EMBEDDINGS_OUT)/similarity_cache*/; \
		rm -rf $(EMBEDDINGS_OUT)/keyword_embeddings_cache/; \
		echo "🧹 Cleaned ALL generated files"; \
	else \
		echo "❌ Clean cancelled"; \
	fi

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "🔬 Pragmatic Moral Governance Analysis Pipeline"
	@echo "================================================"
	@echo ""
	@echo "📑 This pipeline analyzes Chinese 'Promotion of Civilised Behaviour' documents"
	@echo "   using keyword-based similarity analysis and LLM reliability assessment."
	@echo ""
	@echo "🎯 Main Targets:"
	@echo "  all                    - Generate publication essentials (default)"
	@echo "  publication-essentials - Generate figures + reliability analysis"
	@echo "  final-figures          - Generate ONLY the 3 final publication figures"
	@echo "  krippendorff-all       - Calculate Krippendorff's alpha for all levels"
	@echo "  keyword-similarities   - Generate keyword similarity data"
	@echo "  embeddings            - Generate all embeddings (expensive!)"
	@echo "  verify                - Check status of all pipeline components"
	@echo "  help                  - Show this help message"
	@echo ""
	@echo "📊 Individual Reliability Targets:"
	@echo "  krippendorff-sentence - Alpha analysis for sentence level"
	@echo "  krippendorff-article  - Alpha analysis for article level"
	@echo "  krippendorff-section  - Alpha analysis for section level"
	@echo ""
	@echo "🧹 Clean Targets:"
	@echo "  clean-graphs          - Remove only visualizations (safe)"
	@echo "  clean-similarities    - Remove similarity caches"
	@echo "  clean-scores          - Remove concept scores database"
	@echo "  clean-embeddings      - Remove embeddings (⚠️  expensive! requires confirmation)"
	@echo "  clean-all             - Remove everything (⚠️  very expensive! requires confirmation)"
	@echo ""
	@echo "📋 Quick Start:"
	@echo "  make verify           # Check what files exist"
	@echo "  make final-figures    # Generate 3 publication figures"
	@echo "  make all              # Generate everything needed for publication"
	@echo ""
	@echo "⚙️ System Requirements:"
	@echo "  - R with required packages (automatically installed)"
	@echo "  - Python 3.11+ with uv for dependency management"
	@echo "  - OpenAI API key in .env file (for LLM reliability analysis only)"
	@echo "  - Copy .env.example to .env and configure paths"

.PHONY: all final-figures publication-essentials krippendorff-all krippendorff-sentence krippendorff-article krippendorff-section keyword-similarities embeddings verify help \
        clean-all clean-graphs clean-similarities clean-scores clean-embeddings