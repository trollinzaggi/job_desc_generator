# Job Description Generator - Corpus Analysis

Phase 1 analysis tools for evaluating JD corpus quality, structure, and content patterns.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your JSON files in jd_data/

# 3. Discover your data schema
python run_analysis.py --discover

# 4. Configure config.py (see Configuration section below)

# 5. Run analysis
python run_analysis.py --all -o analysis_output
# OR use the interactive notebook
jupyter notebook notebooks/corpus_analysis.ipynb
```

---

## Configuration

Configuration is centralized in `config.py` with **two parts**:

### Part 1: Field Mapping

Maps your source JSON fields to standard names:

```python
JD_FIELD_MAPPING = FieldMapping(
    jd_text="content.description",      # Path to JD text
    jd_id="job_id",                      # Unique identifier
    org_unit="metadata.business_unit",   # Organization unit
    level="metadata.grade",              # Job level
    title="metadata.title",              # Job title
    
    # Add custom fields for additional data
    custom_fields={
        "jd_expertise": "content.expertise_section",
        "rank": "metadata.rank",
    }
)
```

### Part 2: Analysis Configuration

Specifies **which mapped fields** to use for each analysis:

```python
ANALYSIS_CONFIG = AnalysisConfig(
    # Which text field to analyze (default: "jd_text")
    primary_text_field="jd_text",
    
    # Analyze multiple text fields
    additional_text_fields=["jd_expertise"],
    
    # Stratification dimensions for sampling
    stratify_by_primary="org_unit",
    stratify_by_secondary="level",
    
    # Fields to analyze against clusters
    cluster_metadata_fields=["org_unit", "level", "title"],
)
```

### Common Configuration Changes

| Want to... | Change this |
|------------|-------------|
| Analyze expertise section instead of full JD | `primary_text_field="jd_expertise"` |
| Stratify by rank instead of org_unit | `stratify_by_primary="rank"` |
| Add hiring_manager to cluster analysis | Add to `cluster_metadata_fields` |
| Analyze both full JD and expertise | `additional_text_fields=["jd_expertise"]` |

See [docs/FIELD_MAPPING.md](docs/FIELD_MAPPING.md) for complete configuration guide.

---

## Project Structure

```
job_desc_generator/
├── src/
│   ├── data_loaders/           # Data loading utilities
│   │   ├── base_loader.py      # Base class with field mapping
│   │   ├── json_file_loader.py # Load from JSON files
│   │   ├── cosmos_loader.py    # Load from Azure Cosmos DB
│   │   └── schema_discovery.py # Schema analysis
│   └── analysis/               # Analysis modules
│       ├── data_cleaner.py     # Data cleaning
│       ├── quality_baseline.py # Phase 1.1: Quality evaluation
│       ├── structure_analysis.py # Phase 1.2: Structure parsing
│       └── content_clustering.py # Phase 1.3: Clustering
├── notebooks/
│   └── corpus_analysis.ipynb   # Interactive analysis notebook
├── docs/
│   └── FIELD_MAPPING.md        # Configuration guide
├── config.py                   # YOUR CONFIGURATION (edit this)
├── run_analysis.py             # CLI entry point
├── jd_data/                    # Your data goes here
└── requirements.txt
```

---

## Analysis Phases

### Phase 1.1: JD Quality Baseline

Creates stratified sample for human evaluation.

```bash
python run_analysis.py --phase 1.1

# After filling in the CSV:
python run_analysis.py --analyze-quality
```

- Uses `stratify_by_primary` and `stratify_by_secondary` from config
- Exports fields specified in `quality_export_fields`

### Phase 1.2: Structural Consistency

Parses JDs to extract section headers and analyze patterns.

```bash
python run_analysis.py --phase 1.2
```

- Analyzes `primary_text_field` from config
- Also analyzes each field in `additional_text_fields`

### Phase 1.3: Content Clustering

Generates embeddings and clusters JDs by content similarity.

```bash
python run_analysis.py --phase 1.3
```

- Embeds `primary_text_field` from config
- Analyzes clusters against `cluster_metadata_fields`
- Computes purity using `cluster_purity_field`

---

## Output Files

```
analysis_output/
├── phase_1_1_quality/
│   ├── jd_quality_evaluation.csv    # Fill this in
│   ├── jd_quality_evaluation.json
│   └── quality_analysis_results.json
├── phase_1_2_structure/
│   ├── parsed_jd_structures.json
│   ├── structure_consistency.json
│   └── structure_consistency_{field}.json  # For additional text fields
└── phase_1_3_clustering/
    ├── embeddings.npy
    ├── cluster_assignments.csv
    ├── cluster_composition.json
    ├── cluster_archetypes.json
    ├── visualization_data.csv
    └── cluster_assignments_{field}.csv  # For additional text fields
```

---

## Requirements

**Core:** pandas, numpy, scikit-learn

**Embeddings:** sentence-transformers (local/free) OR openai/cohere (API)

**Visualization:** matplotlib, seaborn, plotly, umap-learn

**Notebook:** jupyter

```bash
pip install -r requirements.txt
```

---

## Programmatic Usage

```python
from src.data_loaders import JSONFileLoader
from src.analysis import (
    DataCleaner,
    StructureAnalyzer,
    EmbeddingGenerator,
    ContentClusterer,
    ClusterAnalyzer,
)
from config import JSON_CONFIG, JD_FIELD_MAPPING, ANALYSIS_CONFIG

# Load data
loader = JSONFileLoader(
    data_path=JSON_CONFIG["data_path"],
    field_mapping=JD_FIELD_MAPPING,
)
df = loader.load_as_dataframe()

# Clean using configured text field
cleaner = DataCleaner(
    text_field=ANALYSIS_CONFIG.primary_text_field,
    id_field=ANALYSIS_CONFIG.id_field,
)
df_clean, stats = cleaner.clean(df)

# Structure analysis
analyzer = StructureAnalyzer()
analyzer.parse_corpus(
    df_clean,
    text_field=ANALYSIS_CONFIG.primary_text_field,
    id_field=ANALYSIS_CONFIG.id_field,
)
analyzer.print_structure_report()

# Clustering
embedder = EmbeddingGenerator()
embeddings, ids = embedder.embed_dataframe(
    df_clean,
    text_field=ANALYSIS_CONFIG.primary_text_field,
    id_field=ANALYSIS_CONFIG.id_field,
)

clusterer = ContentClusterer(embeddings, ids)
result = clusterer.kmeans(n_clusters=20)

# Analyze against configured metadata
cluster_df = clusterer.get_cluster_assignments("kmeans_20")
cluster_analyzer = ClusterAnalyzer(df_clean, cluster_df, id_field=ANALYSIS_CONFIG.id_field)
cluster_analyzer.print_cluster_report(
    metadata_fields=ANALYSIS_CONFIG.get_available_cluster_fields(list(df_clean.columns))
)
```
