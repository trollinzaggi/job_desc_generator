# Configuration Guide

This guide explains how to configure the analysis pipeline in `config.py`.

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: Field Mapping](#part-1-field-mapping)
3. [Part 2: Analysis Configuration](#part-2-analysis-configuration)
4. [Common Scenarios](#common-scenarios)
5. [Validation](#validation)
6. [Reference](#reference)

---

## Overview

Configuration in `config.py` has **two parts**:

| Part | Purpose | When to Change |
|------|---------|----------------|
| `JD_FIELD_MAPPING` | Maps your JSON fields → standard names | When your data structure changes |
| `ANALYSIS_CONFIG` | Tells analysis which fields to use | When you want to analyze different dimensions |

**Key insight**: You map fields ONCE, then change analysis behavior by editing `ANALYSIS_CONFIG`.

---

## Part 1: Field Mapping

Maps your source JSON field paths to standard names.

### Basic Example

Your JSON:
```json
{
  "job_id": "12345",
  "content": {
    "full_description": "We are looking for...",
    "expertise": "Must have 5+ years in Python..."
  },
  "metadata": {
    "business_unit": "Engineering",
    "grade": "L5",
    "job_title": "Software Engineer",
    "rank": "Senior",
    "created_date": "2024-01-15"
  }
}
```

Your config:
```python
JD_FIELD_MAPPING = FieldMapping(
    # Map standard names to your JSON paths (use dot notation)
    jd_text="content.full_description",
    jd_id="job_id",
    org_unit="metadata.business_unit",
    level="metadata.grade",
    title="metadata.job_title",
    posting_date="metadata.created_date",
    
    # Add ANY other fields you want available
    custom_fields={
        "jd_expertise": "content.expertise",
        "rank": "metadata.rank",
        "hiring_manager": "metadata.manager_name",
    }
)
```

### Standard Fields

| Field | Description | Used For |
|-------|-------------|----------|
| `jd_text` | Main JD text | Default text for all analysis |
| `jd_id` | Unique identifier | Tracking records |
| `org_unit` | Organization/business unit | Default stratification |
| `level` | Job level/grade | Default stratification |
| `title` | Job title | Cluster interpretation |
| `function` | Job function/category | Cluster analysis |
| `department` | Department | Additional grouping |
| `location` | Work location | Geographic analysis |
| `posting_date` | Posting date | **Recency analysis** |
| `status` | Active/closed | Filtering |

### Custom Fields

Add ANY additional fields using `custom_fields`:

```python
custom_fields={
    "jd_expertise": "sections.expertise",
    "jd_requirements": "sections.min_requirements",
    "rank": "employee_info.rank",
    "hiring_manager": "metadata.manager",
    "team_size": "metadata.team_info.size",
    "salary_band": "compensation.band",
}
```

These become columns in your DataFrame that you can use in `ANALYSIS_CONFIG`.

---

## Part 2: Analysis Configuration

Specifies **which mapped fields** to use for each analysis operation.

### Full Configuration

```python
ANALYSIS_CONFIG = AnalysisConfig(
    # TEXT ANALYSIS
    primary_text_field="jd_text",           # Main text field to analyze
    additional_text_fields=[],               # Extra text fields to analyze
    
    # ID
    id_field="jd_id",
    
    # STRATIFICATION (Phase 1.1)
    stratify_by_primary="org_unit",          # Primary sampling dimension
    stratify_by_secondary="level",           # Secondary sampling dimension
    
    # RECENCY (Phase 1.1) - NEW
    date_field="posting_date",               # Field for recency analysis
    
    # CLUSTERING (Phase 1.3)
    cluster_metadata_fields=[                # Fields to analyze against clusters
        "org_unit", "level", "title", "function"
    ],
    cluster_purity_field="org_unit",         # Field for purity calculation
    
    # SECTION-LEVEL EMBEDDINGS (Phase 1.3) - NEW
    section_types_to_embed=[                 # Sections to embed separately
        "responsibilities", "required_qualifications", "summary"
    ],
    
    # QUALITY EXPORT (Phase 1.1)
    quality_export_fields=[                  # Fields to show in evaluation
        "title", "org_unit", "level"
    ],
)
```

### What Each Setting Does

| Setting | Description | Analysis Phase |
|---------|-------------|----------------|
| `primary_text_field` | Text field for parsing/embedding | 1.2, 1.3 |
| `additional_text_fields` | Extra text fields to analyze | 1.2, 1.3 |
| `id_field` | ID field for tracking | All |
| `stratify_by_primary` | Primary stratification dimension | 1.1 |
| `stratify_by_secondary` | Secondary stratification dimension | 1.1 |
| `date_field` | Date field for recency analysis | 1.1 |
| `cluster_metadata_fields` | Fields to analyze against clusters | 1.3 |
| `cluster_purity_field` | Field for purity calculation | 1.3 |
| `section_types_to_embed` | Section types for section-level embeddings | 1.3 |
| `quality_export_fields` | Fields in quality evaluation export | 1.1 |

---

## Common Scenarios

### Scenario 1: Analyze `jd_expertise` instead of `jd_text`

**Goal**: Run structure and clustering analysis on just the expertise section.

```python
# Step 1: Make sure jd_expertise is mapped (in Part 1)
JD_FIELD_MAPPING = FieldMapping(
    jd_text="content.full_description",
    custom_fields={
        "jd_expertise": "content.expertise_section",
    }
)

# Step 2: Tell analysis to use it (in Part 2)
ANALYSIS_CONFIG = AnalysisConfig(
    primary_text_field="jd_expertise",  # ← Changed from "jd_text"
)
```

### Scenario 2: Stratify by `rank` and `title` instead of `org_unit` and `level`

**Goal**: Sample JDs proportionally by rank × title combinations.

```python
# Step 1: Map rank and title (in Part 1)
JD_FIELD_MAPPING = FieldMapping(
    title="metadata.job_title",
    custom_fields={
        "rank": "metadata.rank",  # Senior, Manager, Director, etc.
    }
)

# Step 2: Configure stratification (in Part 2)
ANALYSIS_CONFIG = AnalysisConfig(
    stratify_by_primary="rank",     # ← Changed from "org_unit"
    stratify_by_secondary="title",  # ← Changed from "level"
)
```

### Scenario 3: Enable recency analysis

**Goal**: Analyze whether JD quality correlates with posting date.

```python
# Step 1: Map the date field (in Part 1)
JD_FIELD_MAPPING = FieldMapping(
    posting_date="metadata.created_date",  # or "posted_at", etc.
)

# Step 2: Configure the date field (in Part 2)
ANALYSIS_CONFIG = AnalysisConfig(
    date_field="posting_date",  # Must match mapped field name
)
```

The quality report will now include:
- Quality trends by quarter/month
- Recent vs older JD comparison
- Correlation coefficient between date and quality

### Scenario 4: Analyze clusters against different metadata

**Goal**: See how clusters break down by rank, hiring_manager, and department.

```python
# Step 1: Map the fields (in Part 1)
JD_FIELD_MAPPING = FieldMapping(
    custom_fields={
        "rank": "metadata.rank",
        "hiring_manager": "metadata.manager_name",
        "department": "metadata.department",
    }
)

# Step 2: Include in cluster analysis (in Part 2)
ANALYSIS_CONFIG = AnalysisConfig(
    cluster_metadata_fields=[
        "rank",
        "title",
        "hiring_manager",
        "department",
    ],
    cluster_purity_field="rank",  # Also changed purity field
)
```

### Scenario 5: Analyze BOTH full JD and expertise section

**Goal**: Compare structure/clustering results between full JD and expertise section.

```python
# Step 1: Map both (in Part 1)
JD_FIELD_MAPPING = FieldMapping(
    jd_text="content.full_description",
    custom_fields={
        "jd_expertise": "content.expertise",
        "jd_requirements": "content.requirements",
    }
)

# Step 2: Include additional fields (in Part 2)
ANALYSIS_CONFIG = AnalysisConfig(
    primary_text_field="jd_text",
    additional_text_fields=[
        "jd_expertise",
        "jd_requirements",
    ],
)
```

This will:
- Run main analysis on `jd_text`
- Also run structure analysis on `jd_expertise` → `structure_consistency_jd_expertise.json`
- Also run clustering on `jd_expertise` → `cluster_assignments_jd_expertise.csv`
- Same for `jd_requirements`

### Scenario 6: Section-level embeddings

**Goal**: Compare how JDs cluster when looking at specific sections only.

```python
ANALYSIS_CONFIG = AnalysisConfig(
    section_types_to_embed=[
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
    ],
)
```

This creates separate cluster assignments for each section type:
- `cluster_assignments_section_responsibilities.csv`
- `cluster_assignments_section_required_qualifications.csv`

---

## Validation

The config validates itself against your actual data:

```python
# In notebook or code
validation = ANALYSIS_CONFIG.validate(list(df.columns))

print("Errors:", validation["errors"])    # Will cause failure
print("Warnings:", validation["warnings"]) # Will continue with reduced functionality
```

### Example Validation Output

```
Errors: []
Warnings: [
    "stratify_by_secondary 'level' not found - will use single-dimension sampling",
    "date_field 'posting_date' not found - recency analysis will be skipped",
    "cluster_metadata_fields missing: ['function'] - will be skipped"
]
```

---

## Reference

### Quick Examples

**Minimum config (just analyze jd_text):**
```python
JD_FIELD_MAPPING = FieldMapping(
    jd_text="description",
    jd_id="id",
)

ANALYSIS_CONFIG = AnalysisConfig()  # All defaults
```

**Analyze expertise section, stratify by rank, with recency:**
```python
JD_FIELD_MAPPING = FieldMapping(
    jd_text="description",
    jd_id="id",
    posting_date="created_at",
    custom_fields={
        "jd_expertise": "expertise_section",
        "rank": "employee_rank",
    }
)

ANALYSIS_CONFIG = AnalysisConfig(
    primary_text_field="jd_expertise",
    stratify_by_primary="rank",
    stratify_by_secondary="title",
    date_field="posting_date",
    cluster_metadata_fields=["rank", "title"],
    cluster_purity_field="rank",
)
```

**Full config with all features:**
```python
JD_FIELD_MAPPING = FieldMapping(
    jd_text="content.full",
    jd_id="job_id",
    org_unit="meta.org",
    level="meta.grade",
    title="meta.title",
    posting_date="meta.created_at",
    custom_fields={
        "jd_expertise": "content.expertise",
        "jd_requirements": "content.requirements",
        "rank": "meta.rank",
        "hiring_manager": "meta.manager",
    }
)

ANALYSIS_CONFIG = AnalysisConfig(
    primary_text_field="jd_text",
    additional_text_fields=["jd_expertise", "jd_requirements"],
    id_field="jd_id",
    stratify_by_primary="org_unit",
    stratify_by_secondary="level",
    date_field="posting_date",
    cluster_metadata_fields=["org_unit", "level", "title", "rank", "hiring_manager"],
    cluster_purity_field="org_unit",
    section_types_to_embed=["responsibilities", "required_qualifications", "summary"],
    quality_export_fields=["title", "org_unit", "level", "rank"],
)
```

### What You DON'T Need to Change

When you edit `ANALYSIS_CONFIG`, you don't need to:
- Edit any Python files in `src/`
- Pass different parameters to functions
- Modify the notebook cells

The CLI and notebook automatically use your config.

---

## New Features Summary

### Recency Analysis (Phase 1.1)

The quality evaluation now analyzes correlation between JD quality and posting date:

- **By Quarter/Month**: Average quality scores grouped by time period
- **Trend Comparison**: Recent vs older JDs comparison
- **Correlation**: Statistical correlation between date and quality score

Enable by mapping `posting_date` and setting `date_field` in config.

### Section-Level Embeddings (Phase 1.3)

In addition to full-JD embeddings, the pipeline now supports embedding specific sections:

- Embed only "responsibilities" sections across all JDs
- Embed only "required_qualifications" sections
- Compare how JDs cluster differently based on section

This helps identify whether JDs are similar in structure but different in content for specific sections.
