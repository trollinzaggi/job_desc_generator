"""
Configuration for JD Corpus Analysis

This file has TWO parts:
1. JD_FIELD_MAPPING - Maps your JSON fields to standard names
2. ANALYSIS_CONFIG - Tells analysis which fields to use

WORKFLOW:
1. Run: python run_analysis.py --discover
2. Edit JD_FIELD_MAPPING below with your field paths
3. Edit ANALYSIS_CONFIG to customize analysis behavior
4. Run: python run_analysis.py --all

See docs/FIELD_MAPPING.md for detailed documentation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from src.data_loaders import FieldMapping


# =============================================================================
# PART 1: FIELD MAPPING
# =============================================================================
# Maps your source JSON fields to standard names.
# Use dot notation for nested fields (e.g., "metadata.org_unit")
#
# Run `python run_analysis.py --discover` to see your schema.

JD_FIELD_MAPPING = FieldMapping(
    # ----- PRIMARY FIELDS (required) -----
    jd_text=None,                    # Path to main JD text
    jd_id=None,                      # Path to unique identifier
    
    # ----- METADATA FIELDS (recommended) -----
    org_unit=None,                   # Organization unit / business unit
    level=None,                      # Job level / grade
    title=None,                      # Job title
    function=None,                   # Job function / category
    department=None,                 # Department
    location=None,                   # Location
    posting_date=None,               # Posting date (for recency analysis)
    status=None,                     # Status (active/closed)
    
    # ----- PRE-PARSED SECTIONS (optional) -----
    responsibilities=None,
    required_qualifications=None,
    preferred_qualifications=None,
    team_description=None,
    
    # ----- CUSTOM FIELDS -----
    # Add any additional fields you want available for analysis.
    # These become columns in your DataFrame.
    #
    # Examples:
    #   "jd_expertise": "content.expertise_section",
    #   "jd_min_requirements": "content.minimum_requirements",
    #   "rank": "metadata.employee_rank",
    #   "hiring_manager": "metadata.manager_name",
    #   "salary_band": "compensation.band",
    custom_fields={
        # Add your custom fields here
    }
)


# =============================================================================
# PART 2: ANALYSIS CONFIGURATION
# =============================================================================
# Specifies WHICH mapped fields to use for each analysis operation.
# Change these to analyze different dimensions WITHOUT editing code.

@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""
    
    # ===== TEXT ANALYSIS =====
    # Which text field to use for structure parsing and embeddings
    primary_text_field: str = "jd_text"
    
    # Additional text fields to analyze separately (optional)
    # e.g., ["jd_expertise", "jd_min_requirements"]
    additional_text_fields: List[str] = field(default_factory=list)
    
    # ===== IDENTIFICATION =====
    id_field: str = "jd_id"
    
    # ===== STRATIFICATION (Phase 1.1) =====
    # Fields for stratified sampling - should be categorical
    stratify_by_primary: str = "org_unit"
    stratify_by_secondary: str = "level"
    
    # ===== RECENCY ANALYSIS (Phase 1.1) =====
    # Date field for analyzing quality trends over time
    date_field: str = "posting_date"
    
    # ===== CLUSTER ANALYSIS (Phase 1.3) =====
    # Metadata fields to analyze against clusters
    cluster_metadata_fields: List[str] = field(default_factory=lambda: [
        "org_unit", "level", "title", "function"
    ])
    
    # Field for cluster purity calculation
    cluster_purity_field: str = "org_unit"
    
    # ===== SECTION-LEVEL EMBEDDINGS (Phase 1.3) =====
    # Section types to embed separately for comparison
    section_types_to_embed: List[str] = field(default_factory=lambda: [
        "responsibilities", "required_qualifications", "summary"
    ])
    
    # ===== QUALITY EVALUATION (Phase 1.1) =====
    # Fields to include in quality evaluation export
    quality_export_fields: List[str] = field(default_factory=lambda: [
        "title", "org_unit", "level"
    ])
    
    # ===== VALIDATION =====
    def validate(self, available_fields: List[str]) -> Dict[str, List[str]]:
        """Validate config against available data fields."""
        errors = []
        warnings = []
        
        # Required fields
        if self.primary_text_field not in available_fields:
            errors.append(f"primary_text_field '{self.primary_text_field}' not found in data")
        
        if self.id_field not in available_fields:
            errors.append(f"id_field '{self.id_field}' not found in data")
        
        # Stratification fields
        if self.stratify_by_primary not in available_fields:
            warnings.append(
                f"stratify_by_primary '{self.stratify_by_primary}' not found - "
                "will use random sampling"
            )
        
        if self.stratify_by_secondary not in available_fields:
            warnings.append(
                f"stratify_by_secondary '{self.stratify_by_secondary}' not found - "
                "will use single-dimension sampling"
            )
        
        # Date field for recency
        if self.date_field not in available_fields:
            warnings.append(
                f"date_field '{self.date_field}' not found - "
                "recency analysis will be skipped"
            )
        
        # Cluster metadata fields
        missing = [f for f in self.cluster_metadata_fields if f not in available_fields]
        if missing:
            warnings.append(f"cluster_metadata_fields missing: {missing} - will be skipped")
        
        # Additional text fields
        missing_text = [f for f in self.additional_text_fields if f not in available_fields]
        if missing_text:
            warnings.append(f"additional_text_fields missing: {missing_text}")
        
        return {"errors": errors, "warnings": warnings}
    
    def get_available_cluster_fields(self, available_fields: List[str]) -> List[str]:
        """Get cluster metadata fields that exist in data."""
        return [f for f in self.cluster_metadata_fields if f in available_fields]
    
    def get_available_stratify_fields(self, available_fields: List[str]) -> tuple:
        """Get stratification fields that exist in data."""
        primary = self.stratify_by_primary if self.stratify_by_primary in available_fields else None
        secondary = self.stratify_by_secondary if self.stratify_by_secondary in available_fields else None
        return primary, secondary


# ----- YOUR ANALYSIS CONFIGURATION -----
# Modify this to change analysis behavior.

ANALYSIS_CONFIG = AnalysisConfig(
    # Which text field to analyze
    primary_text_field="jd_text",
    
    # Additional text fields (analyzed separately)
    additional_text_fields=[
        # "jd_expertise",
        # "jd_min_requirements",
    ],
    
    # ID field
    id_field="jd_id",
    
    # Stratification for quality sampling
    # Change to use different categorical fields
    stratify_by_primary="org_unit",      # e.g., "rank", "function"
    stratify_by_secondary="level",       # e.g., "title", "department"
    
    # Date field for recency analysis
    date_field="posting_date",
    
    # Fields to analyze against clusters
    cluster_metadata_fields=[
        "org_unit",
        "level",
        "title",
        "function",
        # Add custom fields here:
        # "rank",
        # "hiring_manager",
    ],
    
    # Field for cluster purity calculation
    cluster_purity_field="org_unit",     # e.g., "function", "rank"
    
    # Section types for section-level embeddings
    section_types_to_embed=[
        "responsibilities",
        "required_qualifications",
        "summary",
    ],
    
    # Fields shown in quality evaluation export
    quality_export_fields=[
        "title",
        "org_unit",
        "level",
        # Add fields to see during evaluation:
        # "rank",
        # "department",
    ],
)


# =============================================================================
# PART 3: DATA SOURCE CONFIGURATION
# =============================================================================

# JSON File Configuration
JSON_CONFIG = {
    "data_path": "jd_data",           # Path to folder or file
    "content_key": "content",          # Key containing JD list
    "file_pattern": "*.json",
    "encoding": "utf-8",
}

# Cosmos DB Configuration (optional)
COSMOS_CONFIG = {
    "database_name": "your_database",
    "container_name": "your_container",
    "query": "SELECT * FROM c",
    "max_item_count": 100,
    "endpoint_env": "COSMOS_ENDPOINT",
    "key_env": "COSMOS_KEY",
    "connection_string_env": "COSMOS_CONNECTION_STRING",
}


# =============================================================================
# PART 4: OUTPUT CONFIGURATION
# =============================================================================

OUTPUT_CONFIG = {
    # Root directory for all analysis outputs
    "root_dir": "analysis_output",
    
    # Subdirectories for each phase (relative to root_dir)
    "schema_discovery": "schema_discovery",
    "phase_1_1_quality": "phase_1_1_quality",
    "phase_1_2_structure": "phase_1_2_structure",
    "phase_1_3_clustering": "phase_1_3_clustering",
}


def get_output_path(*subdirs: str) -> Path:
    """
    Get output path relative to the configured root directory.
    
    Args:
        *subdirs: Subdirectory names to append to root
        
    Returns:
        Path object for the output location
        
    Usage:
        from config import get_output_path
        
        # Get root output directory
        root = get_output_path()
        
        # Get phase-specific directory
        phase_dir = get_output_path("phase_1_1_quality")
        
        # Get specific file path
        file_path = get_output_path("phase_1_1_quality", "results.json")
    """
    root = Path(OUTPUT_CONFIG["root_dir"])
    
    if subdirs:
        return root.joinpath(*subdirs)
    return root


def get_phase_output_path(phase: str) -> Path:
    """
    Get output path for a specific analysis phase.
    
    Args:
        phase: One of "schema_discovery", "phase_1_1_quality", 
               "phase_1_2_structure", "phase_1_3_clustering"
               
    Returns:
        Path object for the phase output directory
    """
    root = Path(OUTPUT_CONFIG["root_dir"])
    subdir = OUTPUT_CONFIG.get(phase, phase)
    return root / subdir


# =============================================================================
# PART 5: EMBEDDING CONFIGURATION (Azure OpenAI)
# =============================================================================

EMBEDDING_CONFIG = {
    # Your Azure OpenAI deployment name for embeddings (REQUIRED)
    "deployment_name": None,  # e.g., "text-embedding-ada-002"
    
    # Azure endpoint (or set AZURE_OPENAI_ENDPOINT env var)
    "azure_endpoint": None,  # e.g., "https://your-resource.openai.azure.com"
    
    # API key (or set AZURE_OPENAI_API_KEY env var)
    "api_key": None,
    
    # API version
    "api_version": "2024-02-01",
}


def get_embedding_generator():
    """
    Factory function to create an EmbeddingGenerator from config.
    
    Usage:
        from config import get_embedding_generator
        generator = get_embedding_generator()
        embeddings = generator.embed(["text1", "text2"])
    """
    from src.analysis.content_clustering import EmbeddingGenerator
    
    return EmbeddingGenerator(
        deployment_name=EMBEDDING_CONFIG["deployment_name"],
        azure_endpoint=EMBEDDING_CONFIG.get("azure_endpoint"),
        api_key=EMBEDDING_CONFIG.get("api_key"),
        api_version=EMBEDDING_CONFIG.get("api_version", "2024-02-01"),
    )


# =============================================================================
# PART 6: CHAT/LLM CONFIGURATION (Azure OpenAI - for extraction & naming)
# =============================================================================

CHAT_CONFIG = {
    # Your Azure OpenAI deployment name for chat/completions (REQUIRED for archetype pipeline)
    "deployment_name": None,  # e.g., "gpt-4o", "gpt-4-turbo"
    
    # Azure endpoint (or set AZURE_OPENAI_ENDPOINT env var)
    # Can be same as EMBEDDING_CONFIG if using same resource
    "azure_endpoint": None,
    
    # API key (or set AZURE_OPENAI_API_KEY env var)
    "api_key": None,
    
    # API version
    "api_version": "2024-02-01",
}


# =============================================================================
# PART 7: ARCHETYPE PIPELINE CONFIGURATION
# =============================================================================

ARCHETYPE_CONFIG = {
    # Output subdirectory for archetype pipeline (relative to OUTPUT_CONFIG root_dir)
    "output_subdir": "archetypes",
    
    # Extraction settings
    "extraction": {
        # Fields to extract from (from your JD_FIELD_MAPPING)
        "job_description_field": "jd_text",
        "expertise_field": None,  # e.g., "expertise" if you have it
        "team_description_field": None,  # e.g., "team_description"
        
        # Metadata fields to capture with extraction
        "metadata_fields": ["title", "level", "division", "function", "org_unit"],
    },
    
    # Clustering settings
    "clustering": {
        "algorithm": "hdbscan",
        "min_cluster_size": 5,
        "min_samples": 3,
    },
    
    # Feature engineering default
    "features": {
        "approach": "skills_only",  # "skills_only", "contextual", "structured"
        "include_division": False,
        "include_function": False,
    },
}


def get_archetype_output_path(*subdirs: str) -> Path:
    """
    Get output path for archetype pipeline.
    
    Args:
        *subdirs: Additional subdirectories
        
    Returns:
        Path object
    """
    root = Path(OUTPUT_CONFIG["root_dir"])
    archetype_dir = root / ARCHETYPE_CONFIG["output_subdir"]
    
    if subdirs:
        return archetype_dir.joinpath(*subdirs)
    return archetype_dir
