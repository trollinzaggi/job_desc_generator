"""
Job Archetype Pipeline

This module provides functionality for:
- Phase 1: LLM-based requirement extraction
- Phase 2: Feature engineering for clustering
- Phase 3: HDBSCAN clustering with experiments
- Phase 4: Aggregation into job archetypes
- Phase 5: LLM-based archetype naming
"""

from .extraction import RequirementExtractor, ExtractionResult
from .feature_engineering import FeatureEngineer, FeatureConfig
from .clustering import ArchetypeClusterer, ClusteringConfig, ExperimentRunner
from .aggregation import ArchetypeAggregator, JobArchetype
from .naming import ArchetypeNamer

__all__ = [
    # Phase 1
    "RequirementExtractor",
    "ExtractionResult",
    # Phase 2
    "FeatureEngineer", 
    "FeatureConfig",
    # Phase 3
    "ArchetypeClusterer",
    "ClusteringConfig",
    "ExperimentRunner",
    # Phase 4
    "ArchetypeAggregator",
    "JobArchetype",
    # Phase 5
    "ArchetypeNamer",
]
