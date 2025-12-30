"""
Phase 2: Feature Engineering

Build configurable feature vectors from extracted requirements for clustering.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path
from enum import Enum

from .extraction import ExtractionResult


class FeatureApproach(str, Enum):
    """Feature engineering approaches for experimentation."""
    
    # Option A: Embed extracted requirements as text only
    SKILLS_ONLY = "skills_only"
    
    # Option B: Structured feature vector with one-hot encoding
    STRUCTURED = "structured"
    
    # Option C: Embed skills with division/function as text context
    CONTEXTUAL = "contextual"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Which approach to use
    approach: FeatureApproach = FeatureApproach.SKILLS_ONLY
    
    # What to include in text embedding
    include_skills: bool = True
    include_licenses: bool = True
    include_certifications: bool = True
    include_tools: bool = True
    include_education: bool = True
    include_experience: bool = True
    
    # For STRUCTURED approach: which categorical fields to one-hot encode
    categorical_fields: List[str] = field(default_factory=lambda: [
        "division", "function"
    ])
    
    # For STRUCTURED approach: weights for combining features
    # Skills embedding weight (remainder goes to categorical)
    skills_weight: float = 0.7
    
    # For CONTEXTUAL approach: which fields to prepend as context
    context_fields: List[str] = field(default_factory=lambda: [
        "division", "function"
    ])
    
    # Experiment metadata
    experiment_id: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["approach"] = self.approach.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureConfig":
        if "approach" in data and isinstance(data["approach"], str):
            data["approach"] = FeatureApproach(data["approach"])
        return cls(**data)


@dataclass
class FeatureOutput:
    """Output from feature engineering."""
    
    # Feature vectors
    features: np.ndarray  # Shape: (n_samples, feature_dim)
    
    # JD IDs in same order as features
    ids: List[str]
    
    # Configuration used
    config: FeatureConfig
    
    # Feature metadata
    feature_dim: int = 0
    n_samples: int = 0
    
    # For STRUCTURED approach: mapping of categorical values
    categorical_mappings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.n_samples, self.feature_dim = self.features.shape
    
    def save(self, output_dir: str) -> None:
        """Save features and config to files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        np.save(path / "features.npy", self.features)
        
        # Save IDs
        with open(path / "feature_ids.json", "w") as f:
            json.dump(self.ids, f)
        
        # Save config and metadata
        metadata = {
            "config": self.config.to_dict(),
            "feature_dim": self.feature_dim,
            "n_samples": self.n_samples,
            "categorical_mappings": self.categorical_mappings,
        }
        with open(path / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, output_dir: str) -> "FeatureOutput":
        """Load features from files."""
        path = Path(output_dir)
        
        features = np.load(path / "features.npy")
        
        with open(path / "feature_ids.json", "r") as f:
            ids = json.load(f)
        
        with open(path / "feature_metadata.json", "r") as f:
            metadata = json.load(f)
        
        config = FeatureConfig.from_dict(metadata["config"])
        
        return cls(
            features=features,
            ids=ids,
            config=config,
            categorical_mappings=metadata.get("categorical_mappings", {}),
        )


class FeatureEngineer:
    """
    Build feature vectors from extracted requirements.
    
    Supports multiple approaches for experimentation.
    """
    
    def __init__(
        self,
        embedding_generator=None,
    ):
        """
        Initialize feature engineer.
        
        Args:
            embedding_generator: EmbeddingGenerator instance for text embedding.
                                If None, will create from config when needed.
        """
        self._embedding_generator = embedding_generator
    
    def _get_embedding_generator(self):
        """Get or create embedding generator."""
        if self._embedding_generator is None:
            from config import get_embedding_generator
            self._embedding_generator = get_embedding_generator()
        return self._embedding_generator
    
    def _build_requirements_text(
        self,
        extraction: ExtractionResult,
        config: FeatureConfig,
    ) -> str:
        """Build text representation of requirements for embedding."""
        parts = []
        
        if config.include_licenses:
            all_licenses = extraction.licenses.required + extraction.licenses.preferred
            if all_licenses:
                parts.append(f"Licenses: {', '.join(all_licenses)}")
        
        if config.include_certifications:
            all_certs = extraction.certifications.required + extraction.certifications.preferred
            if all_certs:
                parts.append(f"Certifications: {', '.join(all_certs)}")
        
        if config.include_skills:
            skills = extraction.get_all_skills_flat()
            if skills:
                parts.append(f"Skills: {', '.join(skills)}")
        
        if config.include_tools:
            all_tools = extraction.tools.required + extraction.tools.preferred
            if all_tools:
                parts.append(f"Tools: {', '.join(all_tools)}")
        
        if config.include_education:
            edu_parts = extraction.education.level.required + extraction.education.level.preferred
            edu_parts.extend(extraction.education.fields)
            if edu_parts:
                parts.append(f"Education: {', '.join(edu_parts)}")
        
        if config.include_experience:
            exp_parts = []
            if extraction.experience.years_min:
                exp_parts.append(f"{extraction.experience.years_min}+ years")
            exp_parts.extend(extraction.experience.specific.required)
            exp_parts.extend(extraction.experience.specific.preferred)
            if exp_parts:
                parts.append(f"Experience: {', '.join(exp_parts)}")
        
        return "; ".join(parts) if parts else "No requirements extracted"
    
    def _build_contextual_text(
        self,
        extraction: ExtractionResult,
        config: FeatureConfig,
    ) -> str:
        """Build text with context fields prepended."""
        context_parts = []
        
        for field_name in config.context_fields:
            value = extraction.metadata.get(field_name)
            if value:
                context_parts.append(f"[{field_name}: {value}]")
        
        requirements_text = self._build_requirements_text(extraction, config)
        
        if context_parts:
            return " ".join(context_parts) + " " + requirements_text
        return requirements_text
    
    def _build_categorical_features(
        self,
        extractions: List[ExtractionResult],
        config: FeatureConfig,
    ) -> tuple:
        """Build one-hot encoded categorical features."""
        # Collect unique values for each categorical field
        value_sets: Dict[str, set] = {f: set() for f in config.categorical_fields}
        
        for extraction in extractions:
            for field_name in config.categorical_fields:
                value = extraction.metadata.get(field_name)
                if value:
                    value_sets[field_name].add(str(value))
        
        # Create mappings
        mappings: Dict[str, Dict[str, int]] = {}
        total_dim = 0
        
        for field_name in config.categorical_fields:
            sorted_values = sorted(value_sets[field_name])
            mappings[field_name] = {v: i for i, v in enumerate(sorted_values)}
            total_dim += len(sorted_values)
        
        # Build one-hot vectors
        n_samples = len(extractions)
        categorical_features = np.zeros((n_samples, total_dim))
        
        offset = 0
        for field_name in config.categorical_fields:
            field_mapping = mappings[field_name]
            field_dim = len(field_mapping)
            
            for i, extraction in enumerate(extractions):
                value = extraction.metadata.get(field_name)
                if value and str(value) in field_mapping:
                    idx = field_mapping[str(value)]
                    categorical_features[i, offset + idx] = 1.0
            
            offset += field_dim
        
        return categorical_features, mappings
    
    def build_features(
        self,
        extractions: List[ExtractionResult],
        config: FeatureConfig,
        show_progress: bool = True,
    ) -> FeatureOutput:
        """
        Build feature vectors from extractions.
        
        Args:
            extractions: List of ExtractionResult objects
            config: FeatureConfig specifying the approach
            show_progress: Print progress updates
            
        Returns:
            FeatureOutput with features and metadata
        """
        ids = [e.jd_id for e in extractions]
        
        if config.approach == FeatureApproach.SKILLS_ONLY:
            return self._build_skills_only(extractions, config, ids, show_progress)
        
        elif config.approach == FeatureApproach.CONTEXTUAL:
            return self._build_contextual(extractions, config, ids, show_progress)
        
        elif config.approach == FeatureApproach.STRUCTURED:
            return self._build_structured(extractions, config, ids, show_progress)
        
        else:
            raise ValueError(f"Unknown approach: {config.approach}")
    
    def _build_skills_only(
        self,
        extractions: List[ExtractionResult],
        config: FeatureConfig,
        ids: List[str],
        show_progress: bool,
    ) -> FeatureOutput:
        """Build features using Option A: skills embedding only."""
        if show_progress:
            print(f"Building features with approach: SKILLS_ONLY")
        
        # Build text for each extraction
        texts = [self._build_requirements_text(e, config) for e in extractions]
        
        if show_progress:
            print(f"Embedding {len(texts)} requirement texts...")
        
        # Generate embeddings
        embedder = self._get_embedding_generator()
        embeddings = embedder.embed(texts, show_progress=show_progress)
        
        return FeatureOutput(
            features=embeddings,
            ids=ids,
            config=config,
        )
    
    def _build_contextual(
        self,
        extractions: List[ExtractionResult],
        config: FeatureConfig,
        ids: List[str],
        show_progress: bool,
    ) -> FeatureOutput:
        """Build features using Option C: contextual embedding."""
        if show_progress:
            print(f"Building features with approach: CONTEXTUAL")
            print(f"Context fields: {config.context_fields}")
        
        # Build contextual text for each extraction
        texts = [self._build_contextual_text(e, config) for e in extractions]
        
        if show_progress:
            print(f"Embedding {len(texts)} contextual texts...")
        
        # Generate embeddings
        embedder = self._get_embedding_generator()
        embeddings = embedder.embed(texts, show_progress=show_progress)
        
        return FeatureOutput(
            features=embeddings,
            ids=ids,
            config=config,
        )
    
    def _build_structured(
        self,
        extractions: List[ExtractionResult],
        config: FeatureConfig,
        ids: List[str],
        show_progress: bool,
    ) -> FeatureOutput:
        """Build features using Option B: structured with one-hot."""
        if show_progress:
            print(f"Building features with approach: STRUCTURED")
            print(f"Categorical fields: {config.categorical_fields}")
            print(f"Skills weight: {config.skills_weight}")
        
        # Build skills embeddings
        texts = [self._build_requirements_text(e, config) for e in extractions]
        
        if show_progress:
            print(f"Embedding {len(texts)} requirement texts...")
        
        embedder = self._get_embedding_generator()
        skill_embeddings = embedder.embed(texts, show_progress=show_progress)
        
        # Build categorical features
        if show_progress:
            print("Building categorical features...")
        
        categorical_features, mappings = self._build_categorical_features(
            extractions, config
        )
        
        if show_progress:
            print(f"Categorical feature dim: {categorical_features.shape[1]}")
            for field_name, field_mapping in mappings.items():
                print(f"  {field_name}: {len(field_mapping)} unique values")
        
        # Normalize and combine
        # Normalize skill embeddings to unit norm
        skill_norms = np.linalg.norm(skill_embeddings, axis=1, keepdims=True)
        skill_norms = np.where(skill_norms > 0, skill_norms, 1)  # Avoid division by zero
        skill_embeddings_normalized = skill_embeddings / skill_norms
        
        # Apply weights
        skill_weighted = skill_embeddings_normalized * config.skills_weight
        categorical_weighted = categorical_features * (1 - config.skills_weight)
        
        # Concatenate
        combined = np.concatenate([skill_weighted, categorical_weighted], axis=1)
        
        if show_progress:
            print(f"Combined feature dim: {combined.shape[1]}")
        
        return FeatureOutput(
            features=combined,
            ids=ids,
            config=config,
            categorical_mappings=mappings,
        )
    
    def build_multiple_configs(
        self,
        extractions: List[ExtractionResult],
        configs: List[FeatureConfig],
        output_dir: str,
        show_progress: bool = True,
    ) -> Dict[str, FeatureOutput]:
        """
        Build features with multiple configurations for experimentation.
        
        Args:
            extractions: List of ExtractionResult objects
            configs: List of FeatureConfig to try
            output_dir: Directory to save all feature sets
            show_progress: Print progress updates
            
        Returns:
            Dict mapping experiment_id to FeatureOutput
        """
        results = {}
        
        for i, config in enumerate(configs):
            exp_id = config.experiment_id or f"exp_{i:02d}"
            
            if show_progress:
                print(f"\n{'='*50}")
                print(f"Building features for: {exp_id}")
                print(f"{'='*50}")
            
            output = self.build_features(extractions, config, show_progress)
            
            # Save to experiment subdirectory
            exp_output_dir = Path(output_dir) / exp_id
            output.save(str(exp_output_dir))
            
            results[exp_id] = output
        
        return results


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_experiment_configs() -> List[FeatureConfig]:
    """Get preset configurations for the experiment matrix."""
    return [
        # Experiment 1: Skills only (baseline)
        FeatureConfig(
            approach=FeatureApproach.SKILLS_ONLY,
            experiment_id="exp_01_skills_only",
            description="Skills embedding only - baseline",
        ),
        
        # Experiment 2: Contextual with division and function
        FeatureConfig(
            approach=FeatureApproach.CONTEXTUAL,
            context_fields=["division", "function"],
            experiment_id="exp_02_contextual",
            description="Skills with division/function as text context",
        ),
        
        # Experiment 3: Structured with light division weight
        FeatureConfig(
            approach=FeatureApproach.STRUCTURED,
            categorical_fields=["division"],
            skills_weight=0.8,
            experiment_id="exp_03_structured_light",
            description="Structured with division (20% weight)",
        ),
        
        # Experiment 4: Structured with heavy division weight
        FeatureConfig(
            approach=FeatureApproach.STRUCTURED,
            categorical_fields=["division"],
            skills_weight=0.6,
            experiment_id="exp_04_structured_heavy",
            description="Structured with division (40% weight)",
        ),
        
        # Experiment 5: Structured with division and function
        FeatureConfig(
            approach=FeatureApproach.STRUCTURED,
            categorical_fields=["division", "function"],
            skills_weight=0.5,
            experiment_id="exp_05_structured_both",
            description="Structured with division + function (50% weight)",
        ),
    ]
