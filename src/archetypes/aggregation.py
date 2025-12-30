"""
Phase 4: Aggregation

Aggregate extracted requirements within clusters to build job archetypes.
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import Counter
from datetime import datetime

from .extraction import ExtractionResult, RequirementCategory
from .clustering import ClusteringResult


@dataclass
class FrequencyMap:
    """Map of items to their frequency in the cluster."""
    required: Dict[str, float] = field(default_factory=dict)
    preferred: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "required": self.required,
            "preferred": self.preferred,
        }


@dataclass
class SkillsAggregate:
    """Aggregated skills by type."""
    technical: FrequencyMap = field(default_factory=FrequencyMap)
    domain: FrequencyMap = field(default_factory=FrequencyMap)
    soft: FrequencyMap = field(default_factory=FrequencyMap)
    
    def to_dict(self) -> Dict[str, Dict]:
        return {
            "technical": self.technical.to_dict(),
            "domain": self.domain.to_dict(),
            "soft": self.soft.to_dict(),
        }


@dataclass
class ExperienceAggregate:
    """Aggregated experience requirements."""
    years_min_median: Optional[float] = None
    years_preferred_median: Optional[float] = None
    years_range: List[int] = field(default_factory=list)  # [min, max]
    specific: FrequencyMap = field(default_factory=FrequencyMap)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "years": {
                "min_median": self.years_min_median,
                "preferred_median": self.years_preferred_median,
                "range": self.years_range,
            },
            "specific": self.specific.to_dict(),
        }


@dataclass
class JobArchetype:
    """
    A job archetype aggregated from a cluster of similar JDs.
    """
    
    # Identification
    cluster_id: int
    archetype_id: Optional[str] = None  # Set by naming phase
    label: Optional[str] = None  # LLM-generated name
    
    # Size
    member_count: int = 0
    member_ids: List[str] = field(default_factory=list)
    
    # Distributions
    level_distribution: Dict[str, float] = field(default_factory=dict)
    division_distribution: Dict[str, float] = field(default_factory=dict)
    function_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Representative titles
    representative_titles: List[str] = field(default_factory=list)
    
    # Aggregated requirements (with frequencies)
    licenses: FrequencyMap = field(default_factory=FrequencyMap)
    certifications: FrequencyMap = field(default_factory=FrequencyMap)
    skills: SkillsAggregate = field(default_factory=SkillsAggregate)
    education_levels: FrequencyMap = field(default_factory=FrequencyMap)
    education_fields: Dict[str, float] = field(default_factory=dict)
    experience: ExperienceAggregate = field(default_factory=ExperienceAggregate)
    tools: FrequencyMap = field(default_factory=FrequencyMap)
    languages: FrequencyMap = field(default_factory=FrequencyMap)
    
    # Centroid (for similarity matching)
    centroid_embedding: Optional[List[float]] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "archetype_id": self.archetype_id,
            "label": self.label,
            "member_count": self.member_count,
            "member_ids": self.member_ids,
            "level_distribution": self.level_distribution,
            "division_distribution": self.division_distribution,
            "function_distribution": self.function_distribution,
            "representative_titles": self.representative_titles,
            "requirements": {
                "licenses": self.licenses.to_dict(),
                "certifications": self.certifications.to_dict(),
                "skills": self.skills.to_dict(),
                "education": {
                    "levels": self.education_levels.to_dict(),
                    "fields": self.education_fields,
                },
                "experience": self.experience.to_dict(),
                "tools": self.tools.to_dict(),
                "languages": self.languages.to_dict(),
            },
            "centroid_embedding": self.centroid_embedding,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobArchetype":
        """Create from dictionary."""
        archetype = cls(
            cluster_id=data["cluster_id"],
            archetype_id=data.get("archetype_id"),
            label=data.get("label"),
            member_count=data.get("member_count", 0),
            member_ids=data.get("member_ids", []),
            level_distribution=data.get("level_distribution", {}),
            division_distribution=data.get("division_distribution", {}),
            function_distribution=data.get("function_distribution", {}),
            representative_titles=data.get("representative_titles", []),
            centroid_embedding=data.get("centroid_embedding"),
            created_at=data.get("created_at", ""),
            last_updated=data.get("last_updated", ""),
        )
        
        # Parse requirements
        reqs = data.get("requirements", {})
        
        if "licenses" in reqs:
            archetype.licenses = FrequencyMap(
                required=reqs["licenses"].get("required", {}),
                preferred=reqs["licenses"].get("preferred", {}),
            )
        
        if "certifications" in reqs:
            archetype.certifications = FrequencyMap(
                required=reqs["certifications"].get("required", {}),
                preferred=reqs["certifications"].get("preferred", {}),
            )
        
        if "skills" in reqs:
            skills_data = reqs["skills"]
            archetype.skills = SkillsAggregate(
                technical=FrequencyMap(
                    required=skills_data.get("technical", {}).get("required", {}),
                    preferred=skills_data.get("technical", {}).get("preferred", {}),
                ),
                domain=FrequencyMap(
                    required=skills_data.get("domain", {}).get("required", {}),
                    preferred=skills_data.get("domain", {}).get("preferred", {}),
                ),
                soft=FrequencyMap(
                    required=skills_data.get("soft", {}).get("required", {}),
                    preferred=skills_data.get("soft", {}).get("preferred", {}),
                ),
            )
        
        if "tools" in reqs:
            archetype.tools = FrequencyMap(
                required=reqs["tools"].get("required", {}),
                preferred=reqs["tools"].get("preferred", {}),
            )
        
        if "languages" in reqs:
            archetype.languages = FrequencyMap(
                required=reqs["languages"].get("required", {}),
                preferred=reqs["languages"].get("preferred", {}),
            )
        
        if "education" in reqs:
            edu = reqs["education"]
            archetype.education_levels = FrequencyMap(
                required=edu.get("levels", {}).get("required", {}),
                preferred=edu.get("levels", {}).get("preferred", {}),
            )
            archetype.education_fields = edu.get("fields", {})
        
        if "experience" in reqs:
            exp = reqs["experience"]
            years = exp.get("years", {})
            archetype.experience = ExperienceAggregate(
                years_min_median=years.get("min_median"),
                years_preferred_median=years.get("preferred_median"),
                years_range=years.get("range", []),
                specific=FrequencyMap(
                    required=exp.get("specific", {}).get("required", {}),
                    preferred=exp.get("specific", {}).get("preferred", {}),
                ),
            )
        
        return archetype
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Archetype: {self.label or f'Cluster {self.cluster_id}'}",
            f"Members: {self.member_count}",
        ]
        
        if self.representative_titles:
            lines.append(f"Common titles: {', '.join(self.representative_titles[:3])}")
        
        # Top skills
        all_skills = {}
        for skill_type in [self.skills.technical, self.skills.domain, self.skills.soft]:
            all_skills.update(skill_type.required)
            all_skills.update(skill_type.preferred)
        
        top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_skills:
            skills_str = ", ".join([f"{s[0]} ({s[1]:.0%})" for s in top_skills])
            lines.append(f"Top skills: {skills_str}")
        
        return "\n".join(lines)


class ArchetypeAggregator:
    """
    Aggregate extracted requirements into job archetypes.
    """
    
    def __init__(
        self,
        extractions: List[ExtractionResult],
        clustering_result: ClusteringResult,
        features: Optional[np.ndarray] = None,
    ):
        """
        Initialize aggregator.
        
        Args:
            extractions: List of ExtractionResult objects
            clustering_result: ClusteringResult with cluster assignments
            features: Optional feature vectors for computing centroids
        """
        self.extractions = extractions
        self.clustering_result = clustering_result
        self.features = features
        
        # Build lookup
        self.extraction_lookup = {e.jd_id: e for e in extractions}
        
        # Build cluster to JD mapping
        self.cluster_jds: Dict[int, List[str]] = {}
        for jd_id, label in zip(clustering_result.ids, clustering_result.labels):
            if label not in self.cluster_jds:
                self.cluster_jds[label] = []
            self.cluster_jds[label].append(jd_id)
    
    def _compute_frequency(
        self,
        items: List[str],
        total: int,
    ) -> Dict[str, float]:
        """Compute frequency of each item."""
        if total == 0:
            return {}
        
        counts = Counter(items)
        return {item: count / total for item, count in counts.most_common()}
    
    def _aggregate_requirement_category(
        self,
        extractions: List[ExtractionResult],
        get_category_fn,
    ) -> FrequencyMap:
        """Aggregate a requirement category across extractions."""
        total = len(extractions)
        
        required_items = []
        preferred_items = []
        
        for ext in extractions:
            category = get_category_fn(ext)
            if category:
                required_items.extend(category.required)
                preferred_items.extend(category.preferred)
        
        return FrequencyMap(
            required=self._compute_frequency(required_items, total),
            preferred=self._compute_frequency(preferred_items, total),
        )
    
    def _aggregate_distribution(
        self,
        extractions: List[ExtractionResult],
        field_name: str,
    ) -> Dict[str, float]:
        """Aggregate distribution of a metadata field."""
        total = len(extractions)
        if total == 0:
            return {}
        
        values = []
        for ext in extractions:
            value = ext.metadata.get(field_name)
            if value:
                values.append(str(value))
        
        return self._compute_frequency(values, total)
    
    def _get_representative_titles(
        self,
        extractions: List[ExtractionResult],
        n: int = 5,
    ) -> List[str]:
        """Get most common titles."""
        titles = []
        for ext in extractions:
            title = ext.metadata.get("title")
            if title:
                titles.append(str(title))
        
        counts = Counter(titles)
        return [title for title, _ in counts.most_common(n)]
    
    def _compute_centroid(
        self,
        jd_ids: List[str],
    ) -> Optional[List[float]]:
        """Compute centroid embedding for cluster."""
        if self.features is None:
            return None
        
        # Get indices for JD IDs
        id_to_idx = {jd_id: i for i, jd_id in enumerate(self.clustering_result.ids)}
        
        indices = [id_to_idx[jd_id] for jd_id in jd_ids if jd_id in id_to_idx]
        
        if not indices:
            return None
        
        cluster_features = self.features[indices]
        centroid = np.mean(cluster_features, axis=0)
        
        return centroid.tolist()
    
    def aggregate_cluster(
        self,
        cluster_id: int,
    ) -> JobArchetype:
        """
        Aggregate a single cluster into an archetype.
        
        Args:
            cluster_id: Cluster label
            
        Returns:
            JobArchetype
        """
        jd_ids = self.cluster_jds.get(cluster_id, [])
        extractions = [
            self.extraction_lookup[jd_id]
            for jd_id in jd_ids
            if jd_id in self.extraction_lookup
        ]
        
        if not extractions:
            return JobArchetype(
                cluster_id=cluster_id,
                member_count=0,
            )
        
        archetype = JobArchetype(
            cluster_id=cluster_id,
            member_count=len(extractions),
            member_ids=jd_ids,
        )
        
        # Distributions
        archetype.level_distribution = self._aggregate_distribution(extractions, "level")
        archetype.division_distribution = self._aggregate_distribution(extractions, "division")
        archetype.function_distribution = self._aggregate_distribution(extractions, "function")
        
        # Representative titles
        archetype.representative_titles = self._get_representative_titles(extractions)
        
        # Licenses
        archetype.licenses = self._aggregate_requirement_category(
            extractions, lambda e: e.licenses
        )
        
        # Certifications
        archetype.certifications = self._aggregate_requirement_category(
            extractions, lambda e: e.certifications
        )
        
        # Skills
        archetype.skills = SkillsAggregate(
            technical=self._aggregate_requirement_category(
                extractions, lambda e: e.skills.technical
            ),
            domain=self._aggregate_requirement_category(
                extractions, lambda e: e.skills.domain
            ),
            soft=self._aggregate_requirement_category(
                extractions, lambda e: e.skills.soft
            ),
        )
        
        # Education
        archetype.education_levels = self._aggregate_requirement_category(
            extractions, lambda e: e.education.level
        )
        
        # Education fields
        all_fields = []
        for ext in extractions:
            all_fields.extend(ext.education.fields)
        archetype.education_fields = self._compute_frequency(all_fields, len(extractions))
        
        # Experience
        years_min_values = [
            ext.experience.years_min
            for ext in extractions
            if ext.experience.years_min is not None
        ]
        years_pref_values = [
            ext.experience.years_preferred
            for ext in extractions
            if ext.experience.years_preferred is not None
        ]
        
        archetype.experience = ExperienceAggregate(
            years_min_median=np.median(years_min_values) if years_min_values else None,
            years_preferred_median=np.median(years_pref_values) if years_pref_values else None,
            years_range=[
                min(years_min_values) if years_min_values else 0,
                max(years_pref_values or years_min_values or [0]),
            ],
            specific=self._aggregate_requirement_category(
                extractions, lambda e: e.experience.specific
            ),
        )
        
        # Tools
        archetype.tools = self._aggregate_requirement_category(
            extractions, lambda e: e.tools
        )
        
        # Languages
        archetype.languages = self._aggregate_requirement_category(
            extractions, lambda e: e.languages
        )
        
        # Centroid
        archetype.centroid_embedding = self._compute_centroid(jd_ids)
        
        return archetype
    
    def aggregate_all(
        self,
        exclude_noise: bool = True,
        show_progress: bool = True,
    ) -> List[JobArchetype]:
        """
        Aggregate all clusters into archetypes.
        
        Args:
            exclude_noise: Skip noise cluster (-1)
            show_progress: Print progress
            
        Returns:
            List of JobArchetype
        """
        archetypes = []
        
        cluster_ids = sorted(self.cluster_jds.keys())
        
        for cluster_id in cluster_ids:
            if exclude_noise and cluster_id == -1:
                continue
            
            if show_progress:
                print(f"Aggregating cluster {cluster_id}...")
            
            archetype = self.aggregate_cluster(cluster_id)
            archetypes.append(archetype)
        
        if show_progress:
            print(f"\nCreated {len(archetypes)} archetypes")
        
        return archetypes
    
    def save_archetypes(
        self,
        archetypes: List[JobArchetype],
        output_dir: str,
    ) -> None:
        """Save archetypes to files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        data = [a.to_dict() for a in archetypes]
        with open(path / "archetypes.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for a in archetypes:
            # Get top skills
            all_skills = {}
            for skill_type in [a.skills.technical, a.skills.domain, a.skills.soft]:
                all_skills.update(skill_type.required)
                all_skills.update(skill_type.preferred)
            top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:5]
            
            summary_data.append({
                "cluster_id": a.cluster_id,
                "archetype_id": a.archetype_id,
                "label": a.label,
                "member_count": a.member_count,
                "representative_titles": "; ".join(a.representative_titles[:3]),
                "top_skills": "; ".join([s[0] for s in top_skills]),
                "top_division": list(a.division_distribution.keys())[0] if a.division_distribution else "",
            })
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(path / "archetypes_summary.csv", index=False)
        
        print(f"Saved archetypes to: {path}/")
    
    @staticmethod
    def load_archetypes(path: str) -> List[JobArchetype]:
        """Load archetypes from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return [JobArchetype.from_dict(d) for d in data]
