"""
Phase 3: Clustering with HDBSCAN

Cluster JDs using HDBSCAN with experiment framework for comparison.
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


@dataclass
class ClusteringConfig:
    """Configuration for clustering."""
    
    # Algorithm: "hdbscan", "kmeans", "hierarchical"
    algorithm: str = "hdbscan"
    
    # HDBSCAN parameters
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"
    
    # K-means parameters (if using kmeans)
    n_clusters: Optional[int] = None
    
    # Experiment metadata
    experiment_id: Optional[str] = None
    feature_experiment_id: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusteringConfig":
        return cls(**data)


@dataclass
class ClusteringResult:
    """Results from clustering."""
    
    # Cluster assignments (-1 for noise in HDBSCAN)
    labels: np.ndarray
    
    # JD IDs in same order as labels
    ids: List[str]
    
    # Metrics
    n_clusters: int = 0
    n_noise: int = 0
    noise_ratio: float = 0.0
    silhouette_score: Optional[float] = None
    
    # Cluster sizes
    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    
    # HDBSCAN specific
    probabilities: Optional[np.ndarray] = None
    outlier_scores: Optional[np.ndarray] = None
    
    # Config used
    config: Optional[ClusteringConfig] = None
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        if self.n_clusters == 0:
            unique_labels = set(self.labels) - {-1}
            self.n_clusters = len(unique_labels)
        
        if self.n_noise == 0:
            self.n_noise = int(np.sum(self.labels == -1))
        
        if self.noise_ratio == 0.0 and len(self.labels) > 0:
            self.noise_ratio = self.n_noise / len(self.labels)
        
        if not self.cluster_sizes:
            for label in self.labels:
                self.cluster_sizes[int(label)] = self.cluster_sizes.get(int(label), 0) + 1
    
    def get_assignments_df(self) -> pd.DataFrame:
        """Get cluster assignments as DataFrame."""
        df = pd.DataFrame({
            "jd_id": self.ids,
            "cluster": self.labels,
        })
        
        if self.probabilities is not None:
            df["probability"] = self.probabilities
        
        if self.outlier_scores is not None:
            df["outlier_score"] = self.outlier_scores
        
        return df
    
    def get_cluster_ids(self, cluster_label: int) -> List[str]:
        """Get JD IDs for a specific cluster."""
        return [
            self.ids[i] for i, label in enumerate(self.labels)
            if label == cluster_label
        ]
    
    def get_noise_ids(self) -> List[str]:
        """Get JD IDs marked as noise."""
        return self.get_cluster_ids(-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "noise_ratio": self.noise_ratio,
            "silhouette_score": self.silhouette_score,
            "cluster_sizes": self.cluster_sizes,
            "config": self.config.to_dict() if self.config else None,
            "timestamp": self.timestamp,
        }
    
    def save(self, output_dir: str) -> None:
        """Save clustering results to files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save assignments
        self.get_assignments_df().to_csv(path / "cluster_assignments.csv", index=False)
        
        # Save labels as numpy
        np.save(path / "cluster_labels.npy", self.labels)
        
        if self.probabilities is not None:
            np.save(path / "cluster_probabilities.npy", self.probabilities)
        
        if self.outlier_scores is not None:
            np.save(path / "outlier_scores.npy", self.outlier_scores)
        
        # Save metadata
        with open(path / "clustering_metadata.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, output_dir: str) -> "ClusteringResult":
        """Load clustering results from files."""
        path = Path(output_dir)
        
        # Load labels
        labels = np.load(path / "cluster_labels.npy")
        
        # Load assignments for IDs
        assignments_df = pd.read_csv(path / "cluster_assignments.csv")
        ids = assignments_df["jd_id"].astype(str).tolist()
        
        # Load optional arrays
        probabilities = None
        if (path / "cluster_probabilities.npy").exists():
            probabilities = np.load(path / "cluster_probabilities.npy")
        
        outlier_scores = None
        if (path / "outlier_scores.npy").exists():
            outlier_scores = np.load(path / "outlier_scores.npy")
        
        # Load metadata
        with open(path / "clustering_metadata.json", "r") as f:
            metadata = json.load(f)
        
        config = None
        if metadata.get("config"):
            config = ClusteringConfig.from_dict(metadata["config"])
        
        return cls(
            labels=labels,
            ids=ids,
            n_clusters=metadata.get("n_clusters", 0),
            n_noise=metadata.get("n_noise", 0),
            noise_ratio=metadata.get("noise_ratio", 0.0),
            silhouette_score=metadata.get("silhouette_score"),
            cluster_sizes=metadata.get("cluster_sizes", {}),
            probabilities=probabilities,
            outlier_scores=outlier_scores,
            config=config,
            timestamp=metadata.get("timestamp", ""),
        )


class ArchetypeClusterer:
    """
    Cluster JDs using HDBSCAN or other algorithms.
    """
    
    def __init__(self, features: np.ndarray, ids: List[str]):
        """
        Initialize clusterer.
        
        Args:
            features: Feature vectors (n_samples, feature_dim)
            ids: JD IDs in same order as features
        """
        self.features = features
        self.ids = ids
    
    def cluster_hdbscan(
        self,
        config: Optional[ClusteringConfig] = None,
        **kwargs,
    ) -> ClusteringResult:
        """
        Cluster using HDBSCAN.
        
        Args:
            config: ClusteringConfig with parameters
            **kwargs: Override config parameters
            
        Returns:
            ClusteringResult
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan required. Install with: pip install hdbscan")
        
        if config is None:
            config = ClusteringConfig(algorithm="hdbscan")
        
        # Apply kwargs overrides
        min_cluster_size = kwargs.get("min_cluster_size", config.min_cluster_size)
        min_samples = kwargs.get("min_samples", config.min_samples)
        cluster_selection_epsilon = kwargs.get(
            "cluster_selection_epsilon", config.cluster_selection_epsilon
        )
        metric = kwargs.get("metric", config.metric)
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            gen_min_span_tree=True,
        )
        
        labels = clusterer.fit_predict(self.features)
        
        # Calculate silhouette score (excluding noise)
        silhouette = None
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) > 1:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(
                    self.features[non_noise_mask],
                    labels[non_noise_mask],
                )
            except Exception:
                pass
        
        return ClusteringResult(
            labels=labels,
            ids=self.ids,
            silhouette_score=silhouette,
            probabilities=clusterer.probabilities_,
            outlier_scores=clusterer.outlier_scores_,
            config=config,
        )
    
    def cluster_kmeans(
        self,
        n_clusters: int,
        config: Optional[ClusteringConfig] = None,
        random_state: int = 42,
    ) -> ClusteringResult:
        """
        Cluster using K-means.
        
        Args:
            n_clusters: Number of clusters
            config: ClusteringConfig
            random_state: Random seed
            
        Returns:
            ClusteringResult
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        if config is None:
            config = ClusteringConfig(algorithm="kmeans", n_clusters=n_clusters)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.features)
        
        silhouette = silhouette_score(self.features, labels) if n_clusters > 1 else None
        
        return ClusteringResult(
            labels=labels,
            ids=self.ids,
            silhouette_score=silhouette,
            config=config,
        )
    
    def find_optimal_hdbscan_params(
        self,
        min_cluster_sizes: List[int] = [3, 5, 10, 15, 20],
        min_samples_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Grid search for optimal HDBSCAN parameters.
        
        Args:
            min_cluster_sizes: Values to try for min_cluster_size
            min_samples_values: Values to try for min_samples
            
        Returns:
            Dict with best params and all results
        """
        results = []
        
        for mcs in min_cluster_sizes:
            for ms in min_samples_values:
                if ms > mcs:
                    continue  # min_samples should be <= min_cluster_size
                
                try:
                    result = self.cluster_hdbscan(
                        min_cluster_size=mcs,
                        min_samples=ms,
                    )
                    
                    results.append({
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "n_clusters": result.n_clusters,
                        "noise_ratio": result.noise_ratio,
                        "silhouette_score": result.silhouette_score,
                    })
                    
                except Exception as e:
                    results.append({
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "error": str(e),
                    })
        
        # Find best (balance between silhouette and noise ratio)
        valid_results = [r for r in results if r.get("silhouette_score") is not None]
        
        if valid_results:
            # Score: silhouette - noise_ratio (penalize high noise)
            for r in valid_results:
                r["score"] = (r["silhouette_score"] or 0) - r["noise_ratio"]
            
            best = max(valid_results, key=lambda x: x["score"])
        else:
            best = None
        
        return {
            "best_params": best,
            "all_results": results,
        }


@dataclass
class ExperimentResult:
    """Results from a clustering experiment."""
    
    experiment_id: str
    feature_config_id: str
    clustering_config: ClusteringConfig
    
    # Metrics
    n_clusters: int
    n_noise: int
    noise_ratio: float
    silhouette_score: Optional[float]
    cluster_sizes: Dict[int, int]
    
    # Division purity (if computed)
    division_purity: Optional[Dict[str, Any]] = None
    
    # Sample clusters for inspection
    sample_clusters: List[Dict[str, Any]] = field(default_factory=list)
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "feature_config_id": self.feature_config_id,
            "clustering_config": self.clustering_config.to_dict(),
            "metrics": {
                "n_clusters": self.n_clusters,
                "n_noise": self.n_noise,
                "noise_ratio": round(self.noise_ratio, 4),
                "silhouette_score": round(self.silhouette_score, 4) if self.silhouette_score else None,
                "cluster_size_mean": np.mean(list(self.cluster_sizes.values())) if self.cluster_sizes else 0,
                "cluster_size_std": np.std(list(self.cluster_sizes.values())) if self.cluster_sizes else 0,
            },
            "division_purity": self.division_purity,
            "sample_clusters": self.sample_clusters,
            "timestamp": self.timestamp,
        }


class ExperimentRunner:
    """
    Run clustering experiments with different configurations.
    """
    
    def __init__(
        self,
        extractions: List,  # List[ExtractionResult]
        metadata_df: Optional[pd.DataFrame] = None,
        id_field: str = "jd_id",
    ):
        """
        Initialize experiment runner.
        
        Args:
            extractions: List of ExtractionResult objects
            metadata_df: DataFrame with metadata for analysis
            id_field: ID field name
        """
        self.extractions = extractions
        self.metadata_df = metadata_df
        self.id_field = id_field
        
        # Build extraction lookup
        self.extraction_lookup = {e.jd_id: e for e in extractions}
    
    def run_experiment(
        self,
        features: np.ndarray,
        ids: List[str],
        feature_config_id: str,
        clustering_config: ClusteringConfig,
        experiment_id: Optional[str] = None,
        compute_purity: bool = True,
        purity_field: str = "division",
        n_sample_clusters: int = 5,
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            features: Feature vectors
            ids: JD IDs
            feature_config_id: ID of feature configuration used
            clustering_config: Clustering configuration
            experiment_id: Experiment ID (auto-generated if None)
            compute_purity: Whether to compute division purity
            purity_field: Field to compute purity against
            n_sample_clusters: Number of clusters to sample for inspection
            
        Returns:
            ExperimentResult
        """
        if experiment_id is None:
            experiment_id = f"{feature_config_id}_{clustering_config.algorithm}"
        
        # Run clustering
        clusterer = ArchetypeClusterer(features, ids)
        
        if clustering_config.algorithm == "hdbscan":
            result = clusterer.cluster_hdbscan(clustering_config)
        elif clustering_config.algorithm == "kmeans":
            result = clusterer.cluster_kmeans(
                n_clusters=clustering_config.n_clusters,
                config=clustering_config,
            )
        else:
            raise ValueError(f"Unknown algorithm: {clustering_config.algorithm}")
        
        # Compute purity if requested
        division_purity = None
        if compute_purity and self.metadata_df is not None:
            division_purity = self._compute_purity(result, purity_field)
        
        # Sample clusters for inspection
        sample_clusters = self._sample_clusters(result, n_sample_clusters)
        
        return ExperimentResult(
            experiment_id=experiment_id,
            feature_config_id=feature_config_id,
            clustering_config=clustering_config,
            n_clusters=result.n_clusters,
            n_noise=result.n_noise,
            noise_ratio=result.noise_ratio,
            silhouette_score=result.silhouette_score,
            cluster_sizes=result.cluster_sizes,
            division_purity=division_purity,
            sample_clusters=sample_clusters,
        )
    
    def _compute_purity(
        self,
        result: ClusteringResult,
        purity_field: str,
    ) -> Dict[str, Any]:
        """Compute cluster purity against a metadata field."""
        if self.metadata_df is None:
            return None
        
        if purity_field not in self.metadata_df.columns:
            return None
        
        # Build mapping from ID to label
        id_to_cluster = dict(zip(result.ids, result.labels))
        
        # Merge with metadata
        df = self.metadata_df.copy()
        df["cluster"] = df[self.id_field].astype(str).map(id_to_cluster)
        df = df.dropna(subset=["cluster"])
        df["cluster"] = df["cluster"].astype(int)
        
        # Exclude noise
        df = df[df["cluster"] != -1]
        
        if len(df) == 0:
            return None
        
        # Compute purity per cluster
        cluster_purities = {}
        total_correct = 0
        
        for cluster_id in df["cluster"].unique():
            cluster_data = df[df["cluster"] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Most common value
            mode_value = cluster_data[purity_field].mode()
            if len(mode_value) == 0:
                continue
            
            mode_value = mode_value.iloc[0]
            mode_count = (cluster_data[purity_field] == mode_value).sum()
            
            purity = mode_count / len(cluster_data)
            cluster_purities[int(cluster_id)] = {
                "dominant_value": str(mode_value),
                "purity": round(purity, 3),
                "size": len(cluster_data),
            }
            
            total_correct += mode_count
        
        overall_purity = total_correct / len(df) if len(df) > 0 else 0
        
        return {
            "field": purity_field,
            "overall_purity": round(overall_purity, 3),
            "cluster_purities": cluster_purities,
            "mean_purity": np.mean([c["purity"] for c in cluster_purities.values()]) if cluster_purities else 0,
        }
    
    def _sample_clusters(
        self,
        result: ClusteringResult,
        n_clusters: int,
    ) -> List[Dict[str, Any]]:
        """Sample clusters for inspection."""
        samples = []
        
        # Get cluster labels sorted by size
        sorted_clusters = sorted(
            [(k, v) for k, v in result.cluster_sizes.items() if k != -1],
            key=lambda x: x[1],
            reverse=True,
        )
        
        for cluster_id, size in sorted_clusters[:n_clusters]:
            cluster_ids = result.get_cluster_ids(cluster_id)
            
            # Get sample extractions
            sample_extractions = []
            for jd_id in cluster_ids[:3]:
                if jd_id in self.extraction_lookup:
                    ext = self.extraction_lookup[jd_id]
                    sample_extractions.append({
                        "jd_id": jd_id,
                        "skills": ext.get_all_skills_flat()[:10],
                        "metadata": ext.metadata,
                    })
            
            # Get skill frequency in cluster
            all_skills = []
            for jd_id in cluster_ids:
                if jd_id in self.extraction_lookup:
                    all_skills.extend(self.extraction_lookup[jd_id].get_all_skills_flat())
            
            from collections import Counter
            skill_freq = Counter(all_skills).most_common(10)
            
            samples.append({
                "cluster_id": cluster_id,
                "size": size,
                "top_skills": skill_freq,
                "sample_jds": sample_extractions,
            })
        
        return samples
    
    def run_experiment_matrix(
        self,
        feature_outputs: Dict,  # Dict[str, FeatureOutput]
        clustering_configs: List[ClusteringConfig],
        output_dir: str,
        show_progress: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run full experiment matrix.
        
        Args:
            feature_outputs: Dict mapping feature_config_id to FeatureOutput
            clustering_configs: List of clustering configs to try
            output_dir: Directory to save results
            show_progress: Print progress
            
        Returns:
            List of ExperimentResult
        """
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for feature_id, feature_output in feature_outputs.items():
            for cluster_config in clustering_configs:
                exp_id = f"{feature_id}_{cluster_config.algorithm}"
                if cluster_config.algorithm == "kmeans":
                    exp_id += f"_k{cluster_config.n_clusters}"
                
                if show_progress:
                    print(f"\nRunning: {exp_id}")
                
                result = self.run_experiment(
                    features=feature_output.features,
                    ids=feature_output.ids,
                    feature_config_id=feature_id,
                    clustering_config=cluster_config,
                    experiment_id=exp_id,
                )
                
                results.append(result)
                
                if show_progress:
                    print(f"  Clusters: {result.n_clusters}, Noise: {result.noise_ratio:.1%}, "
                          f"Silhouette: {result.silhouette_score:.4f}" if result.silhouette_score else "N/A")
        
        # Save all results
        all_results_data = [r.to_dict() for r in results]
        with open(output_path / "experiment_results.json", "w") as f:
            json.dump(all_results_data, f, indent=2, default=str)
        
        # Save comparison table
        comparison_df = pd.DataFrame([
            {
                "experiment_id": r.experiment_id,
                "feature_config": r.feature_config_id,
                "algorithm": r.clustering_config.algorithm,
                "n_clusters": r.n_clusters,
                "noise_ratio": round(r.noise_ratio, 4),
                "silhouette_score": round(r.silhouette_score, 4) if r.silhouette_score else None,
                "division_purity": r.division_purity.get("overall_purity") if r.division_purity else None,
            }
            for r in results
        ])
        comparison_df.to_csv(output_path / "experiment_comparison.csv", index=False)
        
        if show_progress:
            print(f"\nResults saved to: {output_path}/")
        
        return results
