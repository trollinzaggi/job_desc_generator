"""
Phase 1.3: JD Content Clustering

Components:
- Embedding generation (full JD and section-level)
- K-means and hierarchical clustering
- UMAP/t-SNE visualization
- Cluster analysis against metadata
"""

from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import json
from pathlib import Path

if TYPE_CHECKING:
    from .structure_analysis import JDStructure


@dataclass
class ClusterResult:
    """Results from a clustering run."""
    
    method: str
    n_clusters: int
    labels: np.ndarray
    silhouette_score: float
    cluster_sizes: Dict[int, int]
    
    # Optional: cluster centers for k-means
    centers: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "silhouette_score": round(self.silhouette_score, 4),
            "cluster_sizes": self.cluster_sizes,
        }


@dataclass
class SectionEmbeddings:
    """Container for section-level embeddings."""
    
    jd_id: str
    section_type: str
    content: str
    embedding: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jd_id": self.jd_id,
            "section_type": self.section_type,
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content,
        }


class EmbeddingGenerator:
    """
    Generate embeddings for JD text using Azure OpenAI.
    """
    
    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-02-01",
    ):
        """
        Initialize embedding generator.
        
        Args:
            deployment_name: Azure OpenAI deployment name (required)
            azure_endpoint: Azure OpenAI endpoint URL (or set AZURE_OPENAI_ENDPOINT env var)
            api_key: API key (or set AZURE_OPENAI_API_KEY env var)
            api_version: Azure OpenAI API version
        
        Example:
            generator = EmbeddingGenerator(
                deployment_name="text-embedding-ada-002",
                azure_endpoint="https://your-resource.openai.azure.com",
                api_key="your-api-key",
            )
        """
        import os
        
        self.deployment_name = deployment_name
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._client = None
        
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key or set AZURE_OPENAI_API_KEY env var."
            )
        if not self.azure_endpoint:
            raise ValueError(
                "Endpoint required. Pass azure_endpoint or set AZURE_OPENAI_ENDPOINT env var."
            )
    
    def _get_client(self):
        """Lazy load the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise ImportError("openai required. Install with: pip install openai")
            
            self._client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
            )
        return self._client
    
    def embed(self, texts: List[str], show_progress: bool = True, batch_size: int = 100) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Print progress updates
            batch_size: Number of texts per API call
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        client = self._get_client()
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 0]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        valid_embeddings = []
        total_batches = (len(valid_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(valid_texts), batch_size):
            if show_progress:
                print(f"Embedding batch {i//batch_size + 1}/{total_batches}")
            
            batch = valid_texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.deployment_name,
            )
            batch_embeddings = [e.embedding for e in response.data]
            valid_embeddings.extend(batch_embeddings)
        
        valid_embeddings = np.array(valid_embeddings)
        
        # If all texts were valid, return directly
        if len(valid_indices) == len(texts):
            return valid_embeddings
        
        # Otherwise, create full array with zeros for invalid texts
        embedding_dim = valid_embeddings.shape[1]
        full_embeddings = np.zeros((len(texts), embedding_dim))
        for new_idx, orig_idx in enumerate(valid_indices):
            full_embeddings[orig_idx] = valid_embeddings[new_idx]
        
        return full_embeddings
    
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_field: str = "jd_text",
        id_field: str = "jd_id",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for texts in a DataFrame.
        
        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        texts = df[text_field].fillna("").tolist()
        ids = df[id_field].astype(str).tolist()
        
        embeddings = self.embed(texts)
        
        return embeddings, ids
    
    def embed_sections(
        self,
        parsed_jds: List["JDStructure"],
        section_types: Optional[List[str]] = None,
        min_content_length: int = 50,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate embeddings for specific sections of parsed JDs.
        
        This enables section-level clustering to compare how similar JDs are
        in specific sections (e.g., responsibilities vs qualifications).
        
        Args:
            parsed_jds: List of JDStructure objects from StructureParser
            section_types: Section types to embed (e.g., ["responsibilities", "required_qualifications"])
                          If None, embeds all section types found
            min_content_length: Minimum content length to embed
            show_progress: Print progress updates
            
        Returns:
            Dict mapping section_type to:
                - embeddings: numpy array
                - ids: list of jd_ids (only JDs that have this section)
                - contents: list of section contents
        """
        # Collect sections by type
        sections_by_type: Dict[str, List[Dict]] = {}
        
        for jd in parsed_jds:
            for section in jd.sections:
                section_type = section.get("section_type", "other")
                content = section.get("content", "")
                
                # Filter by requested types
                if section_types and section_type not in section_types:
                    continue
                
                # Filter by content length
                if len(content) < min_content_length:
                    continue
                
                if section_type not in sections_by_type:
                    sections_by_type[section_type] = []
                
                sections_by_type[section_type].append({
                    "jd_id": jd.jd_id,
                    "content": content,
                })
        
        # Embed each section type
        results = {}
        
        for section_type, sections in sections_by_type.items():
            if show_progress:
                print(f"\nEmbedding {section_type} sections ({len(sections)} JDs)...")
            
            texts = [s["content"] for s in sections]
            ids = [s["jd_id"] for s in sections]
            
            if len(texts) < 2:
                print(f"  Skipping {section_type}: only {len(texts)} samples")
                continue
            
            embeddings = self.embed(texts, show_progress=show_progress)
            
            results[section_type] = {
                "embeddings": embeddings,
                "ids": ids,
                "contents": texts,
                "count": len(texts),
            }
            
            if show_progress:
                print(f"  Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
        
        return results
    
    def embed_sections_concatenated(
        self,
        parsed_jds: List["JDStructure"],
        section_types: List[str],
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate concatenated embeddings for multiple sections.
        
        Creates a combined embedding by concatenating embeddings of specified sections.
        JDs missing any section get zero vectors for that section.
        
        Args:
            parsed_jds: List of JDStructure objects
            section_types: Ordered list of section types to concatenate
            show_progress: Print progress updates
            
        Returns:
            Tuple of (concatenated embeddings, list of jd_ids)
        """
        # Build lookup: jd_id -> {section_type: content}
        jd_sections: Dict[str, Dict[str, str]] = {}
        
        for jd in parsed_jds:
            jd_sections[jd.jd_id] = {}
            for section in jd.sections:
                section_type = section.get("section_type", "other")
                if section_type in section_types:
                    jd_sections[jd.jd_id][section_type] = section.get("content", "")
        
        # Get all JD IDs
        all_ids = list(jd_sections.keys())
        
        # Embed each section type
        section_embeddings: Dict[str, np.ndarray] = {}
        embedding_dim = None
        
        for section_type in section_types:
            if show_progress:
                print(f"Embedding {section_type}...")
            
            texts = []
            for jd_id in all_ids:
                content = jd_sections[jd_id].get(section_type, "")
                texts.append(content)
            
            embeddings = self.embed(texts, show_progress=False)
            section_embeddings[section_type] = embeddings
            
            if embedding_dim is None:
                embedding_dim = embeddings.shape[1]
        
        # Concatenate
        concatenated = np.concatenate(
            [section_embeddings[st] for st in section_types],
            axis=1
        )
        
        if show_progress:
            print(f"\nConcatenated embedding shape: {concatenated.shape}")
            print(f"  {len(section_types)} sections × {embedding_dim} dims = {concatenated.shape[1]} total dims")
        
        return concatenated, all_ids


class ContentClusterer:
    """
    Cluster JDs by content similarity using embeddings.
    """
    
    def __init__(self, embeddings: np.ndarray, ids: List[str]):
        """
        Initialize clusterer with pre-computed embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            ids: List of JD IDs corresponding to each embedding
        """
        self.embeddings = embeddings
        self.ids = ids
        
        self.results: Dict[str, ClusterResult] = {}
    
    def kmeans(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
    ) -> ClusterResult:
        """
        Run K-means clustering.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            ClusterResult
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        sil_score = silhouette_score(self.embeddings, labels) if n_clusters > 1 else 0
        
        cluster_sizes = dict(pd.Series(labels).value_counts().sort_index())
        
        result = ClusterResult(
            method="kmeans",
            n_clusters=n_clusters,
            labels=labels,
            silhouette_score=sil_score,
            cluster_sizes=cluster_sizes,
            centers=kmeans.cluster_centers_,
        )
        
        self.results[f"kmeans_{n_clusters}"] = result
        return result
    
    def hierarchical(
        self,
        n_clusters: int = 10,
        linkage: str = "ward",
    ) -> ClusterResult:
        """
        Run hierarchical/agglomerative clustering.
        
        Args:
            n_clusters: Number of clusters
            linkage: Linkage method ("ward", "complete", "average", "single")
            
        Returns:
            ClusterResult
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = clustering.fit_predict(self.embeddings)
        
        sil_score = silhouette_score(self.embeddings, labels) if n_clusters > 1 else 0
        
        cluster_sizes = dict(pd.Series(labels).value_counts().sort_index())
        
        result = ClusterResult(
            method=f"hierarchical_{linkage}",
            n_clusters=n_clusters,
            labels=labels,
            silhouette_score=sil_score,
            cluster_sizes=cluster_sizes,
        )
        
        self.results[f"hierarchical_{linkage}_{n_clusters}"] = result
        return result
    
    def find_optimal_k(
        self,
        k_range: range = range(5, 51, 5),
        method: str = "kmeans",
    ) -> Dict[int, float]:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            k_range: Range of k values to test
            method: Clustering method to use
            
        Returns:
            Dict mapping k to silhouette score
        """
        scores = {}
        
        for k in k_range:
            print(f"Testing k={k}...")
            
            if method == "kmeans":
                result = self.kmeans(n_clusters=k)
            else:
                result = self.hierarchical(n_clusters=k)
            
            scores[k] = result.silhouette_score
        
        # Find best k
        best_k = max(scores, key=scores.get)
        print(f"\nBest k={best_k} with silhouette score={scores[best_k]:.4f}")
        
        return scores
    
    def reduce_dimensions(
        self,
        method: str = "umap",
        n_components: int = 2,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            method: "umap" or "tsne"
            n_components: Number of dimensions (2 or 3)
            random_state: Random seed
            
        Returns:
            Reduced embeddings array
        """
        if method == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn required. Install with: pip install umap-learn")
            
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
            )
            reduced = reducer.fit_transform(self.embeddings)
        
        elif method == "tsne":
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError("scikit-learn required.")
            
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=30,
            )
            reduced = reducer.fit_transform(self.embeddings)
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'umap' or 'tsne'")
        
        return reduced
    
    def get_cluster_assignments(self, result_key: str) -> pd.DataFrame:
        """
        Get cluster assignments as DataFrame.
        
        Args:
            result_key: Key from self.results (e.g., "kmeans_10")
            
        Returns:
            DataFrame with jd_id and cluster columns
        """
        if result_key not in self.results:
            raise ValueError(f"Result not found: {result_key}. Available: {list(self.results.keys())}")
        
        result = self.results[result_key]
        
        return pd.DataFrame({
            "jd_id": self.ids,
            "cluster": result.labels,
        })


class ClusterAnalyzer:
    """
    Analyze cluster composition against metadata.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        cluster_assignments: pd.DataFrame,
        id_field: str = "jd_id",
    ):
        """
        Initialize analyzer.
        
        Args:
            df: Original DataFrame with JD data and metadata
            cluster_assignments: DataFrame with jd_id and cluster columns
            id_field: ID field name
        """
        # Merge cluster assignments with original data
        df = df.copy()
        df[id_field] = df[id_field].astype(str)
        cluster_assignments = cluster_assignments.copy()
        cluster_assignments["jd_id"] = cluster_assignments["jd_id"].astype(str)
        
        self.df = df.merge(
            cluster_assignments,
            left_on=id_field,
            right_on="jd_id",
            how="inner",
        )
    
    def analyze_cluster_composition(
        self,
        metadata_fields: List[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze what metadata values are in each cluster.
        
        Args:
            metadata_fields: Fields to analyze (e.g., ["org_unit", "level", "function"])
            
        Returns:
            Dict mapping cluster ID to composition analysis
        """
        if metadata_fields is None:
            metadata_fields = ["org_unit", "level", "function"]
        
        # Filter to available fields
        metadata_fields = [f for f in metadata_fields if f in self.df.columns]
        
        cluster_analysis = {}
        
        for cluster_id in sorted(self.df["cluster"].unique()):
            cluster_data = self.df[self.df["cluster"] == cluster_id]
            
            analysis = {
                "size": len(cluster_data),
                "percentage": round(len(cluster_data) / len(self.df) * 100, 2),
            }
            
            for field in metadata_fields:
                value_counts = cluster_data[field].value_counts()
                top_values = value_counts.head(5).to_dict()
                
                # Dominant value (if any)
                if len(value_counts) > 0:
                    top_value = value_counts.index[0]
                    top_pct = value_counts.iloc[0] / len(cluster_data) * 100
                    
                    analysis[field] = {
                        "top_values": top_values,
                        "dominant": top_value if top_pct > 40 else None,
                        "dominant_pct": round(top_pct, 1),
                        "unique_values": len(value_counts),
                    }
            
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def find_cluster_archetypes(
        self,
        text_field: str = "jd_text",
        title_field: str = "title",
        n_examples: int = 3,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Find representative examples for each cluster.
        
        Args:
            text_field: Field containing JD text
            title_field: Field containing job title
            n_examples: Number of examples per cluster
            
        Returns:
            Dict with archetype information per cluster
        """
        archetypes = {}
        
        for cluster_id in sorted(self.df["cluster"].unique()):
            cluster_data = self.df[self.df["cluster"] == cluster_id]
            
            # Sample examples
            sample_size = min(n_examples, len(cluster_data))
            examples = cluster_data.sample(n=sample_size, random_state=42)
            
            # Get common title patterns
            if title_field in cluster_data.columns:
                common_titles = cluster_data[title_field].value_counts().head(5).to_dict()
            else:
                common_titles = {}
            
            archetype = {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "common_titles": common_titles,
                "examples": [],
            }
            
            for _, row in examples.iterrows():
                example = {
                    "title": row.get(title_field, "Unknown"),
                    "text_preview": str(row.get(text_field, ""))[:500] + "...",
                }
                archetype["examples"].append(example)
            
            archetypes[cluster_id] = archetype
        
        return archetypes
    
    def compute_cluster_purity(
        self,
        label_field: str,
    ) -> Dict[str, Any]:
        """
        Compute cluster purity with respect to a label field.
        
        High purity means clusters align well with the label.
        
        Args:
            label_field: Metadata field to compute purity against
            
        Returns:
            Purity metrics
        """
        if label_field not in self.df.columns:
            raise ValueError(f"Field not found: {label_field}")
        
        cluster_purities = {}
        total_correct = 0
        
        for cluster_id in self.df["cluster"].unique():
            cluster_data = self.df[self.df["cluster"] == cluster_id]
            
            # Most common label in this cluster
            mode_label = cluster_data[label_field].mode().iloc[0] if len(cluster_data) > 0 else None
            mode_count = (cluster_data[label_field] == mode_label).sum()
            
            purity = mode_count / len(cluster_data) if len(cluster_data) > 0 else 0
            cluster_purities[cluster_id] = {
                "dominant_label": mode_label,
                "purity": round(purity, 3),
                "size": len(cluster_data),
            }
            
            total_correct += mode_count
        
        overall_purity = total_correct / len(self.df) if len(self.df) > 0 else 0
        
        return {
            "overall_purity": round(overall_purity, 3),
            "cluster_purities": cluster_purities,
        }
    
    def print_cluster_report(self, metadata_fields: List[str] = None) -> None:
        """Print human-readable cluster analysis report."""
        composition = self.analyze_cluster_composition(metadata_fields)
        
        if not composition:
            print("No clusters to analyze.")
            return
        
        print("=" * 60)
        print("CLUSTER COMPOSITION REPORT")
        print("=" * 60)
        
        for cluster_id, analysis in composition.items():
            print(f"\n{'='*40}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'='*40}")
            print(f"Size: {analysis.get('size', 0)} JDs ({analysis.get('percentage', 0)}%)")
            
            for field, field_analysis in analysis.items():
                if field in ["size", "percentage"]:
                    continue
                
                if not isinstance(field_analysis, dict):
                    continue
                
                print(f"\n{field.upper()}:")
                dominant = field_analysis.get("dominant")
                dominant_pct = field_analysis.get("dominant_pct", 0)
                
                if dominant:
                    print(f"  Dominant: {dominant} ({dominant_pct}%)")
                else:
                    print(f"  No dominant value (top: {dominant_pct}%)")
                
                print(f"  Unique values: {field_analysis.get('unique_values', 0)}")
                
                top_values = field_analysis.get("top_values", {})
                if top_values:
                    print("  Top values:")
                    for value, count in list(top_values.items())[:3]:
                        print(f"    - {value}: {count}")
    
    def export_results(self, output_dir: Optional[str] = None) -> None:
        """Export cluster analysis results to files.
        
        Args:
            output_dir: Directory to save results. If None, uses OUTPUT_CONFIG from config.py
        """
        if output_dir is None:
            try:
                from config import get_phase_output_path
                output_path = get_phase_output_path("phase_1_3_clustering")
            except ImportError:
                output_path = Path("cluster_analysis")
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Cluster assignments
        self.df[["jd_id", "cluster"]].to_csv(
            output_path / "cluster_assignments.csv", index=False
        )
        
        # Composition analysis
        composition = self.analyze_cluster_composition()
        with open(output_path / "cluster_composition.json", "w") as f:
            json.dump(composition, f, indent=2, default=str)
        
        # Archetypes
        archetypes = self.find_cluster_archetypes()
        with open(output_path / "cluster_archetypes.json", "w") as f:
            json.dump(archetypes, f, indent=2, default=str)
        
        print(f"Results exported to: {output_path}/")


def create_visualization_data(
    reduced_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    ids: List[str],
    metadata_df: Optional[pd.DataFrame] = None,
    id_field: str = "jd_id",
) -> pd.DataFrame:
    """
    Create DataFrame for visualization tools.
    
    Args:
        reduced_embeddings: 2D or 3D reduced embeddings
        cluster_labels: Cluster assignments
        ids: JD IDs
        metadata_df: Optional metadata to include
        id_field: ID field name
        
    Returns:
        DataFrame ready for plotting
    """
    viz_df = pd.DataFrame({
        "jd_id": ids,
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "cluster": cluster_labels,
    })
    
    if reduced_embeddings.shape[1] > 2:
        viz_df["z"] = reduced_embeddings[:, 2]
    
    # Merge metadata if provided
    if metadata_df is not None:
        metadata_df = metadata_df.copy()
        metadata_df[id_field] = metadata_df[id_field].astype(str)
        viz_df = viz_df.merge(metadata_df, left_on="jd_id", right_on=id_field, how="left")
    
    return viz_df
