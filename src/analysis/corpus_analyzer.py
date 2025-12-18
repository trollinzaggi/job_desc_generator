"""
Corpus analysis pipeline for JD data.

NOTE: This module is deprecated in favor of run_analysis.py which uses
the centralized configuration from config.py. This module is maintained
for backward compatibility and programmatic usage.

For new projects, use:
    python run_analysis.py --all

For programmatic usage with config:
    from config import JD_FIELD_MAPPING, ANALYSIS_CONFIG
    analyzer = CorpusAnalyzer.from_config(
        data_path="jd_data",
        field_mapping=JD_FIELD_MAPPING,
        analysis_config=ANALYSIS_CONFIG,
    )
    results = analyzer.run_full_analysis()
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import pandas as pd

from ..data_loaders import JSONFileLoader, CosmosDBLoader, FieldMapping, SchemaDiscovery
from .data_cleaner import DataCleaner, CleaningStats

if TYPE_CHECKING:
    from config import AnalysisConfig


@dataclass
class CorpusStats:
    """Statistics about the JD corpus."""
    
    total_jds: int = 0
    unique_titles: int = 0
    unique_org_units: int = 0
    unique_levels: int = 0
    
    # Text statistics
    avg_text_length: float = 0
    min_text_length: int = 0
    max_text_length: int = 0
    
    # Distribution of metadata
    title_distribution: Dict[str, int] = field(default_factory=dict)
    org_unit_distribution: Dict[str, int] = field(default_factory=dict)
    level_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Date range
    earliest_date: Optional[str] = None
    latest_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jds": self.total_jds,
            "unique_titles": self.unique_titles,
            "unique_org_units": self.unique_org_units,
            "unique_levels": self.unique_levels,
            "text_stats": {
                "avg_length": self.avg_text_length,
                "min_length": self.min_text_length,
                "max_length": self.max_text_length,
            },
            "date_range": {
                "earliest": self.earliest_date,
                "latest": self.latest_date,
            },
            "distributions": {
                "titles_top_20": dict(list(self.title_distribution.items())[:20]),
                "org_units_top_20": dict(list(self.org_unit_distribution.items())[:20]),
                "levels": self.level_distribution,
            }
        }


class CorpusAnalyzer:
    """
    Main analysis pipeline for JD corpus.
    
    Workflow:
        1. Initialize with data source configuration
        2. Run schema discovery (if needed)
        3. Load and clean data
        4. Compute corpus statistics
        5. Export results
    
    Recommended usage with config.py:
        analyzer = CorpusAnalyzer.from_config(
            data_path="jd_data",
            field_mapping=JD_FIELD_MAPPING,
            analysis_config=ANALYSIS_CONFIG,
        )
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        field_mapping: Optional[FieldMapping] = None,
        cosmos_config: Optional[Dict] = None,
        content_key: str = "content",
        # New: analysis config support
        text_field: str = "jd_text",
        id_field: str = "jd_id",
        org_unit_field: str = "org_unit",
        level_field: str = "level",
        title_field: str = "title",
        date_field: str = "posting_date",
    ):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to JSON files (for local data)
            field_mapping: FieldMapping configuration
            cosmos_config: Cosmos DB configuration (alternative to data_path)
            content_key: Key containing JD list in JSON files
            text_field: Column containing JD text (from ANALYSIS_CONFIG)
            id_field: Column containing JD ID (from ANALYSIS_CONFIG)
            org_unit_field: Column containing org unit
            level_field: Column containing level
            title_field: Column containing title
            date_field: Column containing posting date
        """
        self.data_path = data_path
        self.field_mapping = field_mapping
        self.cosmos_config = cosmos_config
        self.content_key = content_key
        
        # Field configuration (can be overridden by ANALYSIS_CONFIG)
        self.text_field = text_field
        self.id_field = id_field
        self.org_unit_field = org_unit_field
        self.level_field = level_field
        self.title_field = title_field
        self.date_field = date_field
        
        # Analysis artifacts
        self.loader = None
        self.schema: Optional[SchemaDiscovery] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.clean_df: Optional[pd.DataFrame] = None
        self.cleaning_stats: Optional[CleaningStats] = None
        self.corpus_stats: Optional[CorpusStats] = None
    
    @classmethod
    def from_config(
        cls,
        data_path: str,
        field_mapping: FieldMapping,
        analysis_config: "AnalysisConfig",
        content_key: str = "content",
    ) -> "CorpusAnalyzer":
        """
        Create analyzer from config.py configuration.
        
        Args:
            data_path: Path to JSON files
            field_mapping: JD_FIELD_MAPPING from config.py
            analysis_config: ANALYSIS_CONFIG from config.py
            content_key: Key containing JD list in JSON files
            
        Returns:
            Configured CorpusAnalyzer instance
        """
        return cls(
            data_path=data_path,
            field_mapping=field_mapping,
            content_key=content_key,
            text_field=analysis_config.primary_text_field,
            id_field=analysis_config.id_field,
            org_unit_field=analysis_config.stratify_by_primary,
            level_field=analysis_config.stratify_by_secondary,
            title_field="title",  # Standard field
            date_field="posting_date",  # Standard field
        )
    
    def _create_loader(self):
        """Create appropriate loader based on configuration."""
        if self.data_path:
            self.loader = JSONFileLoader(
                data_path=self.data_path,
                content_key=self.content_key,
                field_mapping=self.field_mapping,
            )
        elif self.cosmos_config:
            self.loader = CosmosDBLoader.from_env(**self.cosmos_config)
            self.loader.field_mapping = self.field_mapping
        else:
            raise ValueError("Either data_path or cosmos_config must be provided")
    
    # =========================================================================
    # STEP 1: Schema Discovery
    # =========================================================================
    
    def discover_schema(self, sample_size: int = 100) -> SchemaDiscovery:
        """
        Discover data schema (run this first if you don't know field names).
        
        Args:
            sample_size: Number of records to sample
            
        Returns:
            SchemaDiscovery object
        """
        if self.loader is None:
            self._create_loader()
        
        self.schema = self.loader.discover_schema(sample_size=sample_size)
        return self.schema
    
    def print_schema(self) -> None:
        """Print discovered schema."""
        if self.schema is None:
            self.discover_schema()
        
        print(self.schema.print_schema_tree())
        
        print("\nSuggested field mappings:")
        for field_name, path in self.schema.suggest_field_mapping().items():
            print(f"  {field_name}: {path}")
    
    # =========================================================================
    # STEP 2: Load Data
    # =========================================================================
    
    def load_data(self, apply_mapping: bool = True) -> pd.DataFrame:
        """
        Load data from source.
        
        Args:
            apply_mapping: Whether to apply field mapping
            
        Returns:
            DataFrame with loaded records
        """
        if self.loader is None:
            self._create_loader()
        
        self.raw_df = self.loader.load_as_dataframe(apply_mapping=apply_mapping)
        print(f"Loaded {len(self.raw_df)} records")
        return self.raw_df
    
    # =========================================================================
    # STEP 3: Clean Data
    # =========================================================================
    
    def clean_data(
        self,
        text_field: Optional[str] = None,
        min_text_length: int = 50,
        categorical_fields: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, CleaningStats]:
        """
        Clean and transform the data.
        
        Args:
            text_field: Column containing JD text (defaults to self.text_field)
            min_text_length: Minimum text length to keep
            categorical_fields: Fields to analyze distributions for
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning statistics)
        """
        if self.raw_df is None:
            self.load_data()
        
        # Use configured text field if not specified
        text_field = text_field or self.text_field
        
        # Default categorical fields from config
        if categorical_fields is None:
            categorical_fields = [
                self.org_unit_field, self.level_field, 
                self.title_field, "function", "status"
            ]
        
        cleaner = DataCleaner(
            text_field=text_field,
            id_field=self.id_field,
            min_text_length=min_text_length,
        )
        
        self.clean_df, self.cleaning_stats = cleaner.clean(
            self.raw_df,
            categorical_fields=categorical_fields,
        )
        
        print(self.cleaning_stats.print_report())
        return self.clean_df, self.cleaning_stats
    
    # =========================================================================
    # STEP 4: Analyze Corpus
    # =========================================================================
    
    def analyze_corpus(
        self,
        text_field: Optional[str] = None,
        title_field: Optional[str] = None,
        org_unit_field: Optional[str] = None,
        level_field: Optional[str] = None,
        date_field: Optional[str] = None,
    ) -> CorpusStats:
        """
        Compute corpus statistics.
        
        Args:
            text_field: Column containing JD text
            title_field: Column containing job title
            org_unit_field: Column containing org unit
            level_field: Column containing job level
            date_field: Column containing posting date
            
        Returns:
            CorpusStats object
        """
        # Use configured fields if not specified
        text_field = text_field or self.text_field
        title_field = title_field or self.title_field
        org_unit_field = org_unit_field or self.org_unit_field
        level_field = level_field or self.level_field
        date_field = date_field or self.date_field
        
        if self.clean_df is None:
            self.clean_data(text_field=text_field)
        
        df = self.clean_df
        self.corpus_stats = CorpusStats()
        
        # Basic counts
        self.corpus_stats.total_jds = len(df)
        
        # Unique values
        if title_field in df.columns:
            self.corpus_stats.unique_titles = df[title_field].nunique()
            self.corpus_stats.title_distribution = df[title_field].value_counts().to_dict()
        
        if org_unit_field in df.columns:
            self.corpus_stats.unique_org_units = df[org_unit_field].nunique()
            self.corpus_stats.org_unit_distribution = df[org_unit_field].value_counts().to_dict()
        
        if level_field in df.columns:
            self.corpus_stats.unique_levels = df[level_field].nunique()
            self.corpus_stats.level_distribution = df[level_field].value_counts().to_dict()
        
        # Text statistics
        if text_field in df.columns:
            text_lengths = df[text_field].str.len()
            self.corpus_stats.avg_text_length = round(text_lengths.mean(), 1)
            self.corpus_stats.min_text_length = int(text_lengths.min())
            self.corpus_stats.max_text_length = int(text_lengths.max())
        
        # Date range
        if date_field in df.columns:
            dates = pd.to_datetime(df[date_field], errors="coerce")
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                self.corpus_stats.earliest_date = str(valid_dates.min())
                self.corpus_stats.latest_date = str(valid_dates.max())
        
        return self.corpus_stats
    
    def print_corpus_stats(self) -> None:
        """Print corpus statistics."""
        if self.corpus_stats is None:
            self.analyze_corpus()
        
        stats = self.corpus_stats
        
        print("=" * 60)
        print("CORPUS STATISTICS")
        print("=" * 60)
        print(f"\nTotal JDs: {stats.total_jds}")
        print(f"Unique Titles: {stats.unique_titles}")
        print(f"Unique Org Units: {stats.unique_org_units}")
        print(f"Unique Levels: {stats.unique_levels}")
        print(f"\nText Length: avg={stats.avg_text_length}, min={stats.min_text_length}, max={stats.max_text_length}")
        
        if stats.earliest_date:
            print(f"\nDate Range: {stats.earliest_date} to {stats.latest_date}")
        
        print("\nTop 10 Titles:")
        for title, count in list(stats.title_distribution.items())[:10]:
            print(f"  {title}: {count}")
        
        print("\nTop 10 Org Units:")
        for org, count in list(stats.org_unit_distribution.items())[:10]:
            print(f"  {org}: {count}")
        
        print("\nLevel Distribution:")
        for level, count in stats.level_distribution.items():
            print(f"  {level}: {count}")
    
    # =========================================================================
    # STEP 5: Run Full Analysis
    # =========================================================================
    
    def run_full_analysis(
        self,
        text_field: Optional[str] = None,
        categorical_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            text_field: Column containing JD text (defaults to config)
            categorical_fields: Fields to analyze distributions for
            
        Returns:
            Dictionary containing all analysis results
        """
        # Use configured text field if not specified
        text_field = text_field or self.text_field
        
        print("=" * 60)
        print("RUNNING FULL CORPUS ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load
        print("\n[1/4] Loading data...")
        self.load_data()
        
        # Step 2: Clean
        print("\n[2/4] Cleaning data...")
        self.clean_data(text_field=text_field, categorical_fields=categorical_fields)
        
        # Step 3: Analyze
        print("\n[3/4] Computing corpus statistics...")
        self.analyze_corpus(text_field=text_field)
        
        # Step 4: Compile results
        print("\n[4/4] Compiling results...")
        results = {
            "cleaning_stats": self.cleaning_stats.to_dict(),
            "corpus_stats": self.corpus_stats.to_dict(),
            "sample_records": self.clean_df.head(5).to_dict(orient="records"),
        }
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return results
    
    # =========================================================================
    # STEP 6: Export Results
    # =========================================================================
    
    def export_results(self, output_dir: Optional[str] = None) -> None:
        """
        Export analysis results to files.
        
        Args:
            output_dir: Directory to save results. If None, uses OUTPUT_CONFIG from config.py
        """
        if output_dir is None:
            try:
                from config import get_output_path
                output_path = get_output_path()
            except ImportError:
                output_path = Path("analysis_output")
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export cleaning stats
        if self.cleaning_stats:
            with open(output_path / "cleaning_stats.json", "w") as f:
                json.dump(self.cleaning_stats.to_dict(), f, indent=2)
            print(f"Saved: {output_path}/cleaning_stats.json")
        
        # Export corpus stats
        if self.corpus_stats:
            with open(output_path / "corpus_stats.json", "w") as f:
                json.dump(self.corpus_stats.to_dict(), f, indent=2)
            print(f"Saved: {output_path}/corpus_stats.json")
        
        # Export cleaned data (without _raw field)
        if self.clean_df is not None:
            # Remove raw column if exists
            export_df = self.clean_df.drop(columns=["_raw"], errors="ignore")
            export_df.to_csv(output_path / "cleaned_jds.csv", index=False)
            print(f"Saved: {output_path}/cleaned_jds.csv")
        
        # Export schema
        if self.schema:
            with open(output_path / "schema.json", "w") as f:
                f.write(self.schema.to_json(indent=2))
            print(f"Saved: {output_path}/schema.json")
        
        print(f"\nAll results exported to: {output_path}/")
    
    def get_clean_dataframe(self) -> pd.DataFrame:
        """Get the cleaned DataFrame for further analysis."""
        if self.clean_df is None:
            raise ValueError("Run clean_data() first")
        return self.clean_df
