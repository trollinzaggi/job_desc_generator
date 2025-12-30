#!/usr/bin/env python3
"""
Job Archetype Pipeline Runner

This script runs the complete archetype generation pipeline:
1. Extract requirements from JDs using LLM
2. Build features for clustering
3. Cluster JDs using HDBSCAN
4. Aggregate clusters into archetypes
5. Generate archetype names using LLM

Usage:
    # Run full pipeline
    python run_archetype_pipeline.py --all
    
    # Run specific phases
    python run_archetype_pipeline.py --extract          # Phase 1
    python run_archetype_pipeline.py --features         # Phase 2
    python run_archetype_pipeline.py --cluster          # Phase 3
    python run_archetype_pipeline.py --aggregate        # Phase 4
    python run_archetype_pipeline.py --name             # Phase 5
    
    # Run experiments
    python run_archetype_pipeline.py --experiments      # Feature/cluster experiments
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def get_archetype_output_path() -> Path:
    """Get output path for archetype pipeline."""
    try:
        from config import get_output_path
        return get_output_path("archetypes")
    except ImportError:
        return Path("analysis_output/archetypes")


def load_data():
    """Load JD data using existing data loader."""
    from config import JSON_CONFIG, JD_FIELD_MAPPING
    from src.data_loaders import JSONFileLoader
    from src.analysis import DataCleaner
    from config import ANALYSIS_CONFIG
    
    print("Loading data...")
    loader = JSONFileLoader(
        data_path=JSON_CONFIG["data_path"],
        content_key=JSON_CONFIG["content_key"],
        field_mapping=JD_FIELD_MAPPING,
    )
    
    df = loader.load_as_dataframe()
    print(f"  Loaded {len(df)} records")
    
    # Clean
    cleaner = DataCleaner(
        text_field=ANALYSIS_CONFIG.primary_text_field,
        id_field=ANALYSIS_CONFIG.id_field,
        min_text_length=50,
    )
    
    df_clean, stats = cleaner.clean(df)
    print(f"  After cleaning: {len(df_clean)} records")
    
    return df_clean


def run_extraction(df: pd.DataFrame, output_dir: Path, args):
    """Phase 1: Extract requirements from JDs."""
    from src.archetypes import RequirementExtractor
    from config import ANALYSIS_CONFIG
    
    print("\n" + "="*60)
    print("PHASE 1: REQUIREMENT EXTRACTION")
    print("="*60)
    
    output_path = output_dir / "phase_1_extraction"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for existing extraction
    extraction_file = output_path / "extracted_requirements.json"
    if extraction_file.exists() and not args.force:
        print(f"  Found existing extraction: {extraction_file}")
        print("  Use --force to re-extract")
        return RequirementExtractor.load_results(str(extraction_file))
    
    # Get field names from config
    jd_id_field = ANALYSIS_CONFIG.id_field
    jd_text_field = ANALYSIS_CONFIG.primary_text_field
    
    # Try to find expertise and team description fields
    expertise_field = None
    team_desc_field = None
    
    for field in df.columns:
        if "expertise" in field.lower():
            expertise_field = field
        if "team" in field.lower() and "desc" in field.lower():
            team_desc_field = field
    
    # Metadata fields to capture
    metadata_fields = [
        f for f in ["title", "level", "division", "function", "org_unit", "department"]
        if f in df.columns
    ]
    
    print(f"  ID field: {jd_id_field}")
    print(f"  Text field: {jd_text_field}")
    print(f"  Expertise field: {expertise_field}")
    print(f"  Team description field: {team_desc_field}")
    print(f"  Metadata fields: {metadata_fields}")
    
    # Initialize extractor
    extractor = RequirementExtractor()
    
    # Run extraction
    results = extractor.extract_batch(
        df=df,
        jd_id_field=jd_id_field,
        job_description_field=jd_text_field,
        expertise_field=expertise_field,
        team_description_field=team_desc_field,
        metadata_fields=metadata_fields,
        output_path=str(extraction_file),
        save_interval=50,
    )
    
    # Summary
    successful = sum(1 for r in results if r.extraction_success)
    print(f"\n  Extraction complete: {successful}/{len(results)} successful")
    print(f"  Results saved to: {extraction_file}")
    
    return results


def run_features(extractions, output_dir: Path, args):
    """Phase 2: Build features for clustering."""
    from src.archetypes import FeatureEngineer, FeatureConfig, FeatureApproach
    from src.archetypes.feature_engineering import get_experiment_configs
    
    print("\n" + "="*60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*60)
    
    output_path = output_dir / "phase_2_features"
    output_path.mkdir(parents=True, exist_ok=True)
    
    engineer = FeatureEngineer()
    
    if args.experiments:
        # Run all experiment configurations
        print("  Running experiment matrix...")
        configs = get_experiment_configs()
        feature_outputs = engineer.build_multiple_configs(
            extractions=extractions,
            configs=configs,
            output_dir=str(output_path),
        )
        return feature_outputs
    else:
        # Run single configuration (default: skills only)
        config = FeatureConfig(
            approach=FeatureApproach.SKILLS_ONLY,
            experiment_id="default",
        )
        
        print(f"  Building features with approach: {config.approach.value}")
        output = engineer.build_features(extractions, config)
        output.save(str(output_path / "default"))
        
        return {"default": output}


def run_clustering(feature_outputs, extractions, df, output_dir: Path, args):
    """Phase 3: Cluster JDs."""
    from src.archetypes import ArchetypeClusterer, ClusteringConfig, ExperimentRunner
    
    print("\n" + "="*60)
    print("PHASE 3: CLUSTERING")
    print("="*60)
    
    output_path = output_dir / "phase_3_clustering"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.experiments:
        # Run experiment matrix
        print("  Running clustering experiments...")
        
        runner = ExperimentRunner(
            extractions=extractions,
            metadata_df=df,
            id_field="jd_id",
        )
        
        # Clustering configs to try
        clustering_configs = [
            ClusteringConfig(
                algorithm="hdbscan",
                min_cluster_size=5,
                min_samples=3,
            ),
            ClusteringConfig(
                algorithm="hdbscan",
                min_cluster_size=10,
                min_samples=5,
            ),
        ]
        
        results = runner.run_experiment_matrix(
            feature_outputs=feature_outputs,
            clustering_configs=clustering_configs,
            output_dir=str(output_path),
        )
        
        # Print comparison
        print("\n  Experiment Results:")
        print("  " + "-"*50)
        for r in results:
            sil = f"{r.silhouette_score:.4f}" if r.silhouette_score else "N/A"
            print(f"  {r.experiment_id}: clusters={r.n_clusters}, noise={r.noise_ratio:.1%}, silhouette={sil}")
        
        return results
    else:
        # Run single clustering
        feature_output = list(feature_outputs.values())[0]
        
        clusterer = ArchetypeClusterer(
            features=feature_output.features,
            ids=feature_output.ids,
        )
        
        # Find optimal params
        print("  Finding optimal HDBSCAN parameters...")
        param_search = clusterer.find_optimal_hdbscan_params()
        
        if param_search["best_params"]:
            best = param_search["best_params"]
            print(f"  Best params: min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}")
            print(f"    Clusters: {best['n_clusters']}, Noise: {best['noise_ratio']:.1%}")
        
        # Run with best params or default
        result = clusterer.cluster_hdbscan()
        result.save(str(output_path / "default"))
        
        print(f"\n  Clustering complete:")
        print(f"    Clusters: {result.n_clusters}")
        print(f"    Noise: {result.n_noise} ({result.noise_ratio:.1%})")
        print(f"    Silhouette: {result.silhouette_score:.4f}" if result.silhouette_score else "")
        
        return result


def run_aggregation(extractions, clustering_result, features, output_dir: Path, args):
    """Phase 4: Aggregate into archetypes."""
    from src.archetypes import ArchetypeAggregator
    from src.archetypes.clustering import ClusteringResult
    
    print("\n" + "="*60)
    print("PHASE 4: AGGREGATION")
    print("="*60)
    
    output_path = output_dir / "phase_4_aggregation"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load clustering result if needed
    if clustering_result is None:
        clustering_path = output_dir / "phase_3_clustering" / "default"
        clustering_result = ClusteringResult.load(str(clustering_path))
    
    # Aggregate
    aggregator = ArchetypeAggregator(
        extractions=extractions,
        clustering_result=clustering_result,
        features=features,
    )
    
    archetypes = aggregator.aggregate_all()
    
    # Save
    aggregator.save_archetypes(archetypes, str(output_path))
    
    print(f"\n  Created {len(archetypes)} archetypes")
    
    # Print summary
    for archetype in archetypes[:5]:
        print(f"\n  Cluster {archetype.cluster_id}:")
        print(f"    Members: {archetype.member_count}")
        print(f"    Titles: {', '.join(archetype.representative_titles[:2])}")
    
    if len(archetypes) > 5:
        print(f"\n  ... and {len(archetypes) - 5} more")
    
    return archetypes


def run_naming(archetypes, output_dir: Path, args):
    """Phase 5: Generate archetype names."""
    from src.archetypes import ArchetypeNamer
    
    print("\n" + "="*60)
    print("PHASE 5: NAMING")
    print("="*60)
    
    output_path = output_dir / "phase_5_naming"
    output_path.mkdir(parents=True, exist_ok=True)
    
    namer = ArchetypeNamer()
    
    # Generate names
    print("  Generating archetype names...")
    archetypes = namer.name_all_archetypes(archetypes)
    
    # Standardize skills
    print("\n  Standardizing skills...")
    archetypes, skill_mappings = namer.standardize_archetype_skills(archetypes)
    
    # Save
    namer.save_skill_mappings(skill_mappings, str(output_path / "skill_mappings.json"))
    
    # Save final archetypes
    from src.archetypes.aggregation import ArchetypeAggregator
    ArchetypeAggregator.save_archetypes = staticmethod(
        lambda a, p: None  # Already handled below
    )
    
    # Save to final location
    final_data = [a.to_dict() for a in archetypes]
    with open(output_path / "final_archetypes.json", "w") as f:
        json.dump(final_data, f, indent=2, default=str)
    
    # Print results
    print(f"\n  Named {len(archetypes)} archetypes:")
    for archetype in archetypes[:10]:
        print(f"    - {archetype.label} ({archetype.member_count} members)")
    
    if len(archetypes) > 10:
        print(f"    ... and {len(archetypes) - 10} more")
    
    print(f"\n  Results saved to: {output_path}/")
    
    return archetypes


def main():
    parser = argparse.ArgumentParser(
        description="Job Archetype Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Phase arguments
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--extract", action="store_true", help="Phase 1: Extract requirements")
    parser.add_argument("--features", action="store_true", help="Phase 2: Build features")
    parser.add_argument("--cluster", action="store_true", help="Phase 3: Cluster JDs")
    parser.add_argument("--aggregate", action="store_true", help="Phase 4: Aggregate archetypes")
    parser.add_argument("--name", action="store_true", help="Phase 5: Name archetypes")
    
    # Options
    parser.add_argument("--experiments", action="store_true", help="Run experiment matrix")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs exist")
    parser.add_argument("--output", "-o", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    # Default to --all if no phase specified
    if not any([args.all, args.extract, args.features, args.cluster, args.aggregate, args.name]):
        args.all = True
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else get_archetype_output_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("JOB ARCHETYPE PIPELINE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    
    # Load data
    df = load_data()
    
    # Track outputs
    extractions = None
    feature_outputs = None
    clustering_result = None
    features = None
    archetypes = None
    
    # Phase 1: Extraction
    if args.all or args.extract:
        extractions = run_extraction(df, output_dir, args)
    else:
        # Load existing
        extraction_file = output_dir / "phase_1_extraction" / "extracted_requirements.json"
        if extraction_file.exists():
            from src.archetypes import RequirementExtractor
            extractions = RequirementExtractor.load_results(str(extraction_file))
            print(f"\n  Loaded {len(extractions)} existing extractions")
    
    # Phase 2: Features
    if args.all or args.features:
        if extractions is None:
            raise ValueError("Extractions required. Run --extract first.")
        feature_outputs = run_features(extractions, output_dir, args)
        # Get features array for later
        features = list(feature_outputs.values())[0].features if feature_outputs else None
    else:
        # Load existing
        feature_path = output_dir / "phase_2_features" / "default"
        if feature_path.exists():
            from src.archetypes.feature_engineering import FeatureOutput
            feature_output = FeatureOutput.load(str(feature_path))
            feature_outputs = {"default": feature_output}
            features = feature_output.features
            print(f"\n  Loaded existing features: {features.shape}")
    
    # Phase 3: Clustering
    if args.all or args.cluster:
        if feature_outputs is None:
            raise ValueError("Features required. Run --features first.")
        clustering_result = run_clustering(feature_outputs, extractions, df, output_dir, args)
        
        # Handle experiment results vs single result
        if isinstance(clustering_result, list):
            # Use best experiment result
            clustering_result = clustering_result[0]  # Or select best
    else:
        # Load existing
        clustering_path = output_dir / "phase_3_clustering" / "default"
        if clustering_path.exists():
            from src.archetypes.clustering import ClusteringResult
            clustering_result = ClusteringResult.load(str(clustering_path))
            print(f"\n  Loaded existing clustering: {clustering_result.n_clusters} clusters")
    
    # Phase 4: Aggregation
    if args.all or args.aggregate:
        if extractions is None or clustering_result is None:
            raise ValueError("Extractions and clustering required. Run previous phases first.")
        archetypes = run_aggregation(extractions, clustering_result, features, output_dir, args)
    else:
        # Load existing
        archetype_file = output_dir / "phase_4_aggregation" / "archetypes.json"
        if archetype_file.exists():
            from src.archetypes.aggregation import ArchetypeAggregator
            archetypes = ArchetypeAggregator.load_archetypes(str(archetype_file))
            print(f"\n  Loaded {len(archetypes)} existing archetypes")
    
    # Phase 5: Naming
    if args.all or args.name:
        if archetypes is None:
            raise ValueError("Archetypes required. Run --aggregate first.")
        archetypes = run_naming(archetypes, output_dir, args)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
