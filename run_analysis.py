"""
Phase 1 Corpus Analysis Pipeline

Uses configuration from config.py for field mappings and analysis settings.

Usage:
    python run_analysis.py --discover     # Discover schema
    python run_analysis.py --all          # Run all phases
    python run_analysis.py --phase 1.1    # Run specific phase
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loaders import JSONFileLoader
from src.analysis import (
    DataCleaner,
    QualityEvaluator,
    StructureAnalyzer,
    ContentClusterer,
    ClusterAnalyzer,
    create_visualization_data,
)
from config import (
    JSON_CONFIG, 
    JD_FIELD_MAPPING, 
    ANALYSIS_CONFIG,
    OUTPUT_CONFIG,
    get_output_path,
    get_phase_output_path,
)


# Constants
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_K_VALUES = [10, 20, 30, 50]
MIN_CLUSTER_SIZE = 10


def check_config():
    """Check if configuration is valid."""
    mapping = JD_FIELD_MAPPING.to_dict()
    
    if not mapping:
        print("[ERROR] No field mappings configured in config.py")
        print("        Run 'python run_analysis.py --discover' first")
        return False
    
    errors = []
    
    if ANALYSIS_CONFIG.primary_text_field not in mapping:
        errors.append(
            f"primary_text_field '{ANALYSIS_CONFIG.primary_text_field}' not in JD_FIELD_MAPPING"
        )
    
    if ANALYSIS_CONFIG.id_field not in mapping:
        errors.append(
            f"id_field '{ANALYSIS_CONFIG.id_field}' not in JD_FIELD_MAPPING"
        )
    
    if errors:
        print("[ERROR] Configuration errors:")
        for err in errors:
            print(f"    - {err}")
        return False
    
    print("[OK] Field mappings configured:")
    for field, path in mapping.items():
        print(f"    {field} <- {path}")
    
    print(f"\n[OK] Analysis config:")
    print(f"    Primary text field: {ANALYSIS_CONFIG.primary_text_field}")
    print(f"    ID field: {ANALYSIS_CONFIG.id_field}")
    print(f"    Stratify by: {ANALYSIS_CONFIG.stratify_by_primary}, {ANALYSIS_CONFIG.stratify_by_secondary}")
    print(f"    Cluster metadata: {ANALYSIS_CONFIG.cluster_metadata_fields}")
    
    return True


def run_schema_discovery():
    """Discover schema structure."""
    print("=" * 60)
    print("SCHEMA DISCOVERY")
    print("=" * 60)
    
    loader = JSONFileLoader(
        data_path=JSON_CONFIG["data_path"],
        content_key=JSON_CONFIG["content_key"],
    )
    
    print(f"File stats: {loader.get_file_stats()}")
    print(f"Total records: {loader.count_records()}")
    
    schema = loader.discover_schema(sample_size=100)
    print("\n" + schema.print_schema_tree())
    
    print("\nSuggested mappings:")
    for field, path in schema.suggest_field_mapping().items():
        print(f"    {field}: \"{path}\"")
    
    # Use configured output path
    output_dir = get_phase_output_path("schema_discovery")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "schema_discovery_output.json"
    with open(output_file, "w") as f:
        f.write(schema.to_json(indent=2))
    print(f"\nSchema exported to: {output_file}")


def load_and_clean_data():
    """Load and clean the JD corpus using config."""
    print("\n[LOADING DATA]")
    
    loader = JSONFileLoader(
        data_path=JSON_CONFIG["data_path"],
        content_key=JSON_CONFIG["content_key"],
        field_mapping=JD_FIELD_MAPPING,
    )
    
    df = loader.load_as_dataframe()
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    validation = ANALYSIS_CONFIG.validate(list(df.columns))
    if validation["errors"]:
        print("\n[ERROR] Configuration errors:")
        for err in validation["errors"]:
            print(f"    - {err}")
        raise ValueError("Fix configuration errors before proceeding")
    
    if validation["warnings"]:
        print("\n[WARNING] Configuration warnings:")
        for warn in validation["warnings"]:
            print(f"    - {warn}")
    
    print("\n[CLEANING DATA]")
    cleaner = DataCleaner(
        text_field=ANALYSIS_CONFIG.primary_text_field,
        id_field=ANALYSIS_CONFIG.id_field,
        min_text_length=50
    )
    df_clean, stats = cleaner.clean(df)
    print(stats.print_report())
    
    return df_clean


def get_date_field(df):
    """Get the date field if available in the dataframe."""
    if ANALYSIS_CONFIG.date_field in df.columns:
        return ANALYSIS_CONFIG.date_field
    return None


def create_evaluator(df):
    """Create a QualityEvaluator with proper field configuration."""
    primary, secondary = ANALYSIS_CONFIG.get_available_stratify_fields(list(df.columns))
    date_field = get_date_field(df)
    
    return QualityEvaluator(
        df=df,
        text_field=ANALYSIS_CONFIG.primary_text_field,
        id_field=ANALYSIS_CONFIG.id_field,
        org_unit_field=primary or ANALYSIS_CONFIG.id_field,
        level_field=secondary or ANALYSIS_CONFIG.id_field,
        date_field=date_field or ANALYSIS_CONFIG.date_field,
    )


def run_phase_1_1(df):
    """Phase 1.1: JD Quality Baseline"""
    print("\n" + "=" * 60)
    print("PHASE 1.1: JD QUALITY BASELINE")
    print("=" * 60)
    
    primary, secondary = ANALYSIS_CONFIG.get_available_stratify_fields(list(df.columns))
    date_field = get_date_field(df)
    
    evaluator = create_evaluator(df)
    
    if primary and secondary:
        print(f"\nStratifying by: {primary} x {secondary}")
        sample = evaluator.create_evaluation_sample(n=DEFAULT_SAMPLE_SIZE)
    elif primary:
        print(f"\nStratifying by: {primary} only")
        sample = evaluator.create_evaluation_sample(n=DEFAULT_SAMPLE_SIZE)
    else:
        print("\nNo stratification fields available - using random sampling")
        sample = df.sample(n=min(DEFAULT_SAMPLE_SIZE, len(df)), random_state=42)
        evaluator.sample_df = sample
    
    print(f"Created sample of {len(sample)} JDs")
    
    phase_dir = get_phase_output_path("phase_1_1_quality")
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    export_fields = [ANALYSIS_CONFIG.id_field, ANALYSIS_CONFIG.primary_text_field]
    export_fields.extend([
        f for f in ANALYSIS_CONFIG.quality_export_fields 
        if f in df.columns
    ])
    if date_field:
        export_fields.append(date_field)
    
    evaluator.export_for_evaluation(
        str(phase_dir / "jd_quality_evaluation.json"),
        include_fields=export_fields
    )
    evaluator.export_for_evaluation_csv(str(phase_dir / "jd_quality_evaluation.csv"))
    
    print(f"\n[OK] Sample exported to: {phase_dir}/")
    print("\nNEXT STEPS:")
    print("1. Open jd_quality_evaluation.csv")
    print("2. Score each JD on eval_* columns (1-5 scale)")
    print("3. Run: python run_analysis.py --analyze-quality")


def run_phase_1_2(df):
    """Phase 1.2: JD Structural Consistency"""
    print("\n" + "=" * 60)
    print("PHASE 1.2: JD STRUCTURAL CONSISTENCY")
    print("=" * 60)
    
    print(f"\nAnalyzing text field: {ANALYSIS_CONFIG.primary_text_field}")
    
    analyzer = StructureAnalyzer()
    parsed_jds = analyzer.parse_corpus(
        df, 
        text_field=ANALYSIS_CONFIG.primary_text_field,
        id_field=ANALYSIS_CONFIG.id_field
    )
    
    analyzer.print_structure_report()
    
    structure_clusters = analyzer.cluster_by_structure(n_clusters=5)
    print(f"\nStructure clustering silhouette: {structure_clusters['silhouette_score']}")
    
    phase_dir = get_phase_output_path("phase_1_2_structure")
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.export_parsed_jds(str(phase_dir / "parsed_jd_structures.json"))
    
    consistency = analyzer.measure_consistency()
    with open(phase_dir / "structure_consistency.json", "w") as f:
        json.dump(consistency, f, indent=2, default=str)
    
    with open(phase_dir / "structure_clusters.json", "w") as f:
        json.dump(structure_clusters, f, indent=2, default=str)
    
    print(f"\n[OK] Results exported to: {phase_dir}/")
    
    for text_field in ANALYSIS_CONFIG.additional_text_fields:
        if text_field in df.columns:
            print(f"\n[ADDITIONAL ANALYSIS: {text_field}]")
            additional_analyzer = StructureAnalyzer()
            additional_analyzer.parse_corpus(
                df,
                text_field=text_field,
                id_field=ANALYSIS_CONFIG.id_field
            )
            
            add_consistency = additional_analyzer.measure_consistency()
            with open(phase_dir / f"structure_consistency_{text_field}.json", "w") as f:
                json.dump(add_consistency, f, indent=2, default=str)
            print(f"  Exported: structure_consistency_{text_field}.json")
    
    return parsed_jds


def run_phase_1_3(df, parsed_jds=None):
    """Phase 1.3: JD Content Clustering"""
    print("\n" + "=" * 60)
    print("PHASE 1.3: JD CONTENT CLUSTERING")
    print("=" * 60)
    
    phase_dir = get_phase_output_path("phase_1_3_clustering")
    phase_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEmbedding text field: {ANALYSIS_CONFIG.primary_text_field}")
    
    # Import embedding config
    from config import get_embedding_generator
    embedder = get_embedding_generator()
    
    embeddings, ids = embedder.embed_dataframe(
        df, 
        text_field=ANALYSIS_CONFIG.primary_text_field,
        id_field=ANALYSIS_CONFIG.id_field
    )
    print(f"Generated {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
    
    np.save(phase_dir / "embeddings.npy", embeddings)
    with open(phase_dir / "embedding_ids.json", "w") as f:
        json.dump(ids, f)
    
    clusterer = ContentClusterer(embeddings, ids)
    
    k_scores = {}
    valid_k_values = [k for k in DEFAULT_K_VALUES if k < len(df)]
    
    if not valid_k_values:
        print(f"[WARNING] Dataset too small for clustering (n={len(df)})")
        print(f"          Need at least {min(DEFAULT_K_VALUES) + 1} records")
        return
    
    for k in valid_k_values:
        result = clusterer.kmeans(n_clusters=k)
        k_scores[k] = result.silhouette_score
        print(f"  k={k}: silhouette={result.silhouette_score:.4f}")
    
    best_k = max(k_scores, key=k_scores.get)
    print(f"\nBest k={best_k}")
    
    best_result = clusterer.results[f"kmeans_{best_k}"]
    cluster_df = clusterer.get_cluster_assignments(f"kmeans_{best_k}")
    
    metadata_fields = ANALYSIS_CONFIG.get_available_cluster_fields(list(df.columns))
    print(f"\nAnalyzing clusters against: {metadata_fields}")
    
    cluster_analyzer = ClusterAnalyzer(
        df, cluster_df, 
        id_field=ANALYSIS_CONFIG.id_field
    )
    cluster_analyzer.print_cluster_report(metadata_fields=metadata_fields)
    
    if ANALYSIS_CONFIG.cluster_purity_field in df.columns:
        purity = cluster_analyzer.compute_cluster_purity(ANALYSIS_CONFIG.cluster_purity_field)
        print(f"\nCluster purity (vs {ANALYSIS_CONFIG.cluster_purity_field}): {purity['overall_purity']:.3f}")
    
    try:
        reduced = clusterer.reduce_dimensions(method="umap", n_components=2)
        viz_df = create_visualization_data(
            reduced, best_result.labels, ids, df,
            id_field=ANALYSIS_CONFIG.id_field
        )
        viz_df.to_csv(phase_dir / "visualization_data.csv", index=False)
        print("Visualization data saved")
    except ImportError:
        print("UMAP not installed - skipping visualization")
    
    cluster_df.to_csv(phase_dir / "cluster_assignments.csv", index=False)
    
    with open(phase_dir / "k_scores.json", "w") as f:
        json.dump(k_scores, f, indent=2)
    
    composition = cluster_analyzer.analyze_cluster_composition(metadata_fields)
    with open(phase_dir / "cluster_composition.json", "w") as f:
        json.dump(composition, f, indent=2, default=str)
    
    title_field = "title" if "title" in df.columns else ANALYSIS_CONFIG.id_field
    archetypes = cluster_analyzer.find_cluster_archetypes(
        text_field=ANALYSIS_CONFIG.primary_text_field,
        title_field=title_field
    )
    with open(phase_dir / "cluster_archetypes.json", "w") as f:
        json.dump(archetypes, f, indent=2, default=str)
    
    print(f"\n[OK] Results exported to: {phase_dir}/")
    
    if parsed_jds:
        run_section_level_clustering(embedder, parsed_jds, phase_dir)
    
    for text_field in ANALYSIS_CONFIG.additional_text_fields:
        if text_field in df.columns:
            print(f"\n[ADDITIONAL CLUSTERING: {text_field}]")
            add_embeddings, add_ids = embedder.embed_dataframe(
                df, text_field=text_field, id_field=ANALYSIS_CONFIG.id_field
            )
            
            add_clusterer = ContentClusterer(add_embeddings, add_ids)
            add_result = add_clusterer.kmeans(n_clusters=best_k)
            
            add_cluster_df = add_clusterer.get_cluster_assignments(f"kmeans_{best_k}")
            add_cluster_df.to_csv(
                phase_dir / f"cluster_assignments_{text_field}.csv", 
                index=False
            )
            print(f"  Silhouette: {add_result.silhouette_score:.4f}")
            print(f"  Exported: cluster_assignments_{text_field}.csv")


def run_section_level_clustering(embedder, parsed_jds, phase_dir: Path):
    """Run clustering on individual sections."""
    print("\n[SECTION-LEVEL EMBEDDINGS]")
    
    section_types = ANALYSIS_CONFIG.section_types_to_embed
    section_results = embedder.embed_sections(
        parsed_jds,
        section_types=section_types,
        min_content_length=50,
    )
    
    for section_type, data in section_results.items():
        if data["count"] < MIN_CLUSTER_SIZE:
            print(f"  Skipping {section_type}: only {data['count']} samples (need {MIN_CLUSTER_SIZE})")
            continue
        
        n_clusters = min(10, max(2, data["count"] // 5))
        print(f"\n  Clustering {section_type} ({data['count']} JDs, k={n_clusters})...")
        
        section_clusterer = ContentClusterer(data["embeddings"], data["ids"])
        section_result = section_clusterer.kmeans(n_clusters=n_clusters)
        
        section_cluster_df = section_clusterer.get_cluster_assignments(f"kmeans_{n_clusters}")
        section_cluster_df.to_csv(
            phase_dir / f"cluster_assignments_section_{section_type}.csv",
            index=False
        )
        print(f"    Silhouette: {section_result.silhouette_score:.4f}")


def analyze_imported_quality():
    """Import and analyze completed quality evaluations."""
    phase_dir = get_phase_output_path("phase_1_1_quality")
    csv_path = phase_dir / "jd_quality_evaluation.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return
    
    print("=" * 60)
    print("ANALYZING QUALITY EVALUATIONS")
    print("=" * 60)
    
    df = load_and_clean_data()
    evaluator = create_evaluator(df)
    evaluator.create_evaluation_sample(n=DEFAULT_SAMPLE_SIZE)
    
    count = evaluator.import_evaluations(str(csv_path))
    
    if count == 0:
        print("No evaluations found. Fill in the eval_* columns.")
        return
    
    evaluator.print_quality_report()
    
    results = evaluator.analyze_quality()
    with open(phase_dir / "quality_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    gold_df = evaluator.get_gold_standard_jds()
    if len(gold_df) > 0:
        gold_df.to_csv(phase_dir / "gold_standard_jds.csv", index=False)
        print("\n[OK] Gold standard JDs saved")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Corpus Analysis")
    parser.add_argument("--discover", action="store_true", help="Run schema discovery")
    parser.add_argument("--phase", choices=["1.1", "1.2", "1.3"], help="Run specific phase")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--analyze-quality", action="store_true", help="Import quality evaluations")
    
    args = parser.parse_args()
    
    # Ensure output root directory exists
    output_root = get_output_path()
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"[CONFIG] Output directory: {output_root.absolute()}")
    
    if args.discover:
        run_schema_discovery()
        return
    
    if args.analyze_quality:
        analyze_imported_quality()
        return
    
    if args.phase or args.all:
        if not check_config():
            return
        
        df = load_and_clean_data()
        parsed_jds = None
        
        if args.phase == "1.1" or args.all:
            run_phase_1_1(df)
        
        if args.phase == "1.2" or args.all:
            parsed_jds = run_phase_1_2(df)
        
        if args.phase == "1.3" or args.all:
            run_phase_1_3(df, parsed_jds)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {output_root.absolute()}")
        return
    
    parser.print_help()
    print("\n" + "=" * 60)
    print("QUICK START")
    print("=" * 60)
    print(f"""
1. Put JSON files in jd_data/
2. Discover schema:     python run_analysis.py --discover
3. Edit config.py:
   - Set JD_FIELD_MAPPING with your field paths
   - Set ANALYSIS_CONFIG with your analysis preferences
   - Set OUTPUT_CONFIG["root_dir"] for output location
4. Run analysis:        python run_analysis.py --all

Current output directory: {OUTPUT_CONFIG['root_dir']}
""")


if __name__ == "__main__":
    main()
