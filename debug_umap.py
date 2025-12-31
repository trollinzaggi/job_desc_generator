#!/usr/bin/env python3
"""
Debug script for UMAP visualization issues.

Run this to diagnose why UMAP points aren't showing:
    python debug_umap.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd

def main():
    print("="*60)
    print("UMAP VISUALIZATION DEBUGGER")
    print("="*60)
    
    from config import get_output_path
    
    # 1. Check feature outputs
    print("\n1. Checking feature outputs...")
    feature_dir = get_output_path("archetypes", "phase_2_features")
    
    if not feature_dir.exists():
        print(f"   ❌ Feature directory not found: {feature_dir}")
        print("   Run: python run_archetype_pipeline.py --features")
        return
    
    feature_subdirs = [d for d in feature_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(feature_subdirs)} feature outputs: {[d.name for d in feature_subdirs]}")
    
    # Load first feature output
    feature_path = feature_subdirs[0] if feature_subdirs else None
    if not feature_path:
        print("   ❌ No feature outputs found")
        return
    
    features_file = feature_path / "features.npy"
    ids_file = feature_path / "feature_ids.json"
    
    if not features_file.exists():
        print(f"   ❌ Features file not found: {features_file}")
        return
    
    features = np.load(features_file)
    print(f"   ✅ Features shape: {features.shape}")
    
    if ids_file.exists():
        with open(ids_file) as f:
            feature_ids = json.load(f)
        print(f"   ✅ Feature IDs: {len(feature_ids)}")
        print(f"      Sample IDs: {feature_ids[:3]}")
        print(f"      ID type: {type(feature_ids[0])}")
    else:
        print("   ❌ Feature IDs file not found")
        feature_ids = None
    
    # 2. Check clustering results
    print("\n2. Checking clustering results...")
    cluster_dir = get_output_path("archetypes", "phase_3_clustering")
    
    if not cluster_dir.exists():
        print(f"   ❌ Clustering directory not found: {cluster_dir}")
        print("   Run: python run_archetype_pipeline.py --cluster")
        return
    
    cluster_subdirs = [d for d in cluster_dir.iterdir() if d.is_dir() and (d / "cluster_labels.npy").exists()]
    print(f"   Found {len(cluster_subdirs)} clustering results: {[d.name for d in cluster_subdirs]}")
    
    cluster_path = cluster_subdirs[0] if cluster_subdirs else None
    if not cluster_path:
        print("   ❌ No clustering results found")
        return
    
    labels = np.load(cluster_path / "cluster_labels.npy")
    print(f"   ✅ Cluster labels: {len(labels)}")
    print(f"      Unique clusters: {len(set(labels) - {-1})} + noise")
    print(f"      Noise points: {sum(labels == -1)}")
    
    # Load cluster IDs
    assignments_file = cluster_path / "cluster_assignments.csv"
    if assignments_file.exists():
        assignments_df = pd.read_csv(assignments_file)
        cluster_ids = assignments_df['jd_id'].astype(str).tolist()
        print(f"   ✅ Cluster IDs: {len(cluster_ids)}")
        print(f"      Sample IDs: {cluster_ids[:3]}")
        print(f"      ID type: {type(cluster_ids[0])}")
    else:
        print("   ❌ Cluster assignments file not found")
        cluster_ids = None
    
    # 3. Check ID matching
    print("\n3. Checking ID matching...")
    if feature_ids and cluster_ids:
        # Convert both to strings for comparison
        feature_ids_str = set(str(i) for i in feature_ids)
        cluster_ids_str = set(str(i) for i in cluster_ids)
        
        overlap = feature_ids_str & cluster_ids_str
        print(f"   Feature IDs: {len(feature_ids_str)}")
        print(f"   Cluster IDs: {len(cluster_ids_str)}")
        print(f"   Overlap: {len(overlap)}")
        
        if len(overlap) == 0:
            print("\n   ❌ NO ID OVERLAP! This is the problem.")
            print("      Feature ID samples:", list(feature_ids_str)[:5])
            print("      Cluster ID samples:", list(cluster_ids_str)[:5])
            
            # Check if they're the same but different format
            if len(feature_ids) == len(cluster_ids):
                print("\n   ℹ️  Same count - might be positional match possible")
        elif len(overlap) < len(feature_ids_str):
            print(f"\n   ⚠️  Partial overlap: {len(overlap)}/{len(feature_ids_str)} IDs match")
        else:
            print("\n   ✅ All IDs match!")
    
    # 4. Test UMAP
    print("\n4. Testing UMAP computation...")
    try:
        import umap
        print("   ✅ umap-learn is installed")
        
        # Quick test
        print("   Computing UMAP (this may take a minute)...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(features)
        print(f"   ✅ UMAP computed: {embedding.shape}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(embedding)):
            print("   ⚠️  UMAP contains NaN values!")
        if np.any(np.isinf(embedding)):
            print("   ⚠️  UMAP contains Inf values!")
        
        # Check range
        print(f"   X range: [{embedding[:, 0].min():.2f}, {embedding[:, 0].max():.2f}]")
        print(f"   Y range: [{embedding[:, 1].min():.2f}, {embedding[:, 1].max():.2f}]")
        
    except ImportError:
        print("   ❌ umap-learn not installed")
        print("   Install with: pip install umap-learn")
    
    # 5. Summary
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if feature_ids and cluster_ids:
        feature_ids_str = set(str(i) for i in feature_ids)
        cluster_ids_str = set(str(i) for i in cluster_ids)
        overlap = feature_ids_str & cluster_ids_str
        
        if len(overlap) == 0:
            print("""
❌ PROBLEM: Feature IDs and Cluster IDs don't match!

This happens when:
1. Features and clustering were run on different data
2. ID format changed between phases (string vs int)

SOLUTION: Re-run both phases on the same data:
    python run_archetype_pipeline.py --features --cluster --force
""")
        elif len(features) != len(labels):
            print(f"""
❌ PROBLEM: Feature count ({len(features)}) != Label count ({len(labels)})

SOLUTION: Re-run clustering on the current features:
    python run_archetype_pipeline.py --cluster --force
""")
        else:
            print("""
✅ Data looks OK. The issue might be in the notebook.

Try running this test plot:
    python debug_umap.py --plot
""")
    
    # Optional: create a test plot
    if "--plot" in sys.argv:
        print("\nCreating test plot...")
        try:
            import plotly.express as px
            
            # Build plot data
            if feature_ids and cluster_ids and len(feature_ids) == len(cluster_ids):
                # Use positional matching
                plot_df = pd.DataFrame({
                    'jd_id': feature_ids,
                    'x': embedding[:, 0],
                    'y': embedding[:, 1],
                    'cluster': labels,
                })
            else:
                # Try ID matching
                id_to_cluster = {str(k): v for k, v in zip(cluster_ids, labels)}
                plot_df = pd.DataFrame({
                    'jd_id': [str(i) for i in feature_ids],
                    'x': embedding[:, 0],
                    'y': embedding[:, 1],
                })
                plot_df['cluster'] = plot_df['jd_id'].map(id_to_cluster).fillna(-2).astype(int)
            
            plot_df['cluster_label'] = plot_df['cluster'].apply(
                lambda x: 'Unmatched' if x == -2 else ('Noise' if x == -1 else f'Cluster {x}')
            )
            
            fig = px.scatter(
                plot_df, x='x', y='y',
                color='cluster_label',
                title=f'UMAP Test Plot ({len(plot_df)} points)',
                hover_data=['jd_id', 'cluster']
            )
            fig.update_traces(marker=dict(size=6, opacity=0.7))
            
            # Save to HTML
            output_file = "debug_umap_plot.html"
            fig.write_html(output_file)
            print(f"✅ Plot saved to: {output_file}")
            print("   Open this file in a browser to view the plot.")
            
        except ImportError as e:
            print(f"❌ Could not create plot: {e}")
            print("   Install plotly: pip install plotly")


if __name__ == "__main__":
    main()
