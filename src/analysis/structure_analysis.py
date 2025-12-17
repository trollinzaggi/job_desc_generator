"""
Phase 1.2: JD Structural Consistency Analysis

Components:
- Section header extraction using patterns and heuristics
- Common section identification
- Structural consistency measurement
- Structure-based clustering
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


@dataclass
class JDStructure:
    """Represents the extracted structure of a single JD."""
    
    jd_id: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    
    # Detected sections with content
    # Each section: {"header": str, "content": str, "start_pos": int, "end_pos": int}
    
    def section_names(self) -> List[str]:
        """Get list of section header names."""
        return [s["header"] for s in self.sections]
    
    def has_section(self, pattern: str) -> bool:
        """Check if JD has a section matching pattern."""
        pattern_lower = pattern.lower()
        return any(pattern_lower in s["header"].lower() for s in self.sections)
    
    def get_section_content(self, pattern: str) -> Optional[str]:
        """Get content of section matching pattern."""
        pattern_lower = pattern.lower()
        for s in self.sections:
            if pattern_lower in s["header"].lower():
                return s["content"]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jd_id": self.jd_id,
            "num_sections": len(self.sections),
            "section_names": self.section_names(),
            "sections": self.sections,
        }


class StructureParser:
    """
    Extract section structure from JD text.
    
    Uses multiple strategies:
    1. Common header patterns (e.g., "Responsibilities:", "Requirements:")
    2. Formatting cues (bold, caps, line breaks)
    3. Bullet point groupings
    """
    
    # Common section header patterns
    STANDARD_SECTIONS = {
        "summary": [
            r"(?:job\s+)?(?:summary|overview|description|about\s+(?:the\s+)?(?:role|position|job))",
            r"position\s+summary",
            r"role\s+(?:summary|overview)",
        ],
        "responsibilities": [
            r"(?:key\s+)?responsibilities",
            r"(?:job\s+)?duties",
            r"what\s+you(?:'ll|\s+will)\s+do",
            r"your\s+responsibilities",
            r"the\s+role",
            r"accountabilities",
        ],
        "required_qualifications": [
            r"(?:required|minimum|basic)\s+qualifications",
            r"requirements",
            r"what\s+you(?:'ll|\s+will)\s+(?:need|bring)",
            r"must\s+have",
            r"required\s+(?:skills|experience)",
            r"you\s+have",
        ],
        "preferred_qualifications": [
            r"(?:preferred|desired|nice\s+to\s+have)\s+qualifications",
            r"preferred\s+(?:skills|experience)",
            r"bonus\s+(?:points|qualifications)",
            r"it(?:'s|\s+is)\s+a\s+plus",
            r"ideally",
        ],
        "education": [
            r"education(?:al)?\s+(?:requirements?|background)",
            r"degree\s+requirements?",
        ],
        "experience": [
            r"(?:work\s+)?experience",
            r"years?\s+(?:of\s+)?experience",
        ],
        "skills": [
            r"(?:required\s+|key\s+)?skills",
            r"technical\s+skills",
            r"competenc(?:y|ies)",
        ],
        "benefits": [
            r"benefits",
            r"what\s+we\s+offer",
            r"perks",
            r"compensation",
        ],
        "team_info": [
            r"(?:about\s+)?(?:the\s+)?team",
            r"who\s+we\s+are",
            r"our\s+team",
        ],
        "company_info": [
            r"(?:about\s+)?(?:the\s+)?company",
            r"about\s+us",
            r"who\s+we\s+are",
        ],
        "location": [
            r"location",
            r"work\s+location",
            r"where\s+you(?:'ll|\s+will)\s+work",
        ],
        "application": [
            r"how\s+to\s+apply",
            r"application\s+(?:process|instructions?)",
            r"to\s+apply",
        ],
    }
    
    def __init__(
        self,
        custom_patterns: Optional[Dict[str, List[str]]] = None,
        min_section_length: int = 20,
    ):
        """
        Initialize parser.
        
        Args:
            custom_patterns: Additional section patterns to recognize
            min_section_length: Minimum characters for a valid section
        """
        self.patterns = {**self.STANDARD_SECTIONS}
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        self.min_section_length = min_section_length
        
        # Compile patterns
        self._compiled_patterns = {}
        for section_type, patterns in self.patterns.items():
            combined = "|".join(f"(?:{p})" for p in patterns)
            self._compiled_patterns[section_type] = re.compile(
                combined, re.IGNORECASE
            )
    
    def _find_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Find potential section headers in text.
        
        Returns list of {"text": str, "start": int, "end": int, "type": str}
        """
        headers = []
        
        # Strategy 1: Lines ending with colon
        colon_pattern = re.compile(r'^([A-Z][^:\n]{2,50}):[ \t]*$', re.MULTILINE)
        for match in colon_pattern.finditer(text):
            headers.append({
                "text": match.group(1).strip(),
                "start": match.start(),
                "end": match.end(),
                "type": "colon",
            })
        
        # Strategy 2: ALL CAPS lines (likely headers)
        caps_pattern = re.compile(r'^([A-Z][A-Z\s]{3,50})$', re.MULTILINE)
        for match in caps_pattern.finditer(text):
            header_text = match.group(1).strip()
            if len(header_text.split()) <= 6:  # Not too many words
                headers.append({
                    "text": header_text,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "caps",
                })
        
        # Strategy 3: Lines matching known patterns
        for section_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Check if this is at start of line or after newline
                start = match.start()
                line_start = text.rfind('\n', 0, start) + 1
                prefix = text[line_start:start].strip()
                
                if len(prefix) < 10:  # Near start of line
                    headers.append({
                        "text": match.group(0).strip(),
                        "start": match.start(),
                        "end": match.end(),
                        "type": section_type,
                    })
        
        # Deduplicate and sort by position
        seen_starts = set()
        unique_headers = []
        for h in sorted(headers, key=lambda x: x["start"]):
            # Allow some tolerance for overlapping matches
            if not any(abs(h["start"] - s) < 10 for s in seen_starts):
                unique_headers.append(h)
                seen_starts.add(h["start"])
        
        return unique_headers
    
    def _classify_header(self, header_text: str) -> str:
        """Classify a header into a standard section type."""
        header_lower = header_text.lower()
        
        for section_type, pattern in self._compiled_patterns.items():
            if pattern.search(header_lower):
                return section_type
        
        return "other"
    
    def parse(self, text: str, jd_id: str = "") -> JDStructure:
        """
        Parse JD text into structured sections.
        
        Args:
            text: JD text content
            jd_id: Optional JD identifier
            
        Returns:
            JDStructure with extracted sections
        """
        if not text:
            return JDStructure(jd_id=jd_id, raw_text=text)
        
        headers = self._find_headers(text)
        
        if not headers:
            # No clear structure - treat entire text as one section
            return JDStructure(
                jd_id=jd_id,
                sections=[{
                    "header": "content",
                    "content": text.strip(),
                    "start_pos": 0,
                    "end_pos": len(text),
                    "section_type": "unstructured",
                }],
                raw_text=text,
            )
        
        # Extract sections between headers
        sections = []
        for i, header in enumerate(headers):
            start = header["end"]
            end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
            
            content = text[start:end].strip()
            
            if len(content) >= self.min_section_length:
                sections.append({
                    "header": header["text"],
                    "content": content,
                    "start_pos": header["start"],
                    "end_pos": end,
                    "section_type": self._classify_header(header["text"]),
                })
        
        # Check for content before first header
        if headers and headers[0]["start"] > self.min_section_length:
            preamble = text[:headers[0]["start"]].strip()
            if len(preamble) >= self.min_section_length:
                sections.insert(0, {
                    "header": "preamble",
                    "content": preamble,
                    "start_pos": 0,
                    "end_pos": headers[0]["start"],
                    "section_type": "summary",
                })
        
        return JDStructure(jd_id=jd_id, sections=sections, raw_text=text)


class StructureAnalyzer:
    """
    Analyze structural patterns across JD corpus.
    
    Outputs:
    - Common section types and their frequency
    - Structural consistency metrics
    - Structure-based clustering
    """
    
    def __init__(self, parser: Optional[StructureParser] = None):
        self.parser = parser or StructureParser()
        self.parsed_jds: List[JDStructure] = []
    
    def parse_corpus(
        self,
        df: pd.DataFrame,
        text_field: str = "jd_text",
        id_field: str = "jd_id",
        show_progress: bool = True,
    ) -> List[JDStructure]:
        """
        Parse all JDs in the corpus.
        
        Args:
            df: DataFrame with JD text
            text_field: Column containing JD text
            id_field: Column containing JD ID
            show_progress: Print progress updates
            
        Returns:
            List of JDStructure objects
        """
        self.parsed_jds = []
        total = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            if show_progress and i % 100 == 0:
                print(f"Parsing JDs: {i}/{total}")
            
            text = row.get(text_field, "")
            jd_id = str(row.get(id_field, i))
            
            structure = self.parser.parse(text, jd_id)
            self.parsed_jds.append(structure)
        
        if show_progress:
            print(f"Parsed {len(self.parsed_jds)} JDs")
        
        return self.parsed_jds
    
    def get_section_frequency(self) -> Dict[str, int]:
        """Get frequency of each section type across corpus."""
        section_counts = Counter()
        
        for jd in self.parsed_jds:
            for section in jd.sections:
                section_type = section.get("section_type", "other")
                section_counts[section_type] += 1
        
        return dict(section_counts.most_common())
    
    def get_section_coverage(self) -> Dict[str, float]:
        """Get % of JDs that have each section type."""
        if not self.parsed_jds:
            return {}
        
        section_presence = defaultdict(int)
        
        for jd in self.parsed_jds:
            seen_types = set()
            for section in jd.sections:
                section_type = section.get("section_type", "other")
                if section_type not in seen_types:
                    section_presence[section_type] += 1
                    seen_types.add(section_type)
        
        total = len(self.parsed_jds)
        return {
            section: round(count / total, 3)
            for section, count in sorted(
                section_presence.items(),
                key=lambda x: -x[1]
            )
        }
    
    def get_structure_patterns(self, min_support: int = 5) -> List[Tuple[tuple, int]]:
        """
        Find common section ordering patterns.
        
        Args:
            min_support: Minimum occurrences for a pattern
            
        Returns:
            List of (section_sequence, count) sorted by frequency
        """
        pattern_counts = Counter()
        
        for jd in self.parsed_jds:
            section_types = tuple(
                s.get("section_type", "other") for s in jd.sections
            )
            pattern_counts[section_types] += 1
        
        return [
            (pattern, count)
            for pattern, count in pattern_counts.most_common()
            if count >= min_support
        ]
    
    def measure_consistency(self) -> Dict[str, Any]:
        """
        Measure structural consistency across corpus.
        
        Returns metrics on how consistently JDs are structured.
        """
        if not self.parsed_jds:
            return {}
        
        # Number of sections distribution
        num_sections = [len(jd.sections) for jd in self.parsed_jds]
        
        # Most common structure
        patterns = self.get_structure_patterns()
        
        # Calculate consistency score
        # High score = most JDs follow similar structure
        if patterns:
            top_pattern_count = patterns[0][1]
            consistency_score = top_pattern_count / len(self.parsed_jds)
        else:
            consistency_score = 0
        
        # Expected sections coverage
        expected_sections = ["responsibilities", "required_qualifications", "summary"]
        coverage = self.get_section_coverage()
        expected_coverage = {
            s: coverage.get(s, 0) for s in expected_sections
        }
        
        return {
            "total_jds": len(self.parsed_jds),
            "num_sections_stats": {
                "min": min(num_sections),
                "max": max(num_sections),
                "mean": round(sum(num_sections) / len(num_sections), 2),
                "median": sorted(num_sections)[len(num_sections) // 2],
            },
            "unique_structures": len(patterns),
            "top_structure_coverage": round(consistency_score, 3),
            "top_5_structures": patterns[:5],
            "expected_sections_coverage": expected_coverage,
            "section_coverage": coverage,
        }
    
    def get_structure_vectors(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert JD structures to binary vectors for clustering.
        
        Returns:
            Tuple of (DataFrame with binary vectors, list of section types)
        """
        # Get all unique section types
        all_types = set()
        for jd in self.parsed_jds:
            for section in jd.sections:
                all_types.add(section.get("section_type", "other"))
        
        section_types = sorted(all_types)
        
        # Create binary vectors
        vectors = []
        for jd in self.parsed_jds:
            jd_types = {s.get("section_type", "other") for s in jd.sections}
            vector = [1 if t in jd_types else 0 for t in section_types]
            vectors.append({"jd_id": jd.jd_id, **dict(zip(section_types, vector))})
        
        return pd.DataFrame(vectors), section_types
    
    def cluster_by_structure(
        self,
        n_clusters: int = 5,
    ) -> Dict[str, Any]:
        """
        Cluster JDs by their structural similarity.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Clustering results with cluster assignments and characteristics
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn required. Install with: pip install scikit-learn")
        
        vectors_df, section_types = self.get_structure_vectors()
        X = vectors_df[section_types].values
        
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        vectors_df["cluster"] = clusters
        
        # Analyze clusters
        cluster_profiles = {}
        for cluster_id in range(n_clusters):
            cluster_mask = vectors_df["cluster"] == cluster_id
            cluster_data = vectors_df[cluster_mask]
            
            # Average section presence in this cluster
            section_presence = {
                section: cluster_data[section].mean()
                for section in section_types
            }
            
            # Most common sections (>50% of cluster)
            common_sections = [
                s for s, pct in section_presence.items() if pct > 0.5
            ]
            
            cluster_profiles[f"cluster_{cluster_id}"] = {
                "size": int(cluster_mask.sum()),
                "percentage": round(cluster_mask.sum() / len(vectors_df), 3),
                "common_sections": common_sections,
                "section_presence": {k: round(v, 2) for k, v in section_presence.items()},
            }
        
        # Calculate silhouette score
        if n_clusters > 1:
            sil_score = silhouette_score(X, clusters)
        else:
            sil_score = 0
        
        return {
            "n_clusters": n_clusters,
            "silhouette_score": round(sil_score, 3),
            "cluster_profiles": cluster_profiles,
            "assignments": vectors_df[["jd_id", "cluster"]].to_dict(orient="records"),
        }
    
    def print_structure_report(self) -> None:
        """Print human-readable structure analysis report."""
        consistency = self.measure_consistency()
        
        print("=" * 60)
        print("JD STRUCTURAL CONSISTENCY REPORT")
        print("=" * 60)
        
        print(f"\nTotal JDs Analyzed: {consistency['total_jds']}")
        
        stats = consistency["num_sections_stats"]
        print(f"\nSections per JD:")
        print(f"  Min: {stats['min']}, Max: {stats['max']}")
        print(f"  Mean: {stats['mean']}, Median: {stats['median']}")
        
        print(f"\nUnique Structure Patterns: {consistency['unique_structures']}")
        print(f"Top Pattern Coverage: {consistency['top_structure_coverage']:.1%}")
        
        print("\nTOP 5 STRUCTURE PATTERNS")
        print("-" * 40)
        for pattern, count in consistency["top_5_structures"]:
            pct = count / consistency["total_jds"] * 100
            pattern_str = " → ".join(pattern) if pattern else "(no sections)"
            print(f"  [{count:4d}] ({pct:5.1f}%) {pattern_str}")
        
        print("\nSECTION COVERAGE")
        print("-" * 40)
        for section, coverage in consistency["section_coverage"].items():
            bar_filled = int(coverage * 20)
            bar = "#" * bar_filled + "." * (20 - bar_filled)
            print(f"  {section:25} [{bar}] {coverage:.1%}")
        
        print("\nEXPECTED SECTIONS COVERAGE")
        print("-" * 40)
        for section, coverage in consistency["expected_sections_coverage"].items():
            if coverage > 0.7:
                status = "[OK]  "
            elif coverage > 0.4:
                status = "[WARN]"
            else:
                status = "[MISS]"
            print(f"  {status} {section:25} {coverage:.1%}")
    
    def export_parsed_jds(self, output_path: str = "parsed_jd_structures.json") -> None:
        """Export parsed structures to JSON."""
        import json
        
        data = [jd.to_dict() for jd in self.parsed_jds]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported {len(data)} parsed JD structures to: {output_path}")
