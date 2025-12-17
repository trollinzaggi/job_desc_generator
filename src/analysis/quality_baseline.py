"""
Phase 1.1: JD Quality Baseline Analysis

Components:
- Stratified sampling by org unit and level
- Quality rubric definition and scoring
- Human evaluation support (export for review, import scores)
- Pattern analysis (quality vs metadata correlations, including recency)
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import json
from pathlib import Path


@dataclass
class QualityRubric:
    """
    Quality rubric for evaluating JDs.
    
    Each dimension is scored 1-5:
        1 = Very Poor
        2 = Poor  
        3 = Adequate
        4 = Good
        5 = Excellent
    """
    
    # Completeness: Does the JD contain all expected sections?
    completeness: Optional[int] = None
    completeness_notes: str = ""
    
    # Clarity: Is the language clear and unambiguous?
    clarity: Optional[int] = None
    clarity_notes: str = ""
    
    # Specificity: Are requirements specific vs generic?
    specificity: Optional[int] = None
    specificity_notes: str = ""
    
    # Compliance: Does it follow company/legal standards?
    compliance: Optional[int] = None
    compliance_notes: str = ""
    
    # Actionability: Could a recruiter use this to screen candidates?
    actionability: Optional[int] = None
    actionability_notes: str = ""
    
    # Overall assessment
    overall_score: Optional[int] = None
    overall_notes: str = ""
    
    # Would you use this as a template?
    is_gold_standard: bool = False
    
    def average_score(self) -> Optional[float]:
        """Calculate average of all dimension scores."""
        scores = [
            self.completeness, self.clarity, self.specificity,
            self.compliance, self.actionability
        ]
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return None
        return round(sum(valid_scores) / len(valid_scores), 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "completeness": self.completeness,
            "completeness_notes": self.completeness_notes,
            "clarity": self.clarity,
            "clarity_notes": self.clarity_notes,
            "specificity": self.specificity,
            "specificity_notes": self.specificity_notes,
            "compliance": self.compliance,
            "compliance_notes": self.compliance_notes,
            "actionability": self.actionability,
            "actionability_notes": self.actionability_notes,
            "overall_score": self.overall_score,
            "overall_notes": self.overall_notes,
            "is_gold_standard": self.is_gold_standard,
            "average_score": self.average_score(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QualityRubric":
        return cls(
            completeness=data.get("completeness"),
            completeness_notes=data.get("completeness_notes", ""),
            clarity=data.get("clarity"),
            clarity_notes=data.get("clarity_notes", ""),
            specificity=data.get("specificity"),
            specificity_notes=data.get("specificity_notes", ""),
            compliance=data.get("compliance"),
            compliance_notes=data.get("compliance_notes", ""),
            actionability=data.get("actionability"),
            actionability_notes=data.get("actionability_notes", ""),
            overall_score=data.get("overall_score"),
            overall_notes=data.get("overall_notes", ""),
            is_gold_standard=data.get("is_gold_standard", False),
        )


class StratifiedSampler:
    """
    Create stratified samples of JDs by org unit and level.
    
    Ensures representation across different segments of your corpus.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        org_unit_field: str = "org_unit",
        level_field: str = "level",
        id_field: str = "jd_id",
    ):
        self.df = df
        self.org_unit_field = org_unit_field
        self.level_field = level_field
        self.id_field = id_field
    
    def get_strata_distribution(self) -> pd.DataFrame:
        """Get distribution of records across org_unit x level combinations."""
        if self.org_unit_field not in self.df.columns or self.level_field not in self.df.columns:
            available = list(self.df.columns)
            raise ValueError(
                f"Required fields not found. Available: {available}"
            )
        
        distribution = self.df.groupby(
            [self.org_unit_field, self.level_field]
        ).size().reset_index(name="count")
        
        distribution["percentage"] = (
            distribution["count"] / distribution["count"].sum() * 100
        ).round(2)
        
        return distribution.sort_values("count", ascending=False)
    
    def sample_stratified(
        self,
        n: int = 100,
        min_per_stratum: int = 1,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Create a stratified sample.
        
        Args:
            n: Total sample size
            min_per_stratum: Minimum samples per stratum (if available)
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with sampled records
        """
        random.seed(random_state)
        
        # Group by strata
        grouped = self.df.groupby([self.org_unit_field, self.level_field])
        
        # Calculate proportional allocation
        strata_sizes = grouped.size()
        total = len(self.df)
        
        samples = []
        remaining = n
        
        # First pass: ensure minimum per stratum
        for (org, level), group in grouped:
            stratum_n = min(min_per_stratum, len(group), remaining)
            if stratum_n > 0:
                samples.append(group.sample(n=stratum_n, random_state=random_state))
                remaining -= stratum_n
        
        # Second pass: proportional allocation of remaining
        if remaining > 0:
            for (org, level), group in grouped:
                already_sampled = min_per_stratum if len(group) >= min_per_stratum else len(group)
                available = len(group) - already_sampled
                
                if available <= 0:
                    continue
                
                # Proportional allocation
                proportion = len(group) / total
                additional = min(int(proportion * remaining), available)
                
                if additional > 0:
                    # Sample from records not already selected
                    already_ids = samples[-1][self.id_field].tolist() if samples else []
                    available_records = group[~group[self.id_field].isin(already_ids)]
                    
                    if len(available_records) > 0:
                        samples.append(
                            available_records.sample(
                                n=min(additional, len(available_records)),
                                random_state=random_state
                            )
                        )
        
        result = pd.concat(samples, ignore_index=True).drop_duplicates(subset=[self.id_field])
        
        # If we still need more, random sample from remaining
        if len(result) < n:
            remaining_ids = set(self.df[self.id_field]) - set(result[self.id_field])
            remaining_df = self.df[self.df[self.id_field].isin(remaining_ids)]
            additional_needed = n - len(result)
            
            if len(remaining_df) > 0:
                additional = remaining_df.sample(
                    n=min(additional_needed, len(remaining_df)),
                    random_state=random_state
                )
                result = pd.concat([result, additional], ignore_index=True)
        
        return result.head(n)
    
    def sample_by_strata(
        self,
        strata_config: Dict[Tuple[str, str], int],
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Sample specific counts from specific strata.
        
        Args:
            strata_config: Dict mapping (org_unit, level) -> count
            random_state: Random seed
            
        Returns:
            DataFrame with sampled records
        """
        samples = []
        
        for (org, level), count in strata_config.items():
            mask = (
                (self.df[self.org_unit_field] == org) &
                (self.df[self.level_field] == level)
            )
            stratum = self.df[mask]
            
            if len(stratum) == 0:
                print(f"Warning: No records found for ({org}, {level})")
                continue
            
            sample_n = min(count, len(stratum))
            samples.append(stratum.sample(n=sample_n, random_state=random_state))
        
        return pd.concat(samples, ignore_index=True)


class QualityEvaluator:
    """
    Manage quality evaluation workflow.
    
    Workflow:
        1. Create stratified sample
        2. Export sample for human evaluation
        3. Import evaluation scores
        4. Analyze quality patterns (including recency correlation)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        text_field: str = "jd_text",
        id_field: str = "jd_id",
        org_unit_field: str = "org_unit",
        level_field: str = "level",
        date_field: str = "posting_date",
    ):
        self.df = df
        self.text_field = text_field
        self.id_field = id_field
        self.org_unit_field = org_unit_field
        self.level_field = level_field
        self.date_field = date_field
        
        self.sample_df: Optional[pd.DataFrame] = None
        self.evaluations: Dict[str, QualityRubric] = {}
    
    def create_evaluation_sample(
        self,
        n: int = 100,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Create stratified sample for evaluation."""
        sampler = StratifiedSampler(
            self.df,
            org_unit_field=self.org_unit_field,
            level_field=self.level_field,
            id_field=self.id_field,
        )
        
        self.sample_df = sampler.sample_stratified(n=n, random_state=random_state)
        return self.sample_df
    
    def export_for_evaluation(
        self,
        output_path: str = "jd_quality_evaluation.json",
        include_fields: Optional[List[str]] = None,
    ) -> None:
        """
        Export sample JDs for human evaluation.
        
        Creates a JSON file with JDs and blank rubric fields to fill in.
        """
        if self.sample_df is None:
            raise ValueError("Create sample first with create_evaluation_sample()")
        
        # Default fields to include
        if include_fields is None:
            include_fields = [
                self.id_field, self.text_field,
                self.org_unit_field, self.level_field, "title", self.date_field
            ]
        
        # Filter to available fields
        include_fields = [f for f in include_fields if f in self.sample_df.columns]
        
        export_data = []
        for _, row in self.sample_df.iterrows():
            record = {
                "jd_data": {f: row.get(f) for f in include_fields},
                "evaluation": QualityRubric().to_dict(),
            }
            export_data.append(record)
        
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Exported {len(export_data)} JDs for evaluation to: {output_path}")
        print("\nInstructions:")
        print("1. Open the file and review each JD")
        print("2. Score each dimension 1-5 (1=Very Poor, 5=Excellent)")
        print("3. Add notes explaining your scores")
        print("4. Mark exceptional JDs as 'is_gold_standard': true")
        print("5. Import completed evaluations with import_evaluations()")
    
    def export_for_evaluation_csv(
        self,
        output_path: str = "jd_quality_evaluation.csv",
    ) -> None:
        """
        Export as CSV for spreadsheet-based evaluation.
        """
        if self.sample_df is None:
            raise ValueError("Create sample first with create_evaluation_sample()")
        
        # Create evaluation columns
        eval_df = self.sample_df.copy()
        
        # Add rubric columns
        rubric_cols = [
            "eval_completeness", "eval_completeness_notes",
            "eval_clarity", "eval_clarity_notes",
            "eval_specificity", "eval_specificity_notes",
            "eval_compliance", "eval_compliance_notes",
            "eval_actionability", "eval_actionability_notes",
            "eval_overall_score", "eval_overall_notes",
            "eval_is_gold_standard",
        ]
        
        for col in rubric_cols:
            eval_df[col] = ""
        
        eval_df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")
        print("\nScore each eval_* column 1-5, add notes, mark gold standards as TRUE")
    
    def import_evaluations(self, input_path: str) -> int:
        """
        Import completed evaluations.
        
        Args:
            input_path: Path to JSON or CSV file with evaluations
            
        Returns:
            Number of evaluations imported
        """
        path = Path(input_path)
        
        if path.suffix == ".json":
            return self._import_json_evaluations(input_path)
        elif path.suffix == ".csv":
            return self._import_csv_evaluations(input_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _import_json_evaluations(self, input_path: str) -> int:
        """Import from JSON format."""
        with open(input_path) as f:
            data = json.load(f)
        
        count = 0
        for record in data:
            jd_id = record["jd_data"].get(self.id_field)
            if jd_id and record.get("evaluation"):
                self.evaluations[str(jd_id)] = QualityRubric.from_dict(record["evaluation"])
                count += 1
        
        print(f"Imported {count} evaluations")
        return count
    
    def _import_csv_evaluations(self, input_path: str) -> int:
        """Import from CSV format."""
        df = pd.read_csv(input_path)
        
        count = 0
        for _, row in df.iterrows():
            jd_id = row.get(self.id_field)
            if pd.isna(jd_id):
                continue
            
            # Check if any evaluation was done
            if pd.isna(row.get("eval_completeness")) and pd.isna(row.get("eval_overall_score")):
                continue
            
            rubric = QualityRubric(
                completeness=int(row["eval_completeness"]) if pd.notna(row.get("eval_completeness")) else None,
                completeness_notes=str(row.get("eval_completeness_notes", "")),
                clarity=int(row["eval_clarity"]) if pd.notna(row.get("eval_clarity")) else None,
                clarity_notes=str(row.get("eval_clarity_notes", "")),
                specificity=int(row["eval_specificity"]) if pd.notna(row.get("eval_specificity")) else None,
                specificity_notes=str(row.get("eval_specificity_notes", "")),
                compliance=int(row["eval_compliance"]) if pd.notna(row.get("eval_compliance")) else None,
                compliance_notes=str(row.get("eval_compliance_notes", "")),
                actionability=int(row["eval_actionability"]) if pd.notna(row.get("eval_actionability")) else None,
                actionability_notes=str(row.get("eval_actionability_notes", "")),
                overall_score=int(row["eval_overall_score"]) if pd.notna(row.get("eval_overall_score")) else None,
                overall_notes=str(row.get("eval_overall_notes", "")),
                is_gold_standard=str(row.get("eval_is_gold_standard", "")).upper() == "TRUE",
            )
            
            self.evaluations[str(jd_id)] = rubric
            count += 1
        
        print(f"Imported {count} evaluations")
        return count
    
    def analyze_quality(self) -> Dict[str, Any]:
        """
        Analyze quality patterns from evaluations.
        
        Includes correlation analysis against:
        - org_unit
        - level
        - recency (posting_date)
        
        Returns:
            Dictionary with quality analysis results
        """
        if not self.evaluations:
            raise ValueError("No evaluations loaded. Import evaluations first.")
        
        # Merge evaluations with sample data
        eval_records = []
        for jd_id, rubric in self.evaluations.items():
            record = rubric.to_dict()
            record[self.id_field] = jd_id
            eval_records.append(record)
        
        eval_df = pd.DataFrame(eval_records)
        
        # Merge with original sample to get metadata
        if self.sample_df is not None:
            self.sample_df[self.id_field] = self.sample_df[self.id_field].astype(str)
            merge_fields = [self.id_field]
            
            # Add available metadata fields
            for field in [self.org_unit_field, self.level_field, self.date_field]:
                if field in self.sample_df.columns:
                    merge_fields.append(field)
            
            merged = eval_df.merge(
                self.sample_df[merge_fields],
                on=self.id_field,
                how="left"
            )
        else:
            merged = eval_df
        
        # Overall statistics
        results = {
            "total_evaluated": len(self.evaluations),
            "overall_stats": {
                "mean_completeness": merged["completeness"].mean(),
                "mean_clarity": merged["clarity"].mean(),
                "mean_specificity": merged["specificity"].mean(),
                "mean_compliance": merged["compliance"].mean(),
                "mean_actionability": merged["actionability"].mean(),
                "mean_average_score": merged["average_score"].mean(),
            },
            "gold_standard_count": sum(1 for r in self.evaluations.values() if r.is_gold_standard),
            "gold_standard_ids": [
                jd_id for jd_id, r in self.evaluations.items() if r.is_gold_standard
            ],
        }
        
        # Quality by org unit
        if self.org_unit_field in merged.columns:
            by_org = merged.groupby(self.org_unit_field)["average_score"].agg(["mean", "count"])
            results["quality_by_org_unit"] = by_org.to_dict()
        
        # Quality by level
        if self.level_field in merged.columns:
            by_level = merged.groupby(self.level_field)["average_score"].agg(["mean", "count"])
            results["quality_by_level"] = by_level.to_dict()
        
        # Quality by recency (posting date)
        if self.date_field in merged.columns:
            recency_analysis = self._analyze_quality_by_recency(merged)
            if recency_analysis:
                results["quality_by_recency"] = recency_analysis
        
        # Distribution of scores
        results["score_distribution"] = {
            "completeness": merged["completeness"].value_counts().to_dict(),
            "clarity": merged["clarity"].value_counts().to_dict(),
            "specificity": merged["specificity"].value_counts().to_dict(),
            "compliance": merged["compliance"].value_counts().to_dict(),
            "actionability": merged["actionability"].value_counts().to_dict(),
        }
        
        # Identify problem areas (low-scoring dimensions)
        overall_means = results["overall_stats"]
        problem_areas = [
            dim.replace("mean_", "") 
            for dim, score in overall_means.items() 
            if score and score < 3.0
        ]
        results["problem_areas"] = problem_areas
        
        return results
    
    def _analyze_quality_by_recency(self, merged_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze quality correlation with posting date (recency).
        
        Args:
            merged_df: DataFrame with evaluations and metadata
            
        Returns:
            Dictionary with recency analysis or None if no valid dates
        """
        if self.date_field not in merged_df.columns:
            return None
        
        # Parse dates
        df = merged_df.copy()
        df["_parsed_date"] = pd.to_datetime(df[self.date_field], errors="coerce")
        
        # Filter to records with valid dates and scores
        valid_mask = df["_parsed_date"].notna() & df["average_score"].notna()
        df_valid = df[valid_mask]
        
        if len(df_valid) < 5:
            return None
        
        results = {}
        
        # Quality by quarter
        df_valid["_quarter"] = df_valid["_parsed_date"].dt.to_period("Q")
        by_quarter = df_valid.groupby("_quarter")["average_score"].agg(["mean", "count"])
        by_quarter = by_quarter[by_quarter["count"] >= 2]  # Need at least 2 samples
        
        if len(by_quarter) > 0:
            results["by_quarter"] = {
                str(period): {"mean": round(row["mean"], 2), "count": int(row["count"])}
                for period, row in by_quarter.iterrows()
            }
        
        # Quality by month (if enough data)
        df_valid["_month"] = df_valid["_parsed_date"].dt.to_period("M")
        by_month = df_valid.groupby("_month")["average_score"].agg(["mean", "count"])
        by_month = by_month[by_month["count"] >= 2]
        
        if len(by_month) >= 3:
            results["by_month"] = {
                str(period): {"mean": round(row["mean"], 2), "count": int(row["count"])}
                for period, row in by_month.iterrows()
            }
        
        # Quality trend: recent vs older
        median_date = df_valid["_parsed_date"].median()
        recent = df_valid[df_valid["_parsed_date"] >= median_date]
        older = df_valid[df_valid["_parsed_date"] < median_date]
        
        if len(recent) >= 3 and len(older) >= 3:
            recent_mean = recent["average_score"].mean()
            older_mean = older["average_score"].mean()
            
            results["trend_comparison"] = {
                "recent_mean": round(recent_mean, 2),
                "recent_count": len(recent),
                "older_mean": round(older_mean, 2),
                "older_count": len(older),
                "difference": round(recent_mean - older_mean, 2),
                "trend": "improving" if recent_mean > older_mean else "declining" if recent_mean < older_mean else "stable",
            }
        
        # Correlation coefficient (if enough variation)
        try:
            # Convert dates to numeric (days since earliest)
            earliest = df_valid["_parsed_date"].min()
            df_valid["_days_since"] = (df_valid["_parsed_date"] - earliest).dt.days
            
            correlation = df_valid["_days_since"].corr(df_valid["average_score"])
            
            if pd.notna(correlation):
                results["correlation"] = {
                    "coefficient": round(correlation, 3),
                    "interpretation": (
                        "positive (newer JDs score higher)" if correlation > 0.1
                        else "negative (older JDs score higher)" if correlation < -0.1
                        else "no significant correlation"
                    ),
                }
        except Exception:
            pass
        
        # Date range info
        results["date_range"] = {
            "earliest": str(df_valid["_parsed_date"].min().date()),
            "latest": str(df_valid["_parsed_date"].max().date()),
            "span_days": (df_valid["_parsed_date"].max() - df_valid["_parsed_date"].min()).days,
        }
        
        return results
    
    def print_quality_report(self) -> None:
        """Print human-readable quality report."""
        results = self.analyze_quality()
        
        print("=" * 60)
        print("JD QUALITY BASELINE REPORT")
        print("=" * 60)
        
        print(f"\nTotal JDs Evaluated: {results['total_evaluated']}")
        print(f"Gold Standard JDs: {results['gold_standard_count']}")
        
        print("\nOVERALL QUALITY SCORES (1-5 scale)")
        print("-" * 40)
        for dim, score in results["overall_stats"].items():
            dim_name = dim.replace("mean_", "").replace("_", " ").title()
            if score:
                bar_filled = int(score)
                bar = "#" * bar_filled + "." * (5 - bar_filled)
                print(f"  {dim_name:20} [{bar}] {score:.2f}")
        
        if results.get("problem_areas"):
            print(f"\n  [WARNING] Problem Areas (score < 3.0): {', '.join(results['problem_areas'])}")
        
        if "quality_by_org_unit" in results:
            print("\nQUALITY BY ORG UNIT")
            print("-" * 40)
            means = results["quality_by_org_unit"].get("mean", {})
            counts = results["quality_by_org_unit"].get("count", {})
            for org in sorted(means.keys(), key=lambda x: means[x], reverse=True)[:10]:
                print(f"  {org:30} {means[org]:.2f} (n={counts[org]})")
        
        if "quality_by_level" in results:
            print("\nQUALITY BY LEVEL")
            print("-" * 40)
            means = results["quality_by_level"].get("mean", {})
            counts = results["quality_by_level"].get("count", {})
            for level in sorted(means.keys(), key=lambda x: means.get(x, 0), reverse=True):
                if means.get(level):
                    print(f"  {level:30} {means[level]:.2f} (n={counts[level]})")
        
        # Recency analysis
        if "quality_by_recency" in results:
            recency = results["quality_by_recency"]
            print("\nQUALITY BY RECENCY")
            print("-" * 40)
            
            if "date_range" in recency:
                print(f"  Date range: {recency['date_range']['earliest']} to {recency['date_range']['latest']}")
            
            if "trend_comparison" in recency:
                trend = recency["trend_comparison"]
                print(f"\n  Recent JDs (n={trend['recent_count']}): {trend['recent_mean']:.2f}")
                print(f"  Older JDs (n={trend['older_count']}):  {trend['older_mean']:.2f}")
                print(f"  Trend: {trend['trend'].upper()} ({trend['difference']:+.2f})")
            
            if "correlation" in recency:
                corr = recency["correlation"]
                print(f"\n  Correlation coefficient: {corr['coefficient']}")
                print(f"  Interpretation: {corr['interpretation']}")
            
            if "by_quarter" in recency:
                print("\n  By Quarter:")
                for period, stats in sorted(recency["by_quarter"].items()):
                    print(f"    {period}: {stats['mean']:.2f} (n={stats['count']})")
        
        if results["gold_standard_ids"]:
            print(f"\nGOLD STANDARD JD IDs:")
            print("-" * 40)
            for jd_id in results["gold_standard_ids"][:10]:
                print(f"  - {jd_id}")
    
    def get_gold_standard_jds(self) -> pd.DataFrame:
        """Get DataFrame of gold standard JDs."""
        gold_ids = [
            jd_id for jd_id, rubric in self.evaluations.items()
            if rubric.is_gold_standard
        ]
        
        if self.sample_df is not None:
            self.sample_df[self.id_field] = self.sample_df[self.id_field].astype(str)
            return self.sample_df[self.sample_df[self.id_field].isin(gold_ids)]
        
        return pd.DataFrame()
