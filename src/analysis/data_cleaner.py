"""
Data transformation and cleaning utilities for JD corpus.

Tracks cleaning statistics so you can understand data quality before analysis.
"""

import re
import html
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter
import pandas as pd


@dataclass
class CleaningStats:
    """Statistics collected during data cleaning."""
    
    total_records: int = 0
    records_with_text: int = 0
    records_missing_text: int = 0
    
    # Text cleaning stats
    html_tags_removed: int = 0
    extra_whitespace_cleaned: int = 0
    encoding_issues_fixed: int = 0
    
    # Field-level stats
    field_fill_rates: Dict[str, float] = field(default_factory=dict)
    field_value_counts: Dict[str, Counter] = field(default_factory=dict)
    
    # Text length stats
    text_length_before: List[int] = field(default_factory=list)
    text_length_after: List[int] = field(default_factory=list)
    
    # Records dropped
    dropped_reasons: Counter = field(default_factory=Counter)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "total_records": self.total_records,
            "records_with_text": self.records_with_text,
            "records_missing_text": self.records_missing_text,
            "text_coverage_rate": round(self.records_with_text / max(self.total_records, 1), 3),
            "cleaning_operations": {
                "html_tags_removed": self.html_tags_removed,
                "extra_whitespace_cleaned": self.extra_whitespace_cleaned,
                "encoding_issues_fixed": self.encoding_issues_fixed,
            },
            "field_fill_rates": self.field_fill_rates,
            "text_length_stats": {
                "before": {
                    "min": min(self.text_length_before) if self.text_length_before else 0,
                    "max": max(self.text_length_before) if self.text_length_before else 0,
                    "avg": round(sum(self.text_length_before) / len(self.text_length_before), 1) if self.text_length_before else 0,
                },
                "after": {
                    "min": min(self.text_length_after) if self.text_length_after else 0,
                    "max": max(self.text_length_after) if self.text_length_after else 0,
                    "avg": round(sum(self.text_length_after) / len(self.text_length_after), 1) if self.text_length_after else 0,
                },
            },
            "dropped_records": dict(self.dropped_reasons),
        }
    
    def print_report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "DATA CLEANING REPORT",
            "=" * 60,
            "",
            "RECORD COUNTS",
            "-" * 40,
            f"  Total records processed: {self.total_records}",
            f"  Records with text: {self.records_with_text}",
            f"  Records missing text: {self.records_missing_text}",
            f"  Text coverage rate: {self.records_with_text / max(self.total_records, 1):.1%}",
            "",
            "CLEANING OPERATIONS",
            "-" * 40,
            f"  HTML tags removed: {self.html_tags_removed} records",
            f"  Whitespace cleaned: {self.extra_whitespace_cleaned} records",
            f"  Encoding issues fixed: {self.encoding_issues_fixed} records",
            "",
        ]
        
        if self.text_length_before:
            lines.extend([
                "TEXT LENGTH STATISTICS",
                "-" * 40,
                f"  Before cleaning:",
                f"    Min: {min(self.text_length_before)}, Max: {max(self.text_length_before)}, Avg: {sum(self.text_length_before) / len(self.text_length_before):.0f}",
                f"  After cleaning:",
                f"    Min: {min(self.text_length_after)}, Max: {max(self.text_length_after)}, Avg: {sum(self.text_length_after) / len(self.text_length_after):.0f}",
                "",
            ])
        
        if self.field_fill_rates:
            lines.extend([
                "FIELD FILL RATES",
                "-" * 40,
            ])
            for field_name, rate in sorted(self.field_fill_rates.items(), key=lambda x: -x[1]):
                lines.append(f"  {field_name}: {rate:.1%}")
            lines.append("")
        
        if self.dropped_reasons:
            lines.extend([
                "DROPPED RECORDS",
                "-" * 40,
            ])
            for reason, count in self.dropped_reasons.most_common():
                lines.append(f"  {reason}: {count}")
            lines.append("")
        
        return "\n".join(lines)


class DataCleaner:
    """
    Clean and transform JD data with statistics tracking.
    
    Usage:
        cleaner = DataCleaner(text_field="jd_text")
        df_clean, stats = cleaner.clean(df_raw)
        print(stats.print_report())
    """
    
    def __init__(
        self,
        text_field: str = "jd_text",
        id_field: str = "jd_id",
        min_text_length: int = 50,
        drop_missing_text: bool = True,
    ):
        """
        Initialize the cleaner.
        
        Args:
            text_field: Name of the column containing JD text
            id_field: Name of the column containing JD ID
            min_text_length: Minimum text length to keep (after cleaning)
            drop_missing_text: Whether to drop records with no text
        """
        self.text_field = text_field
        self.id_field = id_field
        self.min_text_length = min_text_length
        self.drop_missing_text = drop_missing_text
        self.stats = CleaningStats()
    
    def _clean_html(self, text: str) -> Tuple[str, bool]:
        """Remove HTML tags and decode entities."""
        if not text:
            return text, False
        
        original = text
        # Decode HTML entities
        text = html.unescape(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        changed = text != original
        return text, changed
    
    def _clean_whitespace(self, text: str) -> Tuple[str, bool]:
        """Normalize whitespace."""
        if not text:
            return text, False
        
        original = text
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        
        changed = text != original
        return text, changed
    
    def _fix_encoding(self, text: str) -> Tuple[str, bool]:
        """Fix common encoding issues."""
        if not text:
            return text, False
        
        original = text
        # Common encoding artifacts
        replacements = [
            ('â€™', "'"),
            ('â€œ', '"'),
            ('â€', '"'),
            ('â€"', '—'),
            ('â€"', '-'),
            ('Â', ''),
            ('\x00', ''),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        changed = text != original
        return text, changed
    
    def clean_text(self, text: Optional[str]) -> Optional[str]:
        """Apply all text cleaning operations."""
        if text is None or not isinstance(text, str):
            return None
        
        original_length = len(text)
        self.stats.text_length_before.append(original_length)
        
        # Apply cleaning steps
        text, html_changed = self._clean_html(text)
        text, ws_changed = self._clean_whitespace(text)
        text, enc_changed = self._fix_encoding(text)
        
        # Track stats
        if html_changed:
            self.stats.html_tags_removed += 1
        if ws_changed:
            self.stats.extra_whitespace_cleaned += 1
        if enc_changed:
            self.stats.encoding_issues_fixed += 1
        
        self.stats.text_length_after.append(len(text) if text else 0)
        
        return text
    
    def _compute_field_fill_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute fill rate for each column."""
        fill_rates = {}
        for col in df.columns:
            if col.startswith("_"):
                continue
            non_null = df[col].notna().sum()
            fill_rates[col] = round(non_null / len(df), 3) if len(df) > 0 else 0
        return fill_rates
    
    def _compute_value_counts(
        self, 
        df: pd.DataFrame, 
        categorical_fields: List[str],
        max_values: int = 20
    ) -> Dict[str, Counter]:
        """Compute value counts for categorical fields."""
        value_counts = {}
        for col in categorical_fields:
            if col in df.columns:
                counts = df[col].value_counts().head(max_values).to_dict()
                value_counts[col] = Counter(counts)
        return value_counts
    
    def clean(
        self,
        df: pd.DataFrame,
        categorical_fields: Optional[List[str]] = None,
        custom_cleaners: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[pd.DataFrame, CleaningStats]:
        """
        Clean the DataFrame and return cleaned data with statistics.
        
        Args:
            df: Input DataFrame (from loader.load_as_dataframe())
            categorical_fields: Fields to compute value distributions for
            custom_cleaners: Dict of {field_name: cleaner_function} for custom cleaning
            
        Returns:
            Tuple of (cleaned DataFrame, CleaningStats)
        """
        self.stats = CleaningStats()
        self.stats.total_records = len(df)
        
        df = df.copy()
        
        # Clean text field
        if self.text_field in df.columns:
            df[self.text_field] = df[self.text_field].apply(self.clean_text)
            
            # Count records with/without text
            has_text = df[self.text_field].notna() & (df[self.text_field].str.len() > 0)
            self.stats.records_with_text = has_text.sum()
            self.stats.records_missing_text = (~has_text).sum()
            
            # Drop missing text if configured
            if self.drop_missing_text:
                missing_mask = ~has_text
                self.stats.dropped_reasons["missing_text"] = missing_mask.sum()
                df = df[has_text]
            
            # Drop short text
            if self.min_text_length > 0:
                short_mask = df[self.text_field].str.len() < self.min_text_length
                self.stats.dropped_reasons["text_too_short"] = short_mask.sum()
                df = df[~short_mask]
        
        # Apply custom cleaners
        if custom_cleaners:
            for field_name, cleaner_func in custom_cleaners.items():
                if field_name in df.columns:
                    df[field_name] = df[field_name].apply(cleaner_func)
        
        # Compute field fill rates
        self.stats.field_fill_rates = self._compute_field_fill_rates(df)
        
        # Compute value counts for categorical fields
        if categorical_fields:
            self.stats.field_value_counts = self._compute_value_counts(df, categorical_fields)
        
        return df, self.stats
    
    def get_text_length_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Get distribution of text lengths."""
        if self.text_field not in df.columns:
            return pd.Series()
        
        lengths = df[self.text_field].str.len()
        return lengths.describe()
