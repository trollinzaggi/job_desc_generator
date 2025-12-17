"""
Schema discovery utilities for exploring JSON structures without exposing sensitive data.
"""

from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import json


class SchemaDiscovery:
    """
    Discover and analyze JSON schema structures without exposing actual values.
    
    Safe to share output - contains only field paths, types, and statistics.
    """
    
    def __init__(self):
        self._field_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "null_count": 0,
            "types": set(),
            "sample_lengths": [],
            "nested_keys": set(),
        })
    
    def analyze_record(self, record: Dict[str, Any], prefix: str = "") -> None:
        """Analyze a single record to build schema understanding."""
        for key, value in record.items():
            field_path = f"{prefix}.{key}" if prefix else key
            stats = self._field_stats[field_path]
            stats["count"] += 1
            
            if value is None:
                stats["null_count"] += 1
                stats["types"].add("null")
            elif isinstance(value, dict):
                stats["types"].add("object")
                stats["nested_keys"].update(value.keys())
                self.analyze_record(value, prefix=field_path)
            elif isinstance(value, list):
                stats["types"].add("array")
                stats["sample_lengths"].append(len(value))
                for item in value[:3]:
                    if isinstance(item, dict):
                        self.analyze_record(item, prefix=f"{field_path}[*]")
            elif isinstance(value, str):
                stats["types"].add("string")
                stats["sample_lengths"].append(len(value))
            elif isinstance(value, bool):
                stats["types"].add("boolean")
            elif isinstance(value, int):
                stats["types"].add("integer")
            elif isinstance(value, float):
                stats["types"].add("number")
            else:
                stats["types"].add(type(value).__name__)
    
    def analyze_records(self, records: List[Dict[str, Any]]) -> "SchemaDiscovery":
        """Analyze multiple records. Returns self for chaining."""
        for record in records:
            self.analyze_record(record)
        return self
    
    def get_schema_summary(self) -> Dict[str, Dict]:
        """Get schema summary (safe to share - no actual values)."""
        summary = {}
        for field_path, stats in self._field_stats.items():
            field_summary = {
                "count": stats["count"],
                "null_count": stats["null_count"],
                "null_rate": round(stats["null_count"] / stats["count"], 3) if stats["count"] > 0 else 0,
                "types": list(stats["types"]),
            }
            
            if "string" in stats["types"] and stats["sample_lengths"]:
                lengths = stats["sample_lengths"]
                field_summary["string_length"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "avg": round(sum(lengths) / len(lengths), 1),
                }
            
            if "array" in stats["types"] and stats["sample_lengths"]:
                lengths = stats["sample_lengths"]
                field_summary["array_length"] = {
                    "min": min(lengths),
                    "max": max(lengths),
                    "avg": round(sum(lengths) / len(lengths), 1),
                }
            
            if stats["nested_keys"]:
                field_summary["nested_keys"] = list(stats["nested_keys"])
            
            summary[field_path] = field_summary
        
        return summary
    
    def print_schema_tree(self, indent: int = 2) -> str:
        """Get a tree-like visualization of the schema."""
        lines = ["Schema Structure:", "=" * 50]
        sorted_fields = sorted(self._field_stats.keys())
        
        for field_path in sorted_fields:
            stats = self._field_stats[field_path]
            depth = field_path.count(".") + field_path.count("[")
            prefix = " " * (depth * indent)
            
            field_name = field_path.split(".")[-1] if "." in field_path else field_path
            types_str = ", ".join(sorted(stats["types"]))
            
            null_info = ""
            if stats["null_count"] > 0 and stats["count"] > 0:
                null_rate = stats["null_count"] / stats["count"]
                null_info = f" (null: {null_rate:.1%})"
            
            lines.append(f"{prefix}|-- {field_name}: {types_str}{null_info}")
        
        return "\n".join(lines)
    
    def get_field_paths(self, type_filter: Optional[str] = None) -> List[str]:
        """Get all field paths, optionally filtered by type."""
        if type_filter is None:
            return list(self._field_stats.keys())
        
        return [
            path for path, stats in self._field_stats.items()
            if type_filter in stats["types"]
        ]
    
    def get_likely_text_fields(self, min_avg_length: int = 100) -> List[Tuple[str, Dict]]:
        """Identify fields likely to contain JD text content."""
        candidates = []
        
        for field_path, stats in self._field_stats.items():
            if "string" not in stats["types"] or not stats["sample_lengths"]:
                continue
            
            avg_length = sum(stats["sample_lengths"]) / len(stats["sample_lengths"])
            if avg_length >= min_avg_length:
                candidates.append((field_path, {
                    "avg_length": round(avg_length, 1),
                    "max_length": max(stats["sample_lengths"]),
                    "null_rate": round(stats["null_count"] / stats["count"], 3),
                }))
        
        return sorted(candidates, key=lambda x: x[1]["avg_length"], reverse=True)
    
    def get_likely_id_fields(self) -> List[str]:
        """Identify fields likely to be identifiers."""
        indicators = ["id", "_id", "uuid", "guid", "key", "code", "number"]
        return [
            path for path in self._field_stats.keys()
            if any(ind in path.lower() for ind in indicators)
        ]
    
    def get_likely_metadata_fields(self) -> List[str]:
        """Identify fields likely to be metadata."""
        indicators = [
            "org", "unit", "department", "dept", "division",
            "level", "grade", "band", "tier",
            "title", "role", "position",
            "function", "category", "type",
            "location", "city", "country", "region",
            "date", "created", "posted", "updated", "modified",
            "status", "state", "active",
            "manager", "team", "group"
        ]
        return [
            path for path in self._field_stats.keys()
            if any(ind in path.lower() for ind in indicators)
        ]
    
    def suggest_field_mapping(self) -> Dict[str, str]:
        """Suggest field mappings based on discovered schema."""
        suggestions = {}
        
        text_candidates = self.get_likely_text_fields()
        if text_candidates:
            suggestions["jd_text"] = text_candidates[0][0]
        
        id_candidates = self.get_likely_id_fields()
        if id_candidates:
            suggestions["jd_id"] = id_candidates[0]
        
        metadata_candidates = self.get_likely_metadata_fields()
        metadata_mapping = {
            "org_unit": ["org", "unit", "organization"],
            "department": ["department", "dept", "division"],
            "level": ["level", "grade", "band", "tier"],
            "title": ["title", "job_title", "position_title", "role_title"],
            "function": ["function", "job_function", "category"],
            "location": ["location", "city", "office"],
            "posting_date": ["date", "posted", "created", "posting"],
            "status": ["status", "state"],
        }
        
        for standard_field, indicators in metadata_mapping.items():
            for candidate in metadata_candidates:
                if any(ind in candidate.lower() for ind in indicators):
                    if standard_field not in suggestions:
                        suggestions[standard_field] = candidate
                    break
        
        return suggestions
    
    def to_json(self, indent: int = 2) -> str:
        """Export schema summary as JSON string."""
        return json.dumps(self.get_schema_summary(), indent=indent, default=str)
    
    def reset(self) -> None:
        """Reset all collected statistics."""
        self._field_stats.clear()
