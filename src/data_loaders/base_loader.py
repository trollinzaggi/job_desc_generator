"""
Base class for JD data loaders with schema discovery and field mapping capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator
import pandas as pd
from dataclasses import dataclass, field

from .utils import flatten_dict, get_nested_value


@dataclass
class FieldMapping:
    """Configuration for mapping source fields to standardized fields."""
    
    # Required fields - set these to your actual field paths
    # Use dot notation for nested fields, e.g., "metadata.org_unit"
    jd_text: Optional[str] = None
    jd_id: Optional[str] = None
    
    # Metadata fields
    org_unit: Optional[str] = None
    department: Optional[str] = None
    level: Optional[str] = None
    title: Optional[str] = None
    function: Optional[str] = None
    location: Optional[str] = None
    posting_date: Optional[str] = None
    status: Optional[str] = None
    
    # Optional structured sections (if JD is pre-parsed)
    responsibilities: Optional[str] = None
    required_qualifications: Optional[str] = None
    preferred_qualifications: Optional[str] = None
    team_description: Optional[str] = None
    
    # Additional fields you want to capture
    custom_fields: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary, excluding None values."""
        result = {
            "jd_text": self.jd_text,
            "jd_id": self.jd_id,
            "org_unit": self.org_unit,
            "department": self.department,
            "level": self.level,
            "title": self.title,
            "function": self.function,
            "location": self.location,
            "posting_date": self.posting_date,
            "status": self.status,
            "responsibilities": self.responsibilities,
            "required_qualifications": self.required_qualifications,
            "preferred_qualifications": self.preferred_qualifications,
            "team_description": self.team_description,
        }
        result.update(self.custom_fields)
        return {k: v for k, v in result.items() if v is not None}


class BaseJDLoader(ABC):
    """
    Abstract base class for loading job description data from various sources.
    
    Subclasses must implement:
        - _load_raw_data(): Load raw data from source
        - _iterate_records(): Iterate through individual JD records
    """
    
    def __init__(self, field_mapping: Optional[FieldMapping] = None):
        """
        Initialize the loader with optional field mapping.
        
        Args:
            field_mapping: FieldMapping object specifying how source fields
                          map to standardized fields. If None, raw data is returned.
        """
        self.field_mapping = field_mapping
    
    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from the source. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _iterate_records(self, raw_data: Any) -> Generator[Dict[str, Any], None, None]:
        """Iterate through individual JD records from raw data."""
        pass
    
    def _apply_mapping(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mapping to transform a raw record to standardized format."""
        if self.field_mapping is None:
            return record
        
        mapping = self.field_mapping.to_dict()
        result = {}
        
        for standard_field, source_path in mapping.items():
            if source_path:
                value = get_nested_value(record, source_path)
                result[standard_field] = value
        
        result["_raw"] = record
        return result
    
    def load(self, apply_mapping: bool = True) -> List[Dict[str, Any]]:
        """Load all JD records."""
        raw_data = self._load_raw_data()
        records = list(self._iterate_records(raw_data))
        
        if apply_mapping and self.field_mapping:
            return [self._apply_mapping(r) for r in records]
        return records
    
    def load_as_dataframe(self, apply_mapping: bool = True) -> pd.DataFrame:
        """Load JD records as a pandas DataFrame."""
        records = self.load(apply_mapping=apply_mapping)
        
        if not apply_mapping or not self.field_mapping:
            flattened = [flatten_dict(r) for r in records]
            return pd.DataFrame(flattened)
        
        clean_records = [{k: v for k, v in r.items() if k != "_raw"} for r in records]
        return pd.DataFrame(clean_records)
    
    def sample(self, n: int = 5, apply_mapping: bool = False) -> List[Dict[str, Any]]:
        """Load a sample of records for inspection."""
        raw_data = self._load_raw_data()
        records = []
        
        for i, record in enumerate(self._iterate_records(raw_data)):
            if i >= n:
                break
            if apply_mapping and self.field_mapping:
                records.append(self._apply_mapping(record))
            else:
                records.append(record)
        return records
    
    def count_records(self) -> int:
        """Count total number of JD records."""
        raw_data = self._load_raw_data()
        return sum(1 for _ in self._iterate_records(raw_data))
    
    def discover_schema(self, sample_size: int = 100) -> "SchemaDiscovery":
        """
        Discover the schema of the data.
        
        Args:
            sample_size: Number of records to analyze
            
        Returns:
            SchemaDiscovery object with analysis results
        """
        from .schema_discovery import SchemaDiscovery
        
        discovery = SchemaDiscovery()
        records = self.sample(n=sample_size, apply_mapping=False)
        discovery.analyze_records(records)
        return discovery
    
    def load_chunked(
        self, 
        chunk_size: int = 1000,
        apply_mapping: bool = True
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Load records in chunks for memory-efficient processing."""
        raw_data = self._load_raw_data()
        chunk = []
        
        for record in self._iterate_records(raw_data):
            if apply_mapping and self.field_mapping:
                record = self._apply_mapping(record)
            chunk.append(record)
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        
        if chunk:
            yield chunk
