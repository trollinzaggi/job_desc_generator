"""
JSON file loader for reading JD data from local JSON files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Union

from .base_loader import BaseJDLoader, FieldMapping


class JSONFileLoader(BaseJDLoader):
    """
    Load JD data from JSON files in a directory.
    
    Supports multiple JSON structures:
    - {"content": [list of JD objects]}  (configurable key)
    - [list of JD objects]
    - {single JD object}
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        content_key: str = "content",
        file_pattern: str = "*.json",
        field_mapping: Optional[FieldMapping] = None,
        encoding: str = "utf-8"
    ):
        """
        Initialize the JSON file loader.
        
        Args:
            data_path: Path to directory or single JSON file
            content_key: Key containing JD list in each file (default: "content")
            file_pattern: Glob pattern for matching files (default: "*.json")
            field_mapping: Optional FieldMapping for transforming records
            encoding: File encoding (default: "utf-8")
        """
        super().__init__(field_mapping)
        self.data_path = Path(data_path)
        self.content_key = content_key
        self.file_pattern = file_pattern
        self.encoding = encoding
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
    
    def _get_json_files(self) -> List[Path]:
        """Get list of JSON files to process."""
        if self.data_path.is_file():
            return [self.data_path]
        
        files = list(self.data_path.glob(self.file_pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' found in {self.data_path}"
            )
        return sorted(files)
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from all JSON files."""
        all_data = []
        
        for file_path in self._get_json_files():
            with open(file_path, "r", encoding=self.encoding) as f:
                try:
                    data = json.load(f)
                    all_data.append({
                        "_source_file": str(file_path),
                        "_data": data
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
                    continue
        
        return all_data
    
    def _iterate_records(self, raw_data: List[Dict]) -> Generator[Dict[str, Any], None, None]:
        """Iterate through individual JD records from loaded files."""
        for file_data in raw_data:
            source_file = file_data["_source_file"]
            data = file_data["_data"]
            
            if isinstance(data, dict):
                if self.content_key in data and isinstance(data[self.content_key], list):
                    for record in data[self.content_key]:
                        if isinstance(record, dict):
                            record["_source_file"] = source_file
                            yield record
                else:
                    data["_source_file"] = source_file
                    yield data
            elif isinstance(data, list):
                for record in data:
                    if isinstance(record, dict):
                        record["_source_file"] = source_file
                        yield record
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about the JSON files."""
        files = self._get_json_files()
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            "num_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [str(f.name) for f in files[:10]],
            "files_truncated": len(files) > 10
        }
