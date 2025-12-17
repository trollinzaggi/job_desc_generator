"""
Utility functions for data manipulation.
"""

from typing import Any, Dict


def flatten_dict(d: Dict[str, Any], prefix: str = "", separator: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary to dot-notation keys.
    
    Args:
        d: Dictionary to flatten
        prefix: Prefix for keys (used internally)
        separator: Separator for nested keys (default: ".")
        
    Returns:
        Flattened dictionary
        
    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items = {}
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, separator))
        elif isinstance(value, list):
            # For lists, store as-is (don't flatten)
            items[new_key] = value
        else:
            items[new_key] = value
    
    return items


def get_nested_value(data: Dict, path: str, separator: str = ".") -> Any:
    """
    Extract a value from a nested dictionary using dot notation.
    
    Args:
        data: The dictionary to extract from
        path: Dot-separated path, e.g., "metadata.org_unit"
        separator: Path separator (default: ".")
    
    Returns:
        The value at the path, or None if not found
        
    Example:
        >>> get_nested_value({"a": {"b": 1}}, "a.b")
        1
    """
    keys = path.split(separator)
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return None
        else:
            return None
    
    return current
