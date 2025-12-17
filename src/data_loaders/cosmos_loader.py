"""
Cosmos DB loader for reading JD data from Azure Cosmos DB.
"""

from typing import Any, Dict, List, Optional, Generator

from .base_loader import BaseJDLoader, FieldMapping


class CosmosDBLoader(BaseJDLoader):
    """
    Load JD data from Azure Cosmos DB.
    
    Supports both connection string and endpoint/key authentication.
    """
    
    # Cosmos DB system fields to exclude
    SYSTEM_FIELDS = {"_rid", "_self", "_etag", "_attachments", "_ts"}
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        connection_string: Optional[str] = None,
        database_name: str = None,
        container_name: str = None,
        query: Optional[str] = None,
        field_mapping: Optional["FieldMapping"] = None,
        max_item_count: int = 100,
        remove_system_fields: bool = True,
    ):
        """
        Initialize the Cosmos DB loader.
        
        Args:
            endpoint: Cosmos DB endpoint URL (use with key)
            key: Cosmos DB key (use with endpoint)
            connection_string: Full connection string (alternative to endpoint/key)
            database_name: Name of the database
            container_name: Name of the container
            query: SQL query to filter results (default: SELECT * FROM c)
            field_mapping: Optional FieldMapping for transforming records
            max_item_count: Items per page for pagination
            remove_system_fields: Remove Cosmos system fields from records
        """
        super().__init__(field_mapping)
        
        self.endpoint = endpoint
        self.key = key
        self.connection_string = connection_string
        self.database_name = database_name
        self.container_name = container_name
        self.query = query or "SELECT * FROM c"
        self.max_item_count = max_item_count
        self.remove_system_fields = remove_system_fields
        
        if not database_name or not container_name:
            raise ValueError("database_name and container_name are required")
        
        if not connection_string and not (endpoint and key):
            raise ValueError("Either connection_string or (endpoint + key) must be provided")
        
        self._client = None
        self._container = None
    
    def _get_client(self):
        """Get or create Cosmos DB client (lazy initialization)."""
        if self._client is None:
            try:
                from azure.cosmos import CosmosClient
            except ImportError:
                raise ImportError(
                    "azure-cosmos package is required. "
                    "Install with: pip install azure-cosmos"
                )
            
            if self.connection_string:
                self._client = CosmosClient.from_connection_string(self.connection_string)
            else:
                self._client = CosmosClient(self.endpoint, credential=self.key)
        
        return self._client
    
    def _get_container(self):
        """Get the Cosmos DB container."""
        if self._container is None:
            client = self._get_client()
            database = client.get_database_client(self.database_name)
            self._container = database.get_container_client(self.container_name)
        return self._container
    
    def _clean_record(self, record: Dict) -> Dict:
        """Remove system fields from a record if configured."""
        if self.remove_system_fields:
            return {k: v for k, v in record.items() if k not in self.SYSTEM_FIELDS}
        return record
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load all data from Cosmos DB."""
        container = self._get_container()
        items = list(container.query_items(
            query=self.query,
            enable_cross_partition_query=True
        ))
        return [self._clean_record(item) for item in items]
    
    def _iterate_records(self, raw_data: List[Dict]) -> Generator[Dict[str, Any], None, None]:
        """Iterate through Cosmos DB records."""
        for record in raw_data:
            yield record
    
    def count_records(self) -> int:
        """Count total records using a COUNT query (more efficient)."""
        container = self._get_container()
        count_query = "SELECT VALUE COUNT(1) FROM c"
        result = list(container.query_items(
            query=count_query,
            enable_cross_partition_query=True
        ))
        return result[0] if result else 0
    
    def sample(self, n: int = 5, apply_mapping: bool = False) -> List[Dict[str, Any]]:
        """Load a sample using TOP query (more efficient than base class)."""
        sample_query = f"SELECT TOP {n} * FROM c"
        container = self._get_container()
        
        records = list(container.query_items(
            query=sample_query,
            enable_cross_partition_query=True
        ))
        
        result = [self._clean_record(r) for r in records]
        
        if apply_mapping and self.field_mapping:
            result = [self._apply_mapping(r) for r in result]
        
        return result
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return basic info."""
        try:
            container = self._get_container()
            properties = container.read()
            count = self.count_records()
            
            return {
                "status": "connected",
                "database": self.database_name,
                "container": self.container_name,
                "partition_key": str(properties.get("partitionKey", {}).get("paths", [])),
                "record_count": count
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
    
    @classmethod
    def from_env(
        cls,
        database_name: str,
        container_name: str,
        endpoint_env: str = "COSMOS_ENDPOINT",
        key_env: str = "COSMOS_KEY",
        connection_string_env: str = "COSMOS_CONNECTION_STRING",
        **kwargs
    ) -> "CosmosDBLoader":
        """Create a loader using environment variables for credentials."""
        import os
        
        connection_string = os.environ.get(connection_string_env)
        if connection_string:
            return cls(
                connection_string=connection_string,
                database_name=database_name,
                container_name=container_name,
                **kwargs
            )
        
        endpoint = os.environ.get(endpoint_env)
        key = os.environ.get(key_env)
        
        if endpoint and key:
            return cls(
                endpoint=endpoint,
                key=key,
                database_name=database_name,
                container_name=container_name,
                **kwargs
            )
        
        raise ValueError(
            f"Set either {connection_string_env} or both {endpoint_env} and {key_env}"
        )
