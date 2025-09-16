from typing import Protocol, Dict, List, Any, Optional, Tuple
from abc import abstractmethod


class Searchable(Protocol):
    """Protocol for memories that support advanced search operations.
    
    This interface extends basic search capabilities with semantic search,
    metadata filtering, and similarity-based retrieval.
    """
    
    def search_by_similarity(self, query: str, threshold: float = 0.8, top_k: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for data based on semantic similarity.
        
        Args:
            query (str): Search query
            threshold (float): Minimum similarity threshold
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Tuple[str, Any, float]]: List of (key, data, similarity_score)
        """
        ...
    
    def search_by_metadata(self, filters: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Search for data based on metadata filters.
        
        Args:
            filters (Dict[str, Any]): Metadata filters
            
        Returns:
            List[Tuple[str, Any]]: List of matching (key, data) pairs
        """
        ...
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Tuple[str, Any]]:
        """Search for data by tags.
        
        Args:
            tags (List[str]): List of tags to search for
            match_all (bool): True to match all tags, False to match any
            
        Returns:
            List[Tuple[str, Any]]: List of (key, data) with matching tags
        """
        ...


class Persistent(Protocol):
    """Protocol for memories that persist data across sessions.
    
    This interface provides capabilities for saving and loading memory state,
    backup/restore operations, and data durability.
    """
    
    def save_to_disk(self, path: str) -> bool:
        """Save memory state to disk.
        
        Args:
            path (str): File path to save to
            
        Returns:
            bool: True if save was successful
        """
        ...
    
    def load_from_disk(self, path: str) -> bool:
        """Load memory state from disk.
        
        Args:
            path (str): File path to load from
            
        Returns:
            bool: True if load was successful
        """
        ...
    
    def backup(self, backup_path: str, compress: bool = True) -> bool:
        """Create backup of memory.
        
        Args:
            backup_path (str): Backup file path
            compress (bool): Whether to compress backup
            
        Returns:
            bool: True if backup was successful
        """
        ...
    
    def restore(self, backup_path: str) -> bool:
        """Restore memory from backup.
        
        Args:
            backup_path (str): Backup file path
            
        Returns:
            bool: True if restore was successful
        """
        ...
    
    def get_persistence_info(self) -> Dict[str, Any]:
        """Get information about persistence status.
        
        Returns:
            Dict[str, Any]: Persistence information (last_saved, file_size, etc.)
        """
        ...


class Cacheable(Protocol):
    """Protocol for memories with caching capabilities.
    
    This interface provides cache management, TTL support,
    and cache performance optimization.
    """
    
    def set_ttl(self, key: str, seconds: int) -> bool:
        """Set time-to-live (TTL) for a key.
        
        Args:
            key (str): Key to set TTL for
            seconds (int): Time to live in seconds
            
        Returns:
            bool: True if TTL was set successfully
        """
        ...
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining time-to-live for a key.
        
        Args:
            key (str): Key to check
            
        Returns:
            Optional[int]: Remaining seconds, None if no TTL
        """
        ...
    
    def refresh_cache(self, key: str) -> bool:
        """Refresh cache for a key.
        
        Args:
            key (str): Key to refresh
            
        Returns:
            bool: True if refresh was successful
        """
        ...
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics information.
        
        Returns:
            Dict[str, Any]: Cache statistics (hit_rate, miss_rate, size, etc.)
        """
        ...
    
    def evict_expired(self) -> int:
        """Remove expired entries.
        
        Returns:
            int: Number of entries that were evicted
        """
        ...


class Configurable(Protocol):
    """Protocol for memories with runtime configuration.
    
    This interface allows dynamic configuration updates,
    parameter tuning, and behavior modification at runtime.
    """
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update runtime configuration.
        
        Args:
            config (Dict[str, Any]): New configuration
            
        Returns:
            bool: True if update was successful
        """
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Dict[str, Any]: Current configuration
        """
        ...
    
    def reset_config(self) -> bool:
        """Reset to default configuration.
        
        Returns:
            bool: True if reset was successful
        """
        ...
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration validity.
        
        Args:
            config (Dict[str, Any]): Configuration to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        ...


class Monitorable(Protocol):
    """Protocol for memories with monitoring and metrics capabilities.
    
    This interface provides performance monitoring, health checks,
    and operational metrics collection.
    """
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics.
        
        Returns:
            Dict[str, float]: Metrics (latency, throughput, error_rate, etc.)
        """
        ...
    
    def reset_metrics(self) -> None:
        """Reset all metrics to 0."""
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """Check memory health status.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        ...
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report.
        
        Returns:
            Dict[str, Any]: Performance report
        """
        ...


class Compressible(Protocol):
    """Protocol for memories with compression capabilities.
    
    This interface provides data compression and decompression
    to optimize storage space and memory usage.
    """
    
    def compress(self) -> int:
        """Compress data in memory.
        
        Returns:
            int: Number of bytes saved
        """
        ...
    
    def decompress(self) -> bool:
        """Decompress data in memory.
        
        Returns:
            bool: True if decompression was successful
        """
        ...
    
    def get_compression_ratio(self) -> float:
        """Get current compression ratio.
        
        Returns:
            float: Compression ratio (0.0 - 1.0)
        """
        ...
    
    def set_compression_level(self, level: int) -> bool:
        """Set compression level.
        
        Args:
            level (int): Compression level (1-9, 9 is highest compression)
            
        Returns:
            bool: True if setting was successful
        """
        ...


class Versionable(Protocol):
    """Protocol for memories with versioning capabilities.
    
    This interface provides version control for stored data,
    allowing rollback and history tracking.
    """
    
    def store_version(self, key: str, data: Any, version: str, metadata: Dict = None) -> bool:
        """Store a specific version of data.
        
        Args:
            key (str): Key to identify data
            data (Any): Data to store
            version (str): Version identifier
            metadata (Dict): Additional metadata
            
        Returns:
            bool: True if storage was successful
        """
        ...
    
    def retrieve_version(self, key: str, version: str) -> Optional[Any]:
        """Retrieve a specific version of data.
        
        Args:
            key (str): Key to identify data
            version (str): Version to retrieve
            
        Returns:
            Optional[Any]: Data for that version, None if not found
        """
        ...
    
    def list_versions(self, key: str) -> List[str]:
        """List all versions of a key.
        
        Args:
            key (str): Key to check
            
        Returns:
            List[str]: List of version identifiers
        """
        ...
    
    def rollback(self, key: str, version: str) -> bool:
        """Rollback to an older version.
        
        Args:
            key (str): Key to rollback
            version (str): Version to rollback to
            
        Returns:
            bool: True if rollback was successful
        """
        ...
    
    def delete_version(self, key: str, version: str) -> bool:
        """Delete a specific version.
        
        Args:
            key (str): Key of the data
            version (str): Version to delete
            
        Returns:
            bool: True if deletion was successful
        """
        ...
    
    def get_version_metadata(self, key: str, version: str) -> Optional[Dict[str, Any]]:
        """Get metadata of a version.
        
        Args:
            key (str): Key of the data
            version (str): Version to get metadata for
            
        Returns:
            Optional[Dict[str, Any]]: Version metadata, None if not found
        """
        ...