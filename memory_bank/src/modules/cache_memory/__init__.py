#!/usr/bin/env python3
"""
Cache Memory Module

A production-ready cache memory implementation with Redis backend,
semantic search capabilities, and comprehensive configuration management.

Main Components:
- CacheMemory: Main cache implementation with BaseMemory and multiple interfaces
- EmbeddingService: Text embedding operations using HuggingFace models
- Configuration classes: Environment-based configuration management

Usage:
    from cache_memory import CacheMemory, CacheMemoryConfig
    
    config = CacheMemoryConfig.from_env()
    cache = CacheMemory(config)
    
    # Store data
    cache.store("key", "value", metadata={"type": "text"})
    
    # Retrieve data
    result = cache.retrieve("key")
    
    # Semantic search
    results = cache.search("query text", limit=10)
"""

# Import configuration classes
from .config import (
    RedisConfig,
    EmbeddingConfig,
    CacheConfig,
    CacheMemoryConfig
)

# Import main service classes
from .cache_memory import CacheMemory
from .embedding_service import EmbeddingService

# Version information
__version__ = "1.0.0"
__author__ = "Cache Memory Team"
__description__ = "Production-ready cache memory with semantic search"

# Public API exports
__all__ = [
    # Configuration classes
    "RedisConfig",
    "EmbeddingConfig", 
    "CacheConfig",
    "CacheMemoryConfig",
    
    # Service classes
    "CacheMemory",
    "EmbeddingService",
    
    # Module metadata
    "__version__",
    "__author__",
    "__description__"
]

# Module-level convenience functions
def create_cache_from_env() -> CacheMemory:
    """Create a CacheMemory instance from environment variables
    
    Returns:
        CacheMemory: Configured cache memory instance
        
    Raises:
        ValueError: If required environment variables are missing
        ConnectionError: If Redis connection fails
    """
    config = CacheMemoryConfig.from_env()
    return CacheMemory(config)

def create_embedding_service_from_env() -> EmbeddingService:
    """Create an EmbeddingService instance from environment variables
    
    Returns:
        EmbeddingService: Configured embedding service instance
        
    Raises:
        ValueError: If required environment variables are missing
        RuntimeError: If model initialization fails
    """
    embedding_config = EmbeddingConfig.from_env()
    return EmbeddingService(embedding_config)

# Add convenience functions to __all__
__all__.extend([
    "create_cache_from_env",
    "create_embedding_service_from_env"
])

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Cache Memory module v{__version__} initialized")