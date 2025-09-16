
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import redis
from sklearn.metrics.pairwise import cosine_similarity

from core.base_memory import BaseMemory
from core.interfaces import Searchable, Persistent, Cacheable, Configurable, Monitorable, Compressible
from .config import CacheMemoryConfig
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class CacheMemory(BaseMemory, Searchable, Persistent, Cacheable, Configurable, Monitorable, Compressible):
    """Cache Memory implementation with Redis backend and semantic search"""
    
    def __init__(self, config: CacheMemoryConfig):
        """Initialize cache memory with configuration
        
        Args:
            config (CacheMemoryConfig): Configuration object
        """
        self.config = config
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "deletes": 0,
            "total_size": 0,
            "last_reset": time.time()
        }
        
        # Initialize Redis connection
        self._init_redis()
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(config.embedding)
        
        logger.info("Cache Memory initialized successfully")
    
    def _init_redis(self) -> None:
        """Initialize Redis connection with authentication"""
        try:
            redis_url = self.config.redis.url
            if self.config.redis.password and "@" not in redis_url:
                redis_url = redis_url.replace(
                    "redis://localhost:6379",
                    f"redis://:{self.config.redis.password}@localhost:6379"
                )
            
            self.redis_client = redis.from_url(
                redis_url,
                max_connections=self.config.redis.max_connections,
                socket_timeout=self.config.redis.socket_timeout
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate Redis cache key from input key
        
        Args:
            key (str): Input key
            
        Returns:
            str: Redis cache key
        """
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.config.redis.cache_prefix}:{key_hash}"
    
    # BaseMemory implementation
    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Store data in cache with optional metadata
        
        Args:
            key (str): Unique identifier for the data
            data (Any): Data to store
            metadata (Optional[Dict]): Additional metadata
            
        Returns:
            bool: True if storage was successful
        """
        try:
            cache_key = self._generate_cache_key(key)
            
            # Generate embedding for semantic search
            embedding = self.embedding_service.embed_text(key)
            
            # Prepare cache data
            cache_data = {
                "key": key,
                "data": json.dumps(data) if not isinstance(data, str) else data,
                "metadata": json.dumps(metadata or {}),
                "embedding": embedding.tobytes(),
                "timestamp": time.time()
            }
            
            # Store in Redis
            self.redis_client.hset(cache_key, mapping=cache_data)
            self.redis_client.expire(cache_key, self.config.redis.cache_ttl)
            
            self._stats["stores"] += 1
            self._update_total_size()
            
            logger.debug(f"Stored data for key: {key[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data for key {key}: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by exact key match
        
        Args:
            key (str): Key to retrieve
            
        Returns:
            Optional[Any]: Retrieved data or None if not found
        """
        try:
            cache_key = self._generate_cache_key(key)
            cached_data = self.redis_client.hgetall(cache_key)
            
            if not cached_data:
                self._stats["misses"] += 1
                return None
            
            self._stats["hits"] += 1
            
            # Parse data
            data = cached_data.get(b"data", b"").decode("utf-8")
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
                
        except Exception as e:
            logger.error(f"Failed to retrieve data for key {key}: {e}")
            self._stats["misses"] += 1
            return None
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[str, Any]]:
        """Search for data using basic text matching
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            List[Tuple[str, Any]]: List of (key, data) pairs
        """
        try:
            pattern = f"{self.config.redis.cache_prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            results = []
            for redis_key in keys[:limit]:
                cached_data = self.redis_client.hgetall(redis_key)
                if cached_data:
                    key = cached_data.get(b"key", b"").decode("utf-8")
                    if query.lower() in key.lower():
                        data = cached_data.get(b"data", b"").decode("utf-8")
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            pass
                        results.append((key, data))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete(self, key: str) -> bool:
        """Delete data by key
        
        Args:
            key (str): Key to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            cache_key = self._generate_cache_key(key)
            result = self.redis_client.delete(cache_key)
            
            if result > 0:
                self._stats["deletes"] += 1
                self._update_total_size()
                logger.debug(f"Deleted key: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached data
        
        Returns:
            bool: True if clearing was successful
        """
        try:
            pattern = f"{self.config.redis.cache_prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
            
            # Reset stats
            self._stats.update({
                "hits": 0,
                "misses": 0,
                "stores": 0,
                "deletes": 0,
                "total_size": 0,
                "last_reset": time.time()
            })
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        self._update_total_size()
        
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "miss_rate": 1 - hit_rate,
            "total_requests": total_requests
        }
    
    def _update_total_size(self) -> None:
        """Update total cache size statistics"""
        try:
            pattern = f"{self.config.redis.cache_prefix}:*"
            keys = self.redis_client.keys(pattern)
            self._stats["total_size"] = len(keys)
        except Exception as e:
            logger.error(f"Failed to update cache size: {e}")
    
    # Searchable interface implementation
    def search_by_similarity(self, query: str, threshold: float = 0.8, top_k: int = 10) -> List[Tuple[str, Any, float]]:
        """Search for data based on semantic similarity
        
        Args:
            query (str): Search query
            threshold (float): Minimum similarity threshold
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Tuple[str, Any, float]]: List of (key, data, similarity_score)
        """
        try:
            # Use configured threshold if not provided
            if threshold == 0.8:
                threshold = self.config.embedding.similarity_threshold
            
            pattern = f"{self.config.redis.cache_prefix}:*"
            cache_keys = self.redis_client.keys(pattern)
            
            if not cache_keys:
                return []
            
            # Generate embedding for query
            query_embedding = self.embedding_service.embed_text(query).reshape(1, -1)
            
            results = []
            for cache_key in cache_keys:
                cached_data = self.redis_client.hgetall(cache_key)
                if not cached_data:
                    continue
                
                # Get stored embedding
                embedding_bytes = cached_data.get(b"embedding")
                if not embedding_bytes:
                    continue
                
                stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
                
                # Calculate similarity
                similarity = cosine_similarity(query_embedding, stored_embedding)[0][0]
                
                if similarity >= threshold:
                    key = cached_data.get(b"key", b"").decode("utf-8")
                    data = cached_data.get(b"data", b"").decode("utf-8")
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                    
                    results.append((key, data, float(similarity)))
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def search_by_metadata(self, filters: Dict[str, Any]) -> List[Tuple[str, Any]]:
        """Search for data based on metadata filters
        
        Args:
            filters (Dict[str, Any]): Metadata filters
            
        Returns:
            List[Tuple[str, Any]]: List of matching (key, data) pairs
        """
        try:
            pattern = f"{self.config.redis.cache_prefix}:*"
            cache_keys = self.redis_client.keys(pattern)
            
            results = []
            for cache_key in cache_keys:
                cached_data = self.redis_client.hgetall(cache_key)
                if not cached_data:
                    continue
                
                # Parse metadata
                metadata_str = cached_data.get(b"metadata", b"{}").decode("utf-8")
                try:
                    metadata = json.loads(metadata_str)
                except json.JSONDecodeError:
                    continue
                
                # Check if all filters match
                match = True
                for filter_key, filter_value in filters.items():
                    if metadata.get(filter_key) != filter_value:
                        match = False
                        break
                
                if match:
                    key = cached_data.get(b"key", b"").decode("utf-8")
                    data = cached_data.get(b"data", b"").decode("utf-8")
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                    
                    results.append((key, data))
            
            return results
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[Tuple[str, Any]]:
        """Search for data by tags
        
        Args:
            tags (List[str]): List of tags to search for
            match_all (bool): True to match all tags, False to match any
            
        Returns:
            List[Tuple[str, Any]]: List of (key, data) with matching tags
        """
        try:
            pattern = f"{self.config.redis.cache_prefix}:*"
            cache_keys = self.redis_client.keys(pattern)
            
            results = []
            for cache_key in cache_keys:
                cached_data = self.redis_client.hgetall(cache_key)
                if not cached_data:
                    continue
                
                # Parse metadata to get tags
                metadata_str = cached_data.get(b"metadata", b"{}").decode("utf-8")
                try:
                    metadata = json.loads(metadata_str)
                    item_tags = metadata.get("tags", [])
                except json.JSONDecodeError:
                    continue
                
                # Check tag matching
                if match_all:
                    match = all(tag in item_tags for tag in tags)
                else:
                    match = any(tag in item_tags for tag in tags)
                
                if match:
                    key = cached_data.get(b"key", b"").decode("utf-8")
                    data = cached_data.get(b"data", b"").decode("utf-8")
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                    
                    results.append((key, data))
            
            return results
            
        except Exception as e:
            logger.error(f"Tag search failed: {e}")
            return []
    
    # Persistent interface implementation
    def save_to_disk(self, path: str) -> bool:
        """Save memory state to disk
        
        Args:
            path (str): File path to save to
            
        Returns:
            bool: True if save was successful
        """
        try:
            # Export all cache data
            pattern = f"{self.config.redis.cache_prefix}:*"
            cache_keys = self.redis_client.keys(pattern)
            
            export_data = {
                "config": self.config.to_dict(),
                "stats": self._stats,
                "data": []
            }
            
            for cache_key in cache_keys:
                cached_data = self.redis_client.hgetall(cache_key)
                if cached_data:
                    # Convert bytes to strings for JSON serialization
                    item_data = {}
                    for k, v in cached_data.items():
                        if k == b"embedding":
                            # Convert embedding to list for JSON
                            embedding = np.frombuffer(v, dtype=np.float32)
                            item_data[k.decode()] = embedding.tolist()
                        else:
                            item_data[k.decode()] = v.decode("utf-8")
                    
                    export_data["data"].append(item_data)
            
            # Save to file
            with open(path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cache saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache to {path}: {e}")
            return False
    
    def load_from_disk(self, path: str) -> bool:
        """Load memory state from disk
        
        Args:
            path (str): File path to load from
            
        Returns:
            bool: True if load was successful
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                import_data = json.load(f)
            
            # Clear existing cache
            self.clear()
            
            # Restore data
            for item_data in import_data.get("data", []):
                cache_key = self._generate_cache_key(item_data["key"])
                
                # Convert embedding back to bytes
                if "embedding" in item_data:
                    embedding = np.array(item_data["embedding"], dtype=np.float32)
                    item_data["embedding"] = embedding.tobytes()
                
                # Store in Redis
                self.redis_client.hset(cache_key, mapping=item_data)
                self.redis_client.expire(cache_key, self.config.redis.cache_ttl)
            
            # Restore stats
            if "stats" in import_data:
                self._stats.update(import_data["stats"])
            
            logger.info(f"Cache loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cache from {path}: {e}")
            return False
    
    # Additional interface implementations would continue here...
    # For brevity, implementing key methods from each interface
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure memory with parameters
        
        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters
        """
        # Update configuration dynamically
        if "cache_ttl" in config:
            self.config.redis.cache_ttl = config["cache_ttl"]
        if "similarity_threshold" in config:
            self.config.embedding.similarity_threshold = config["similarity_threshold"]
        if "max_size" in config:
            self.config.cache.max_size = config["max_size"]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        stats = self.get_stats()
        return {
            "hit_rate": stats["hit_rate"],
            "miss_rate": stats["miss_rate"],
            "total_requests": float(stats["total_requests"]),
            "cache_size": float(stats["total_size"]),
            "uptime": time.time() - stats["last_reset"]
        }
    
    def compress(self) -> int:
        """Compress data in memory
        
        Returns:
            int: Number of bytes saved
        """
        # Placeholder for compression implementation
        logger.info("Compression not implemented yet")
        return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Check memory health status
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            # Test Redis connection
            self.redis_client.ping()
            redis_healthy = True
        except Exception:
            redis_healthy = False
        
        stats = self.get_stats()
        
        return {
            "healthy": redis_healthy,
            "redis_connected": redis_healthy,
            "cache_size": stats["total_size"],
            "hit_rate": stats["hit_rate"],
            "uptime": time.time() - stats["last_reset"]
        }