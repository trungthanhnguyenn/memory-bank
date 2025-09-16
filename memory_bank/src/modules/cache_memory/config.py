import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv


@dataclass
class RedisConfig:
    """Redis configuration for cache storage"""
    url: str = "redis://localhost:6379"
    password: str = "mypassword"
    cache_prefix: str = "memory_cache"
    cache_ttl: int = 3600  # 1 hour
    max_connections: int = 10
    socket_timeout: int = 30
    
    @classmethod
    def from_env(cls, config_file: str = ".env") -> "RedisConfig":
        """Create Redis config from environment variables"""
        if not os.path.isabs(config_file):
            # If relative path, make it relative to the project root
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), config_file)
        load_dotenv(config_file)
        return cls(
            url=os.getenv("REDIS_URL", cls.url),
            password=os.getenv("REDIS_PASSWORD", cls.password),
            cache_prefix=os.getenv("CACHE_PREFIX", cls.cache_prefix),
            cache_ttl=int(os.getenv("CACHE_TTL", cls.cache_ttl)),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", cls.max_connections)),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", cls.socket_timeout))
        )


@dataclass
class EmbeddingConfig:
    """Embedding model configuration for semantic search"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.8
    device: str = "cpu"
    batch_size: int = 32
    max_seq_length: int = 512
    
    @classmethod
    def from_env(cls, config_file: str = ".env") -> "EmbeddingConfig":
        """Create embedding config from environment variables"""
        if not os.path.isabs(config_file):
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), config_file)
        load_dotenv(config_file)
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", cls.model_name),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", cls.similarity_threshold)),
            device=os.getenv("EMBEDDING_DEVICE", cls.device),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", cls.batch_size)),
            max_seq_length=int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", cls.max_seq_length))
        )


@dataclass
class CacheConfig:
    """Cache-specific configuration"""
    max_size: int = 10000  # Maximum number of cached items
    eviction_policy: str = "lru"  # lru, lfu, fifo
    compression_enabled: bool = True
    compression_level: int = 6
    enable_stats: bool = True
    stats_interval: int = 300  # seconds
    
    @classmethod
    def from_env(cls, config_file: str = ".env") -> "CacheConfig":
        """Create cache config from environment variables"""
        if not os.path.isabs(config_file):
            config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), config_file)
        load_dotenv(config_file)
        return cls(
            max_size=int(os.getenv("CACHE_MAX_SIZE", cls.max_size)),
            eviction_policy=os.getenv("CACHE_EVICTION_POLICY", cls.eviction_policy),
            compression_enabled=os.getenv("CACHE_COMPRESSION_ENABLED", str(cls.compression_enabled)).lower() == "true",
            compression_level=int(os.getenv("CACHE_COMPRESSION_LEVEL", cls.compression_level)),
            enable_stats=os.getenv("CACHE_ENABLE_STATS", str(cls.enable_stats)).lower() == "true",
            stats_interval=int(os.getenv("CACHE_STATS_INTERVAL", cls.stats_interval))
        )


@dataclass
class CacheMemoryConfig:
    """Complete cache memory configuration"""
    redis: RedisConfig
    embedding: EmbeddingConfig
    cache: CacheConfig
    
    @classmethod
    def from_env(cls, config_file: str = ".env") -> "CacheMemoryConfig":
        """Create complete cache memory config from environment"""
        return cls(
            redis=RedisConfig.from_env(config_file),
            embedding=EmbeddingConfig.from_env(config_file),
            cache=CacheConfig.from_env(config_file)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for easy serialization"""
        return {
            "redis": self.redis.__dict__,
            "embedding": self.embedding.__dict__,
            "cache": self.cache.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration values"""
        # Validate Redis config
        if not self.redis.url or not self.redis.cache_prefix:
            return False
        
        # Validate embedding config
        if not self.embedding.model_name or self.embedding.similarity_threshold < 0 or self.embedding.similarity_threshold > 1:
            return False
        
        # Validate cache config
        if self.cache.max_size <= 0 or self.cache.eviction_policy not in ["lru", "lfu", "fifo"]:
            return False
        
        return True