#!/usr/bin/env python3
"""
Cache Memory API Server
FastAPI server providing REST endpoints for cache memory operations
"""

import logging
import os
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import cache memory modules
import sys
sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from modules.cache_memory import CacheMemory, CacheMemoryConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global cache instance
cache_memory: Optional[CacheMemory] = None


def _generate_session_key(key: str, session: Optional[str] = None) -> str:
    """Generate a session-aware cache key
    
    Args:
        key (str): Original cache key
        session (Optional[str]): Session identifier
        
    Returns:
        str: Session-aware cache key
    """
    if session:
        return f"session:{session}:{key}"
    return key


def _search_by_session(query: str, session: Optional[str], threshold: float, top_k: int):
    """Search cache with session filtering
    
    Args:
        query (str): Search query
        session (Optional[str]): Session identifier  
        threshold (float): Similarity threshold
        top_k (int): Maximum results
        
    Returns:
        List[Tuple[str, Any, float]]: List of (key, data, similarity_score)
    """
    if not cache_memory:
        return []
    
    if session:
        # For session-aware search, we need to search using the session-aware key format
        # Since embeddings are generated from the full key including session prefix,
        # we need to construct the session-aware search key
        session_query = f"session:{session}:llm:{query}"
        
        # Use the session-aware key for embedding comparison
        all_results = cache_memory.search_by_similarity(
            query=session_query,
            threshold=threshold,
            top_k=top_k
        )
        
        # Filter results by session and remove session prefix from keys
        session_prefix = f"session:{session}:"
        filtered_results = []
        
        for key, data, similarity_score in all_results:
            if key.startswith(session_prefix):
                # Remove session prefix from key for response
                original_key = key[len(session_prefix):]
                filtered_results.append((original_key, data, similarity_score))
        
        return filtered_results
    else:
        # No session filter, search normally but exclude session-specific entries
        all_results = cache_memory.search_by_similarity(
            query=query,
            threshold=threshold,
            top_k=1000  # Get more results to filter out sessions
        )
        
        # Filter out session-specific entries
        non_session_results = []
        for key, data, similarity_score in all_results:
            if not key.startswith("session:"):
                non_session_results.append((key, data, similarity_score))
                
                if len(non_session_results) >= top_k:
                    break
        
        return non_session_results


# Pydantic models for API requests/responses
class CacheLookupRequest(BaseModel):
    """Request model for cache lookup"""
    query: str = Field(..., description="Query text to search for")
    similarity_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Minimum similarity threshold")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    session: Optional[str] = Field(None, description="Session identifier for cache isolation")


class CacheStoreRequest(BaseModel):
    """Request model for storing data in cache"""
    key: str = Field(..., description="Cache key")
    data: Any = Field(..., description="Data to store")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    tags: Optional[List[str]] = Field(None, description="Optional tags")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    session: Optional[str] = Field(None, description="Session identifier for cache isolation")


class CacheRetrieveRequest(BaseModel):
    """Request model for retrieving data from cache"""
    key: str = Field(..., description="Cache key to retrieve")
    session: Optional[str] = Field(None, description="Session identifier for cache isolation")


class CacheSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    session: Optional[str] = Field(None, description="Session identifier for cache isolation")


class CacheLookupResponse(BaseModel):
    """Response model for cache lookup"""
    found: bool = Field(..., description="Whether a match was found")
    data: Optional[Any] = Field(None, description="Retrieved data if found")
    similarity_score: Optional[float] = Field(None, description="Similarity score if found")
    key: Optional[str] = Field(None, description="Cache key if found")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata if found")


class CacheStoreResponse(BaseModel):
    """Response model for cache store operation"""
    success: bool = Field(..., description="Whether the operation was successful")
    key: str = Field(..., description="The cache key used")
    message: Optional[str] = Field(None, description="Additional message")


class CacheRetrieveResponse(BaseModel):
    """Response model for cache retrieve operation"""
    found: bool = Field(..., description="Whether the key was found")
    data: Optional[Any] = Field(None, description="Retrieved data if found")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata if found")
    key: str = Field(..., description="The cache key")


class CacheSearchResponse(BaseModel):
    """Response model for cache search operation"""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of results found")
    query: str = Field(..., description="Original search query")


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    total_keys: int = Field(..., description="Total number of keys in cache")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage statistics")
    hit_rate: float = Field(..., description="Cache hit rate")
    uptime: float = Field(..., description="Cache uptime in seconds")
    config: Dict[str, Any] = Field(..., description="Current cache configuration")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    cache_connected: bool = Field(..., description="Whether cache is connected")
    embedding_model_loaded: bool = Field(..., description="Whether embedding model is loaded")
    version: str = Field(..., description="API version")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown"""
    # Startup
    global cache_memory
    try:
        logger.info("Initializing Cache Memory API...")
        
        # Load configuration from environment
        config = CacheMemoryConfig.from_env()
        
        # Initialize cache memory
        cache_memory = CacheMemory(config)
        
        # Test connections
        health = cache_memory.health_check()
        if not health.get("redis_connected", False):
            raise RuntimeError("Failed to connect to Redis")
        
        logger.info("Cache Memory API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Cache Memory API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cache Memory API...")
    if cache_memory:
        try:
            # Cleanup if needed
            pass
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Cache Memory API",
    description="REST API for Cache Memory operations with semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    try:
        if cache_memory:
            health = cache_memory.health_check()
            cache_connected = health.get("redis_connected", False)
            # Check if embedding service is available
            try:
                cache_memory.embedding_service.embed_text("test")
                embedding_loaded = True
            except:
                embedding_loaded = False
        else:
            cache_connected = False
            embedding_loaded = False
        
        return HealthResponse(
            status="healthy" if cache_connected and embedding_loaded else "degraded",
            cache_connected=cache_connected,
            embedding_model_loaded=embedding_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint"""
    return await root()


@app.post("/cache/lookup", response_model=CacheLookupResponse)
async def cache_lookup(request: CacheLookupRequest):
    """Lookup cached data using semantic search"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Perform session-aware semantic search
        results = _search_by_session(
            query=request.query,
            session=request.session,
            threshold=request.similarity_threshold,
            top_k=request.limit
        )
        
        if results:
            # Return the best match - results are tuples of (key, data, similarity_score)
            key, data, similarity_score = results[0]
            return CacheLookupResponse(
                found=True,
                data=data,
                similarity_score=similarity_score,
                key=key,
                metadata=None  # metadata not returned in this format
            )
        else:
            return CacheLookupResponse(found=False)
            
    except Exception as e:
        logger.error(f"Cache lookup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache lookup failed: {str(e)}"
        )


@app.post("/cache/store", response_model=CacheStoreResponse)
async def cache_store(request: CacheStoreRequest):
    """Store data in cache"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Generate session-aware cache key
        session_key = _generate_session_key(request.key, request.session)
        
        # Store data in cache with session-aware key
        success = cache_memory.store(
            key=session_key,
            data=request.data,
            metadata=request.metadata
        )
        
        if success:
            return CacheStoreResponse(
                success=True,
                key=session_key,
                message="Data stored successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store data"
            )
            
    except Exception as e:
        logger.error(f"Cache store failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache store failed: {str(e)}"
        )


@app.post("/cache/retrieve", response_model=CacheRetrieveResponse)
async def cache_retrieve(request: CacheRetrieveRequest):
    """Retrieve data from cache by key"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Generate session-aware cache key
        session_key = _generate_session_key(request.key, request.session)
        
        # Retrieve data from cache using session-aware key
        data = cache_memory.retrieve(session_key)
        
        if data is not None:
            # Get metadata if available (would need session-aware get_metadata method)
            # For now, skip metadata retrieval
            metadata = None
            
            return CacheRetrieveResponse(
                found=True,
                data=data,
                metadata=metadata,
                key=request.key  # Return original key, not session key
            )
        else:
            return CacheRetrieveResponse(
                found=False,
                data=None,
                metadata=None,
                key=request.key
            )
            
    except Exception as e:
        logger.error(f"Cache retrieve failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache retrieve failed: {str(e)}"
        )


@app.post("/cache/search", response_model=CacheSearchResponse)
async def cache_search(request: CacheSearchRequest):
    """Perform semantic search in cache"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Perform session-aware semantic search
        results = _search_by_session(
            query=request.query,
            session=request.session,
            threshold=request.similarity_threshold,
            top_k=request.limit
        )
        
        # Convert tuples to dict format for response
        formatted_results = []
        for key, data, similarity_score in results:
            formatted_results.append({
                "key": key,
                "data": data,
                "similarity_score": similarity_score,
                "metadata": None
            })
        
        # Filter by tags if provided (skip for now since metadata not available in this format)
        # TODO: Implement tag filtering when metadata is properly returned
        
        return CacheSearchResponse(
            results=formatted_results,
            total_found=len(formatted_results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Cache search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache search failed: {str(e)}"
        )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    """Get cache statistics"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Get cache statistics
        stats = cache_memory.get_stats()
        
        return CacheStatsResponse(
            total_keys=stats.get('total_keys', 0),
            memory_usage=stats.get('memory_usage', {}),
            hit_rate=stats.get('hit_rate', 0.0),
            uptime=stats.get('uptime', 0.0),
            config=stats.get('config', {})
        )
        
    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache stats failed: {str(e)}"
        )


@app.delete("/cache/clear")
async def cache_clear():
    """Clear all cache data"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Clear cache
        deleted_count = cache_memory.clear()
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Cleared {deleted_count} cache entries"
        }
        
    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {str(e)}"
        )


@app.delete("/cache/key/{key}")
async def cache_delete_key(key: str, session: Optional[str] = None):
    """Delete specific cache key"""
    try:
        if not cache_memory:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cache service not available"
            )
        
        # Generate session-aware cache key
        session_key = _generate_session_key(key, session)
        
        # Delete specific key using session-aware key
        success = cache_memory.delete(session_key)
        
        if success:
            return {
                "success": True,
                "key": key,
                "message": f"Key '{key}' deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Key '{key}' not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache delete key failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache delete key failed: {str(e)}"
        )


def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Memory API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=1234, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Cache Memory API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "cache_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()