#!/usr/bin/env python3
"""
LLM Workflow with Cache Memory Integration
Production-ready example using cache API
"""

import os
import logging
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowConfig:
    """Configuration for LLM workflow"""
    
    def __init__(self):
        """Initialize config from environment variables"""
        self.openai_api_key = os.getenv("API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
        self.base_url = os.getenv("BASE_URL")
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        self.similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.8"))
        self.cache_api_url = os.getenv("CACHE_API_URL", "http://localhost:1234")
        self.session_memory_cached = os.getenv("SESSION_MEMORY_CACHED")
        
        if not self.openai_api_key:
            raise ValueError("API_KEY environment variable is required")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'WorkflowConfig':
        instance = cls.__new__(cls)
        instance.openai_api_key = config_dict["openai_api_key"]
        instance.model_name = config_dict.get("model_name", "gpt-3.5-turbo")
        instance.base_url = config_dict.get("base_url")
        instance.temperature = config_dict.get("temperature", 0.7)
        instance.similarity_threshold = config_dict.get("similarity_threshold", 0.8)
        instance.cache_api_url = config_dict.get("cache_api_url", "http://localhost:1234")
        instance.session_memory_cached = config_dict.get("session_memory_cached")
        return instance


class LLMWorkflow:
    """Production LLM workflow with cache integration"""
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow with config"""
        self.config = config
        self._init_llm()
        self._init_cache()
        self._init_stats()
    
    def _init_llm(self):
        """Initialize LLM client"""
        llm_kwargs = {
            "model_name": self.config.model_name,
            "openai_api_key": self.config.openai_api_key,
            "temperature": self.config.temperature
        }
        
        if self.config.base_url:
            llm_kwargs["openai_api_base"] = self.config.base_url
        
        self.llm = ChatOpenAI(**llm_kwargs)
    
    def _init_cache(self):
        """Initialize cache API client"""
        self.cache_client = httpx.Client(base_url=self.config.cache_api_url, timeout=30.0)
        try:
            response = self.cache_client.get("/health")
            if response.status_code == 200:
                health_data = response.json()
                self.cache_enabled = health_data.get("cache_connected", False)
                if self.cache_enabled:
                    logger.info("Cache API connected successfully")
                else:
                    logger.warning("Cache API unhealthy - running without cache")
            else:
                logger.warning("Cache API not responding - running without cache")
                self.cache_enabled = False
        except Exception as e:
            logger.error(f"Cache API connection failed: {e}")
            self.cache_enabled = False
    
    def _init_stats(self):
        """Initialize statistics tracking"""
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _search_cache(self, query: str) -> Optional[str]:
        """Search cache using API"""
        if not self.cache_enabled:
            return None
        
        try:
            lookup_data = {
                "query": query,
                "similarity_threshold": self.config.similarity_threshold,
                "limit": 1
            }
            if self.config.session_memory_cached:
                lookup_data["session"] = self.config.session_memory_cached
            
            response = self.cache_client.post("/cache/lookup", json=lookup_data)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("found"):
                    cached_data = data.get("data", {})
                    return cached_data.get("response")
            
            return None
        except Exception as e:
            logger.error(f"Cache search failed: {e}")
            return None
    
    def _store_cache(self, query: str, response: str) -> bool:
        """Store data in cache using API"""
        if not self.cache_enabled:
            return False
        
        try:
            cache_key = f"llm:{query}"
            store_data = {
                "key": cache_key,
                "data": {"query": query, "response": response},
                "metadata": {"model": self.config.model_name}
            }
            if self.config.session_memory_cached:
                store_data["session"] = self.config.session_memory_cached
            
            api_response = self.cache_client.post("/cache/store", json=store_data)
            
            return api_response.status_code == 200 and api_response.json().get("success", False)
        except Exception as e:
            logger.error(f"Cache store failed: {e}")
            return False
    
    def _call_llm(self, query: str) -> str:
        """Call LLM API"""
        template = f"""You are a helpful AI assistant. Please provide a clear and concise answer.

Question: {query}

Answer:"""
        
        result = self.llm.invoke(template)
        return result.content if hasattr(result, 'content') else str(result)
    
    def process(self, query: str) -> Dict[str, Any]:
        """Process query with cache lookup"""
        self.stats["total_queries"] += 1
        
        # Try cache first
        cached_response = self._search_cache(query)
        if cached_response:
            self.stats["cache_hits"] += 1
            return {
                "response": cached_response,
                "cache_hit": True
            }
        
        # Cache miss - call LLM
        self.stats["cache_misses"] += 1
        response = self._call_llm(query)
        
        # Store in cache
        self._store_cache(query, response)
        
        return {
            "response": response,
            "cache_hit": False
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        stats = self.stats.copy()
        total = stats["total_queries"]
        if total > 0:
            stats["hit_rate"] = stats["cache_hits"] / total
        else:
            stats["hit_rate"] = 0.0
        return stats
    
    def clear_cache(self) -> int:
        """Clear cache using API"""
        if not self.cache_enabled:
            return 0
        try:
            response = self.cache_client.delete("/cache/clear")
            if response.status_code == 200:
                data = response.json()
                return data.get("deleted_count", 0)
            return 0
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return 0
    
    def __del__(self):
        """Cleanup HTTP client on destruction"""
        if hasattr(self, 'cache_client'):
            self.cache_client.close()


def create_workflow() -> LLMWorkflow:
    """Create workflow from environment config"""
    config = WorkflowConfig()
    return LLMWorkflow(config)


def main():
    """Main function for testing"""
    try:
        workflow = create_workflow()
        
        # Test queries
        test_queries = [
            "What is Python?",
            "Explain machine learning",
            "tell me what is Python?"  
        ]
        
        for query in test_queries:
            result = workflow.process(query)
            print('This is result: ', result)
            status = "CACHE HIT" if result["cache_hit"] else "LLM CALL"
            logger.info(f"[{status}] {query[:50]}...")
        
        # Print stats
        stats = workflow.get_stats()
        logger.info(f"Stats: {stats}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


if __name__ == "__main__":
    # main()
    workflow = create_workflow()
    result = workflow.process("Hey, tell me about what is Java?")
    print(result)



