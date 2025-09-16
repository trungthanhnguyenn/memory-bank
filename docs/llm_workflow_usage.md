# LLM Workflow with Cache Memory - Usage Guide

## Overview

LLM Workflow with Cache Memory is a production-ready example for integrating cache memory with LLM workflow. This module helps improve response speed and reduce API costs by caching LLM responses.

## Installation

### System Requirements

- Python 3.10+
- Redis server (for cache memory)
- OpenAI API key

### Dependencies

```bash
pip install langchain-openai httpx python-dotenv redis fastapi uvicorn sen
```

## Configuration

### Environment Variables

Create a `.env` file or set the following environment variables:

```bash
# OpenAI Configuration
API_KEY="your_openai_api_key_here"
MODEL_NAME="gpt-3.5-turbo"  # Optional
BASE_URL="https://api.openai.com/v1"  # Optional
TEMPERATURE="0.7"  # Optional

# Cache Configuration
CACHE_API_URL="http://localhost:1234"  # Cache API endpoint
CACHE_SIMILARITY_THRESHOLD="0.8"  # Optional, default: 0.8
SESSION_MEMORY_CACHED="your_session_id"  # Optional, for session-based caching
```

### Cache API Setup

The workflow now uses a Cache API instead of direct Redis connection. You need to start the Cache API server:

#### Start Cache API Server
```bash
# Navigate to your project directory
cd /path/to/memory-bank

# Start the FastAPI cache server
uvicorn memory_bank.src.api.cache_api:app --host 0.0.0.0 --port 1234
```

#### Docker Setup (Optional)
```bash
# If using Docker for the cache API
docker run -d --name cache-api -p 1234:1234 your-cache-api-image
```

## Basic Usage

### 1. Import and Initialize

```python
from examples.llm_workflow_with_cached import create_workflow

# Create workflow from environment variables
workflow = create_workflow()

# The workflow will automatically connect to the Cache API
# and fall back to direct LLM calls if cache is unavailable
```

### 2. Simple Query Processing

```python
# Process a single query
result = workflow.process("What is machine learning?")

print(f"Response: {result['response']}")
print(f"Cache hit: {result['cache_hit']}")
```

### 3. Multiple Query Processing

```python
queries = [
    "What is Python?",
    "Explain neural networks",
    "What is Python?"  # Duplicate query to test cache
]

for query in queries:
    result = workflow.process(query)
    status = "CACHE HIT" if result["cache_hit"] else "LLM CALL"
    print(f"[{status}] {query}")
    print(f"Response: {result['response'][:100]}...\n")
```

## Advanced Usage

### 1. Custom Configuration

```python
from examples.llm_workflow_with_cached import WorkflowConfig, LLMWorkflow

# Create custom config
config = WorkflowConfig.from_dict({
    "openai_api_key": "your_api_key",
    "model_name": "gpt-4",
    "temperature": 0.5,
    "similarity_threshold": 0.9,
    "cache_api_url": "http://localhost:1234",
    "session_memory_cached": "custom_session_id"
})

# Initialize workflow with custom config
workflow = LLMWorkflow(config)
```

### 2. Monitoring and Statistics

```python
# Get statistics
stats = workflow.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Clear cache
deleted_count = workflow.clear_cache()
print(f"Cleared {deleted_count} cache entries")
```

### 3. Error Handling

```python
try:
    workflow = create_workflow()
    result = workflow.process("Your question here")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Workflow error: {e}")
```

## API Reference

### WorkflowConfig

#### `__init__()`
Khởi tạo config từ environment variables.

#### `from_dict(config_dict: Dict[str, Any]) -> WorkflowConfig`
Create config from dictionary.

**Parameters:**
- `config_dict`: Dictionary containing configuration
  - `openai_api_key` (str): OpenAI API key
  - `model_name` (str, optional): LLM model name
  - `base_url` (str, optional): Base URL for API
  - `temperature` (float, optional): Temperature for LLM
  - `similarity_threshold` (float, optional): Similarity threshold for cache
  - `cache_api_url` (str, optional): Cache API endpoint URL
  - `session_memory_cached` (str, optional): Session ID for session-based caching

### LLMWorkflow

#### `__init__(config: WorkflowConfig)`
Initialize workflow with config.

**Parameters:**
- `config`: WorkflowConfig instance

#### `process(query: str) -> Dict[str, Any]`
Process query with cache lookup.

**Parameters:**
- `query`: User question

**Returns:**
- Dictionary containing:
  - `response` (str): Response from LLM or cache
  - `cache_hit` (bool): True if retrieved from cache

#### `get_stats() -> Dict[str, Any]`
Get workflow statistics.

**Returns:**
- Dictionary containing:
  - `total_queries` (int): Total number of queries
  - `cache_hits` (int): Number of cache hits
  - `cache_misses` (int): Number of cache misses
  - `hit_rate` (float): Cache hit rate

#### `clear_cache() -> int`
Clear all cache entries.

**Returns:**
- Number of deleted entries

### Helper Functions

#### `create_workflow() -> LLMWorkflow`
Create workflow from environment config.

**Returns:**
- Configured LLMWorkflow instance

## Complete Example

```python
#!/usr/bin/env python3

import os
from examples.llm_workflow_with_cached import create_workflow

def main():
    # Set environment variables
    os.environ["API_KEY"] = "your_api_key_here"
    os.environ["CACHE_API_URL"] = "http://localhost:1234"
    
    try:
        # Create workflow
        workflow = create_workflow()
        
        # Test queries
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "What is artificial intelligence?",  # Duplicate
            "How does deep learning work?"
        ]
        
        print("=== LLM Workflow with Cache Demo ===")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[Query {i}] {query}")
            
            result = workflow.process(query)
            
            if result["cache_hit"]:
                print("✅ CACHE HIT - Fast response!")
            else:
                print("🌐 LLM CALL - New query processed")
            
            print(f"Response: {result['response'][:150]}...")
        
        # Show statistics
        print("\n=== Statistics ===")
        stats = workflow.get_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

#### 1. "API_KEY environment variable is required"
**Solution:** Set the API_KEY environment variable
```bash
export API_KEY="your_api_key_here"
```

#### 2. "Cache API connection failed"
**Cause:** Cache API server not running or misconfigured
**Solution:**
- Check if Cache API is running: `curl http://localhost:1234/health`
- Verify CACHE_API_URL configuration
- Start Cache API server: `uvicorn memory_bank.src.api.cache_api:app --port 1234`

#### 3. "Cache API unhealthy"
**Cause:** Cache API server running but cache backend unavailable
**Solution:**
- Check Cache API logs for errors
- Verify Redis/backend connection in Cache API
- Restart Cache API server if needed

#### 4. Slow Performance
**Cause:** Cache API latency or network issues
**Solution:**
- Check Cache API response times
- Increase CACHE_SIMILARITY_THRESHOLD to reduce cache lookups
- Optimize Cache API server configuration
- Consider running Cache API locally

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Workflow sẽ log chi tiết các hoạt động
workflow = create_workflow()
```

## Best Practices

1. **Environment Management**: Use `.env` file for development, environment variables for production

2. **Cache Strategy**: 
   - Set appropriate similarity threshold (0.8-0.9)
   - Monitor hit rate and adjust threshold
   - Use session-based caching for user-specific contexts
   - Periodically clear cache to avoid stale data

3. **Error Handling**: 
   - Always wrap workflow calls in try-catch
   - Implement graceful fallback when cache API is unavailable
   - Log cache connection issues for monitoring

4. **Performance**: 
   - Run Cache API server close to your application
   - Monitor Cache API response times
   - Consider horizontal scaling for Cache API
   - Use HTTP connection pooling with httpx

5. **Security**: 
   - Don't commit API keys to code
   - Use environment variables or secret management
   - Secure Cache API endpoints in production
   - Implement authentication for Cache API if needed

6. **Monitoring**: 
   - Track cache hit rates and API response times
   - Monitor Cache API health endpoint
   - Set up alerts for cache failures

## Integration with Other Frameworks

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from examples.llm_workflow_with_cached import create_workflow
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize workflow
try:
    workflow = create_workflow()
except Exception as e:
    logger.error(f"Failed to initialize workflow: {e}")
    workflow = None

class ChatRequest(BaseModel):
    query: str
    session_id: str = None

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not available")
    
    try:
        # Update session if provided
        if request.session_id:
            workflow.config.session_memory_cached = request.session_id
        
        result = workflow.process(request.query)
        return {
            "response": result["response"],
            "cache_hit": result["cache_hit"],
            "stats": workflow.get_stats()
        }
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if workflow else "unhealthy",
        "cache_enabled": workflow.cache_enabled if workflow else False
    }

@app.get("/stats")
async def stats_endpoint():
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not available")
    return workflow.get_stats()

@app.delete("/cache")
async def clear_cache_endpoint():
    if not workflow:
        raise HTTPException(status_code=503, detail="Workflow not available")
    deleted = workflow.clear_cache()
    return {"deleted_count": deleted}
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from examples.llm_workflow_with_cached import create_workflow
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Initialize workflow
try:
    workflow = create_workflow()
except Exception as e:
    logger.error(f"Failed to initialize workflow: {e}")
    workflow = None

@app.route('/chat', methods=['POST'])
def chat():
    if not workflow:
        return jsonify({'error': 'Workflow not available'}), 503
    
    data = request.get_json()
    query = data.get('query')
    session_id = data.get('session_id')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        # Update session if provided
        if session_id:
            workflow.config.session_memory_cached = session_id
        
        result = workflow.process(query)
        return jsonify({
            'response': result['response'],
            'cache_hit': result['cache_hit'],
            'stats': workflow.get_stats()
        })
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if workflow else 'unhealthy',
        'cache_enabled': workflow.cache_enabled if workflow else False
    })

@app.route('/stats', methods=['GET'])
def stats():
    if not workflow:
        return jsonify({'error': 'Workflow not available'}), 503
    return jsonify(workflow.get_stats())

@app.route('/cache', methods=['DELETE'])
def clear_cache():
    if not workflow:
        return jsonify({'error': 'Workflow not available'}), 503
    deleted = workflow.clear_cache()
    return jsonify({'deleted_count': deleted})

if __name__ == '__main__':
    app.run(debug=True)
```

## Conclusion

LLM Workflow with Cache Memory provides a production-ready solution for integrating cache with LLM workflow. This module helps:

- Improve response speed through semantic caching
- Reduce API call costs
- Easy integration with existing applications
- Effective monitoring and debugging

With proper configuration, you can achieve high hit rates and good performance for your LLM applications.