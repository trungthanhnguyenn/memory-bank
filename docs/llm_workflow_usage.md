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
pip install langchain-openai httpx python-dotenv redis fastapi uvicorn 
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
