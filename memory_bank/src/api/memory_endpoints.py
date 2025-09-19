"""
Memory Block API Endpoints
FastAPI endpoints for memory block management with WindowMemoryAPI
"""

import logging
import sys
import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.window_memory_api import WindowMemoryAPI
from config.llm_config import LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Memory Block API",
    description="API for creating and retrieving memory blocks",
    version="1.0.0"
)

# Request/Response Models
class MemoryBlockRequest(BaseModel):
    """Request model for memory block operations"""
    user_id: str = Field(..., description="User identifier from docman")
    session_id: str = Field(..., description="Session identifier from docman")
    answer: Optional[str] = Field(None, description="Answer from downstream agents (optional)")

class MemoryBlockResponse(BaseModel):
    """Response model for memory block operations"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: Optional[List[Dict[str, Any]]] = Field(None, description="Memory block data")
    postgres_session: Optional[str] = Field(None, description="Postgres session ID for tracking")

class RetrieveMemoryRequest(BaseModel):
    """Request model for retrieving memory blocks"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")

class RetrieveMemoryResponse(BaseModel):
    """Response model for retrieving memory blocks"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    postgres_session: Optional[str] = Field(None, description="Associated postgres session ID")
    memory_blocks: Optional[List[Dict[str, Any]]] = Field(None, description="Retrieved memory blocks")

@app.post("/api/v1/memory-block", response_model=MemoryBlockResponse)
async def create_or_update_memory_block(request: MemoryBlockRequest):
    """
    Endpoint 1: Create or update memory block
    
    Logic:
    1. Check if postgres_session exists for user_id + session_id
    2. If not exists: create new postgres_session, get history, create summary, save as user role
    3. If exists and answer provided: update with assistant role to complete the block
    4. Save to docman using postgres_session as the session identifier
    
    Args:
        request: MemoryBlockRequest with user_id, session_id, and optional answer
        
    Returns:
        MemoryBlockResponse with operation result and memory block data
    """
    try:
        logger.info(f"Processing memory block request for user_id={request.user_id}, session_id={request.session_id}")
        
        # Initialize WindowMemoryAPI
        memory_api = WindowMemoryAPI(
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Check if this is first call (summary creation) or second call (answer update)
        if request.answer is None:
            # First call: Create summary and save as user role
            logger.info("First call: Creating summary block")
            
            # Generate summary and get/create postgres session
            summary_content = memory_api.summarize_history()
            postgres_session = memory_api._get_or_create_postgres_session()
            
            # Save summary as user message to docman using postgres_session
            # We need to switch to postgres_session for saving
            temp_memory_api = WindowMemoryAPI(
                session_id=postgres_session,  # Use postgres_session for docman operations
                user_id=request.user_id
            )
            
            # Get current history from postgres session (should be empty for new session)
            postgres_history = temp_memory_api._get_conversation_history()
            
            # Save summary as user role
            temp_memory_api._save_summary_to_docman(summary_content, postgres_history)
            
            # Build response
            memory_block = {
                "info_docman": {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "postgres_session": postgres_session
                },
                "summary_block": {
                    "conversation_summary": summary_content,
                    "answer": "",
                    "is_complete": False
                },
                "postgres_session": postgres_session,
                "status": "pending_answer"
            }
            
            return MemoryBlockResponse(
                success=True,
                message="Summary block created successfully, waiting for answer",
                data=[memory_block],
                postgres_session=postgres_session
            )
            
        else:
            # Second call: Update with answer to complete the block
            logger.info("Second call: Updating with answer")
            
            # Get existing postgres session
            postgres_session = memory_api._get_or_create_postgres_session()
            
            # Use postgres session to update docman
            temp_memory_api = WindowMemoryAPI(
                session_id=postgres_session,
                user_id=request.user_id
            )
            
            # Update with answer
            temp_memory_api._update_answer_to_docman(request.answer)
            
            # Get the complete summary for response
            summary_content = memory_api.summarize_history()
            
            # Build complete response
            memory_block = {
                "info_docman": {
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "postgres_session": postgres_session
                },
                "summary_block": {
                    "conversation_summary": summary_content,
                    "answer": request.answer,
                    "is_complete": True
                },
                "postgres_session": postgres_session,
                "status": "complete"
            }
            
            return MemoryBlockResponse(
                success=True,
                message="Memory block completed successfully",
                data=[memory_block],
                postgres_session=postgres_session
            )
            
    except Exception as e:
        logger.error(f"Error processing memory block request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process memory block: {str(e)}"
        )

@app.post("/api/v1/memory-block/retrieve", response_model=RetrieveMemoryResponse)
async def retrieve_memory_blocks(request: RetrieveMemoryRequest):
    """
    Endpoint 2: Retrieve memory blocks for user and session
    
    Logic:
    1. Look up postgres_session using user_id + session_id from postgres DB
    2. If found: retrieve memory blocks from docman using postgres_session
    3. If not found: return empty result with appropriate message
    
    Args:
        request: RetrieveMemoryRequest with user_id and session_id
        
    Returns:
        RetrieveMemoryResponse with retrieved memory blocks or empty result
    """
    try:
        logger.info(f"Retrieving memory blocks for user_id={request.user_id}, session_id={request.session_id}")
        
        # Initialize WindowMemoryAPI
        memory_api = WindowMemoryAPI(
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Check if postgres session exists
        postgres_session = memory_api.postgres_db.select_data(
            "docman_information",
            request.user_id,
            request.session_id
        )
        
        if not postgres_session:
            logger.info(f"No postgres session found for user_id={request.user_id}, session_id={request.session_id}")
            return RetrieveMemoryResponse(
                success=True,
                message="No memory blocks found - postgres session does not exist",
                postgres_session=None,
                memory_blocks=[]
            )
        
        logger.info(f"Found postgres session: {postgres_session}")
        
        # Use postgres session to retrieve history from docman
        temp_memory_api = WindowMemoryAPI(
            session_id=postgres_session,
            user_id=request.user_id
        )
        
        # Get conversation history from docman using postgres session
        memory_history = temp_memory_api._get_conversation_history()
        
        if not memory_history:
            return RetrieveMemoryResponse(
                success=True,
                message="No memory blocks found - history is empty",
                postgres_session=postgres_session,
                memory_blocks=[]
            )
        
        # Format memory blocks for response
        memory_blocks = []
        
        # Group messages in pairs (user question + assistant answer)
        for i in range(0, len(memory_history), 2):
            if i + 1 < len(memory_history):
                user_msg = memory_history[i]
                assistant_msg = memory_history[i + 1]
                
                memory_block = {
                    "conversation_summary": user_msg.get("content", ""),
                    "answer": assistant_msg.get("content", ""),
                    "is_complete": True,
                    "timestamp": None  # Could be added if timestamp is stored in history
                }
                memory_blocks.append(memory_block)
            else:
                # Incomplete block (only user message, no assistant response yet)
                user_msg = memory_history[i]
                memory_block = {
                    "conversation_summary": user_msg.get("content", ""),
                    "answer": "",
                    "is_complete": False,
                    "timestamp": None
                }
                memory_blocks.append(memory_block)
        
        return RetrieveMemoryResponse(
            success=True,
            message=f"Retrieved {len(memory_blocks)} memory block(s)",
            postgres_session=postgres_session,
            memory_blocks=memory_blocks
        )
        
    except Exception as e:
        logger.error(f"Error retrieving memory blocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory blocks: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Memory Block API"}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Block API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8132, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Memory Block API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "memory_endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
