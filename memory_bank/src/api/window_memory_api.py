import hashlib
import json
import logging
import time
import requests

from typing import Dict, Any, Optional, List

from core.base_memory import BaseMemory
from core.interfaces import Searchable, Persistent, Cacheable, Configurable, Monitorable, Compressible
from utils.summary_history import SummaryHistory
from utils.postgres_db import PostgresDB
from config.llm_config import LLMConfig

logger = logging.getLogger(__name__)

BASE_URL = "https://wm5090.pythera.ai/"

class WindowMemoryAPI(BaseMemory, Searchable, Persistent, Cacheable, Configurable, Monitorable, Compressible):
    """Window Memory API implementation with LLM backend and semantic search"""
    
    def __init__(self, session_id: str, user_id: str, llm_config: Optional[LLMConfig] = None):
        """Initialize window memory with configuration
        
        Args:
            session_id (str): Session identifier from docman
            user_id (str): User identifier from docman
            llm_config (Optional[LLMConfig]): LLM configuration for summarization
        """
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "deletes": 0,
            "total_size": 0,
            "last_reset": time.time()
        }

        self.session_id = session_id
        self.user_id = user_id
        
        # Initialize LLM config and summary service
        if llm_config is None:
            llm_config = LLMConfig.from_env()
        self.llm_config = llm_config
        self.summarizer = SummaryHistory(llm_config)
        
        # Initialize Postgres DB
        self.postgres_db = PostgresDB()
        
        # Ensure table exists
        try:
            self.postgres_db.create_table("docman_information")
        except Exception as e:
            logging.warning(f"Could not create table: {e}")
        
        logger.info("Window Memory API initialized successfully")

    @staticmethod
    def _generate_session_id() -> Optional[str]:
        """Generate a temporary session ID using external API"""
        url = f"{BASE_URL}api/v1/sessions"
        payload = {
            "user_id": "temp_user",
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            temp_session_id = response.json()['session_id']
            logging.info(f"Generated temporary session ID: {temp_session_id}")
            return temp_session_id
        except Exception as e:
            logging.error(f"Error generating temporary session ID: {e}")
            return None

    def _get_conversation_history(self) -> List[Dict[str, Any]]:
        """Retrieve conversation history for the current session"""
        url = f"{BASE_URL}appv1/api/v1/sessions/{self.session_id}"
        payload = {
            "user_id": self.user_id,
            "session_id": self.session_id
        }

        try:
            response = requests.get(url, json=payload, timeout=10)
            response.raise_for_status()
            response_data = response.json()
            
            # Handle case where metadata or history doesn't exist
            if 'metadata' in response_data and 'history' in response_data['metadata']:
                history = response_data['metadata']['history']
            else:
                logging.info(f"Warning: No history found in response: {response_data}")
                history = []
        except Exception as e:
            logging.error(f"Error retrieving conversation history: {e}")
            history = []

        return history
    
    def _save_history_prompts(self, question: str, answer: str, history: List):
        """Save conversation history to docman API"""
        # Add new turns to history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        url = f"{BASE_URL}appv1/api/v1/sessions/{self.session_id}"
        payload = {
            "metadata": {
                "history": history
            }
        }
        
        try:
            response = requests.put(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logging.info(f"History saved with status: {result.get('status', 'unknown')}")
        except Exception as e:
            logging.error(f"Error saving history: {e}")
            raise e

    def summarize_history(self) -> str:
        """Summarize the conversation history using LLM"""
        history = self._get_conversation_history()
        summary = self.summarizer.summarize(history)
        return summary
    
    def _get_or_create_postgres_session(self) -> str:
        """Get existing postgres session or create new one"""
        # First check if mapping already exists in Postgres
        existing_postgres_session = self.postgres_db.select_data(
            "docman_information", 
            self.user_id, 
            self.session_id
        )
        
        if existing_postgres_session:
            logging.info(f"Found existing postgres session: {existing_postgres_session}")
            return existing_postgres_session
        
        # Create new postgres session if not exists
        new_postgres_session = self._generate_session_id()
        if not new_postgres_session:
            raise Exception("Failed to generate new postgres session")
            
        # Save mapping to Postgres
        success = self.postgres_db.insert_data(
            "docman_information",
            {"user_id": self.user_id, "session_id": self.session_id},
            {"postgres_info": new_postgres_session}
        )
        
        if success:
            logging.info(f"Created new postgres session mapping: {new_postgres_session}")
        
        return new_postgres_session
    
    def build_block_memory(self, postgres_session: Optional[str] = None, answer: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build a block memory for the given user and session
        
        Args:
            postgres_session (Optional[str]): Existing postgres session ID if provided
            answer (Optional[str]): Answer from downstream agents (empty for first save)
            
        Returns:
            List[Dict[str, Any]]: Memory block with summary, postgres session mapping, and metadata
        """
        # Generate summary from conversation history  
        summary_content = self.summarize_history()
        
        # Get or create postgres session
        if postgres_session:
            final_postgres_session = postgres_session
            logging.info(f"Using provided postgres session: {postgres_session}")
        else:
            final_postgres_session = self._get_or_create_postgres_session()

        # Get current conversation history
        history = self._get_conversation_history()
        
        # Determine answer content
        final_answer = answer if answer else ""
        
        # Save to docman (first time with summary as question, later with answer)
        if final_answer:
            # This is the second save - we have the answer now
            self._save_history_prompts(summary_content, final_answer, history)
            logging.info("Saved complete block with summary and answer to docman")
        else:
            # This is the first save - just the summary for now
            temp_history = history.copy()
            temp_history.append({"role": "user", "content": summary_content})
            # Don't save to docman yet, just prepare for later
            logging.info("Prepared summary block, waiting for answer to complete save")
                
        return [
            {
                "info_docman": {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                },
                "summary_block": {
                    "conversation_summary": summary_content,
                    "answer": final_answer,
                    "is_complete": bool(final_answer)  # Flag to indicate if block is complete
                },
                "postgres_session": final_postgres_session,
                "timestamp": time.time(),
                "status": "complete" if final_answer else "pending_answer"
            }
        ]
    
    def _save_summary_to_docman(self, summary_content: str, history: List) -> None:
        """Save summary as user message to docman (first save)"""
        # Add summary as user question to history
        history.append({"role": "user", "content": summary_content})

        url = f"{BASE_URL}appv1/api/v1/sessions/{self.session_id}"
        payload = {
            "metadata": {
                "history": history
            }
        }
        
        try:
            response = requests.put(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logging.info(f"Summary saved to docman with status: {result.get('status', 'unknown')}")
        except Exception as e:
            logging.error(f"Error saving summary to docman: {e}")
            raise e

    def _update_answer_to_docman(self, answer: str) -> None:
        """Update existing docman record with answer (second save)"""
        # Get current history (should include the summary we saved earlier)
        current_history = self._get_conversation_history()
        
        # Add answer as assistant response
        current_history.append({"role": "assistant", "content": answer})

        url = f"{BASE_URL}appv1/api/v1/sessions/{self.session_id}"
        payload = {
            "metadata": {
                "history": current_history
            }
        }
        
        try:
            response = requests.put(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logging.info(f"Answer updated to docman with status: {result.get('status', 'unknown')}")
        except Exception as e:
            logging.error(f"Error updating answer to docman: {e}")
            raise e

    # BaseMemory interface implementation (required methods)
    def store(self, key: str, data: Any, metadata: Optional[Dict] = None) -> bool:
        """Store data in memory"""
        # Implementation for base memory interface
        return True
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory"""
        # Implementation for base memory interface
        return None
        
    def search(self, query: str, threshold: float = 0.8) -> List[tuple]:
        """Search data in memory"""
        # Implementation for base memory interface
        return []
        
    def delete(self, key: str) -> bool:
        """Delete data from memory"""
        # Implementation for base memory interface
        return True
        
    def clear(self) -> int:
        """Clear all data from memory"""
        # Implementation for base memory interface
        return 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self._stats
    
    # Additional interface implementations (Searchable, Persistent, etc.)
    def search_by_similarity(self, query: str, threshold: float = 0.8, top_k: int = 10) -> List[tuple]:
        """Search by similarity (placeholder implementation)"""
        return []
    
    def search_by_metadata(self, filters: Dict[str, Any]) -> List[tuple]:
        """Search by metadata (placeholder implementation)"""
        return []
    
    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[tuple]:
        """Search by tags (placeholder implementation)"""
        return []
    
    # Persistent interface
    def save_to_disk(self, path: str) -> bool:
        """Save to disk (placeholder implementation)"""
        return True
    
    def load_from_disk(self, path: str) -> bool:
        """Load from disk (placeholder implementation)"""
        return True
    
    def backup(self, backup_path: str, compress: bool = True) -> bool:
        """Backup (placeholder implementation)"""
        return True
    
    def restore(self, backup_path: str) -> bool:
        """Restore (placeholder implementation)"""
        return True
    
    def get_persistence_info(self) -> Dict[str, Any]:
        """Get persistence info (placeholder implementation)"""
        return {}
    
    # Cacheable interface
    def set_ttl(self, key: str, seconds: int) -> bool:
        """Set TTL (placeholder implementation)"""
        return True
    
    def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL (placeholder implementation)"""
        return None
    
    def refresh_cache(self, key: str) -> bool:
        """Refresh cache (placeholder implementation)"""
        return True
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache info (placeholder implementation)"""
        return {}
    
    def evict_expired(self) -> int:
        """Evict expired (placeholder implementation)"""
        return 0
    
    # Configurable interface
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update config (placeholder implementation)"""
        return True
    
    def get_config(self) -> Dict[str, Any]:
        """Get config (placeholder implementation)"""
        return {}
    
    def reset_config(self) -> bool:
        """Reset config (placeholder implementation)"""
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate config (placeholder implementation)"""
        return (True, [])
    
    # Monitorable interface
    def get_metrics(self) -> Dict[str, float]:
        """Get metrics (placeholder implementation)"""
        return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Health check (placeholder implementation)"""
        return {"status": "healthy"}
    
    def reset_metrics(self) -> None:
        """Reset metrics (placeholder implementation)"""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "deletes": 0,
            "total_size": 0,
            "last_reset": time.time()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report (placeholder implementation)"""
        return {
            "summary": "WindowMemoryAPI performance report",
            "stats": self._stats,
            "uptime": time.time() - self._stats["last_reset"]
        }
    
    # Compressible interface
    def compress(self) -> int:
        """Compress (placeholder implementation)"""
        return 0
    
    def decompress(self) -> bool:
        """Decompress (placeholder implementation)"""
        return True
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio (placeholder implementation)"""
        return 1.0
    
    def set_compression_level(self, level: int) -> bool:
        """Set compression level (placeholder implementation)"""
        return True

