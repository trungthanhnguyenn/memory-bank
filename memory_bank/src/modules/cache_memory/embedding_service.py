import logging
from typing import List, Optional, Union
import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for handling text embeddings using HuggingFace models"""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding service with configuration
        
        Args:
            config (EmbeddingConfig): Embedding configuration
        """
        self.config = config
        self._embeddings = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the embedding model"""
        try:
            model_kwargs = {
                'device': self.config.device
            }
            
            encode_kwargs = {
                'batch_size': self.config.batch_size
            }
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            logger.info(f"Embedding model initialized: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Text embedding as numpy array
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return np.zeros(384, dtype=np.float32)  # Default dimension for MiniLM
            
            # Truncate text if too long
            if len(text) > self.config.max_seq_length:
                text = text[:self.config.max_seq_length]
                logger.debug(f"Text truncated to {self.config.max_seq_length} characters")
            
            embedding = self._embeddings.embed_query(text)
            return np.array(embedding, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            # Return zero vector as fallback
            return np.zeros(384, dtype=np.float32)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            List[np.ndarray]: List of text embeddings as numpy arrays
        """
        try:
            if not texts:
                return []
            
            # Filter and truncate texts
            processed_texts = []
            for text in texts:
                if text and text.strip():
                    if len(text) > self.config.max_seq_length:
                        text = text[:self.config.max_seq_length]
                    processed_texts.append(text)
                else:
                    processed_texts.append("")  # Empty string for empty inputs
            
            embeddings = self._embeddings.embed_documents(processed_texts)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            # Return zero vectors as fallback
            return [np.zeros(384, dtype=np.float32) for _ in texts]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are 2D for sklearn
            emb1 = embedding1.reshape(1, -1)
            emb2 = embedding2.reshape(1, -1)
            
            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray],
                         threshold: Optional[float] = None) -> List[tuple]:
        """Find most similar embeddings to a query embedding
        
        Args:
            query_embedding (np.ndarray): Query embedding
            candidate_embeddings (List[np.ndarray]): List of candidate embeddings
            threshold (Optional[float]): Minimum similarity threshold
            
        Returns:
            List[tuple]: List of (index, similarity_score) sorted by similarity
        """
        try:
            if not candidate_embeddings:
                return []
            
            threshold = threshold or self.config.similarity_threshold
            
            similarities = []
            for i, candidate_emb in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate_emb)
                if similarity >= threshold:
                    similarities.append((i, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model
        
        Returns:
            int: Embedding dimension
        """
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.embed_text("test")
            return test_embedding.shape[0]
            
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 384  # Default for MiniLM
    
    def is_model_loaded(self) -> bool:
        """Check if the embedding model is properly loaded
        
        Returns:
            bool: True if model is loaded and ready
        """
        try:
            if self._embeddings is None:
                return False
            
            # Test with a simple embedding
            test_embedding = self.embed_text("test")
            return test_embedding is not None and len(test_embedding) > 0
            
        except Exception:
            return False
    
    def reload_model(self) -> bool:
        """Reload the embedding model
        
        Returns:
            bool: True if reload was successful
        """
        try:
            logger.info("Reloading embedding model...")
            self._embeddings = None
            self._initialize_model()
            return self.is_model_loaded()
            
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the current model
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "max_seq_length": self.config.max_seq_length,
            "similarity_threshold": self.config.similarity_threshold,
            "embedding_dimension": self.get_embedding_dimension(),
            "model_loaded": self.is_model_loaded()
        }
    
    def update_config(self, new_config: EmbeddingConfig) -> bool:
        """Update embedding configuration and reload model if necessary
        
        Args:
            new_config (EmbeddingConfig): New configuration
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Check if model needs to be reloaded
            model_changed = (
                self.config.model_name != new_config.model_name or
                self.config.device != new_config.device or
                self.config.batch_size != new_config.batch_size
            )
            
            self.config = new_config
            
            if model_changed:
                return self.reload_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        try:
            if hasattr(self, '_embeddings') and self._embeddings is not None:
                # Cleanup if needed
                self._embeddings = None
                logger.debug("Embedding service cleaned up")
        except Exception:
            pass  # Ignore cleanup errors