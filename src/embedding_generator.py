from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .logger import logger

class EmbeddingGenerator:
    """Handles embedding generation and caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings with batching"""
        try:
            if not texts:
                raise ValueError("No texts provided for embedding generation")
            
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise