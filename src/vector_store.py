from typing import Tuple
from pathlib import Path
import numpy as np
import faiss
from .logger import logger

class VectorStore:
    """Handles vector storage and retrieval using FAISS"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = None
        self.is_trained = False
    
    def build_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> None:
        """Build FAISS index with optional GPU support"""
        try:
            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")
            
            # Use IndexFlatIP for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy().astype('float32')
            faiss.normalize_L2(embeddings_normalized)
            
            # Add to GPU if requested and available
            if use_gpu and faiss.get_num_gpus() > 0:
                gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                gpu_index.add(embeddings_normalized)
                self.index = gpu_index
                logger.info("Using GPU for FAISS index")
            else:
                self.index.add(embeddings_normalized)
            
            self.is_trained = True
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors"""
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        try:
            # Normalize query embedding
            query_normalized = query_embedding.copy().astype('float32')
            faiss.normalize_L2(query_normalized)
            
            scores, indices = self.index.search(query_normalized, top_k)
            return scores, indices
            
        except Exception as e:
            logger.error(f"Error searching index: {e}")
            raise
    
    def save(self, filepath: Path) -> None:
        """Save FAISS index to disk"""
        if not self.is_trained:
            raise RuntimeError("No index to save")
        
        try:
            faiss.write_index(self.index, str(filepath))
            logger.info(f"Saved FAISS index to {filepath}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            raise
    
    def load(self, filepath: Path) -> None:
        """Load FAISS index from disk"""
        try:
            self.index = faiss.read_index(str(filepath))
            self.is_trained = True
            logger.info(f"Loaded FAISS index from {filepath}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise