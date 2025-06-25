from typing import List, Dict, Any
import tiktoken
from .logger import logger

class TextChunker:
    """Handles text chunking with overlap"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def create_overlapping_chunks(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
        """Create overlapping text chunks with metadata"""
        try:
            tokens = self.encoding.encode(text)
            chunks = []
            
            if len(tokens) <= chunk_size:
                # Single chunk if text is small enough
                return [{
                    "chunk_id": 0,
                    "text": text,
                    "token_count": len(tokens),
                    "start_token": 0,
                    "end_token": len(tokens)
                }]
            
            start = 0
            chunk_id = 0
            
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                chunk_data = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "token_count": len(chunk_tokens),
                    "start_token": start,
                    "end_token": end
                }
                
                chunks.append(chunk_data)
                
                if end >= len(tokens):
                    break
                    
                start = end - overlap
                chunk_id += 1
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise