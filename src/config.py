from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "gpt-4o-mini",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        temperature: float = 0.0,
        max_tokens_openai: int = 500,

    ):
        self.embedding_model = embedding_model
        self.openai_model = openai_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens_openai = max_tokens_openai
