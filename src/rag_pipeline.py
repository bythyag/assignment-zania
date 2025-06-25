from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
import json
from datetime import datetime

from .config import RAGConfig
from .pdf_processor import PDFProcessor
from .text_chunker import TextChunker
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .openai_client import OpenAIClient
from .logger import logger

class RAGPipeline:
    """Production-grade RAG pipeline with comprehensive error handling and logging"""
    
    def __init__(self, config: RAGConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(config.openai_model)
        self.embedding_generator = EmbeddingGenerator(config.embedding_model)
        self.vector_store = VectorStore(self.embedding_generator.embedding_dim)
        self.openai_client = OpenAIClient(openai_api_key)
        
        # Data storage
        self.chunks = []
        self.chunk_metadata = []
        self.processed_files = {}
        
        logger.info("Initialized RAGPipeline")
    
    def process_pdf(self, pdf_path: Path, save_dir: Optional[Path] = None) -> Path:
        """Process PDF and create embeddings with comprehensive error handling"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            if save_dir is None:
                save_dir = pdf_path.parent / "rag_output"
            
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = pdf_path.stem
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Step 1: Extract and convert text
            raw_text = self.pdf_processor.extract_text(pdf_path)
            markdown_text = self.pdf_processor.text_to_markdown(raw_text)
            
            # Step 2: Create chunks
            chunks = self.text_chunker.create_overlapping_chunks(
                markdown_text, 
                self.config.chunk_size, 
                self.config.chunk_overlap
            )
            
            self.chunks = [chunk['text'] for chunk in chunks]
            self.chunk_metadata = chunks
            
            # Step 3: Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(self.chunks)
            
            # Step 4: Build vector index
            self.vector_store.build_index(embeddings)
            
            # Step 5: Save everything
            self._save_processed_data(save_dir, base_name, markdown_text, embeddings)
            
            # Update processed files record
            self.processed_files[str(pdf_path)] = {
                'timestamp': datetime.now().isoformat(),
                'save_dir': str(save_dir),
                'base_name': base_name,
                'num_chunks': len(chunks),
                'config': self.config.to_dict()
            }
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return save_dir
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _save_processed_data(
        self, 
        save_dir: Path, 
        base_name: str, 
        markdown_text: str, 
        embeddings
    ) -> None:
        """Save all processed data to disk"""
        try:
            # Save chunks and metadata
            chunks_file = save_dir / f"{base_name}_chunks.pkl"
            with open(chunks_file, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.chunk_metadata,
                    'embeddings': embeddings,
                    'config': self.config.to_dict(),
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            # Save FAISS index
            index_file = save_dir / f"{base_name}_index.faiss"
            self.vector_store.save(index_file)
            
            # Save markdown
            markdown_file = save_dir / f"{base_name}.md"
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            # Save processing metadata
            metadata_file = save_dir / f"{base_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'num_chunks': len(self.chunks),
                    'embedding_dim': self.embedding_generator.embedding_dim,
                    'config': self.config.to_dict(),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved all processed data to: {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, save_dir: Path, base_name: str) -> None:
        """Load previously processed data"""
        try:
            save_dir = Path(save_dir)
            
            # Load chunks and metadata
            chunks_file = save_dir / f"{base_name}_chunks.pkl"
            if not chunks_file.exists():
                raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
            
            with open(chunks_file, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_metadata = data['metadata']
            
            # Load FAISS index
            index_file = save_dir / f"{base_name}_index.faiss"
            if not index_file.exists():
                raise FileNotFoundError(f"Index file not found: {index_file}")
            
            self.vector_store.load(index_file)
            
            logger.info(f"Loaded {len(self.chunks)} chunks and index from {save_dir}")
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def search_and_answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Search for relevant chunks and generate answer"""
        try:
            if top_k is None:
                top_k = self.config.top_k
            
            if not self.vector_store.is_trained:
                raise RuntimeError("Vector store not ready. Process a PDF first.")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([question])
            
            # Search for similar chunks
            scores, indices = self.vector_store.search(query_embedding, top_k)
            
            # Prepare search results
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # Valid result
                    result = {
                        'chunk_id': int(idx),
                        'text': self.chunks[idx],
                        'score': float(score),
                        'metadata': self.chunk_metadata[idx]
                    }
                    search_results.append(result)
            
            # Generate answer
            if search_results:
                context_text = "\n\n".join([chunk['text'] for chunk in search_results])
                answer = self.openai_client.generate_answer(
                    question, 
                    context_text,
                    self.config.openai_model,
                    self.config.max_tokens_openai,
                    self.config.temperature
                )
                
                result = {
                    "question": question,
                    "answer": answer,
                    "sources": search_results,
                    "num_sources": len(search_results),
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info(f"Processed question: {question}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def batch_process_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple questions in batch"""
        results = []
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)}: {question}")
                result = self.search_and_answer(question)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                error_result = {
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "sources": [],
                    "num_sources": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": True
                }
                results.append(error_result)
        
        return results

def create_rag_pipeline(
    embedding_model: str = "all-MiniLM-L6-v2",
    openai_api_key: Optional[str] = None,
    **config_kwargs
) -> RAGPipeline:
    """Factory function to create RAG pipeline"""
    config = RAGConfig(embedding_model=embedding_model, **config_kwargs)
    return RAGPipeline(config, openai_api_key)