import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
from pathlib import Path
from typing import List, Dict, Any
import json
import dotenv
from src import create_rag_pipeline, setup_logger
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

# Load environment variables
dotenv.load_dotenv()


async def process_pdf_and_answer_questions(
    pdf_path: str, 
    questions: List[str],
    save_dir: str = None,
    force_reprocess: bool = False,
    max_concurrent: int = 3  # Control concurrency level
) -> List[Dict[str, Any]]:
    """
    Main function to process PDF and answer questions asynchronously
    
    Args:
        pdf_path: Path to the PDF file
        questions: List of questions to answer
        save_dir: Directory to save processed files (optional)
        force_reprocess: Force reprocessing even if files exist
        max_concurrent: Maximum number of concurrent question processing
    
    Returns:
        List of question-answer results
    """
    
    # Setup logger
    logger = setup_logger()

    # Add validation
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("File must be a PDF")
    
    if not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Questions must be non-empty strings")
    
    try:
        print("Initializing RAG Pipeline...")
        
        # Create RAG pipeline
        rag = create_rag_pipeline(
            embedding_model="all-MiniLM-L6-v2",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=500,
            chunk_overlap=50,
            top_k=5
        )
        
        # Convert to Path objects
        pdf_path = Path(pdf_path)
        if save_dir is None:
            save_dir = pdf_path.parent / "rag_output"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True, parents=True)
        
        base_name = pdf_path.stem
        chunks_file = save_dir / f"{base_name}_chunks.pkl"
        
        # Check if we need to process or can load existing
        if not force_reprocess and chunks_file.exists():
            print("Loading existing processed data...")
            with tqdm(total=3, desc="Loading", ncols=70) as pbar:
                rag.load_processed_data(save_dir, base_name)
                pbar.update(1)
                pbar.set_description("Chunks loaded")
                pbar.update(1)
                pbar.set_description("Index loaded")
                pbar.update(1)
        else:
            print("Processing PDF document...")
            
            # Extract text
            print("Extracting text from PDF...")
            raw_text = rag.pdf_processor.extract_text(pdf_path)
            markdown_text = rag.pdf_processor.text_to_markdown(raw_text)
            
            # Create chunks
            print("Creating text chunks...")
            chunks = rag.text_chunker.create_overlapping_chunks(
                markdown_text, 
                rag.config.chunk_size, 
                rag.config.chunk_overlap
            )
            print(f"Created {len(chunks)} chunks")
            
            rag.chunks = [chunk['text'] for chunk in chunks]
            rag.chunk_metadata = chunks
            
            # Generate embeddings with progress bar
            print("Generating embeddings...")
            with tqdm(total=len(rag.chunks), desc="Embeddings", ncols=70) as pbar:
                embeddings = rag.embedding_generator.generate_embeddings(rag.chunks)
                pbar.update(len(rag.chunks))
            
            # Build index
            print("Building search index...")
            with tqdm(total=1, desc="Indexing", ncols=70) as pbar:
                rag.vector_store.build_index(embeddings)
                pbar.update(1)
            
            # Save data
            print("Saving processed data...")
            rag._save_processed_data(save_dir, base_name, markdown_text, embeddings)
        
        print(f"\n❓ Processing {len(questions)} questions asynchronously...")
        print("="*60)
        
        # Create async wrapper for search_and_answer
        async def process_question(question_idx, question):
            i = question_idx + 1
            print(f"\nProcessing Question {i}/{len(questions)}: {question}")
            
            try:
                # Need a sync progress bar since we're executing concurrently
                print(f"Searching and generating answer...")
                result = rag.search_and_answer(question)
                
                # Show immediate answer
                if not result.get('error', False):
                    print(f"Answer for Q{i}: {result['answer'][:100]}{'...' if len(result['answer']) > 100 else ''}")
                    print(f"Sources: {result['num_sources']} relevant chunks")
                else:
                    print(f"Error for Q{i}: {result['answer']}")
                
                return result
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                return {
                    "question": question,
                    "answer": str(e),
                    "error": True,
                    "sources": [],
                    "num_sources": 0
                }
        
        # Process questions concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process_question(idx, question):
            async with semaphore:
                return await process_question(idx, question)
        
        # Launch all tasks and gather results
        tasks = [bounded_process_question(idx, q) for idx, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        
        # Save results
        results_file = save_dir / "qa_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        
        # Final Summary
        print(f"\n{'='*60}")
        print("FINAL ANSWERS SUMMARY")
        print(f"{'='*60}")
        
        for i, result in enumerate(results, 1):
            if not result.get('error', False):
                print(f"\nQ{i}: {result['question']}")
                print(f"\nA{i}: {result['answer']}")
            else:
                print(f"\nQ{i}: {result['question']}")
                print(f"ERROR: {result['answer']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

async def async_main():
    """Async main function with example usage"""
    Path("logs").mkdir(exist_ok=True)
    
    print("RAG Question Answering System")
    print("="*50)
    
    pdf_file_path = "/Users/thyag/Desktop/Assignement/assignment-zania/dataset/raw-data/handbook.pdf"
    
    questions = [
        "What is the name of the company?",
        "Who is the CEO of the company?",
        "What is their vacation policy?",
        "What is the termination policy?"
    ]
    
    save_directory = "/Users/thyag/Desktop/Assignement/assignment-zania/dataset/rag_output"

    print(f"Document: {Path(pdf_file_path).name}")
    print(f"Questions: {len(questions)}")
    print(f"Output: {save_directory}")
    print()
    
    # Run the pipeline
    try:
        results = await process_pdf_and_answer_questions(
            pdf_path=pdf_file_path,
            questions=questions,
            save_dir=save_directory,
            force_reprocess=False,
            max_concurrent=3  # Process up to 3 questions concurrently
        )
        
        print(f"\nSuccessfully processed {len(results)} questions!")
        error_count = sum(1 for r in results if r.get('error', False))
        if error_count > 0:
            print(f"{error_count} questions had errors")
        
    except FileNotFoundError as e:
        print(f"Error: PDF file not found. Please check the path: {pdf_file_path}")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Check the logs for more details.")

def main():
    """Main entrypoint that calls the async main function"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()