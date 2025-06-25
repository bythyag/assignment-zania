import logging
from pathlib import Path

def setup_logger(log_file: str = "/Users/thyag/Desktop/Assignement/assignment-zania/logs/rag_pipeline.log", log_level: int = logging.INFO) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    
    return logging.getLogger(__name__)

# Create default logger
logger = setup_logger()