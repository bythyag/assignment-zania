from pathlib import Path
import PyPDF2
from .logger import logger

class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extract text from PDF file with error handling"""
        # Add size check
        file_size_mb = Path(pdf_path).stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # Set max size to 100MB
            raise ValueError(f"PDF too large: {file_size_mb:.2f}MB (max 100MB)")
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
                
                if not text.strip():
                    raise ValueError("No text extracted from PDF")
                
                logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
                return text
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    @staticmethod
    def text_to_markdown(text: str) -> str:
        """Convert plain text to markdown with improved formatting"""
        lines = text.split('\n')
        markdown_text = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Header detection with better heuristics
            if len(line) < 100 and line.isupper() and len(line.split()) > 1:
                markdown_text += f"# {line}\n\n"
            elif line.endswith(':') and len(line) < 80 and not line.count('.') > 2:
                markdown_text += f"## {line}\n\n"
            elif line.startswith(('•', '-', '*')) or line.lstrip().startswith(tuple('123456789')):
                markdown_text += f"{line}\n\n"
            else:
                markdown_text += f"{line}\n\n"
        
        return markdown_text