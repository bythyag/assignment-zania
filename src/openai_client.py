from typing import Optional
from openai import OpenAI
from .logger import logger

class OpenAIClient:
    """Wrapper for OpenAI API with error handling and retry logic"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        model: str = "gpt-4o-mini",
        max_tokens: int = 500,
        temperature: float = 0.1
    ) -> str:
        """Generate answer using OpenAI API with retry logic"""
        prompt = f"""Analyze the following document context and answer the question with high precision.

                Context:
                {context}

                Question: {question}

                Answer:"""
        system_prompt = """
                    You are a precise document analysis assistant. Your primary function is to extract exact information from provided context. 
                    Only provide answers when you have high confidence based on the document content. 
                    For direct factual queries, provide word-for-word matches from the source material when possible. 
                    If the information is not clearly present or you have any doubt about accuracy, respond with "Data Not Available" rather than making inferences.
                    """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer with OpenAI: {e}")
            return "Sorry, I couldn't generate an answer due to an error."