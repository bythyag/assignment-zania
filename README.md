## Zania Assignment - RAG Question Answering System

A RAG system that processes PDF documents and answers questions using OpenAI's GPT models and local embedding search.

GPT Model used in the project: ```gpt-4o-mini```
Local Embedding model used in the project: ```all-MiniLM-L6-v2```

### current features

1. **PDF Processing:** Extract and convert PDF texts to markdown format to maintain document structure. 
2. **Text chunking:** Splits document into overlapping chunks. 
3. **Semantic Search:** Use FAISS for indexing and vector similarity search on local machine.
4. **Question Answering:** Uses OpenAI model to answer questions based on the retrieved contents.

### Thoughts on improvements

1. One of the ways we improved the document processing is through the ```async``` method while looping though the list of questions and also during document chunking and indexing. 
2. To further improve on the confidence of the answer we can do the following:
    1. Add a relevancy agent that can compare the question and the answer to make sure on the relevancy of the answer, or otherwise return 'No Data Found'
    2. We can also filter the texts using cosine similarity score to only pass similar documents with certain threshold. 
3. The project is designed to run locally, but for scaling purpose we can use OpenAI Embedding model and MongoDB or Pinecone vector database.
4. Introduce more agents for sub query generation on each question to get more related documents. 
5. Combine exact word matching and semantic similarity for improved document retrieval. 
6. Use multiple models for fallback such that we don't face any downtime during the production.

### Thoughts on code
1. A separate file for the OpenAI prompts for better version control. 
2. Any embedding model should work during the chunking and indexing process without making major changes in the code. 
3. Separate python file consisting of all the file paths across the codebase. 
4. Separate python file for different features and functions.

### Entry Points
- **app.py**: Simplified entry point with predefined questions and PDF path for easy interpretation and quick testings.
- **main.py**: Core application logic that app.py imports. Use it for custom implementations during the RAG process.
- **Relationship**: app.py imports and calls main.py with hardcoded parameters, while main.py accepts configurable inputs

### How to use

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

3. Basic Usage
Run the application with the list of your questions and pdf path:
```bash
# python app.py
from main import main

# PDF file path
pdf_path = "/Users/thyag/Desktop/Assignement/assignment-zania/dataset/raw-data/handbook.pdf"

# Questions
questions = [
    "What is the name of the company?",
    "Who is the CEO of the company?", 
    "What is their vacation policy?",
    "What is the termination policy?"
]

# Run
main()
```

### Sample Output

```bash
Q1: What is the name of the company?
A1: The name of the company is Zania, Inc.

Q2: Who is the CEO of the company?
A2: The CEO of the company is Shruti Gupta.

Q3: What is their vacation policy?
A3: Zania, Inc. provides paid vacation to all full-time regular employees...
```

### Notes on file directory 

1. Run the ```app.py``` file with list of questions and pdf document. 
2. ```main.py``` can also be used but (1) is more simplified and cleaner in usage.
3. The core modules are present in ```src/``` directory. 
4. ```notebook/``` contains experiments on design and implementation. 
5. ```dataset/```directory contains original dataset and parsed md dataset. 
5. ```logs/``` contains track of all the intermediate activites. 
