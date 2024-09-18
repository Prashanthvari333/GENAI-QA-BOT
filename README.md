# QA Bot

QA Bot is an intelligent question-answering system that allows users to upload documents, build a knowledge base, and ask questions about the content. It uses advanced natural language processing techniques to provide accurate answers based on the uploaded documents.

## Features
- Support for multiple file formats: PDF, DOCX, CSV, TXT, and MD
- Document chunking and preprocessing
- Vector embeddings for efficient similarity search
- Integration with Pinecone for serverless vector database
- Natural language question answering using Cohere's language models
- User-friendly interface built with Gradio

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.7 or higher
- Pinecone API key
- Cohere API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Prashanthvari333/GENAI-QA-BOT.git
   cd GENAI-QA-BOT
   ```
2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3. Install the required packages:
```bash
pip install -r requirements.txt

```
4. Create a .env file in the project root directory and add your API keys:
   ```bash
   PINECONE_API_KEY=your_pinecone_api_key
   COHERE_API_KEY=your_cohere_api_key
   PINECONE_CLOUD=aws  # or gcp, depending on your Pinecone setup
   PINECONE_REGION=us-east-1  # or your preferred region
```
5. Run the application:
```bash
python app.py
```
