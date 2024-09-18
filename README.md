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
#Open your web browser and navigate to the URL provided in the terminal (usually http://127.0.0.1:7860).

#Use the interface to:

Upload a document (PDF, DOCX, CSV, TXT, or MD)
Provide a name for the index (use only lowercase letters, numbers, and hyphens)
Click "Build Bot" to process the document and create the knowledge base
Ask questions about the document in the "Knowledge Bot" tab



#How It Works

Document Processing: When you upload a document, the system chunks it into smaller pieces and preprocesses the text.
Embedding Generation: The preprocessed text chunks are converted into vector embeddings using a SentenceTransformer model.
Vector Storage: The embeddings are stored in a Pinecone vector index for efficient similarity search.
Question Answering: When you ask a question, the system:

Converts your question into a vector embedding
Searches for the most relevant text chunks in the Pinecone index
Uses Cohere's language model to generate an answer based on the retrieved context



#Troubleshooting

If you encounter issues with index creation or deletion, ensure that you have the correct Pinecone API key and permissions.
Make sure you're not exceeding the maximum number of indexes allowed in your Pinecone plan.
If you're having trouble with file uploads, check that the file format is supported and the file is not corrupted.

#Contributing
Contributions to the QA Bot project are welcome! Please follow these steps:

Fork the repository
Create a new branch: git checkout -b feature-branch-name
Make your changes and commit them: git commit -m 'Add some feature'
Push to the branch: git push origin feature-branch-name
Create a pull request

#Acknowledgements

Gradio for the user interface
Pinecone for vector similarity search
Cohere for natural language processing
Sentence-Transformers for text embeddings

#Contact
If you have any questions or feedback, please open an issue on the GitHub repository.
