
# Zyro Dynamics HR Assistant

An AI-powered HR policy chatbot built using
Retrieval-Augmented Generation (RAG).

## Features

- HR document ingestion using PyPDFDirectoryLoader
- Recursive text chunking
- Hugging Face sentence embeddings
- FAISS vector search
- Groq LLM integration
- HR scope guardrail
- Context-grounded answers
- Source document references
- Streamlit chat interface

## Architecture

PDF Documents
    ↓
Document Loading
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Context + Question
    ↓
Groq LLM
    ↓
Answer + Sources

## Technologies

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face
- Groq

## Run Locally

Install dependencies:

pip install -r requirements.txt

Configure your Groq API key in .env:

GROQ_API_KEY=your_api_key

Build the vector index:

python build_index.py

Start the application:

streamlit run app.py

## Project Structure

app.py
build_index.py
rag_pipeline.py
config.py
requirements.txt
zyro-dynamics-hr-corpus/
vectorstore/

## Security

Never commit API keys or private credentials.
