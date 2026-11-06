
from pathlib import Path

# Main project directory
BASE_DIR = Path(__file__).resolve().parent

# HR PDF corpus
CORPUS_DIR = BASE_DIR / "zyro-dynamics-hr-corpus"

# FAISS vector database
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Hugging Face embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunks to retrieve
RETRIEVER_K = 3
