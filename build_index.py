
from config import (
    CORPUS_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore():

    # 1. Check corpus directory
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {CORPUS_DIR}"
        )

    pdf_files = list(CORPUS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {CORPUS_DIR}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    # 2. Load PDF documents
    loader = PyPDFDirectoryLoader(str(CORPUS_DIR))
    documents = loader.load()

    print(f"Loaded {len(documents)} document pages.")

    # 3. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # 4. Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # 5. Create FAISS vector store
    vectorstore = FAISS.from_documents(
        chunks,
        embedding_model
    )

    # 6. Save FAISS index
    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("FAISS index saved successfully!")
    print(f"Location: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
