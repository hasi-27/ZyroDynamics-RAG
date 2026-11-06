
import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Project configuration
from config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    RETRIEVER_K,
)


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================


load_dotenv()

import streamlit as st

# Read API key locally or from Streamlit Cloud secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    try:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        GROQ_API_KEY = None

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Configure it in .env or Streamlit Secrets."
    )

# Make the key available to ChatGroq
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



# ==================================================
# 2. INITIALIZE EMBEDDING MODEL
# ==================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ==================================================
# 3. LOAD FAISS VECTOR STORE
# ==================================================

faiss_index_path = VECTORSTORE_DIR / "index.faiss"

if not faiss_index_path.exists():
    raise FileNotFoundError(
        "FAISS index not found. "
        "Please run build_index.py first."
    )

vectorstore = FAISS.load_local(
    str(VECTORSTORE_DIR),
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS vector store loaded successfully!")


# ==================================================
# 4. INITIALIZE RETRIEVER
# ==================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVER_K
    }
)


# ==================================================
# 5. INITIALIZE GROQ LLM
# ==================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=500
)


# ==================================================
# 6. HR SCOPE GUARDRAIL PROMPT
# ==================================================

GUARDRAIL_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a scope classifier for the Zyro Dynamics
    HR assistant.

    Classify questions about internal company policies
    and employee-related matters as IN_SCOPE.

    IN_SCOPE topics include:

    - Leave policies and holidays
    - Salary and employee benefits
    - Reimbursements
    - Travel and expense policies
    - Domestic and international travel entitlements
    - Travel approvals and budgets
    - Travel advances
    - Expense submissions and reimbursement timelines
    - Attendance and working hours
    - Code of conduct
    - Other internal employee policies

    OUT_OF_SCOPE topics include:

    - General knowledge unrelated to company policies
    - Coding and programming help
    - Unrelated mathematics
    - Unrelated small talk

    If a question concerns any internal Zyro Dynamics
    employee policy, classify it as IN_SCOPE.

    Respond with exactly one word:
    IN_SCOPE or OUT_OF_SCOPE.

    Question: {question}
    """
)


# ==================================================
# 7. RAG ANSWER PROMPT
# ==================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
    You are the Zyro Dynamics HR assistant.

    Answer the employee's question using ONLY the
    information provided in the retrieved context.

    Instructions:

    1. Use only the supplied context.
    2. Do not invent company policies, numbers, or rules.
    3. If the answer is not available in the context,
       clearly say that you do not have that information.
    4. If the context contains relevant information,
       answer clearly and directly.
    5. If multiple policy details are needed, combine
       them carefully without changing their meaning.

    Retrieved Context:
    {context}

    Employee Question:
    {question}

    Answer:
    """
)


# ==================================================
# 8. REFUSAL MESSAGE
# ==================================================

REFUSAL_MESSAGE = (
    "I'm the Zyro Dynamics HR assistant. "
    "I can only help with internal company HR and "
    "employee policies, such as leave, reimbursement, "
    "travel, benefits, and company guidelines."
)


# ==================================================
# 9. FORMAT RETRIEVED DOCUMENTS
# ==================================================

def format_docs(docs):
    """
    Combine retrieved document page contents
    into a single context string.
    """

    return "\n\n".join(
        doc.page_content for doc in docs
    )


# ==================================================
# 10. RAG PIPELINE
# ==================================================

def rag_chain(question: str):
    """
    Retrieve relevant HR document chunks
    and generate an answer using Groq.
    """

    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Handle empty retrieval
    if not docs:
        return {
            "answer": (
                "I couldn't find relevant information "
                "in the available HR documents."
            ),
            "sources": []
        }

    # Convert retrieved documents into context
    context = format_docs(docs)

    # Build RAG chain
    chain = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "answer": answer,
        "sources": docs
    }


# ==================================================
# 11. MAIN CHATBOT FUNCTION
# ==================================================

def ask_bot(question: str):
    """
    Check whether the question is within HR scope.
    If in scope, execute the RAG pipeline.
    Otherwise, return a refusal message.
    """

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    # Create guardrail chain
    guardrail_chain = (
        GUARDRAIL_PROMPT
        | llm
        | StrOutputParser()
    )

    # Classify the question
    verdict = guardrail_chain.invoke({
        "question": question
    }).strip().upper()

    # Reject out-of-scope questions
    if verdict == "OUT_OF_SCOPE":
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": []
        }

    # Handle unexpected classifier responses
    if verdict != "IN_SCOPE":
        return {
            "answer": (
                "I couldn't classify your question. "
                "Please try rephrasing it."
            ),
            "sources": []
        }

    # Run RAG for in-scope questions
    return rag_chain(question)


print("Zyro Dynamics RAG pipeline initialized!")
