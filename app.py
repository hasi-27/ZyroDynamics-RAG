
import streamlit as st
from pathlib import Path

from rag_pipeline import ask_bot
from config import CORPUS_DIR


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #2563eb;
            font-size: 36px;
            font-weight: bold;
        }

        .subtitle {
            text-align: center;
            color: gray;
            font-size: 15px;
        }

        .feature-text {
            font-size: 16px;
            margin-bottom: 15px;
        }

        .source-heading {
            font-weight: bold;
            font-size: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# FIND ORIGINAL SOURCE PDF
# ==========================================

def find_source_pdf(source_path):
    """
    Locate the original PDF inside the corpus directory.

    Supports both Windows and Linux file paths.
    """

    # Extract filename regardless of path separator
    filename = str(source_path).replace("\\", "/").split("/")[-1]

    # Check that the corpus directory exists
    if not CORPUS_DIR.exists():
        return None

    # Search recursively for the matching PDF
    for pdf_path in CORPUS_DIR.rglob("*"):
        if (
            pdf_path.is_file()
            and pdf_path.suffix.lower() == ".pdf"
            and pdf_path.name.lower() == filename.lower()
        ):
            return pdf_path

    return None


# ==========================================
# DISPLAY SOURCE DOCUMENTS
# ==========================================

def display_sources(sources, message_index):

    if not sources:
        return

    with st.expander("📚 View source documents", expanded=False):

        for source_index, doc in enumerate(sources):

            metadata = doc.metadata

            source_path = metadata.get("source", "Unknown document")

            filename = str(source_path).replace("\\", "/").split("/")[-1]

            # PyPDFLoader page numbers are zero-indexed
            page_number = metadata.get("page")

            if page_number is not None:
                page_number = int(page_number) + 1

            st.markdown(
                f"**📄 {filename}**"
            )

            if page_number is not None:
                st.caption(f"Relevant page: {page_number}")

            # Locate the original PDF
            pdf_file = find_source_pdf(source_path)

            if pdf_file is not None:

                try:
                    with open(pdf_file, "rb") as file:
                        pdf_bytes = file.read()

                    st.download_button(
                        label="⬇️ Download original PDF",
                        data=pdf_bytes,
                        file_name=pdf_file.name,
                        mime="application/pdf",
                        key=f"download_{message_index}_{source_index}"
                    )

                except OSError:
                    st.warning("Unable to read the original PDF.")

            else:
                st.warning("Original PDF could not be located.")

            if source_index < len(sources) - 1:
                st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🏢 Zyro Dynamics")

    st.caption("HR Assistant Dashboard")

    st.divider()

    st.markdown("### About")

    st.write(
        """
        An intelligent HR assistant powered by
        Retrieval-Augmented Generation (RAG).

        It retrieves relevant information from
        company policy documents and generates
        answers using an LLM.
        """
    )

    st.divider()

    st.markdown("### Features")

    st.markdown(
        """
        <div class="feature-text">📚 HR document retrieval</div>
        <div class="feature-text">🧠 AI-powered responses</div>
        <div class="feature-text">🛡️ HR scope guardrail</div>
        <div class="feature-text">📄 Source PDF references</div>
        <div class="feature-text">🔍 Original document access</div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # Count available PDF documents
    if CORPUS_DIR.exists():
        pdf_count = sum(
            1 for file in CORPUS_DIR.rglob("*")
            if file.is_file() and file.suffix.lower() == ".pdf"
        )
    else:
        pdf_count = 0

    st.caption(f"Available PDF documents: {pdf_count}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

st.markdown(
    '<div class="main-title">🤖 Zyro Dynamics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Intelligent HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# DISPLAY PREVIOUS CHAT MESSAGES
# ==========================================

for message_index, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display sources for assistant messages
        if message["role"] == "assistant":

            display_sources(
                message.get("sources", []),
                message_index
            )


# ==========================================
# USER INPUT AND RESPONSE GENERATION
# ==========================================

question = st.chat_input("Ask your HR question here...")

if question:

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching HR documents..."):

            try:

                result = ask_bot(question)

                answer = result.get(
                    "answer",
                    "Sorry, I could not generate an answer."
                )

                sources = result.get("sources", [])

            except Exception as error:

                answer = (
                    "Something went wrong while processing "
                    "your question. Please try again."
                )

                sources = []

                st.error(f"Application error: {error}")

        st.markdown(answer)

        display_sources(
            sources,
            len(st.session_state.messages)
        )

    # Save assistant response and sources
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
