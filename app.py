
import streamlit as st
from pathlib import Path
from rag_pipeline import ask_bot
from config import CORPUS_DIR

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Zyro Dynamics HR Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #3B82F6;
    font-size: 34px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SOURCE PDF HELPER
# ==================================================

def find_source_pdf(source_path):
    from pathlib import Path

    filename = Path(source_path).name

    for pdf_path in CORPUS_DIR.rglob("*.pdf"):
        if pdf_path.name == filename:
            return pdf_path

    return None
    """
    Locate the original PDF inside the corpus folder.
    Works even if the stored metadata contains an
    older absolute Windows path.
    """

    filename = Path(source_path).name

    matches = list(CORPUS_DIR.rglob(filename))

    if matches:
        return matches[0]

    return None


def display_sources(sources, message_index):
    """
    Display source filenames, page numbers,
    and downloadable original PDF files.
    """

    if not sources:
        return

    with st.expander("📚 View source documents"):

        for source_index, source in enumerate(sources):

            source_path = source.get("file", "")
            page = source.get("page", "Unknown")

            pdf_path = find_source_pdf(source_path)

            st.markdown(
                f"**📄 {Path(source_path).name}**"
            )

            st.caption(f"Relevant page: {page}")

            if pdf_path and pdf_path.is_file():

                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

                st.download_button(
                    label="⬇️ Open / Download source PDF",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"download_{message_index}_{source_index}"
                )

            else:
                st.warning(
                    "Original PDF could not be located."
                )

            st.divider()


# ==================================================
# 4. HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🤖 Zyro Dynamics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Intelligent HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "Ask questions about Zyro Dynamics HR policies, "
    "leave, travel, reimbursements, benefits, and "
    "company guidelines."
)


# ==================================================
# 5. SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# 6. DISPLAY CHAT HISTORY
# ==================================================

for message_index, message in enumerate(
    st.session_state.messages
):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_sources(
                message.get("sources", []),
                message_index
            )


# ==================================================
# 7. CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask your HR question here..."
)

if question:

    # Display user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            try:

                result = ask_bot(question)

                answer = result["answer"]

                # Extract source metadata
                sources = []

                for doc in result["sources"]:

                    sources.append({
                        "file": doc.metadata.get(
                            "source", "Unknown"
                        ),
                        "page": doc.metadata.get(
                            "page", "Unknown"
                        )
                    })

                # Display answer
                st.markdown(answer)

                # Display source documents
                display_sources(
                    sources,
                    len(st.session_state.messages)
                )

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# ==================================================
# 8. SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Zyro Dynamics")

    st.caption("HR Assistant Dashboard")

    st.divider()

    st.subheader("About")

    st.write(
        "An intelligent HR assistant powered by "
        "Retrieval-Augmented Generation (RAG). "
        "It retrieves information from company "
        "policy documents and generates answers "
        "using an LLM."
    )

    st.divider()

    st.subheader("Features")

    st.write("📚 HR document retrieval")
    st.write("🧠 AI-powered responses")
    st.write("🛡️ HR scope guardrail")
    st.write("📄 Source PDF references")
    st.write("🔍 Original document access")

    st.divider()

    st.caption(
        f"Available PDF documents: "
        f"{len(list(CORPUS_DIR.rglob('*.pdf')))}"
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()
