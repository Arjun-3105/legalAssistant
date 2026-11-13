import streamlit as st
import os
import sys
import tempfile

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore
from api_manager import render_api_sidebar

# ─── Initialize ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_store():
    return DocumentStore()

store = load_store()

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal AI — Document Library", layout="wide")
render_api_sidebar()
st.title("📚 Document Library")
st.write("Manage your legal document corpus. Ingest PDFs individually or in batch, view indexed documents, and manage the vector store.")

st.divider()

# ─── Corpus Stats ────────────────────────────────────────────────────────────

stats = store.get_stats()

col1, col2 = st.columns(2)
col1.metric("📄 Total Documents", stats["total_documents"])
col2.metric("🧩 Total Chunks Indexed", stats["total_chunks"])

st.divider()

# ─── Upload Section ──────────────────────────────────────────────────────────

st.subheader("📥 Ingest Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files to add to your corpus",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to ingest into the vector store"
)

if st.button("🚀 Ingest All Uploaded Files") and uploaded_files:
    progress_bar = st.progress(0)
    results = []

    for idx, file in enumerate(uploaded_files):
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        with st.spinner(f"Ingesting {file.name}..."):
            result = store.ingest_pdf(temp_path)
            results.append(result)

        progress_bar.progress((idx + 1) / len(uploaded_files))

    # Show results
    success_count = sum(1 for r in results if r["status"] == "success")
    skip_count = sum(1 for r in results if r["status"] == "already_exists")

    if success_count > 0:
        st.success(f"✅ Successfully ingested **{success_count}** document(s)")
    if skip_count > 0:
        st.info(f"ℹ️ Skipped **{skip_count}** document(s) (already in corpus)")

    # Show detailed results
    with st.expander("📋 Ingestion Details"):
        for r in results:
            icon = "✅" if r["status"] == "success" else "⏭️" if r["status"] == "already_exists" else "❌"
            st.text(f"{icon} {r['filename']}: {r.get('chunks_added', 0)} chunks | Status: {r['status']}")

    st.rerun()

st.divider()

# ─── Document List ───────────────────────────────────────────────────────────

st.subheader("📄 Indexed Documents")

doc_list = store.get_document_list()

if doc_list:
    for doc in doc_list:
        col1, col2, col3 = st.columns([4, 1, 1])
        col1.text(f"📄 {doc['filename']}")
        col2.text(f"{doc['chunks']} chunks")

        if col3.button("🗑️ Remove", key=f"del_{doc['filename']}"):
            result = store.delete_document(doc['filename'])
            if result["status"] == "deleted":
                st.success(f"Removed {doc['filename']} ({result['chunks_removed']} chunks)")
                st.rerun()
            else:
                st.error(f"Could not remove: {result['status']}")
else:
    st.info("No documents ingested yet. Upload PDFs above to get started.")
    st.markdown("""
    **Getting started:**
    1. Upload one or more legal PDF documents above
    2. Click **Ingest All** to add them to the vector store
    3. Head to the **Summarization & RAG** page to query across your corpus
    """)
