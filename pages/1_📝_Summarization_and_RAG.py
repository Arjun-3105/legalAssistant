import streamlit as st
import os
import sys
import tempfile

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore
from agent import LegalAgent
from ragPipeline import HybridLegalSummarizer
import pdfplumber

# ─── Initialize Components ───────────────────────────────────────────────────

@st.cache_resource
def load_store():
    return DocumentStore()

@st.cache_resource
def load_agent():
    store = load_store()
    return LegalAgent(store)

@st.cache_resource
def load_summarizer():
    return HybridLegalSummarizer()

store = load_store()
agent = load_agent()
summarizer = load_summarizer()

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal AI — Summarizer & RAG", layout="wide")
st.title("📝 Legal Document Summarizer & RAG Query System")
st.write("Upload PDFs to the corpus, then query across all your legal documents or summarize individual files.")

# ─── Sidebar: Corpus Stats ───────────────────────────────────────────────────
with st.sidebar:
    st.subheader("📚 Document Library")
    stats = store.get_stats()
    st.metric("Documents Indexed", stats["total_documents"])
    st.metric("Total Chunks", stats["total_chunks"])

    if stats["document_names"]:
        with st.expander("📄 Indexed Documents"):
            for name in stats["document_names"]:
                st.text(f"• {name}")

st.divider()

# ─── Mode Selection ──────────────────────────────────────────────────────────

mode = st.radio("Select Mode", ["RAG Query (Agent)", "Summarizer"], horizontal=True)

# ─── File Upload & Ingestion ─────────────────────────────────────────────────

uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"], help="Upload to ingest into corpus and/or process")

col1, col2 = st.columns(2)

with col1:
    ingest_btn = st.button("📥 Ingest into Corpus", help="Add this document to your searchable corpus")

with col2:
    process_btn = st.button("🚀 Process Query / Summarize")

# Handle ingestion
if ingest_btn and uploaded_pdf:
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner(f"Ingesting {uploaded_pdf.name} into vector store..."):
        result = store.ingest_pdf(temp_path)

    if result["status"] == "success":
        st.success(f"✅ Ingested **{result['filename']}**: {result['pages']} pages, {result['chunks_added']} chunks indexed")
        st.rerun()
    elif result["status"] == "already_exists":
        st.info(f"ℹ️ **{result['filename']}** is already in the corpus.")
    else:
        st.error(f"❌ Failed to ingest: {result['status']}")

st.divider()

# ─── RAG Query Mode ─────────────────────────────────────────────────────────

if mode == "RAG Query (Agent)":
    st.subheader("🤖 Agent-Powered Legal Query")
    st.write("Ask questions across your entire document corpus. The agent decomposes your query into sub-tasks for accurate retrieval.")

    query_text = st.text_area("Enter your legal question", height=100,
                               placeholder="e.g., What are the indemnification clauses across all uploaded agreements?")

    if process_btn:
        if not query_text:
            st.error("Please enter a query.")
        elif stats["total_documents"] == 0:
            st.error("No documents in corpus. Please ingest at least one PDF first.")
        else:
            with st.spinner("🔄 Agent processing query..."):
                result = agent.query(query_text)

            # ─── Answer ─────────────────────────────────────
            st.subheader("📌 Answer")
            st.write(result["answer"])

            # ─── Metadata ───────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result['confidence']}%")
            col2.metric("Chunks Retrieved", result["chunks_retrieved"])
            col3.metric("Documents Referenced", result["documents_referenced"])

            # ─── Sources ────────────────────────────────────
            if result["sources"]:
                st.subheader("📚 Sources")
                for src in result["sources"]:
                    st.text(f"  {src}")

            # ─── Agent Reasoning Trace ──────────────────────
            with st.expander("🧠 Agent Reasoning Steps", expanded=False):
                for step in result["steps"]:
                    st.markdown(f"**{step['step'].upper()}**: {step['detail']}")
                    if step["result"]:
                        st.caption(f"→ {step['result']}")
                    st.divider()

# ─── Summarizer Mode ────────────────────────────────────────────────────────

elif mode == "Summarizer":
    st.subheader("📄 Document Summarization")
    st.write("Upload a PDF and generate a polished legal summary using RoBERTa extractive ranking + Gemini refinement.")

    word_limit = st.slider("Target Summary Length (words)", min_value=50, max_value=2000, value=250)

    if process_btn:
        if not uploaded_pdf:
            st.error("Please upload a PDF first.")
        else:
            # Extract text
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            with st.spinner("Extracting text from PDF..."):
                full_text = ""
                with pdfplumber.open(temp_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"

            if not full_text.strip():
                st.error("Could not extract text from the PDF.")
            else:
                # Preprocess
                processed = summarizer.preprocess_text(full_text)

                with st.spinner("Generating extractive draft with RoBERTa embeddings..."):
                    draft = summarizer.generate_extractive_draft(processed, max_words=word_limit)

                with st.spinner("Refining with Gemini LLM..."):
                    summary = summarizer.refine_with_llm(draft, max_words=word_limit)

                st.success("✅ Summary generated!")

                st.subheader("📌 Summary")
                st.write(summary)

                # Evaluation metrics
                with st.spinner("Computing evaluation metrics..."):
                    rouge_scores, consistency, coverage = summarizer.calculate_all_scores(processed, summary)

                st.subheader("📊 Evaluation Metrics")
                if rouge_scores:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("ROUGE-1", f"{rouge_scores.get('rouge1', 0):.4f}")
                    col2.metric("ROUGE-2", f"{rouge_scores.get('rouge2', 0):.4f}")
                    col3.metric("ROUGE-L", f"{rouge_scores.get('rougeL', 0):.4f}")

                col4, col5 = st.columns(2)
                col4.metric("Semantic Consistency", f"{consistency:.4f}")
                col5.metric("Legal Keyword Coverage", f"{coverage:.1f}%")
