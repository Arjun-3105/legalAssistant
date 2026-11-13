import streamlit as st
import os
import sys
import tempfile

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore
from agent import LegalAgent
from ragPipeline import HybridLegalSummarizer
from api_manager import render_api_sidebar, get_api_status
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
st.write(
    "A precision legal analysis engine powered by **Hybrid Semantic + Lexical Search**, "
    "**RoBERTa embeddings**, and **Gemini task-decomposing agents**."
)

# ─── Sidebar: API Status & Corpus Stats ───────────────────────────────────────
render_api_sidebar()

with st.sidebar:
    st.markdown("---")
    st.subheader("📚 Document Library")
    stats = store.get_stats()
    st.metric("Documents Indexed", stats["total_documents"])
    st.metric("Total Chunks", stats["total_chunks"])

    if stats["document_names"]:
        with st.expander("📄 Indexed Documents", expanded=False):
            for name in stats["document_names"]:
                st.text(f"• {name}")

# Check API status banner
api_status = get_api_status()
if api_status["is_exhausted"]:
    st.error(
        "🚨 **Backend Gemini API Quota Exhausted (Out of Credits)!**\n\n"
        "The default backend Gemini API has depleted its usage quota. "
        "The system will continue providing hybrid retrieval evidence, but to enable full AI reasoning and synthesis, "
        "**please enter your personal Gemini API key in the left sidebar**."
    )
elif not api_status["configured"]:
    st.warning(
        "⚠️ **Gemini API Key Required**\n\n"
        "Please provide a Gemini API key in the sidebar to enable full AI reasoning, summarization, and query decomposition."
    )

st.divider()

# ─── Mode Selection ──────────────────────────────────────────────────────────

mode = st.radio("Select Operational Mode:", ["🤖 RAG Query (Agent)", "📄 Summarizer"], horizontal=True)

# ─── File Upload & Document Management ───────────────────────────────────────

st.subheader("📎 Document Input")
uploaded_pdf = st.file_uploader(
    "Upload a Legal PDF Document",
    type=["pdf"],
    help="Upload contracts, court opinions, NDAs, or agreements to query or summarize"
)

is_uploaded_indexed = False
if uploaded_pdf:
    stats = store.get_stats()
    is_uploaded_indexed = uploaded_pdf.name in stats["document_names"]
    
    col_status, col_btn = st.columns([3, 1])
    with col_status:
        if is_uploaded_indexed:
            st.success(f"✅ **{uploaded_pdf.name}** is already indexed and searchable in the corpus.")
        else:
            st.info(f"📄 **{uploaded_pdf.name}** uploaded. (Will auto-ingest upon running query/summary).")
    
    with col_btn:
        if not is_uploaded_indexed:
            if st.button("📥 Ingest Now", use_container_width=True):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                    result = store.ingest_pdf(temp_path)
                if result["status"] == "success":
                    st.success(f"✅ Indexed {result['pages']} pages ({result['chunks_added']} chunks)!")
                    st.rerun()

st.divider()

# Helper for auto-ingestion
def ensure_uploaded_file_indexed():
    if uploaded_pdf and uploaded_pdf.name not in store.get_stats()["document_names"]:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        with st.spinner(f"📥 Ingesting {uploaded_pdf.name} into vector store..."):
            result = store.ingest_pdf(temp_path)
        return result
    return None

# ─── RAG Query Mode ─────────────────────────────────────────────────────────

if mode == "🤖 RAG Query (Agent)":
    st.subheader("🤖 Agent-Powered Legal Query")
    st.write(
        "Ask questions across your entire document corpus or a specific contract. "
        "The agent decomposes questions into legal concepts, retrieves clauses using **Hybrid Search (Lexical + Semantic)**, "
        "and synthesizes citation-backed answers."
    )

    # Search Scope Configuration
    stats = store.get_stats()
    scope_options = []
    if uploaded_pdf:
        scope_options.append(f"📄 Current Document Only ({uploaded_pdf.name})")
    scope_options.append(f"🌐 Entire Corpus ({stats['total_documents']} documents)")

    selected_scope = st.radio("Search Scope:", scope_options, horizontal=True)
    filter_file = uploaded_pdf.name if (uploaded_pdf and selected_scope.startswith("📄")) else None

    query_text = st.text_area(
        "Enter your legal question:",
        height=95,
        placeholder="e.g., What are the indemnification provisions, liability caps, and notice periods for termination?"
    )

    col_action, _ = st.columns([2, 4])
    with col_action:
        run_query = st.button("🚀 Run Legal Query", use_container_width=True, type="primary")

    if run_query:
        if not query_text.strip():
            st.error("Please enter a legal query.")
        else:
            # Auto-ingest if uploaded document is selected
            if uploaded_pdf:
                ensure_uploaded_file_indexed()
                stats = store.get_stats()

            if stats["total_documents"] == 0:
                st.error("No documents found in corpus. Please upload and ingest a PDF document first.")
            else:
                with st.spinner("🔄 Agent: Expanding query, retrieving legal clauses, and analyzing evidence..."):
                    result = agent.query(query_text, filter_file=filter_file)

                # ─── Answer Display ──────────────────────────
                st.markdown("### 📌 Legal Analysis & Answer")
                st.markdown(result["answer"])

                st.markdown("---")

                # ─── Metrics Bar ─────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval Confidence", f"{result['confidence']}%")
                col2.metric("Relevant Chunks", result["chunks_retrieved"])
                col3.metric("Documents Cited", result["documents_referenced"])
                col4.metric("Query Category", result["query_type"].replace('_', ' ').title())

                # ─── Sources ─────────────────────────────────
                if result["sources"]:
                    st.subheader("📚 Verified Citations")
                    for src in result["sources"]:
                        st.info(f"📄 {src}")

                # ─── Agent Reasoning Trace ───────────────────
                with st.expander("🧠 Agent Execution Trace (Task Decomposition)", expanded=False):
                    for step in result["steps"]:
                        st.markdown(f"**Step: `{step['step'].upper()}`** — {step['detail']}")
                        if step["result"]:
                            st.caption(f"↳ {step['result']}")
                        st.markdown("---")

# ─── Summarizer Mode ────────────────────────────────────────────────────────

elif mode == "📄 Summarizer":
    st.subheader("📄 Legal Document Summarization")
    st.write(
        "Condense complex legal agreements into structured summaries using "
        "RoBERTa centroid-based extractive ranking and Gemini LLM refinement."
    )

    col_len, _ = st.columns([2, 2])
    with col_len:
        word_limit = st.slider("Target Summary Length (words):", min_value=100, max_value=1500, value=300, step=50)

    col_run, _ = st.columns([2, 4])
    with col_run:
        run_summary = st.button("🚀 Generate Legal Summary", use_container_width=True, type="primary")

    if run_summary:
        if not uploaded_pdf:
            st.error("Please upload a PDF document above to summarize.")
        else:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            with st.spinner("Extracting text from PDF..."):
                full_text = ""
                try:
                    with pdfplumber.open(temp_path) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text()
                            if txt:
                                full_text += txt + "\n"
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

            if not full_text.strip():
                st.error("Could not extract readable text from this PDF.")
            else:
                # Preprocess
                processed = summarizer.preprocess_text(full_text)

                with st.spinner("Generating extractive draft using RoBERTa centroid ranking..."):
                    draft = summarizer.generate_extractive_draft(processed, max_words=word_limit)

                with st.spinner("Refining summary with Gemini LLM..."):
                    summary = summarizer.refine_with_llm(draft, max_words=word_limit)

                st.success("✅ Summary generated successfully!")

                st.subheader("📌 Executive Legal Summary")
                st.markdown(summary)

                # Evaluation metrics
                with st.spinner("Computing evaluation metrics..."):
                    rouge_scores, consistency, coverage = summarizer.calculate_all_scores(processed, summary)

                st.markdown("---")
                st.subheader("📊 Quality Evaluation Metrics")
                if rouge_scores:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("ROUGE-1 F1", f"{rouge_scores.get('rouge1', 0):.4f}")
                    c2.metric("ROUGE-2 F1", f"{rouge_scores.get('rouge2', 0):.4f}")
                    c3.metric("ROUGE-L F1", f"{rouge_scores.get('rougeL', 0):.4f}")

                c4, c5 = st.columns(2)
                c4.metric("Semantic Consistency (Cosine Sim.)", f"{consistency:.4f}")
                c5.metric("Legal Keyword Coverage", f"{coverage:.1f}%")
