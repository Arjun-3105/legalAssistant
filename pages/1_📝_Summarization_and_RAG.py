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

# ─── Page Config & Header ────────────────────────────────────────────────────

st.set_page_config(page_title="Legal AI — Summarizer & RAG", layout="wide")
st.title("📝 Legal Document Summarizer & RAG Query System")
st.write(
    "Precision legal analysis with **Hybrid Semantic + Keyword Search**, "
    "**RoBERTa embeddings**, and **Gemini agent reasoning**."
)

# ─── Sidebar: API + Corpus ────────────────────────────────────────────────────
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

# ─── API Status Banner ────────────────────────────────────────────────────────
api_status = get_api_status()
if api_status["is_exhausted"]:
    st.error(
        "🚨 **Backend Gemini API Quota Exhausted!** "
        "The system still provides hybrid retrieval evidence. "
        "**Add your personal Gemini API key in the sidebar** for full AI synthesis."
    )
elif not api_status["configured"]:
    st.warning(
        "⚠️ **Gemini API Key Required** — Add your key in the sidebar to enable AI reasoning and summaries."
    )

st.divider()

# ─── Mode Selection ──────────────────────────────────────────────────────────
mode = st.radio(
    "Select Operational Mode:",
    ["🤖 RAG Query (Agent)", "📄 Summarizer"],
    horizontal=True
)

# ─── File Upload ──────────────────────────────────────────────────────────────
st.subheader("📎 Document Input")
uploaded_pdf = st.file_uploader(
    "Upload a Legal PDF Document",
    type=["pdf"],
    help="Contracts, NDAs, court opinions, agreements — text-based PDFs work best"
)

is_uploaded_indexed = False
if uploaded_pdf:
    stats = store.get_stats()
    is_uploaded_indexed = uploaded_pdf.name in stats["document_names"]

    col_status, col_btn = st.columns([3, 1])
    with col_status:
        if is_uploaded_indexed:
            st.success(f"✅ **{uploaded_pdf.name}** is already indexed in the corpus.")
        else:
            st.info(f"📄 **{uploaded_pdf.name}** uploaded — not yet indexed.")

    with col_btn:
        if not is_uploaded_indexed:
            if st.button("📥 Ingest Now", use_container_width=True):
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())

                progress_bar = st.progress(0, text="Preparing...")
                status_text = st.empty()

                def ingest_progress(current, total, message):
                    if total > 0:
                        pct = min(int((current / total) * 100), 99)
                        progress_bar.progress(pct, text=message)
                    status_text.caption(message)

                result = store.ingest_pdf(temp_path, progress_callback=ingest_progress)
                progress_bar.progress(100, text="Done!")

                if result["status"] in ("success", "updated"):
                    st.success(
                        f"✅ Indexed **{result['pages']}** pages → **{result['chunks_added']}** chunks"
                        + (" (re-indexed)" if result["status"] == "updated" else "")
                    )
                    st.rerun()
                elif result["status"] == "no_text":
                    st.error(
                        "❌ Could not extract text from this PDF. "
                        "It may be a scanned/image-based PDF. Please use a text-based PDF."
                    )
                else:
                    st.error(f"❌ Ingestion failed: {result['status']}")

st.divider()


# ─── Helper: auto-ingest uploaded file ───────────────────────────────────────
def ensure_file_indexed(show_progress: bool = True) -> dict:
    """
    Ingest the uploaded PDF if it's not already in the corpus.
    Returns the ingestion result dict, or None if already indexed.
    Shows errors directly in the UI.
    """
    if not uploaded_pdf:
        return None
    if uploaded_pdf.name in store.get_stats()["document_names"]:
        return None  # Already done

    temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())

    with st.spinner(f"📥 Indexing {uploaded_pdf.name}..."):
        result = store.ingest_pdf(temp_path)

    if result["status"] == "no_text":
        st.error(
            f"❌ **{uploaded_pdf.name}** could not be parsed — it appears to be a scanned "
            "or image-based PDF. Please upload a text-based PDF."
        )
        st.stop()
    elif result["status"] not in ("success", "updated", "already_exists"):
        st.error(f"❌ Ingestion failed with status: `{result['status']}`")
        st.stop()

    return result


# ─── RAG Query Mode ───────────────────────────────────────────────────────────
if mode == "🤖 RAG Query (Agent)":
    st.subheader("🤖 Agent-Powered Legal Query")
    st.write(
        "Ask questions across your corpus or a specific contract. "
        "The agent classifies your intent, expands legal terminology, "
        "retrieves clauses via **Hybrid Search**, and synthesizes a cited answer."
    )

    # Search scope
    stats = store.get_stats()
    scope_options = []
    if uploaded_pdf:
        scope_options.append(f"📄 This Document Only ({uploaded_pdf.name})")
    scope_options.append(f"🌐 Entire Corpus ({stats['total_documents']} documents)")

    selected_scope = st.radio("Search Scope:", scope_options, horizontal=True)
    filter_file = (
        uploaded_pdf.name
        if (uploaded_pdf and selected_scope.startswith("📄"))
        else None
    )

    query_text = st.text_area(
        "Your legal question:",
        height=95,
        max_chars=2000,
        placeholder="e.g., What are the indemnification clauses, liability caps, and notice periods?"
    )
    if query_text and len(query_text) > 1800:
        st.caption(f"⚠️ Query length: {len(query_text)}/2000 characters")

    col_run, _ = st.columns([2, 4])
    with col_run:
        run_query = st.button("🚀 Run Legal Query", use_container_width=True, type="primary")

    if run_query:
        if not query_text.strip():
            st.error("Please enter a legal question.")
        else:
            # Auto-ingest if needed (stops with error if PDF can't be parsed)
            if uploaded_pdf:
                ensure_file_indexed()
                stats = store.get_stats()

            if stats["total_documents"] == 0:
                st.error(
                    "No documents found in corpus. "
                    "Please upload and ingest a PDF document first."
                )
            else:
                with st.spinner("🔄 Agent: classifying → retrieving → analyzing → synthesizing..."):
                    # agent.query() is now wrapped in try/except internally
                    result = agent.query(query_text.strip(), filter_file=filter_file)

                # ─── Answer ──────────────────────────────────────────────
                st.markdown("### 📌 Legal Analysis & Answer")
                st.markdown(result["answer"])
                st.markdown("---")

                # ─── Metrics ──────────────────────────────────────────────
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Match Score", f"{result.get('match_score', result.get('confidence', 0))}%",
                            help="Heuristic relevance score (not a calibrated probability)")
                col2.metric("Clauses Retrieved", result["chunks_retrieved"])
                col3.metric("Docs Referenced", result["documents_referenced"])
                col4.metric("Query Type", result["query_type"].replace("_", " ").title())

                # ─── Sources ──────────────────────────────────────────────
                if result["sources"]:
                    with st.expander(f"📚 Verified Citations ({len(result['sources'])})", expanded=True):
                        for src in result["sources"]:
                            st.info(f"📄 {src}")

                # ─── Agent Execution Trace ─────────────────────────────────
                with st.expander("🧠 Agent Execution Trace", expanded=False):
                    for step in result["steps"]:
                        icon = {
                            "start": "🏁", "classify": "🏷️", "query_expansion": "🔍",
                            "retrieve": "📂", "analyze": "🔬", "synthesize": "✍️",
                            "complete": "✅", "error": "❌"
                        }.get(step["step"], "•")
                        st.markdown(f"{icon} **`{step['step'].upper()}`** — {step['detail']}")
                        if step.get("result"):
                            st.caption(f"↳ {step['result']}")
                        st.markdown("---")


# ─── Summarizer Mode ──────────────────────────────────────────────────────────
elif mode == "📄 Summarizer":
    st.subheader("📄 Legal Document Summarization")
    st.write(
        "Generate a structured executive summary from a legal PDF using "
        "RoBERTa ranked extraction + Gemini refinement. "
        "Optionally provide a **focus query** to bias extraction toward specific provisions."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        word_limit = st.slider(
            "Target Summary Length (words):",
            min_value=100, max_value=1500, value=350, step=50
        )
    with col_b:
        focus_query = st.text_input(
            "Focus Query (optional):",
            placeholder="e.g., indemnification and liability clauses",
            help="Biases sentence selection toward this topic. Leave blank for a general summary."
        )

    col_run, _ = st.columns([2, 4])
    with col_run:
        run_summary = st.button("🚀 Generate Summary", use_container_width=True, type="primary")

    if run_summary:
        if not uploaded_pdf:
            st.error("Please upload a PDF document above.")
        else:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded_pdf.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            with st.spinner("📄 Extracting and cleaning PDF text..."):
                full_text = ""
                try:
                    with pdfplumber.open(temp_path) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text(x_tolerance=3, y_tolerance=3)
                            if txt:
                                full_text += txt + "\n"
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                    st.stop()

            if not full_text.strip():
                st.error(
                    "Could not extract readable text from this PDF. "
                    "It may be scanned or image-based."
                )
                st.stop()

            # Preprocess preserving clause structure
            with st.spinner("🔧 Preprocessing and cleaning text..."):
                processed = summarizer.preprocess_text(full_text)

            # Choose extraction method based on whether a focus query was given
            if focus_query.strip():
                with st.spinner(f"🔍 Extracting clauses relevant to: '{focus_query}'..."):
                    draft = summarizer.generate_rag_draft(
                        processed, user_query=focus_query.strip(), max_words=word_limit
                    )
                extraction_method = f"Query-Focused RAG Extraction (focus: '{focus_query}')"
            else:
                with st.spinner("📊 Ranking sentences with RoBERTa centroid method..."):
                    draft = summarizer.generate_extractive_draft(processed, max_words=word_limit)
                extraction_method = "RoBERTa Centroid Extractive Ranking"

            if not draft.strip():
                st.error("Could not generate an extractive draft from this document.")
                st.stop()

            with st.spinner("✍️ Refining summary with Gemini LLM..."):
                summary = summarizer.refine_with_llm(
                    draft,
                    max_words=word_limit,
                    user_query=focus_query.strip() if focus_query.strip() else None
                )

            st.success(f"✅ Summary generated! (Method: {extraction_method})")

            st.subheader("📌 Executive Legal Summary")
            st.markdown(summary)

            # Evaluation metrics
            with st.spinner("📊 Computing quality metrics..."):
                try:
                    rouge_scores, consistency, coverage = summarizer.calculate_all_scores(
                        processed, summary
                    )
                except Exception:
                    rouge_scores, consistency, coverage = {}, 0.0, 0.0

            st.markdown("---")
            st.subheader("📊 Quality Evaluation Metrics")
            st.caption(
                "These metrics compare the AI summary against the full extracted text. "
                "Higher ROUGE = more content overlap; higher Semantic Consistency = same meaning preserved."
            )

            if rouge_scores:
                c1, c2, c3 = st.columns(3)
                c1.metric("ROUGE-1 F1", f"{rouge_scores.get('rouge1', 0):.4f}")
                c2.metric("ROUGE-2 F1", f"{rouge_scores.get('rouge2', 0):.4f}")
                c3.metric("ROUGE-L F1", f"{rouge_scores.get('rougeL', 0):.4f}")

            c4, c5 = st.columns(2)
            c4.metric(
                "Semantic Consistency",
                f"{consistency:.4f}",
                help="Cosine similarity between original and summary embeddings"
            )
            c5.metric(
                "Legal Keyword Coverage",
                f"{coverage:.1f}%",
                help="% of legal keywords from the original retained in the summary"
            )
