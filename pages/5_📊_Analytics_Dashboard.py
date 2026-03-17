"""
Analytics Dashboard — Corpus analytics, evaluation metrics, and system overview.
"""

import streamlit as st
import os
import sys
import json
import time

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore
from agent import LegalAgent
from evaluation import evaluate_single

@st.cache_resource
def load_store():
    return DocumentStore()

@st.cache_resource
def load_agent():
    return LegalAgent(load_store())

store = load_store()
agent = load_agent()

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal AI — Analytics", layout="wide")
st.title("📊 Analytics Dashboard")
st.write("Corpus insights, system metrics, and RAG evaluation benchmarks.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Corpus Overview
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("📚 Corpus Overview")

stats = store.get_stats()

col1, col2, col3 = st.columns(3)
col1.metric("📄 Documents Indexed", stats["total_documents"])
col2.metric("🧩 Total Chunks", stats["total_chunks"])
col3.metric("📊 Avg Chunks/Doc", 
            round(stats["total_chunks"] / stats["total_documents"], 1) if stats["total_documents"] > 0 else 0)

if stats["total_documents"] > 0:
    doc_list = store.get_document_list()
    
    # Document size distribution
    st.markdown("#### Document Breakdown")
    chart_data = {doc["filename"][:30]: doc["chunks"] for doc in doc_list}
    st.bar_chart(chart_data)

    # Document table
    with st.expander("📋 Full Document List"):
        for doc in doc_list:
            col1, col2 = st.columns([4, 1])
            col1.text(f"📄 {doc['filename']}")
            col2.text(f"{doc['chunks']} chunks")
else:
    st.info("No documents indexed yet. Head to the **Document Library** to ingest PDFs.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: RAG Evaluation Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🧪 RAG vs. No-RAG Evaluation")
st.write("Compare the factual accuracy of RAG-augmented answers against pure LLM responses.")

# Pre-built evaluation questions (legal domain)
SAMPLE_QUESTIONS = [
    {
        "question": "What are the indemnification obligations in the agreement?",
        "ground_truth": "The indemnifying party shall defend, indemnify, and hold harmless the indemnified party from and against all claims, damages, losses, costs, and expenses arising from any breach of representations, warranties, or obligations under this agreement."
    },
    {
        "question": "What are the termination conditions?",
        "ground_truth": "Either party may terminate this agreement upon thirty days written notice. The agreement shall also terminate automatically upon material breach by either party if such breach remains uncured for fifteen days after written notice."
    },
    {
        "question": "What governing law applies to this contract?",
        "ground_truth": "This agreement shall be governed by and construed in accordance with the laws of the State, without regard to its conflict of laws principles. Any disputes arising under this agreement shall be resolved in the courts of competent jurisdiction."
    }
]

eval_mode = st.radio("Evaluation Mode", ["Quick Demo (Sample Questions)", "Custom Questions"], horizontal=True)

if eval_mode == "Custom Questions":
    st.markdown("**Enter a question and its expected ground truth answer:**")
    custom_q = st.text_input("Question")
    custom_gt = st.text_area("Ground Truth Answer", height=100)
    eval_questions = [{"question": custom_q, "ground_truth": custom_gt}] if custom_q and custom_gt else []
else:
    eval_questions = SAMPLE_QUESTIONS
    with st.expander("📝 Sample Evaluation Questions"):
        for q in SAMPLE_QUESTIONS:
            st.markdown(f"**Q:** {q['question']}")
            st.caption(f"Expected: {q['ground_truth'][:100]}...")
            st.divider()

if st.button("🚀 Run Evaluation") and eval_questions:
    if stats["total_documents"] == 0:
        st.warning("⚠️ No documents in corpus. RAG results will be limited. Ingest documents first for meaningful evaluation.")

    results = []
    progress = st.progress(0)

    for idx, item in enumerate(eval_questions):
        with st.spinner(f"Evaluating question {idx + 1}/{len(eval_questions)}..."):
            # Get RAG answer
            rag_result = agent.query(item["question"])
            rag_answer = rag_result["answer"]

            # Get no-RAG answer
            no_rag_answer = agent.query_without_rag(item["question"])

            # Evaluate
            metrics = evaluate_single(item["ground_truth"], rag_answer, no_rag_answer)
            metrics["question"] = item["question"]
            metrics["rag_answer"] = rag_answer
            metrics["no_rag_answer"] = no_rag_answer
            results.append(metrics)

        progress.progress((idx + 1) / len(eval_questions))

    # ─── Results Display ─────────────────────────────────
    st.subheader("📈 Evaluation Results")

    # Aggregate metrics
    avg_rag = sum(r["rag"]["composite_score"] for r in results) / len(results)
    avg_no_rag = sum(r["no_rag"]["composite_score"] for r in results) / len(results)
    avg_improvement = sum(r["improvement_pct"] for r in results) / len(results)

    col1, col2, col3 = st.columns(3)
    col1.metric("RAG Composite Score", f"{avg_rag:.3f}")
    col2.metric("No-RAG Composite Score", f"{avg_no_rag:.3f}")
    col3.metric("Avg Improvement", f"{avg_improvement:+.1f}%",
                delta=f"{avg_improvement:+.1f}%",
                delta_color="normal")

    # Per-question breakdown
    st.markdown("#### Per-Question Breakdown")

    for idx, r in enumerate(results):
        with st.expander(f"Q{idx + 1}: {r['question'][:80]}... — Improvement: {r['improvement_pct']:+.1f}%"):
            tab1, tab2 = st.tabs(["📊 Metrics", "📝 Answers"])

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**With RAG:**")
                    st.text(f"ROUGE-L:    {r['rag']['rouge']['rougeL']:.4f}")
                    st.text(f"Similarity: {r['rag']['semantic_similarity']:.4f}")
                    st.text(f"Coverage:   {r['rag']['keyword_coverage']:.1f}%")
                    st.text(f"Composite:  {r['rag']['composite_score']:.4f}")

                with col2:
                    st.markdown("**Without RAG:**")
                    st.text(f"ROUGE-L:    {r['no_rag']['rouge']['rougeL']:.4f}")
                    st.text(f"Similarity: {r['no_rag']['semantic_similarity']:.4f}")
                    st.text(f"Coverage:   {r['no_rag']['keyword_coverage']:.1f}%")
                    st.text(f"Composite:  {r['no_rag']['composite_score']:.4f}")

            with tab2:
                st.markdown("**RAG Answer:**")
                st.write(r.get("rag_answer", "N/A"))
                st.divider()
                st.markdown("**No-RAG Answer:**")
                st.write(r.get("no_rag_answer", "N/A"))

    # Summary chart
    chart_data = {
        "Metric": ["ROUGE-L", "Semantic Sim.", "Keyword Cov.", "Composite"],
        "With RAG": [
            sum(r["rag"]["rouge"]["rougeL"] for r in results) / len(results),
            sum(r["rag"]["semantic_similarity"] for r in results) / len(results),
            sum(r["rag"]["keyword_coverage"] for r in results) / len(results) / 100,
            avg_rag
        ],
        "Without RAG": [
            sum(r["no_rag"]["rouge"]["rougeL"] for r in results) / len(results),
            sum(r["no_rag"]["semantic_similarity"] for r in results) / len(results),
            sum(r["no_rag"]["keyword_coverage"] for r in results) / len(results) / 100,
            avg_no_rag
        ]
    }

    import pandas as pd
    df = pd.DataFrame(chart_data).set_index("Metric")
    st.bar_chart(df)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: System Information
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("⚙️ System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Models Loaded:**")
    roberta_path = os.getenv("ROBERTA_MODEL_PATH", "roberta-base")
    st.text(f"• RoBERTa: {roberta_path}")
    st.text(f"• Gemini: models/gemini-2.0-flash")
    st.text(f"• T5 Classifier: valhalla/t5-base-qg-hl")

with col2:
    st.markdown("**Storage:**")
    st.text(f"• ChromaDB: ./chroma_legal_store")
    st.text(f"• Documents: {stats['total_documents']}")
    st.text(f"• Vectors: {stats['total_chunks']}")

st.markdown("---")
st.caption("RoBERTa Legal AI System — Built with Transformers, ChromaDB, and Gemini")
