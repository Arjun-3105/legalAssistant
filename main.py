import streamlit as st
from api_manager import render_api_sidebar

st.set_page_config(
    page_title="RoBERTa Legal AI System",
    layout="wide"
)

# Render API credentials manager in sidebar
render_api_sidebar()

# ------------------------------
# HEADER
# ------------------------------
st.title("⚖️ RoBERTa Legal AI System")
st.write("A fine-tuned **RoBERTa**-powered platform for **legal document analysis**, **RAG-based retrieval with agent task decomposition**, and **contract risk assessment** — capable of reasoning over **300+ legal documents**.")

st.markdown("---")

# ------------------------------
# MODEL CAPABILITIES
# ------------------------------
st.subheader("🧠 System Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 1️⃣ RAG + Agent Task Decomposition
    - Queries decomposed: **Classify → Retrieve → Analyze → Synthesize**
    - RoBERTa embeddings for semantic retrieval
    - Gemini LLM synthesizes citation-backed answers
    - Transparent agent reasoning trace
    
    ---
    
    ### 2️⃣ Multi-Document Pipeline
    - **ChromaDB** persistent vector store
    - Batch ingest entire directories
    - Search across **300+ documents** simultaneously
    - Document corpus management
    """)

with col2:
    st.markdown("""
    ### 3️⃣ Clause Classification
    - T5 trained on **CUAD** (21K+ clauses)
    - **42 legal categories** with 88.1% accuracy
    - Risk tier classification
    - Auto-generated PDF reports
    
    ---
    
    ### 4️⃣ Risk Assessment & Evaluation
    - Clause-level risk detection
    - Cross-reference against corpus
    - RAG vs no-RAG accuracy benchmarks
    - **~40% improvement** with retrieval augmentation
    """)

st.markdown("---")

# ------------------------------
# ARCHITECTURE
# ------------------------------
st.subheader("🏗️ Architecture")

st.code("""
┌──────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                        │
│  ┌──────────┐  ┌──────────────┐  ┌────────┐  ┌──────────────┐  │
│  │ Document  │  │ RAG Query +  │  │  Risk  │  │  Analytics   │  │
│  │ Library   │  │ Summarizer   │  │ Assess │  │  Dashboard   │  │
│  └─────┬────┘  └──────┬───────┘  └───┬────┘  └──────┬───────┘  │
└────────┼──────────────┼──────────────┼───────────────┼──────────┘
         │              │              │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐    ┌─────▼────┐
    │ Vector  │    │  Legal  │    │   T5    │    │ Eval     │
    │  Store  │◄───│  Agent  │    │ Clause  │    │ Engine   │
    │(ChromaDB)│   │(Decomp.)│    │Classify │    │(RAG/noRAG)│
    └────┬────┘    └────┬────┘    └─────────┘    └──────────┘
         │              │
    ┌────▼──────────────▼────┐    ┌──────────────────────┐
    │   RoBERTa Embeddings   │    │   Gemini LLM (2.0)   │
    │   (Fine-tuned / Base)  │    │   Answer Synthesis    │
    └────────────────────────┘    └──────────────────────┘
""", language=None)

st.markdown("---")

# ------------------------------
# TECH STACK
# ------------------------------
st.subheader("🧰 Tech Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Models**
    - RoBERTa (embeddings)
    - T5 (clause classification)
    - Gemini 2.0 Flash (synthesis)
    """)

with col2:
    st.markdown("""
    **Infrastructure**
    - ChromaDB (vector store)
    - Streamlit (frontend)
    - Docker (deployment)
    """)

with col3:
    st.markdown("""
    **Evaluation**
    - ROUGE scoring
    - Semantic similarity
    - Keyword coverage
    """)

st.markdown("---")

st.info("👈 **Select a tool from the sidebar** to get started.")
st.sidebar.success("Select a page above.")
