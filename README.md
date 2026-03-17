# ⚖️ RoBERTa Legal AI System

> An AI-powered legal document analysis platform combining **fine-tuned RoBERTa** embeddings, **Retrieval-Augmented Generation** with agent-style task decomposition, and a **scalable multi-document pipeline** capable of reasoning over **300+ legal documents**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![RoBERTa](https://img.shields.io/badge/Model-RoBERTa-orange?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 🏗️ Architecture

```
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
```

---

## ✨ Key Features

### 🤖 Agent-Style Task Decomposition
Complex legal queries are automatically decomposed into sub-tasks:
1. **Classify** — Determine query type (factual, comparison, risk analysis, summarization, reasoning)
2. **Retrieve** — Semantic search across the document corpus via ChromaDB
3. **Analyze** — Cross-reference and rank evidence across multiple documents
4. **Synthesize** — Gemini LLM generates a grounded, citation-backed answer

Every step is logged transparently in an agent reasoning trace.

### 📚 Scalable Multi-Document Pipeline
- **ChromaDB** persistent vector store for document indexing
- Batch ingest entire directories of legal PDFs
- Semantic search across **300+ documents** simultaneously
- RoBERTa-powered dense embeddings with cosine similarity retrieval

### 🧠 Fine-Tuned Models
- **RoBERTa**: Fine-tuned on legal QA datasets (SQuAD-format) for domain-specific embeddings
- **T5**: Trained on CUAD (Contract Understanding Atticus Dataset) for clause classification across 42 clause types (88.1% accuracy)

### 📊 Evaluation Framework
- Side-by-side RAG vs. no-RAG factual accuracy comparison
- ROUGE, semantic similarity, and legal keyword coverage metrics
- Demonstrates **~40% improvement** in response factual accuracy with RAG

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Gemini API key](https://ai.google.dev/) (for LLM synthesis)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/legalAssistant.git
cd legalAssistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Running the Application

```bash
# Set environment variables
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: point to fine-tuned RoBERTa checkpoint
export ROBERTA_MODEL_PATH="./roberta_legal_finetuned/"

# Launch
streamlit run main.py
```

### Fine-Tuning RoBERTa (Optional)

```bash
python sumTrain.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --do_train --do_eval \
    --train_file data/legal_qa_train.json \
    --predict_file data/legal_qa_dev.json \
    --data_dir ./data/ \
    --output_dir ./roberta_legal_finetuned/ \
    --per_gpu_train_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --max_seq_length 384 \
    --overwrite_output_dir
```

---

## 📁 Project Structure

```
legalAssistant/
├── main.py                 # Streamlit entry point & landing page
├── agent.py                # Agent task decomposition (classify → retrieve → analyze → synthesize)
├── vectorStore.py          # ChromaDB document store + RoBERTa embeddings
├── ragPipeline.py          # Hybrid summarizer (RoBERTa extractive + Gemini abstractive)
├── evaluation.py           # RAG vs no-RAG factual accuracy evaluation
├── sumTrain.py             # RoBERTa fine-tuning script (SQuAD-format QA)
├── config.py               # Centralized configuration & environment management
├── RiskLexis_T5.ipynb      # T5 clause classification training notebook (CUAD dataset)
├── pages/
│   ├── 1_📝_Summarization_and_RAG.py   # Document summarizer + agent RAG queries
│   ├── 2_⚠️_Risk_Assessment.py         # T5-based clause risk analysis + corpus cross-reference
│   ├── 3_📚_Document_Library.py         # Corpus management (ingest, view, delete)
│   ├── 4_🏷️_Clause_Classification.py   # Clause type classification (42 legal categories)
│   └── 5_📊_Analytics_Dashboard.py      # Corpus analytics & evaluation metrics
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Embeddings** | RoBERTa (fine-tuned on legal QA) |
| **Vector Store** | ChromaDB (persistent, cosine similarity) |
| **LLM Synthesis** | Google Gemini 2.0 Flash |
| **Clause Classification** | T5 (fine-tuned on CUAD, 42 categories) |
| **QA Training** | HuggingFace Transformers (SQuAD pipeline) |
| **Frontend** | Streamlit (multi-page app) |
| **Evaluation** | ROUGE, cosine similarity, keyword coverage |
| **Deployment** | Docker + Docker Compose |

---

## 📈 Model Performance

### Clause Classification (T5)
| Metric | Score |
|--------|-------|
| Accuracy | 88.14% |
| Precision | 86.13% |
| Recall | 87.32% |
| F1 Score | 86.21% |

*Trained on CUAD dataset — 21,187 clauses across 42 legal categories*

### RAG Factual Accuracy
| Metric | Without RAG | With RAG | Improvement |
|--------|-------------|----------|-------------|
| ROUGE-L | ~0.25 | ~0.42 | +68% |
| Semantic Similarity | ~0.61 | ~0.78 | +28% |
| Keyword Coverage | ~45% | ~72% | +60% |
| **Composite Score** | ~0.44 | ~0.64 | **+~40%** |

---

## 📄 License

This project is licensed under the MIT License.
