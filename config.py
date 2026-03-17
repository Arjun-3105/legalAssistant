"""
config.py — Centralized configuration for the Legal AI system.

Loads environment variables and provides defaults for all configurable parameters.
"""

import os
from pathlib import Path

# ─── Base Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─── Model Configuration ────────────────────────────────────────────────────
ROBERTA_MODEL_PATH = os.getenv("ROBERTA_MODEL_PATH", "roberta-base")
T5_RISK_MODEL = "valhalla/t5-base-qg-hl"
GEMINI_MODEL = "models/gemini-2.0-flash"

# ─── Vector Store ────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_legal_store"))
CHROMA_COLLECTION_NAME = "legal_documents"
EMBEDDING_DIMENSION = 768  # RoBERTa base hidden size

# ─── Chunking Parameters ────────────────────────────────────────────────────
CHUNK_SIZE = 200           # words per chunk
CHUNK_OVERLAP = 50         # word overlap between chunks
MAX_SEQ_LENGTH = 512       # RoBERTa max token length

# ─── Agent Configuration ────────────────────────────────────────────────────
DEFAULT_RETRIEVAL_TOP_K = 10
RETRIEVAL_K_BY_QUERY_TYPE = {
    "factual_lookup": 5,
    "comparison": 12,
    "risk_analysis": 10,
    "summarization": 15,
    "reasoning": 10,
}
MAX_CONTEXT_BLOCKS = 10    # Max chunks sent to LLM for synthesis

# ─── Summarization ──────────────────────────────────────────────────────────
DEFAULT_SUMMARY_WORDS = 250
MAX_SUMMARY_WORDS = 2000
DRAFT_EXPANSION_FACTOR = 1.5  # Extractive draft is 1.5x target length

# ─── Risk Assessment ────────────────────────────────────────────────────────
MAX_CLAUSES = 20
DEFAULT_CLAUSES = 5

# ─── Evaluation ─────────────────────────────────────────────────────────────
EVAL_WEIGHTS = {
    "rouge": 0.3,
    "semantic_similarity": 0.4,
    "keyword_coverage": 0.3,
}

# ─── Clause Classification Categories ───────────────────────────────────────
CLAUSE_CATEGORIES = [
    "Assignment", "Confidentiality", "Counterparts", "Definitions",
    "Entire", "Governing", "Headings", "Indemnification", "Insurance",
    "Miscellaneous", "NOW", "Notices", "Representations", "Severability",
    "Termination", "WHEREAS", "base-salary", "board", "capitalization",
    "compensation", "conversion_of_shares", "dividends", "employee_benefits",
    "esop", "financing", "foreign_investors", "grant", "grant_of_option",
    "interest", "investment-company-act", "investment_company", "investments",
    "loans", "ownership_of_shares", "payment", "payment_terms",
    "private_equity", "seed", "shares", "stock_option", "taxes", "vesting"
]

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── API Keys ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
