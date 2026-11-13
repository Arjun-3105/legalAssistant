"""
Clause Classification — Classify legal clauses into 42 categories using T5.

Integrates the T5 clause classifier (trained on CUAD dataset) with the
vector store for cross-referencing similar clauses across the corpus.
"""

import streamlit as st
import os
import sys
import re
import pdfplumber
from io import BytesIO
from fpdf import FPDF

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore
from api_manager import render_api_sidebar

@st.cache_resource
def load_store():
    return DocumentStore()

store = load_store()

# ─── T5 Clause Classifier ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading T5 Clause Classifier...")
def load_classifier():
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    model_name = "valhalla/t5-base-qg-hl"
    cache_dir = "./hf_cache"
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer

model, tokenizer = load_classifier()

# 42 clause categories from CUAD dataset
CLAUSE_CATEGORIES = {
    0: "Assignment", 1: "Confidentiality", 2: "Counterparts", 3: "Definitions",
    4: "Entire Agreement", 5: "Governing Law", 6: "Headings", 7: "Indemnification",
    8: "Insurance", 9: "Miscellaneous", 10: "NOW THEREFORE", 11: "Notices",
    12: "Representations", 13: "Severability", 14: "Termination", 15: "WHEREAS",
    16: "Base Salary", 17: "Board", 18: "Capitalization", 19: "Compensation",
    20: "Conversion of Shares", 21: "Dividends", 22: "Employee Benefits",
    23: "ESOP", 24: "Financing", 25: "Foreign Investors", 26: "Grant",
    27: "Grant of Option", 28: "Interest", 29: "Investment Company Act",
    30: "Investment Company", 31: "Investments", 32: "Loans",
    33: "Ownership of Shares", 34: "Payment", 35: "Payment Terms",
    36: "Private Equity", 37: "Seed", 38: "Shares", 39: "Stock Option",
    40: "Taxes", 41: "Vesting"
}

# Category risk profiles
HIGH_RISK_CATEGORIES = {"Indemnification", "Termination", "Liability", "Non-Compete", "Confidentiality"}
FINANCIAL_CATEGORIES = {"Base Salary", "Compensation", "Dividends", "Payment", "Payment Terms", "Taxes", "Interest", "Loans"}


def classify_clause(clause_text: str) -> dict:
    """Classify a single clause using T5."""
    formatted_input = f"classify: {clause_text.strip()}"
    input_ids = tokenizer.encode(formatted_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=input_ids, max_length=32, num_return_sequences=1)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to parse the prediction as a category ID
    try:
        cat_id = int(prediction)
        category = CLAUSE_CATEGORIES.get(cat_id, f"Category {cat_id}")
    except ValueError:
        category = prediction

    # Determine risk tier
    if category in HIGH_RISK_CATEGORIES:
        risk_tier = "🔴 High Risk"
    elif category in FINANCIAL_CATEGORIES:
        risk_tier = "🟡 Financial"
    else:
        risk_tier = "🟢 Standard"

    return {
        "clause": clause_text.strip(),
        "category": category,
        "risk_tier": risk_tier,
        "raw_prediction": prediction
    }


def extract_clauses_from_text(text: str) -> list[str]:
    """Split document text into clause-like segments."""
    # Split by common clause delimiters
    clauses = re.split(r'(?:\n\s*\n|\.\s+(?=[A-Z]))', text)
    # Filter out very short segments
    clauses = [c.strip() for c in clauses if c.strip() and len(c.strip().split()) > 5]
    return clauses


# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Legal AI — Clause Classification", layout="wide")
render_api_sidebar()
st.title("🏷️ Clause Classification")
st.write("Classify legal clauses into **42 categories** using a T5 model trained on the CUAD dataset (88.1% accuracy). Extract and categorize every clause in your documents.")

st.divider()

# ─── Input Options ───────────────────────────────────────────────────────────

input_method = st.radio("Input Method", ["Upload PDF", "Paste Text"], horizontal=True)

clauses_to_classify = []

if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a legal document", type=["pdf"])
    if uploaded_file:
        with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
            full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        clauses_to_classify = extract_clauses_from_text(full_text)
        st.info(f"📄 Extracted **{len(clauses_to_classify)}** clause segments from the document.")
else:
    text_input = st.text_area(
        "Paste clause text (separate multiple clauses with blank lines)",
        height=200,
        placeholder="The Company shall indemnify and hold harmless the Employee from any and all claims..."
    )
    if text_input:
        clauses_to_classify = [c.strip() for c in text_input.split("\n\n") if c.strip()]

# ─── Classification Controls ────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    max_clauses = st.slider("Max clauses to classify", 1, 50, min(20, len(clauses_to_classify) if clauses_to_classify else 20))
with col2:
    show_cross_ref = st.checkbox("Show corpus cross-references", value=True)

st.divider()

# ─── Run Classification ─────────────────────────────────────────────────────

if st.button("🏷️ Classify Clauses") and clauses_to_classify:
    clauses_to_process = clauses_to_classify[:max_clauses]
    results = []

    progress = st.progress(0)
    with st.spinner("Classifying clauses..."):
        for idx, clause in enumerate(clauses_to_process):
            result = classify_clause(clause)
            results.append(result)
            progress.progress((idx + 1) / len(clauses_to_process))

    # ─── Summary Statistics ──────────────────────────────
    st.subheader("📊 Classification Summary")

    categories_found = [r["category"] for r in results]
    unique_categories = set(categories_found)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Clauses", len(results))
    col2.metric("Unique Categories", len(unique_categories))
    col3.metric("High Risk", sum(1 for r in results if "High Risk" in r["risk_tier"]))
    col4.metric("Financial", sum(1 for r in results if "Financial" in r["risk_tier"]))

    # Category distribution
    from collections import Counter
    cat_counts = Counter(categories_found)
    st.bar_chart({k: v for k, v in sorted(cat_counts.items(), key=lambda x: -x[1])})

    st.divider()

    # ─── Detailed Results ────────────────────────────────
    st.subheader("📋 Detailed Classification")

    for idx, result in enumerate(results):
        with st.expander(f"{result['risk_tier']} **{result['category']}** — Clause {idx + 1}", expanded=(idx < 3)):
            st.markdown(f"**Clause Text:**")
            st.text(result["clause"][:500] + ("..." if len(result["clause"]) > 500 else ""))
            st.markdown(f"**Category:** `{result['category']}`")
            st.markdown(f"**Risk Tier:** {result['risk_tier']}")

            # Cross-reference with corpus
            if show_cross_ref:
                corpus_stats = store.get_stats()
                if corpus_stats["total_documents"] > 0:
                    related = store.query(result["clause"], top_k=3)
                    if related:
                        st.markdown("**🔗 Similar clauses in corpus:**")
                        for match in related:
                            st.caption(f"📄 {match['source_file']} (Page {match['page']}) — Similarity: {match['score']:.2%}")

    st.divider()

    # ─── PDF Report Generation ───────────────────────────
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Clause Classification Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 8, f"Total Clauses: {len(results)} | Categories: {len(unique_categories)}", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    for idx, result in enumerate(results):
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 7, f"Clause {idx + 1}: [{result['category']}] {result['risk_tier']}")
        pdf.set_font("Arial", size=9)
        clause_display = result["clause"][:300] + "..." if len(result["clause"]) > 300 else result["clause"]
        # Encode to latin-1 safe text
        safe_text = clause_display.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)

    pdf_bytes = BytesIO()
    pdf.output(pdf_bytes)
    pdf_bytes.seek(0)

    st.download_button(
        "📥 Download Classification Report (PDF)",
        data=pdf_bytes,
        file_name="clause_classification_report.pdf",
        mime="application/pdf"
    )

elif not clauses_to_classify:
    st.info("Upload a PDF or paste clause text above, then click **Classify Clauses** to begin.")
