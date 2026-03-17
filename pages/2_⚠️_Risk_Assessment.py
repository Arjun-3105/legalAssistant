import streamlit as st
import os
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO, StringIO
from fpdf import FPDF
import pdfplumber

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorStore import DocumentStore

@st.cache_resource
def load_store():
    return DocumentStore()

store = load_store()

# Load the model and tokenizer (cached)
@st.cache_resource(show_spinner="Loading NLP Model...")
def load_model():
    model_name = "valhalla/t5-base-qg-hl"

    # use local cache folder to avoid re-downloading every run
    cache_dir = "./hf_cache"

    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    
    return model, tokenizer

# Initialize
model, tokenizer = load_model()

# Streamlit App UI
st.title("Enhanced Contract Risk Assessment Tool")
st.write("Analyze contract clauses for risks and get correction suggestions.")

# File Upload
uploaded_file = st.file_uploader("Upload a contract file (.txt, .docx, .pdf):", type=["txt", "docx", "pdf"])
max_clauses = st.slider("Select the maximum number of clauses to assess:", 1, 20, 5)


def read_uploaded_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    elif file.name.endswith(".pdf"):
        # Try PyPDF2 first
        try:
            reader = PdfReader(file)
            text = "\n".join([
                page.extract_text() or "" 
                for page in reader.pages
            ])
            if text.strip():
                return text
        except:
            pass

        # Fallback: use pdfplumber (better extraction)
        try:
            file.seek(0)
            with pdfplumber.open(file) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                if text.strip():
                    return text
        except:
            pass

        # If all fail
        return ""

    return ""



# Process uploaded file
contract_text = ""
if uploaded_file:
    contract_text = read_uploaded_file(uploaded_file)

# Risk Analysis
if st.button("Assess Risk"):
    if contract_text:
        st.subheader("Risk Assessment Results")
        with st.spinner("Assessing risks..."):
            try:
                clauses = contract_text.split(". ")
                clauses = clauses[:max_clauses]

                results = []

                for idx, clause in enumerate(clauses):
                    formatted_input = f"Assess the risk of the following contract clause: {clause}"
                    
                    # Direct generation instead of pipeline
                    input_ids = tokenizer.encode(formatted_input, return_tensors="pt", max_length=512, truncation=True)
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=128,
                        num_return_sequences=1
                    )
                    
                    risk_assessment = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # simple rule-based mock risk detection
                    text_lower = risk_assessment.lower()
                    if "high" in text_lower:
                        risk_level = "High"
                    elif "medium" in text_lower:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"

                    results.append({
                        "Clause": clause,
                        "Risk Assessment": risk_assessment,
                        "Risk Level": risk_level,
                    })

                # ----- PDF Generation -----
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=11)

                for idx, result in enumerate(results):
                    pdf.multi_cell(0, 10, txt=f"Clause {idx + 1}: {result['Clause']}")
                    pdf.multi_cell(0, 10, txt=f"Risk Level: {result['Risk Level']}")
                    pdf.multi_cell(0, 10, txt=f"Risk Assessment: {result['Risk Assessment']}")
                    pdf.ln(5)

                    # Show in UI
                    st.markdown(f"### Clause {idx + 1}")
                    st.markdown(f"**Clause:** {result['Clause']}")
                    st.markdown(f"**Risk Level:** `{result['Risk Level']}`")
                    st.markdown(f"**Risk Assessment:** {result['Risk Assessment']}")
                    st.write("---")

                # PDF output must use BytesIO, not StringIO
                pdf_bytes = BytesIO()
                pdf.output(pdf_bytes)
                pdf_bytes.seek(0)

                st.download_button(
                    label="Download Results as PDF",
                    data=pdf_bytes,
                    file_name="risk_assessment_results.pdf",
                    mime="application/pdf"
                )

                # ─── Cross-reference with Corpus ─────────────
                corpus_stats = store.get_stats()
                if corpus_stats["total_documents"] > 0:
                    st.divider()
                    st.subheader("🔗 Cross-Reference: Related Clauses from Corpus")
                    st.caption(f"Searching across {corpus_stats['total_documents']} indexed documents...")

                    # Search for the highest-risk clauses in the corpus
                    high_risk = [r for r in results if r["Risk Level"] == "High"]
                    search_clauses = high_risk[:3] if high_risk else results[:2]

                    for clause_result in search_clauses:
                        related = store.query(clause_result["Clause"], top_k=3)
                        if related:
                            with st.expander(f"Related to: \"{clause_result['Clause'][:80]}...\""):
                                for match in related:
                                    st.markdown(f"**Source:** {match['source_file']} (Page {match['page']})")
                                    st.caption(f"Similarity: {match['score']:.2%}")
                                    st.text(match["text"][:300] + "..." if len(match["text"]) > 300 else match["text"])
                                    st.divider()

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a file or input contract text to perform the risk assessment.")
else:
    st.info("Upload a contract file and click 'Assess Risk' to get started.")
