import streamlit as st
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO, StringIO
from fpdf import FPDF

# Load the model and tokenizer (cached)
@st.cache_resource(show_spinner="Loading NLP Model...")
def load_pipeline():
    model_name = "valhalla/t5-base-qg-hl"

    # use local cache folder to avoid re-downloading every run
    cache_dir = "./hf_cache"

    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Initialize pipeline
risk_assessment_pipeline = load_pipeline()

# Streamlit App UI
st.title("Enhanced Contract Risk Assessment Tool")
st.write("Analyze contract clauses for risks and get correction suggestions.")

# File Upload
uploaded_file = st.file_uploader("Upload a contract file (.txt, .docx, .pdf):", type=["txt", "docx", "pdf"])
max_clauses = st.slider("Select the maximum number of clauses to assess:", 1, 20, 5)


import pdfplumber

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

                    generated = risk_assessment_pipeline(
                        formatted_input,
                        max_length=128,
                        num_return_sequences=1
                    )
                    risk_assessment = generated[0]["generated_text"]

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

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a file or input contract text to perform the risk assessment.")
else:
    st.info("Upload a contract file and click 'Assess Risk' to get started.")
