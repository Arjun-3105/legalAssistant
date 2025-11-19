import streamlit as st
import subprocess
import sys
import os

st.set_page_config(
    page_title="T5 Legal AI System",
    layout="wide"
)

# ------------------------------
# HEADER
# ------------------------------
st.title("⚖️ T5 Legal AI System")
st.write("A fine-tuned T5 model designed for **legal document summarization**, **risk detection**, **clause extraction**, and **RAG-based retrieval**.")

st.markdown("---")

# ------------------------------
# MODEL CAPABILITIES SECTION
# ------------------------------
st.subheader("🧠 Model Capabilities")

st.markdown("""
### **1️⃣ Summarization (Fine-tuned T5 Model)**
- Generates high-quality summaries for long legal documents  
- Optimized for contracts, agreements, MoUs  
- Outputs structured, concise, and jargon-aware summaries  

---

### **2️⃣ Risk & Clause Extraction**
- Identifies risky sections, liability issues, indemnity risks  
- Extracts important clauses from agreements  
- Useful for contract review, due diligence, compliance

---

### **3️⃣ RAG Pipeline (Retrieval-Augmented Generation)**
- Accepts queries about your legal document  
- Vector search to retrieve relevant sections  
- LLM (T5) answers using document context (no hallucinations)

---

### **4️⃣ Dual-Mode Inference System**
You can run:
- **Summarization/RAG Mode** → `summRag.py`  
- **Risk & Clause Mode** → `Risk.py`  

Both scripts exist in the **same directory**.
""")

st.markdown("---")

st.subheader("🚀 Choose an Action")

# Button Styles
button_style = """
<style>
div.stButton > button {
    font-size: 18px;
    padding: 0.6rem 1.2rem;
    border-radius: 10px;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# ------------------------------
# BUTTONS TO RUN PY FILES
# ------------------------------

col1, col2 = st.columns(2)
with col1:
    if st.button("📝 Run Summarization / RAG Engine"):
        try:
            st.success("Launching Summarization/RAG engine...")
            subprocess.Popen(["streamlit", "run", "summRag.py"])
        except Exception as e:
            st.error(f"Error launching summRag.py: {e}")

with col2:
    if st.button("⚠️ Run Risk & Clause Extraction Engine"):
        try:
            st.success("Launching Risk/Clause engine...")
            subprocess.Popen(["streamlit", "run", "Risk.py"])
        except Exception as e:
            st.error(f"Error launching Risk.py: {e}")


st.markdown("---")

st.info("ℹ️ Ensure **summRag.py** and **Risk.py** are in the same directory as this Streamlit file.")
