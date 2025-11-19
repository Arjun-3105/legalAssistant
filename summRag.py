import streamlit as st
from gradio_client import Client, handle_file

# Initialize Hugging Face API Client
@st.cache_resource
def load_client():
    return Client("abeergandhi/lexsum")

client = load_client()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="LexSum - Legal Doc Summarizer", layout="wide")
st.title("📄 LexSum – Legal Document Summarizer & RAG Query System")

st.write("Upload a PDF and choose a mode to begin.")

# File uploader
uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])

# Mode selection
mode = st.radio("Select Mode", ["Summarizer", "RAG / Query"])

# Toggle Inputs API (optional)
if st.button("🔄 Sync with API (toggle_inputs)"):
    try:
        word_limit_from_api, default_query = client.predict(
            mode=mode,
            api_name="/toggle_inputs"
        )
        st.success(f"Synced! Word limit: {word_limit_from_api}, Query: {default_query}")
    except Exception as e:
        st.error(f"Error syncing: {e}")

# Mode-specific inputs
if mode == "Summarizer":
    word_limit = st.slider("Word Limit", min_value=50, max_value=2000, value=250)
    query_text = ""  # not used
else:
    word_limit = 250  # not used
    query_text = st.text_input("Enter your RAG Query")

st.divider()

# Submit button
if st.button("🚀 Process Document"):
    if not uploaded_pdf:
        st.error("Please upload a PDF first.")
    else:
        try:
            # Save uploaded PDF to a temporary file
            temp_path = f"temp_{uploaded_pdf.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            st.info("⏳ Processing... please wait. Large files may take 30–70 seconds.")
            with st.spinner("Running AI model..."):

                # MAIN API CALL
                summary, metrics = client.predict(
                    pdf_file=handle_file(temp_path),
                    mode=mode,
                    word_limit=word_limit,
                    query=query_text,
                    api_name="/process_document"
                )

            # Display results
            st.success("Processing complete!")
            
            st.subheader("📌 Summary / Result")
            st.write(summary)

            st.subheader("📊 Evaluation Metrics / Research Output")
            st.write(metrics)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
