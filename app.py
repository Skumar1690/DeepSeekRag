import streamlit as st
from rag_deepseek import answer_query
import os
import tempfile

# Streamlit app configuration
st.set_page_config(page_title="RAG Question-Answering System", page_icon="📚")
st.title("RAG Question-Answering System")
st.write("Upload a PDF and ask questions about its content.")

# Check for sample document
sample_document_path = "sample_document.pdf"
has_sample_document = os.path.exists(sample_document_path)

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for PDF path
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

# Save uploaded PDF
if uploaded_file is not None:
    # Create a temporary file to save the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        st.session_state.pdf_path = tmp_file.name
    st.success("PDF uploaded successfully!")

# Input for user query
query = st.text_input("Enter your question:", "")

# Button to submit query
if st.button("Get Answer"):
    if st.session_state.pdf_path and query:
        with st.spinner("Processing..."):
            try:
                answer = answer_query(st.session_state.pdf_path, query)
                st.write("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please upload a PDF and enter a question.")

# Display sample PDF option
if not uploaded_file and has_sample_document:
    st.write("---")
    st.write("No PDF uploaded. You can use our sample document for testing.")
    if st.button("Use Sample Document"):
        st.session_state.pdf_path = sample_document_path
        st.success("Using sample_document.pdf for testing!")
        
# Display current document being used
if st.session_state.pdf_path:
    doc_name = os.path.basename(st.session_state.pdf_path)
    st.write(f"Currently using: **{doc_name}**")
