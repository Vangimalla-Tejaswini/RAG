import streamlit as st
import os
import tempfile
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

st.title("📄 PDF/TXT-based RAG Assistant")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt", "text"], accept_multiple_files=True
)

if uploaded_files:
    documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load based on file type
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)

            documents.extend(loader.load())

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embeddings)

        query = st.text_input("Ask a question based on your documents:")

        if query:
            results = vector_store.similarity_search(query, k=3)

            st.subheader("📚 Retrieved Chunks:")
            for i, res in enumerate(results, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(res.page_content)
