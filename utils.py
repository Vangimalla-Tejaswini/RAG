# utils.py

import re
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables (like Hugging Face token) from .env
load_dotenv()

# Function to load and chunk documents into smaller pieces
def load_and_chunk_docs(paths):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = []
    for path in paths:
        try:
            docs = TextLoader(path).load()
            chunks.extend(splitter.split_documents(docs))
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return chunks

# Function to create a vector store from document chunks
def create_vector_store(chunks):
    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embedding)
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Function to create the RAG chain (Retrieval Augmented Generation)
def get_rag_chain(vectorstore):
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-small",
            model_kwargs={"temperature": 0.5, "max_length": 150}
        )
        return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_k=3))
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        return None

# Function to route the query either to the calculator or the RAG chain
def route_query(query, rag_chain):
    if re.search(r'\bcalculate\b|\bcompute\b|\bdefine\b', query, re.IGNORECASE):
        return f"[Tool Used: Calculator] Simulated result for: {query}"
    return rag_chain.run(query)

# Streamlit UI for the RAG Q&A assistant
st.set_page_config(page_title="RAG Q&A Assistant")
st.title("🧠 RAG-Powered Multi-Agent Q&A")

# Paths to your documents
DOC_PATHS = ["docs/doc1.txt", "docs/doc2.txt", "docs/doc3.txt"]

# Load & index docs once
chunks = load_and_chunk_docs(DOC_PATHS)

# If docs were successfully loaded and chunked, create the vector store
if chunks:
    vector_store = create_vector_store(chunks)
    if vector_store:
        rag_chain = get_rag_chain(vector_store)
        if rag_chain:
            query = st.text_input("Ask me anything:")
            if query:
                with st.spinner("Thinking..."):
                    answer = route_query(query, rag_chain)
                st.markdown("**Answer:**")
                st.write(answer)
        else:
            st.error("Failed to create RAG chain.")
    else:
        st.error("Failed to create vector store.")
else:
    st.error("Failed to load or chunk documents.")
