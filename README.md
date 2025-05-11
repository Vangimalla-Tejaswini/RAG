# RAG-Powered Multi-Agent Q&A Assistant


## Overview
This project implements a **RAG-powered Multi-Agent Q&A Assistant** that retrieves relevant information from a set of documents and generates answers using **Hugging Face models**. The project uses FAISS for efficient document retrieval and a simple agent workflow to answer various queries.

## Features
- **RAG Pipeline**: Retrieves relevant document chunks based on user queries.
- **LLM Integration**: Generates answers using Hugging Face models.
- **Agent Logic**: Routes queries based on keywords like "define", "calculate", etc.


## Requirements
Install the dependencies with the following command:

pip install -r requirements.txt

## Usage
1. **Prepare your documents**: Add your document text to the `documents/your_documents.txt` file.
2. **Set up your Hugging Face API key**: Add your Hugging Face API key to the `.env` file.
3. **Run the app**:
streamlit run app.py

## 🚀 Live Demo
👉 [Click here to try the app](https://mzj2ckdiderc9svllewvfj.streamlit.app/)