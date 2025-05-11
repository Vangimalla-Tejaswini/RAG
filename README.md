# RAG-Powered Multi-Agent Q&A Assistant

## Overview
This is a simple knowledge assistant that uses a Retrieval-Augmented Generation (RAG) pipeline with agentic logic to answer user queries from a small collection of documents.

## Features
- RAG pipeline with document chunking and vector search (FAISS)
- OpenAI LLM integration for generating answers
- Agent logic for keyword-based query routing
- Streamlit UI for interaction and visualization

## Setup Instructions
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py`: Streamlit interface
- `utils.py`: Core logic for ingestion, vector search, and agent
- `docs/`: Folder containing input `.txt` documents
