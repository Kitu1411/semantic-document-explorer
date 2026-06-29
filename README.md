# Semantic Document Explorer & RAG Q&A

An AI-powered Q&A dashboard built using Python, Streamlit, FAISS (Facebook AI Similarity Search), and the Google Gemini API. This application allows users to upload PDF documents, extract text chunks, index them locally in a vector database, and perform semantic similarity searches to answer questions citing the source page numbers.

## Features

- **PDF Document Parsing** — Extract text, layout properties, and tables from PDF files using PyMuPDF
- **Intelligent Text Chunking** — Split documents into overlapping chunks with page metadata using LangChain
- **Vector Indexing** — Generate embeddings with Sentence-Transformers and build FAISS indexes for fast similarity search
- **Semantic Retrieval** — Find relevant document passages using vector similarity with configurable thresholds
- **AI-Powered Q&A** — Get contextual answers from Google Gemini with cited source pages
- **Interactive Dashboard** — Streamlit UI with chat interface, document viewer, expandable citations, and analytics

## Project Structure

```text
semantic_document_explorer/
│
├── config/
│   └── .env.example          # Template for GEMINI_API_KEY
│
├── database/
│   └── __init__.py
│
├── storage/
│   └── faiss_index/           # FAISS vector database files (git-ignored)
│
├── app/
│   ├── __init__.py
│   ├── main.py                # Streamlit dashboard (Module 6)
│   ├── document_parser.py     # PDF parsing & text chunking (Modules 1 & 2)
│   ├── vector_manager.py      # FAISS index & embeddings (Module 3)
│   └── rag_engine.py          # Context retrieval & Gemini API (Modules 4 & 5)
│
├── tests/
│   ├── create_sample_pdf.py   # Sample PDF generator for testing
│   ├── sample_report.pdf      # Generated test document
│   └── test_rag.py            # Unit tests for all modules
│
├── requirements.txt           # Third-party dependency list
├── .gitignore                 # Files/folders excluded from version control
├── README.md                  # This file
└── run.py                     # CLI script to launch Streamlit
```

## Architecture

| Module | File | Purpose |
|--------|------|---------|
| 1. Document Loader & Parser | `document_parser.py` | PDF byte stream processing, text extraction, metadata logging |
| 2. Text Chunking Engine | `document_parser.py` | Recursive character splitting with page metadata retention |
| 3. Vector Indexing Manager | `vector_manager.py` | FAISS index creation, sentence-transformer embeddings, disk persistence |
| 4. Context Retriever | `rag_engine.py` | FAISS similarity search with L2 distance scoring and threshold filtering |
| 5. RAG Prompt Orchestrator | `rag_engine.py` | Structured prompt construction and Google Gemini API integration |
| 6. Streamlit UI Dashboard | `main.py` | Chat Q&A interface, document viewer, analytics, pipeline controls |

## Setup Instructions

### 1. Prerequisites
- Python 3.10 to 3.12 (Recommended for library stability, though Python 3.14+ is supported with compilation build tools).
- Git installed on your system.
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey)).

### 2. Environment Virtualization
Create and activate the virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Dependencies Installation
```powershell
pip install -r requirements.txt
```

### 4. Configuration
Rename the `.env.example` in the `config/` directory to `.env` and provide your Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...
```

### 5. Running the App
Launch the dashboard via the wrapper runner script:
```powershell
python run.py
```
Or directly via Streamlit:
```powershell
streamlit run app/main.py
```

### 6. Running Tests
```powershell
python -m pytest tests/test_rag.py -v
```

## Usage

1. **Upload a PDF** — Use the sidebar to upload a custom PDF or select the built-in sample financial report.
2. **Process & Index** — Click the "🚀 Process & Index Document" button to parse, chunk, and build the FAISS vector index.
3. **Ask Questions** — Switch to the "💬 Chat Q&A" tab and type questions about the document.
4. **View Sources** — Expand the "📎 Sources" block under each answer to see the cited passages and page numbers.
5. **Explore Analytics** — Check the "📊 Analytics" tab for chunking statistics, index metrics, and query history.

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | PDF text extraction and layout analysis |
| `langchain-text-splitters` | Recursive text chunking |
| `faiss-cpu` | Facebook AI Similarity Search vector database |
| `sentence-transformers` | Text embedding generation (all-MiniLM-L6-v2) |
| `langchain-community` | FAISS vector store integration |
| `langchain` | Document schema and base abstractions |
| `google-generativeai` | Google Gemini API client |
| `streamlit` | Web dashboard framework |
| `python-dotenv` | Environment variable management |
| `pytest` | Unit testing framework |
