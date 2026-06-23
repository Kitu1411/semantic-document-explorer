# Semantic Document Explorer & RAG Q&A

An AI-powered Q&A dashboard built using Python, Streamlit, FAISS (Facebook AI Similarity Search), and the Google Gemini API. This application allows users to upload PDF documents, extract text chunks, index them locally in a vector database, and perform semantic similarity searches to answer questions citing the source page numbers.

## Project Structure

```text
semantic_document_explorer/
│
├── config/
│   └── .env.example        # Template for GEMINI_API_KEY
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py       # SQLite connection logging queries
│   └── logs.db             # Database file (ignored in git)
│
├── storage/
│   └── faiss_index/        # FAISS vector database files (ignored in git)
│
├── app/
│   ├── __init__.py
│   ├── main.py             # Streamlit application main runner
│   ├── document_parser.py  # PDF text extraction and metadata parser
│   ├── vector_manager.py   # Embedding generator and FAISS database interface
│   └── rag_engine.py       # Context retriever and Gemini API orchestrator
│
├── tests/
│   └── test_rag.py         # Unit tests checking text splitting and retrieval
│
├── requirements.txt        # Third-party dependency list
├── .gitignore              # Files/folders to exclude from version control
├── README.md               # Setup and development documentation
└── run.py                  # CLI script to launch Streamlit
```

## Setup Instructions

### 1. Prerequisites
- Python 3.10 to 3.12 (Recommended for library stability, though Python 3.14+ is supported with compilation build tools).
- Git installed on your system.

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
