# Implementation Plan - Phase 1: Setup & Git

This plan covers the initialization and setup of the **AI Q&A from PDF (RAG Based)** project directory, configuring Git, setting up the virtual environment, and installing the required dependencies.

## User Review Required

> [!WARNING]
> **Git Not Found in Path:**
> During initial system checks, the `git` command was not found in the environment paths or standard installation directories. To complete the Git setup portion of this phase, please verify if you have Git installed, or install Git (e.g., from [git-scm.com](https://git-scm.com/)). If you already have it installed, you may need to add it to your system PATH or let us know its installation directory.
>
> **Python 3.14 Compatibility:**
> The active Python version on your system is `3.14.0` (a pre-release/alpha version of Python). Many AI/ML libraries, such as `torch` (dependency of `sentence-transformers`) and `faiss-cpu`, might not have pre-built wheels for Python 3.14.0 yet. This could lead to compilation issues or failures during `pip install`. We will attempt to create the virtual environment and install the dependencies, but if it fails, we may need to use a stable Python version (like 3.10, 3.11, or 3.12).

## Open Questions

1. **Workspace Root vs Subfolder:** Should the project files (e.g., `app/`, `config/`, etc.) be created directly inside the current workspace root directory (`c:\Users\kirtan\OneDrive\Desktop\Internship Project\`), or should we create a subdirectory named `semantic_document_explorer/` as described in the handbook and put all files inside it?
   * *Recommendation:* Create the project structure directly inside `c:\Users\kirtan\OneDrive\Desktop\Internship Project\` to keep it clean and match the workspace scope.
2. **Alternative Python Version:** If dependency installation fails on Python 3.14.0, do you have a stable version of Python (e.g. Python 3.11 or 3.12) installed on your system that we should target?

## Proposed Changes

We will create the directory layout as structured in the mentorship handbook:

### Project Directory Skeleton

#### [NEW] [.gitignore](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/.gitignore)
Create a `.gitignore` to prevent committing virtual environment folders, databases, FAISS index files, and API secrets.
```git
# Virtual Environment
.venv/
venv/
ENV/

# Python caching
__pycache__/
*.pyc
*.pyo
*.pyd

# Environment Variables
.env
.env.local

# Databases and Local Storage
database/logs.db
storage/faiss_index/

# OS files
.DS_Store
Thumbs.db
```

#### [NEW] [requirements.txt](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/requirements.txt)
Define the libraries needed for the project:
```text
pymupdf==1.24.4
langchain-text-splitters==0.2.1
faiss-cpu==1.8.0
sentence-transformers==3.0.1
langchain-community==0.2.4
google-generativeai==0.7.1
streamlit==1.36.0
python-dotenv==1.0.1
pytest==8.2.2
```

#### [NEW] [config/.env.example](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/config/.env.example)
Create a template for environment variables:
```ini
# Gemini API Key configuration
GEMINI_API_KEY=your_gemini_api_key_here
```

#### [NEW] [database/\_\_init\_\_.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/database/__init__.py)
Initialize the database package.

#### [NEW] [app/\_\_init\_\_.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/app/__init__.py)
Initialize the app package.

#### [NEW] [app/document_parser.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/app/document_parser.py)
Placeholder file for PyMuPDF extraction and chunking.

#### [NEW] [app/vector_manager.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/app/vector_manager.py)
Placeholder file for FAISS embedding storage and load/save methods.

#### [NEW] [app/rag_engine.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/app/rag_engine.py)
Placeholder file for retrieval and Gemini API orchestration.

#### [NEW] [app/main.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/app/main.py)
Placeholder file for Streamlit UI interface.

#### [NEW] [tests/test_rag.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/tests/test_rag.py)
Placeholder file for tests.

#### [NEW] [README.md](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/README.md)
Basic README file explaining the setup.

#### [NEW] [run.py](file:///c:/Users/kirtan/OneDrive/Desktop/Internship%20Project/run.py)
A CLI runner wrapper script:
```python
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"])
```

## Verification Plan

### Automated Tests
- Once the virtual environment is set up, run:
  ```powershell
  .venv\Scripts\python -c "import streamlit, pymupdf, langchain, faiss, sentence_transformers, google.generativeai; print('All libraries imported successfully!')"
  ```
  to verify dependencies are functional.

### Manual Verification
- Verify that the `.venv` directory is created.
- Verify directory structure matches the plan.
- If Git is available, verify `git status` shows initialized repository.
