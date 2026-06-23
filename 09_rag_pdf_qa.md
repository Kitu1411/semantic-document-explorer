# Internship Mentorship Handbook: Semantic Document Explorer & RAG Q&A

**Project Name:** AI Q&A from PDF (RAG Based)  
**Assigned Student:** Parmar Kirtan  
**Mentorship Focus:** Learn → Build → Integrate → Test → Document → Present  

---

## 1. Project Overview

### Problem Statement
Unstructured documents, particularly PDF files (manuals, financial reports, research papers, legal contracts), hold critical operational knowledge. However, searching through hundreds of pages for specific clauses or statistics is time-consuming. Standard keyword search (like `Ctrl + F`) is rigid; it cannot understand synonyms, context, or semantic meanings. Users need a system that can read documents, interpret natural language questions, locate relevant sections, and output direct, cited answers.

### Project Goal
The goal of this project is to build **Semantic Document Explorer**, a web-based, RAG-powered (Retrieval-Augmented Generation) document Q&A engine using Python, Streamlit, FAISS (Facebook AI Similarity Search), and the Google Gemini API. The application will allow users to upload multi-page PDF documents, split the text into semantic chunks, generate local vector embeddings, build a vector similarity index, retrieve relevant passages based on natural language queries, and feed them as context to the Gemini API to output accurate, page-cited answers.

### Real-world Applications
* **Financial Analyst Assistant:** Ingesting quarterly earnings reports to quickly compare revenue metrics across tables.
* **Legal Compliance Scanner:** Querying long contract agreements for liability clauses or termination dates.
* **Academic Literature Review:** Summarizing research methodologies and findings across multiple uploaded studies.

### Why This Project Matters
RAG is the dominant architectural pattern in industrial AI engineering. By building this project, interns master document processing pipelines (handling different file encodings and layouts), text chunking mechanics, vector database indexing (using FAISS, the gold standard for local vector search), and context-injected prompt engineering.

### Expected Final MVP
A Streamlit web dashboard that lets users:
1. Drag and drop any multi-page PDF document.
2. Ingest the text and view processing logs (number of pages read, chunks generated).
3. Chat with the document in an interactive chat interface.
4. View **Source Citations** showing the exact page numbers and snippet text blocks that the AI used to answer the question.
5. Review a **Semantic Confidence Meter** indicating how closely the document text matched the user's query.

### Future Enhancements
* Multi-document querying (indexing an entire folder of PDFs and querying across them).
* Supporting scanned PDF files using Tesseract OCR pipelines.

---

## 2. Difficulty Estimation

* **Level:** Intermediate  
* **Why:** Building a basic API call is simple. The project is classified as *Intermediate* because of:
  1. **RAG Pipeline Orchestration:** Interns must coordinate document token parsing, text splitting thresholds, vector generation, and query retrieval using FAISS.
  2. **Metadata Tracking:** For the AI to cite sources (e.g. "Page 12, Paragraph 3"), the indexing pipeline must keep track of text metadata coordinates throughout chunking and retrieval operations.

---

## 3. Skills Required

### Programming
* **Python:** App scripts, file IO operations, dictionary layouts.
* **Streamlit:** File upload handlers, chat elements, sidebar parameters.
* **SQL (SQLite3):** Storing document metadata and historical question logs.

### AI/ML
* **Retrieval-Augmented Generation (RAG):** Context mapping, retrieval thresholds, prompt design.
* **Vector Embeddings:** Generating dense mathematical vectors from text blocks.
* **Vector Databases (FAISS):** Creating index tables, reading, writing, and similarity searching.
* **NLP Chunking:** Document tokenization, splitters, overlap parameters.

### Software Engineering
* **Git & GitHub:** Commit history logs, branch management, and repository structure.
* **Document Processing:** Extracting clean text layouts using PyPDF2 or PyMuPDF.
* **API Integration:** Environmental security configurations (`.env`) and Gemini API connections.

---

## 4. Recommended Tech Stack

| Component | Technology | Why |
|---|---|---|
| **Frontend UI** | **Streamlit** | `st.chat_message` and `st.chat_input` widgets simplify building conversational AI apps. |
| **Document Reader** | **PyMuPDF (fitz)** | Rapid and reliable text extractor that preserves page numbers and text coordinates. |
| **Vector Index** | **FAISS (Local/Embedded)** | Facebook AI Similarity Search. A highly optimized, lightweight library for vector similarity search that runs locally on CPU with zero configuration. |
| **Embeddings Model** | **Hugging Face (`all-MiniLM-L6-v2`)** | A lightweight embedding model. Converts sentences to 384-dimensional dense vectors locally. |
| **AI LLM API** | **Google Gemini Developer API (`gemini-2.5-flash`)** | High processing speed, generous free limits, and excellent at structured text matching. |
| **RAG Orchestrator** | **LangChain** | Provides built-in text splitters (`RecursiveCharacterTextSplitter`) and vector loaders to avoid writing vector search mathematics from scratch. |

---

## 5. System Architecture

### Text-Based Architecture Flowchart

```text
       [Multi-page PDF Document]
                 │
                 ▼
       [PyMuPDF Text Ingestion]
                 │
                 ▼
    [Recursive Text Splitter] (1000 char chunks, 200 overlap)
                 │
                 ▼ (Keep Page Number Metadata)
      [all-MiniLM-L6-v2 Engine] (Convert chunk to 384-dim vector)
                 │
                 ▼
          [Local FAISS Index]
                 │
                 ▼
       [Similarity Search] <─── [User Question / Query]
                 │
                 ▼ (Top 4 chunks + Page numbers fetched)
     [Gemini LLM Prompt Compiler] (Inject text chunks as context)
                 │
                 v
       [Generates Response Text] (Answer citing source pages)
                 │
                 v
       [Streamlit Chat Area] (Visual bubbles showing answers + citations)
```

---

## 6. Development Modules

### Module 1: Document Loader & Parser
* **Purpose:** Process uploaded PDF file streams, extract text per page, and log metadata (file name, page counts).
* **Inputs:** PDF byte buffer.
* **Outputs:** Clean text dictionaries mapped by page numbers.
* **Dependencies:** `pymupdf`

### Module 2: Text Chunking Engine
* **Purpose:** Segment document text into overlapping character chunks while retaining page metadata.
* **Inputs:** Clean text dicts.
* **Outputs:** List of document chunk objects (text content, file name, page number).
* **Dependencies:** `langchain-text-splitters`

### Module 3: Vector Indexing Manager (FAISS)
* **Purpose:** Initialize local FAISS vector store, generate embeddings, and save index files to disk.
* **Inputs:** List of document chunk objects.
* **Outputs:** Local FAISS index files (`index.faiss`, `index.pkl`).
* **Dependencies:** `faiss-cpu`, `sentence-transformers`, `langchain-community`

### Module 4: Context Retriever
* **Purpose:** Vectorize user queries, query the FAISS index, filter chunks below similarity thresholds, and return matching passages.
* **Inputs:** User query string, FAISS vector store.
* **Outputs:** List of relevant text chunks and page numbers.
* **Dependencies:** `faiss-cpu`, `langchain`

### Module 5: RAG Prompt Orchestrator
* **Purpose:** Construct prompts containing user queries and context passages, and call Gemini API.
* **Inputs:** User query, relevant text chunks.
* **Outputs:** AI conversational answer.
* **Dependencies:** `google-generativeai`

### Module 6: Streamlit UI Dashboard
* **Purpose:** Sidebar file uploaders, conversational chat history window, metrics panels, and expandable citation blocks.
* **Inputs:** Streamlit widget interactions.
* **Outputs:** Rendered dashboard pages.
* **Dependencies:** `streamlit`

---

## 7. What Should Be Built vs Reused

| Build Yourself | Reuse Existing |
|---|---|
| Metadata tracking through chunking pipelines | PDF text reader libraries (`pymupdf`) |
| Prompt templates containing context variables | Vector database indices (`FAISS`) |
| Interactive chat bubble formats | Pre-trained Embedding Models (`all-MiniLM-L6-v2`) |
| Custom confidence score metrics | Gemini API Python SDK client wrappers |
| Local database schemas (logs in SQLite) | Streamlit chat message UI templates |

---

## 8. Pre-trained Models and APIs

* **`all-MiniLM-L6-v2` (Sentence Transformers):** Free, local embedding generator. Excellent balance of speed, memory usage, and retrieval accuracy.
* **Google Gemini API (`gemini-2.5-flash`):** Core LLM for chat answers.

---

## 9. Existing Open Source Projects to Study

* **LangChain RAG Quickstart Repository:** Review patterns of document ingestion, vector retrieval, and history injection.
* **Streamlit App Gallery (Generative AI section):** Study how multi-page layouts and conversational chat containers are built.
* **What NOT to copy:** Heavy enterprise RAG platforms (like Dify or Flowise) which run complex Docker compose networks. Keep the implementation self-contained within Streamlit to ensure the student can run and modify all code directly.

---

## 10. Data Sources

* **Financial/Academic PDFs:**
  * Sample annual financial reports (e.g. Apple or Tesla 10-K filings) or open academic papers (from arXiv).
  * Search keywords: `apple 10k report pdf download`.
  * Cost: Free.
* **Synthetic Lecture Notes:** Custom markdown documents representing mock course outlines for testing.

---

## 11. Research Papers for Reference

| Title | Year | Summary | Why Read It |
|---|---|---|---|
| *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | 2020 | Foundational paper introducing the concept of augmenting LLM generators with retrieved document contexts. | Crucial for understanding why RAG reduces LLM hallucinations. |
| *Billion-scale similarity search with GPUs (FAISS)* | 2017 | Details the architectural designs of the FAISS vector similarity search framework. | Provides deep mathematical background on index matching. |

---

## 12. Self-Learning Resources

### Official Documentation
* **LangChain Text Splitters:** [LangChain Docs](https://python.langchain.com/v0.2/docs/how_to/#text-splitters) - Essential for setting chunk sizes and overlap rules.
* **FAISS official documentation:** [FAISS GitHub](https://github.com/facebookresearch/faiss) - Focus on CPU index installations.

### YouTube Crash Courses
* **RAG Pipeline Tutorial in Python:** *RAG from Scratch* (by LangChain or freeCodeCamp).
* **Gemini API RAG Tutorial:** search: "Gemini API PDF Q&A Python".

---

## 13. Industry Standard Folder Structure

```text
semantic_document_explorer/
│
├── config/
│   └── .env.example        # Template for GEMINI_API_KEY
│
├── database/
│   ├── __init__.py
│   ├── db_manager.py       # SQLite connection logging queries
│   └── logs.db             # Database file (ignored in .gitignore)
│
├── storage/
│   └── faiss_index/        # FAISS vector database files (ignored in .gitignore)
│
├── app/
│   ├── __init__.py
│   ├── main.py             # Streamlit application main runner
│   ├── document_parser.py  # PDF text extraction and metadata parser
│   ├── vector_manager.py   # Embedding generator and FAISS database interface
│   └── rag_engine.py       # Context retriever and Gemini API orchestrator
│
├── tests/
│   └── test_rag.py         # Unit tests checking text splitting and retrieval metrics
│
├── requirements.txt
├── .gitignore
├── README.md
└── run.py                  # CLI script launching streamlit (`streamlit run app/main.py`)
```

---

## 14. GitHub Milestones

1. **Milestone 1: Repository Setup & PDF Parser**
   * Structure project directories, configure virtual environment, and develop a script that parses text from a PDF and prints it.
2. **Milestone 2: Chunker & Local Vector Database**
   * Configure LangChain's text splitter, download embedding models, write vectors, and initialize a local FAISS collection.
3. **Milestone 3: Basic Retrieval Engine**
   * Implement similarity searches. Verify that querying a topic returns the exact page/paragraph chunks containing the relevant information.
4. **Milestone 4: Contextual Q&A (RAG)**
   * Link vector retrieval to Gemini API. Design system prompts that instruct Gemini to answer queries *only* using the provided context, citing pages.
5. **Milestone 5: Streamlit UI Implementation**
   * Build the UI: PDF uploader sidebar, conversation container, and citation panels.
6. **Milestone 6: Performance Tuning & Caching**
   * Implement caching optimizations, error prompts, and clean up temporary database logs.
7. **Milestone 7: Unit Testing & Deployment**
   * Write tests checking RAG accuracy, write documentation, deploy, and record the demo video.

---

## 15. MVP Planning

### Must Have
* PDF file upload widget in Streamlit sidebar.
* PDF parsed text chunking (1000 char chunk size, 200 char overlap) and local FAISS DB persistence.
* RAG Q&A chat interface with page citations.
* Visual citation panels (collapsible text expanders displaying source chunks).
* Caching model loaders.

### Nice to Have
* Support for uploading multiple PDFs.
* Search history panel (sidebar listing historical queries and dates).

### Future Scope
* OCR support (parsing scanned text images).

---

## 16. 15-Day Curriculum

| Day | Phase | Learning Goals | Topics to Study | Resources | What to Build | Expected Deliverable | Est. Time |
|---|---|---|---|---|---|---|---|
| **Day 1** | Phase 1 | Setup & Git | Project directory structures, virtual envs, Git branching. | Python Venv Guide. | Set up folders, configure Git, install dependencies. | Project directory with activated virtual environment. | 4 hrs |
| **Day 2** | Phase 1 | PDF Text Parsing | PyMuPDF structure, page arrays, coordinate tables extraction, handling noise. | PyMuPDF docs. | Build `app/document_parser.py` parser class. | Python script extracting PDF page contents and layout properties. | 5 hrs |
| **Day 3** | Phase 1 | Text Chunking strategies | Tokens limits, chunk size, overlapping configs, semantic boundaries. | LangChain splitters. | Write recursive text splitters inside `app/document_parser.py`. | Console script outputting chunks list from a PDF. | 5 hrs |
| **Day 4** | Phase 2 | Vector Embeddings | Vector mathematics, cosine similarity, Sentence Transformers library. | Sentence Transformers docs. | Write embeddings generator to translate text blocks into numeric vectors. | Python script generating embedding float arrays from text blocks. | 6 hrs |
| **Day 5** | Phase 2 | Local Vector Indexing | Vector databases, collections schemas, FAISS initialization. | FAISS documentation. | Build `app/vector_manager.py` initializing local storage and writing indices. | FAISS collection folder populated with document vector logs. | 6 hrs |
| **Day 6** | Phase 2 | Semantic Retrieval | Similarity search thresholds, query vector comparisons. | Vector search guides. | Implement similarity retrieval functions in `app/rag_engine.py`. | Console query returns the top 3 corresponding text chunks. | 6 hrs |
| **Day 7** | Phase 2 | LLM Prompt Injections | System instructions, context variables, formatting assistant outputs. | Gemini Prompt guides. | Connect Gemini API to the retrieval pipeline to answer queries. | RAG script generating text answers containing page references. | 6 hrs |
| **Day 8** | Phase 2 | SQLite logging | SQL schema creation, score logs tables, connections management. | Python SQLite guide. | Build SQLite log helpers in `database/db_manager.py`. | Local SQLite DB saving analysis score records. | 5 hrs |
| **Day 9** | Phase 2 | Streamlit Chat UI | Chat templates, state managers, layout containers. | Streamlit API references. | Code the conversational UI window in `app/main.py`. | Chat UI demonstrating question submissions and responses. | 6 hrs |
| **Day 10**| Phase 3 | App Integration | Connecting page components, loaders, error alerts. | Python code patterns. | Integrate RAG engines, database logs, and conversational UIs. | Dashboard updating chat responses upon file upload. | 7 hrs |
| **Day 11**| Phase 3 | UI Visual Polish | Citation panels, markdown cards, loader overlays. | Streamlit layout guides. | Add collapsible text expanders displaying source chunks. | Dashboard showing visual citation panels. | 6 hrs |
| **Day 12**| Phase 3 | Performance Tuning | Caching functions, index loading, handling connection drops. | Streamlit performance guide. | Implement caching optimization, error prompts, and cleanup logs. | Stable dashboard that loads document vector states instantly. | 6 hrs |
| **Day 13**| Phase 3 | Automated Testing | Pytest structures, mock APIs, database integrity tests. | Pytest docs. | Write test scripts under `tests/` verifying chunking and DB logs. | Passing test logs checking pipeline mechanics. | 6 hrs |
| **Day 14**| Phase 4 | Project Documentation | Documentation workflows, writing README guides. | README templates. | Write `README.md` and complete inline code commenting. | Finished README explaining installation and design features. | 4 hrs |
| **Day 15**| Phase 4 | Presentation Day | Pitching methodologies, slide structures. | Presentation guides. | Design presentation slides detailing Architecture, FAISS models, and RAG pipelines. | Slide deck and final submission. | 5 hrs |

---

## 17. Daily Output Expectations & Developer Code Guides

To ensure success, the intern must refer to the following code guide for FAISS indexing.

### Critical Developer Guide: Document Indexing and Q&A using FAISS
The following code snippet demonstrates how to parse, index, retrieve, and query text chunks using FAISS and the Gemini API:

```python
# app/rag_engine.py
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load and Chunk PDF Document
def index_pdf_document(pdf_path: str, storage_dir: str):
    # Load PDF using PyMuPDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text recursively (1000 char chunk size, 200 char overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Initialize Embedding Model (Runs locally on CPU)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create FAISS Index and Save to Disk
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(storage_dir)
    return db

# Step 2: Query Index and Call Gemini API
def query_rag_engine(query: str, storage_dir: str):
    # Load Embedding Model and FAISS Index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(storage_dir, embeddings, allow_dangerous_deserialization=True)
    
    # Perform Similarity Search (Retrieve top 3 matching chunks)
    retrieved_docs = db.similarity_search(query, k=3)
    
    # Compile Context and Metadata
    context_blocks = []
    citations = []
    
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", 0) + 1 # Page indices are 0-indexed in fitz
        text_content = doc.page_content
        context_blocks.append(f"[Source: Page {page_num}]\n{text_content}")
        citations.append({"page": page_num, "text": text_content})
        
    context = "\n\n".join(context_blocks)
    
    # Call Gemini API
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    system_prompt = f"""
    You are an expert Q&A assistant. Answer the user's question using ONLY the provided document context below:
    ---
    {context}
    ---
    Be factual, objective, and cite the page numbers in your answer when referencing specific details.
    """
    
    response = model.generate_content(system_prompt + f"\n\nQuestion: {query}")
    return response.text, citations

# Mock Usage:
# index_pdf_document("report.pdf", "storage/faiss_index")
# answer, citations = query_rag_engine("What was the net revenue in 2025?", "storage/faiss_index")
# print(f"Answer:\n{answer}\n")
# print(f"Citations:\n{citations}")
```

### Daily Expectations

#### Day 1
* **Learn:** Git workflows, PIP configurations, setup directories.
* **Build:** Setup local project directories and install dependencies.
* **Expected Output:** Local directory with virtual environment set up and active Git tracking.

#### Day 2
* **Learn:** PyMuPDF layout structures, text extraction methods.
* **Build:** Develop `app/document_parser.py` class reading raw PDFs.
* **Expected Output:** Script printing raw text from an uploaded PDF.

#### Day 3
* **Learn:** Document chunking theories, boundaries overlap parameters.
* **Build:** Implement recursive character splitters in `app/document_parser.py`.
* **Expected Output:** Clean chunks list printed with page number logs.

#### Day 4
* **Learn:** Vector embeddings mathematics.
* **Build:** Build an embedding pipeline utilizing local Sentence Transformers.
* **Expected Output:** Test script converting sentences to 384-dimension vector listings.

#### Day 5
* **Learn:** Vector index databases, connection management.
* **Build:** Develop `app/vector_manager.py` initializing local storage and writing indices.
* **Expected Output:** FAISS collection folders created on disk.

#### Day 6
* **Learn:** Semantic similarity searches, cosine distances.
* **Build:** Write similarity search functions inside `app/rag_engine.py`.
* **Expected Output:** Querying returns matching text segments containing page references.

#### Day 7
* **Learn:** Prompt design, context injection.
* **Build:** Construct conversational prompt templates and query Gemini.
* **Expected Output:** Detailed, citation-backed conversational responses.

#### Day 8
* **Learn:** SQLite schema design, queries execution.
* **Build:** Develop SQLite log databases recording Q&A sessions history.
* **Expected Output:** Database schemas configured with sample records.

#### Day 9
* **Learn:** Streamlit chat layouts, components states.
* **Build:** Draft chat UI views in `app/main.py` using conversational widgets.
* **Expected Output:** Interface showing message threads.

#### Day 10
* **Learn:** Component integrations, data connectors.
* **Build:** Connect UI grids, SQLite database logs, and API routes.
* **Expected Output:** Dashboard updating chat responses upon file upload.

#### Day 11
* **Learn:** Collapsible widgets, metrics cards.
* **Build:** Add collapsible text expanders displaying source chunks.
* **Expected Output:** Dashboard displaying visual citation panels.

#### Day 12
* **Learn:** Performance profiling, data caching.
* **Build:** Add caching decorator syntax to data loaders.
* **Expected Output:** Responsive UI and instant dashboard rendering.

#### Day 13
* **Learn:** Pytest mock structures.
* **Build:** Develop tests inside `tests/` verifying data calculations.
* **Expected Output:** Test runners passing all test cases.

#### Day 14
* **Learn:** README documentation standards.
* **Build:** Write `README.md` and complete inline code comments.
* **Expected Output:** Finished README explaining installation and design features.

#### Day 15
* **Learn:** Presentation design, delivery.
* **Build:** Design presentation slides detailing system architecture and milestones.
* **Expected Output:** Final presentation slide deck.

---

## 18. Final Deliverables

For evaluation, each intern must submit:
1. **Source Code:** Complete Python project directories including index pipelines.
2. **GitHub Repository:** Clean history of version control commits.
3. **README.md:** Explaining project context, setup instructions, and database details.
4. **Local Index Files:** Saved FAISS index files.
5. **Live Dashboard Link:** Deployed Streamlit Cloud URL (optional).
6. **Project Report:** Standard PDF detailing findings, chunking analysis, and similarity search tests.
7. **Demo Video:** A 3-minute video showing PDF uploading, Q&A chats, and collapsible citation displays.
8. **Presentation Slides:** Summary slides.

---

## 19. Evaluation Rubric (100 Marks)

| Criteria | Marks | Details |
|---|---|---|
| **Working Features** | **20** | PDF uploading works, FAISS index creates, Q&A answers match context, and citations render. |
| **Code Quality** | **15** | PEP 8 styling, cache configurations, and exception handlers. |
| **AI Implementation** | **15** | Chunking strategy efficacy, FAISS similarity search, and prompt validation. |
| **Documentation** | **10** | Detailed README.md, clean setup instructions, and PDF report. |
| **Git Usage** | **10** | Descriptive commits list, branch management, and repository hygiene. |
| **UI/UX Design** | **10** | Modern widgets layout, visual progress gauges, and clear metrics cards. |
| **Innovation & Extensions** | **10** | Multi-file support, OCR pipeline, or custom themes. |
| **Testing** | **5** | Pytest suites verifying the parser and similarity scoring. |
| **Presentation & Demo** | **5** | Video recording and slides explaining project value and findings. |
