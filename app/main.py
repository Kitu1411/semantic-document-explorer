# Streamlit UI Dashboard (Module 6)
# Complete end-to-end application with Q&A chat, document viewer, and analytics

import streamlit as st
import os
import sys
import time

# Add project root to sys.path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_parser import DocumentParser
from app.vector_manager import VectorManager
from app.rag_engine import RAGEngine


# ──────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────

def setup_page():
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(
        page_title="Semantic Document Explorer & RAG Q&A",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        /* ── Global Theme ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        .stApp {
            font-family: 'Inter', sans-serif;
        }

        /* ── Header Styling ── */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }
        .subheader {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        /* ── Status Badges ── */
        .status-ready {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            background: linear-gradient(135deg, #d1fae5, #a7f3d0);
            color: #065f46;
        }
        .status-pending {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            background: linear-gradient(135deg, #fef3c7, #fde68a);
            color: #92400e;
        }
        .status-error {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 600;
            background: linear-gradient(135deg, #fecaca, #fca5a5);
            color: #991b1b;
        }

        /* ── Chat Styling ── */
        .chat-user {
            background: linear-gradient(135deg, #ede9fe, #ddd6fe);
            border-radius: 1rem;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #7c3aed;
        }
        .chat-assistant {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 1rem;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #0284c7;
        }

        /* ── Citation Block ── */
        .citation-block {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            padding: 0.6rem 0.8rem;
            margin: 0.3rem 0;
            font-size: 0.85rem;
        }
        .citation-header {
            font-weight: 600;
            color: #475569;
            font-size: 0.8rem;
        }

        /* ── Metric Cards ── */
        .metric-card {
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
            border: 1px solid #e2e8f0;
            border-radius: 0.75rem;
            padding: 1rem;
            text-align: center;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.2rem;
        }

        /* ── Pipeline Stage ── */
        .pipeline-stage {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0;
            font-size: 0.9rem;
        }

        /* ── Sidebar Tweaks ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Pipeline state
        "parsed_doc": None,
        "chunks": None,
        "chunk_stats": None,
        "layout_data": None,
        "tables_data": None,
        "index_built": False,
        "pdf_path": None,
        "pdf_name": None,
        # Module instances (cached)
        "vector_manager": None,
        "rag_engine": None,
        # Chat history
        "chat_history": [],
        "query_log": [],
        # UI state
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with upload, settings, and pipeline status."""
    st.sidebar.markdown(
        '<div style="font-size:1.4rem;font-weight:700;'
        'background:linear-gradient(135deg,#667eea,#764ba2);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '📄 Document Explorer</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption("AI-powered document Q&A with RAG")

    st.sidebar.markdown("---")

    # ── Document Upload ──
    st.sidebar.markdown("### 📁 Document Source")
    source_option = st.sidebar.radio(
        "Select source",
        options=["Upload a Custom PDF", "Use Sample Financial Report"],
        label_visibility="collapsed",
    )

    pdf_path = None

    if source_option == "Upload a Custom PDF":
        uploaded_file = st.sidebar.file_uploader(
            "Upload PDF", type=["pdf"], label_visibility="collapsed"
        )
        if uploaded_file is not None:
            uploads_dir = os.path.join(project_root, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            pdf_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.sidebar.info("Upload a PDF to begin.")
    else:
        pdf_path = os.path.join(project_root, "tests", "sample_report.pdf")
        if not os.path.exists(pdf_path):
            try:
                from tests.create_sample_pdf import create_sample_pdf
                create_sample_pdf(pdf_path)
            except Exception as e:
                st.sidebar.error(f"Sample PDF error: {e}")

    # Detect new document upload
    if pdf_path and pdf_path != st.session_state.get("pdf_path"):
        st.session_state["pdf_path"] = pdf_path
        st.session_state["pdf_name"] = os.path.basename(pdf_path)
        # Reset pipeline state for new document
        st.session_state["parsed_doc"] = None
        st.session_state["chunks"] = None
        st.session_state["chunk_stats"] = None
        st.session_state["layout_data"] = None
        st.session_state["tables_data"] = None
        st.session_state["index_built"] = False
        st.session_state["chat_history"] = []
        st.session_state["query_log"] = []

    st.sidebar.markdown("---")

    # ── Chunking Settings ──
    st.sidebar.markdown("### ⚙️ Chunking Settings")
    chunk_size = st.sidebar.slider(
        "Max Chunk Size (chars)",
        min_value=200, max_value=3000, value=1000, step=100,
        help="Maximum characters per text chunk.",
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (chars)",
        min_value=0, max_value=1000, value=200, step=50,
        help="Overlapping characters between adjacent chunks.",
    )
    if chunk_overlap >= chunk_size:
        st.sidebar.warning("Overlap must be smaller than chunk size!")
        chunk_overlap = chunk_size - 50

    st.sidebar.markdown("---")

    # ── Retrieval Settings ──
    st.sidebar.markdown("### 🔍 Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value=1, max_value=10, value=4,
        help="Number of context chunks to retrieve per query.",
    )
    score_threshold = st.sidebar.slider(
        "Distance Threshold",
        min_value=0.5, max_value=2.0, value=1.5, step=0.1,
        help="Max L2 distance (lower = stricter matching).",
    )

    st.sidebar.markdown("---")

    # ── Pipeline Status ──
    st.sidebar.markdown("### 🔄 Pipeline Status")
    _render_pipeline_status()

    # ── API Status ──
    st.sidebar.markdown("---")
    rag = _get_rag_engine()
    if rag.is_api_configured():
        st.sidebar.markdown(
            '<div class="status-ready">🔑 Gemini API: Configured</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="status-error">🔑 Gemini API: Not Set</div>',
            unsafe_allow_html=True,
        )
        st.sidebar.caption("Set `GEMINI_API_KEY` in `config/.env`")

    return chunk_size, chunk_overlap, top_k, score_threshold


def _render_pipeline_status():
    """Show pipeline stage indicators in sidebar."""
    stages = [
        ("📄 PDF Loaded", st.session_state["parsed_doc"] is not None),
        ("🧩 Text Chunked", st.session_state["chunks"] is not None),
        ("🗄️ Index Built", st.session_state["index_built"]),
        ("🔑 API Ready", _get_rag_engine().is_api_configured()),
    ]
    for label, done in stages:
        icon = "✅" if done else "⬜"
        st.sidebar.markdown(f"{icon} {label}")


# ──────────────────────────────────────────────
# Cached Module Instances
# ──────────────────────────────────────────────

def _get_vector_manager() -> VectorManager:
    """Get or create the VectorManager singleton."""
    if st.session_state["vector_manager"] is None:
        st.session_state["vector_manager"] = VectorManager()
    return st.session_state["vector_manager"]


def _get_rag_engine() -> RAGEngine:
    """Get or create the RAGEngine singleton."""
    if st.session_state["rag_engine"] is None:
        st.session_state["rag_engine"] = RAGEngine()
    return st.session_state["rag_engine"]


# ──────────────────────────────────────────────
# Processing Pipeline
# ──────────────────────────────────────────────

def run_ingestion_pipeline(chunk_size: int, chunk_overlap: int):
    """Parse PDF, chunk text, build FAISS index."""
    pdf_path = st.session_state["pdf_path"]
    if not pdf_path or not os.path.exists(pdf_path):
        st.warning("⚠️ No document loaded.")
        return

    parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Step 1: Parse PDF
    with st.spinner("📄 Parsing PDF document..."):
        parsed_doc = parser.parse_pdf(pdf_path)
        layout_data = parser.extract_layout_properties(pdf_path)
        tables_data = parser.extract_tables_text(pdf_path)
        st.session_state["parsed_doc"] = parsed_doc
        st.session_state["layout_data"] = layout_data
        st.session_state["tables_data"] = tables_data

    # Step 2: Chunk text
    with st.spinner("🧩 Chunking text with overlap..."):
        chunks = parser.chunk_text(parsed_doc)
        chunk_stats = parser.get_chunk_statistics(chunks)
        st.session_state["chunks"] = chunks
        st.session_state["chunk_stats"] = chunk_stats

    # Step 3: Build FAISS index
    with st.spinner("🗄️ Building FAISS vector index (this may take a moment)..."):
        vm = _get_vector_manager()
        vm.build_index(chunks)
        vm.save_index()

        # Connect vector store to RAG engine
        rag = _get_rag_engine()
        rag.set_vector_store(vm.get_vector_store())

        st.session_state["index_built"] = True

    st.success(
        f"✅ Pipeline complete! {parsed_doc['total_pages']} pages → "
        f"{chunk_stats['total_chunks']} chunks → FAISS index ready."
    )


# ──────────────────────────────────────────────
# Tab 1: Chat Q&A
# ──────────────────────────────────────────────

def render_chat_tab(top_k: int, score_threshold: float):
    """Render the conversational Q&A chat interface."""
    st.markdown("### 💬 Ask a Question About Your Document")

    if not st.session_state["index_built"]:
        st.info(
            "👆 Process a document first using the button above to enable Q&A."
        )
        return

    rag = _get_rag_engine()

    if not rag.is_api_configured():
        st.warning(
            "⚠️ Gemini API key not configured. "
            "Set `GEMINI_API_KEY` in `config/.env` to enable AI answers."
        )
        st.markdown("You can still test **context retrieval** below.")

    # Display chat history
    for entry in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(entry["query"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(entry["answer"])

            # Expandable citations
            if entry.get("context"):
                with st.expander(
                    f"📎 Sources ({len(entry['context'])} passages)",
                    expanded=False,
                ):
                    for i, ctx in enumerate(entry["context"], 1):
                        st.markdown(
                            f'<div class="citation-block">'
                            f'<div class="citation-header">'
                            f"Passage {i} · Page {ctx['page_number']} · "
                            f"Similarity: {ctx['similarity']:.0%}</div>"
                            f"<p>{ctx['text'][:500]}{'...' if len(ctx['text']) > 500 else ''}</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Response metrics
            if entry.get("response_time"):
                cols = st.columns(3)
                cols[0].caption(f"⏱️ {entry['response_time']}s")
                cols[1].caption(
                    f"📎 {len(entry.get('context', []))} chunks"
                )
                cols[2].caption(f"🤖 {entry.get('model', 'N/A')}")

    # Chat input
    user_query = st.chat_input(
        "Ask a question about the uploaded document...",
        key="chat_input",
    )

    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                result = rag.generate_answer(
                    query=user_query,
                    k=top_k,
                    score_threshold=score_threshold,
                )

            st.markdown(result["answer"])

            # Show citations
            if result["context"]:
                with st.expander(
                    f"📎 Sources ({len(result['context'])} passages)",
                    expanded=False,
                ):
                    for i, ctx in enumerate(result["context"], 1):
                        st.markdown(
                            f'<div class="citation-block">'
                            f'<div class="citation-header">'
                            f"Passage {i} · Page {ctx['page_number']} · "
                            f"Similarity: {ctx['similarity']:.0%}</div>"
                            f"<p>{ctx['text'][:500]}{'...' if len(ctx['text']) > 500 else ''}</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Metrics
            cols = st.columns(3)
            cols[0].caption(f"⏱️ {result['response_time']}s")
            cols[1].caption(f"📎 {len(result['context'])} chunks")
            cols[2].caption(f"🤖 {result['model']}")

        # Save to history
        st.session_state["chat_history"].append(result)
        st.session_state["query_log"].append({
            "query": user_query,
            "chunks_retrieved": len(result["context"]),
            "response_time": result["response_time"],
            "error": result.get("error"),
        })


# ──────────────────────────────────────────────
# Tab 2: Document Viewer
# ──────────────────────────────────────────────

def render_document_tab():
    """Render document parsing results with text, layout, tables, and chunks."""
    parsed_doc = st.session_state["parsed_doc"]
    layout_data = st.session_state["layout_data"]
    tables_data = st.session_state["tables_data"]
    chunks = st.session_state["chunks"]
    chunk_stats = st.session_state["chunk_stats"]

    if parsed_doc is None:
        st.info("📄 Process a document to view its contents here.")
        return

    # Document metadata
    with st.expander("📁 Document Metadata", expanded=False):
        m1, m2 = st.columns(2)
        meta = parsed_doc["metadata"]
        with m1:
            st.write(f"**Filename:** {parsed_doc['file_name']}")
            st.write(f"**Title:** {meta.get('title') or 'N/A'}")
            st.write(f"**Author:** {meta.get('author') or 'N/A'}")
        with m2:
            st.write(f"**Subject:** {meta.get('subject') or 'N/A'}")
            st.write(f"**Producer:** {meta.get('producer') or 'N/A'}")
            st.write(f"**Size:** {parsed_doc['file_size_kb']} KB")

    # Sub-tabs
    sub_text, sub_layout, sub_tables, sub_chunks = st.tabs([
        "📄 Extracted Text",
        "📐 Layout Blocks",
        "📊 Tables",
        "🧩 Chunks",
    ])

    # Page selector
    selected_page = st.selectbox(
        "Select Page",
        options=range(1, parsed_doc["total_pages"] + 1),
        index=0,
    )
    page_data = parsed_doc["pages"][selected_page - 1]
    page_layout = layout_data[selected_page - 1]
    page_tables = tables_data[selected_page - 1]

    with sub_text:
        st.subheader(f"Cleaned Text — Page {selected_page}")
        st.caption(
            f"Characters: {page_data['char_count']} | "
            f"Words: {page_data['word_count']} | "
            f"Images: {page_data['image_count']}"
        )
        st.text_area(
            "Text preview",
            value=page_data["clean_text"],
            height=400,
            disabled=True,
            label_visibility="collapsed",
        )

    with sub_layout:
        st.subheader(f"Layout Analysis — Page {selected_page}")
        st.write(
            f"**Dimensions:** {page_layout['width']} × "
            f"{page_layout['height']} points"
        )
        st.write(
            f"**Fonts:** {', '.join(page_layout['fonts_used']) or 'None'}"
        )
        blocks = []
        for b in page_layout["text_blocks"]:
            blocks.append({
                "Block #": b["block_number"],
                "Position": f"({b['x0']}, {b['y0']}) → ({b['x1']}, {b['y1']})",
                "Preview": (
                    b["text"][:80] + "..."
                    if len(b["text"]) > 80
                    else b["text"]
                ),
            })
        if blocks:
            st.dataframe(blocks, use_container_width=True)
        else:
            st.info("No text blocks detected.")

    with sub_tables:
        st.subheader(f"Tables — Page {selected_page}")
        tables = page_tables["tables"]
        if tables:
            st.success(f"{len(tables)} table(s) detected.")
            for idx, table in enumerate(tables):
                st.markdown(f"**Table {idx + 1}:**")
                st.dataframe(table, use_container_width=True)
        else:
            st.info("No tables detected on this page.")

    with sub_chunks:
        st.subheader("Text Chunks")
        page_chunks = [c for c in chunks if c["page_number"] == selected_page]
        st.markdown(
            f"**{len(page_chunks)} chunks from Page {selected_page}:**"
        )
        for c in page_chunks:
            with st.container(border=True):
                st.markdown(
                    f"**Chunk {c['chunk_id']}** · "
                    f"Page {c['page_number']} · "
                    f"{c['char_count']} chars"
                )
                st.text(c["text"])


# ──────────────────────────────────────────────
# Tab 3: Analytics
# ──────────────────────────────────────────────

def render_analytics_tab():
    """Render chunking statistics, index stats, and query history."""
    chunk_stats = st.session_state["chunk_stats"]
    parsed_doc = st.session_state["parsed_doc"]

    if chunk_stats is None:
        st.info("📊 Process a document to view analytics.")
        return

    st.markdown("### 📈 Pipeline Analytics")

    # Chunking stats
    st.markdown("#### Chunking Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunks", chunk_stats["total_chunks"])
    c2.metric("Avg Size", f"{chunk_stats['avg_chunk_size']} chars")
    c3.metric("Min Size", f"{chunk_stats['min_chunk_size']} chars")
    c4.metric("Max Size", f"{chunk_stats['max_chunk_size']} chars")

    # Chunks per page visualization
    st.markdown("#### Chunks Per Page")
    for page, count in chunk_stats["chunks_per_page"].items():
        bar = "🟩" * count
        st.markdown(f"Page {page}: {bar} ({count})")

    st.markdown("---")

    # FAISS index stats
    st.markdown("#### Vector Index Statistics")
    vm = _get_vector_manager()
    idx_stats = vm.get_index_stats()
    i1, i2, i3 = st.columns(3)
    i1.metric("Total Vectors", idx_stats["total_vectors"])
    i2.metric("Dimensions", idx_stats["embedding_dimension"])
    i3.metric("Model", idx_stats["model_name"])

    st.markdown("---")

    # Chunk size comparison
    st.markdown("#### Chunk Size Comparison")
    st.caption("How different configurations would split this document:")
    parser = DocumentParser()
    comp_data = []
    for size, overlap in [(500, 100), (1000, 200), (2000, 400)]:
        test_chunks = parser.chunk_text(
            parsed_doc, chunk_size=size, chunk_overlap=overlap
        )
        test_stats = parser.get_chunk_statistics(test_chunks)
        comp_data.append({
            "Chunk Size": size,
            "Overlap": overlap,
            "Total Chunks": test_stats["total_chunks"],
            "Avg Size": test_stats["avg_chunk_size"],
            "Min": test_stats["min_chunk_size"],
            "Max": test_stats["max_chunk_size"],
        })
    st.table(comp_data)

    st.markdown("---")

    # Query history
    st.markdown("#### 📝 Query History")
    query_log = st.session_state.get("query_log", [])
    if query_log:
        for i, entry in enumerate(reversed(query_log), 1):
            st.markdown(
                f"**{i}.** {entry['query']} — "
                f"{entry['chunks_retrieved']} chunks, "
                f"{entry['response_time']}s"
            )
    else:
        st.caption("No queries yet. Ask a question in the Chat tab!")


# ──────────────────────────────────────────────
# Main Application
# ──────────────────────────────────────────────

def main():
    setup_page()
    init_session_state()

    # Sidebar
    chunk_size, chunk_overlap, top_k, score_threshold = render_sidebar()

    # Header
    st.markdown(
        '<div class="main-header">📄 Semantic Document Explorer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subheader">'
        "AI-powered document Q&A with RAG, FAISS vector search, "
        "and Google Gemini"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Process Button ──
    pdf_path = st.session_state.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        # Show top metrics if document is processed
        if st.session_state["parsed_doc"] is not None:
            parsed_doc = st.session_state["parsed_doc"]
            chunk_stats = st.session_state["chunk_stats"]
            vm = _get_vector_manager()
            idx_stats = vm.get_index_stats()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📄 Pages", parsed_doc["total_pages"])
            m2.metric("📦 File Size", f"{parsed_doc['file_size_kb']} KB")
            m3.metric("🧩 Chunks", chunk_stats["total_chunks"])
            m4.metric("🗄️ Vectors", idx_stats["total_vectors"])
        else:
            st.markdown(
                f"📎 Document ready: **{os.path.basename(pdf_path)}**"
            )

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            if st.button(
                "🚀 Process & Index Document",
                type="primary",
                use_container_width=True,
            ):
                run_ingestion_pipeline(chunk_size, chunk_overlap)
                st.rerun()

        with col_status:
            if st.session_state["index_built"]:
                st.markdown(
                    '<span class="status-ready">✅ Ready for Q&A</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="status-pending">'
                    "⏳ Click Process to build index"
                    "</span>",
                    unsafe_allow_html=True,
                )
    else:
        st.warning(
            "⚠️ No document loaded. Upload a PDF or select the sample "
            "report from the sidebar."
        )
        return

    st.markdown("---")

    # ── Main Tabs ──
    tab_chat, tab_docs, tab_analytics = st.tabs([
        "💬 Chat Q&A",
        "📄 Document Viewer",
        "📊 Analytics",
    ])

    with tab_chat:
        render_chat_tab(top_k, score_threshold)

    with tab_docs:
        render_document_tab()

    with tab_analytics:
        render_analytics_tab()


if __name__ == "__main__":
    main()
