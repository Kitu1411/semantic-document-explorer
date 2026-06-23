# Streamlit UI Dashboard (Milestone 5)
import streamlit as st
import os
import sys

# Add project root to sys.path to allow importing app modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.document_parser import DocumentParser


def main():
    st.set_page_config(
        page_title="Semantic Document Explorer & RAG Q&A",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Styling helper for headers and subheaders
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }
        .subheader {
            font-size: 1.25rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .status-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.85rem;
            font-weight: 600;
            background-color: #d1fae5;
            color: #065f46;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("🛠️ Project Controls")
    
    st.sidebar.markdown("### Development Milestones")
    st.sidebar.markdown(
        """
        - 🟢 **Day 1:** Project Scaffold & Environment Setup
        - 🟢 **Day 2:** PyMuPDF PDF Text & Layout Parser
        - 🟢 **Day 3:** LangChain Recursive Text Chunking
        - ⚪ **Day 4-15:** Vector embeddings & RAG Q&A (Coming Next)
        """
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Chunking Settings")
    chunk_size = st.sidebar.slider(
        "Max Chunk Size (chars)", 
        min_value=200, 
        max_value=3000, 
        value=1000, 
        step=100,
        help="Maximum characters in each text segment."
    )
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap (chars)", 
        min_value=0, 
        max_value=1000, 
        value=200, 
        step=50,
        help="Number of overlapping characters between adjacent chunks."
    )
    
    if chunk_overlap >= chunk_size:
        st.sidebar.warning("Overlap should be smaller than Chunk Size!")
        chunk_overlap = chunk_size - 50

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Document Source")
    source_option = st.sidebar.radio(
        "Select Document Source",
        options=["Use Sample Financial Report", "Upload a Custom PDF"]
    )

    # Handle PDF source selection
    pdf_path = None
    uploaded_file = None

    if source_option == "Upload a Custom PDF":
        uploaded_file = st.sidebar.file_uploader("Upload a PDF Document", type=["pdf"])
        if uploaded_file is not None:
            # Save uploaded file to 'uploads/' directory which is ignored by git
            uploads_dir = os.path.join(project_root, "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            pdf_path = os.path.join(uploads_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            st.sidebar.info("Please upload a PDF file to begin.")
    else:
        # Default to the sample report
        pdf_path = os.path.join(project_root, "tests", "sample_report.pdf")
        # Generate sample PDF if it doesn't exist
        if not os.path.exists(pdf_path):
            try:
                from tests.create_sample_pdf import create_sample_pdf
                create_sample_pdf(pdf_path)
            except Exception as e:
                st.error(f"Error generating sample PDF: {e}")

    # Header section
    st.markdown('<div class="main-header">📄 Semantic Document Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subheader">Day 2 & Day 3 Demonstration Dashboard (PDF Parsing & Chunking Engine)</div>', 
        unsafe_allow_html=True
    )

    if pdf_path is None or not os.path.exists(pdf_path):
        st.warning("⚠️ No document loaded. Please upload a PDF or select the sample report from the sidebar.")
        return

    # Initialize parser
    parser = DocumentParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Ingestion Pipeline
    try:
        with st.spinner("Processing PDF Document..."):
            parsed_doc = parser.parse_pdf(pdf_path)
            layout_data = parser.extract_layout_properties(pdf_path)
            tables_data = parser.extract_tables_text(pdf_path)
            chunks = parser.chunk_text(parsed_doc)
            chunk_stats = parser.get_chunk_statistics(chunks)
    except Exception as e:
        st.error(f"Failed to parse the PDF document: {e}")
        return

    # Performance / Ingestion overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pages", parsed_doc["total_pages"])
    with col2:
        st.metric("File Size (KB)", f"{parsed_doc['file_size_kb']} KB")
    with col3:
        st.metric("Total Chunks Generated", chunk_stats["total_chunks"])
    with col4:
        st.metric("Avg Chunk Size", f"{chunk_stats['avg_chunk_size']} chars")

    st.markdown("---")

    # Document Level Metadata Display
    with st.expander("📁 Document Metadata"):
        m_col1, m_col2 = st.columns(2)
        meta = parsed_doc["metadata"]
        with m_col1:
            st.write(f"**Filename:** {parsed_doc['file_name']}")
            st.write(f"**Title:** {meta.get('title') or 'N/A'}")
            st.write(f"**Author:** {meta.get('author') or 'N/A'}")
        with m_col2:
            st.write(f"**Subject:** {meta.get('subject') or 'N/A'}")
            st.write(f"**Producer:** {meta.get('producer') or 'N/A'}")
            st.write(f"**Location:** `{parsed_doc['file_path']}`")

    # Tabs for different pipeline outputs
    tab_text, tab_layout, tab_tables, tab_chunks, tab_analytics = st.tabs([
        "📄 Extracted Text", 
        "📐 Layout Blocks", 
        "📊 Extracted Tables", 
        "🧩 Semantic Chunks",
        "📈 Chunking Analytics"
    ])

    # Interactive Page Selector for Page-by-Page viewing
    selected_page_num = st.selectbox(
        "Select Page to Inspect", 
        options=range(1, parsed_doc["total_pages"] + 1),
        index=0
    )
    page_data = parsed_doc["pages"][selected_page_num - 1]
    page_layout = layout_data[selected_page_num - 1]
    page_tables = tables_data[selected_page_num - 1]

    # Tab 1: Extracted Text
    with tab_text:
        st.subheader(f"Cleaned Text Output — Page {selected_page_num}")
        st.caption(
            f"Characters: {page_data['char_count']} | Words: {page_data['word_count']} | Images: {page_data['image_count']}"
        )
        st.text_area(
            "Plain Text Preview", 
            value=page_data["clean_text"], 
            height=400, 
            disabled=True
        )

    # Tab 2: Layout & Fonts
    with tab_layout:
        st.subheader(f"Page Boundaries & Coordinates — Page {selected_page_num}")
        st.write(f"**Dimensions:** {page_layout['width']} x {page_layout['height']} points")
        st.write(f"**Fonts Used:** {', '.join(page_layout['fonts_used']) or 'None'}")
        
        st.markdown("**Positioned Text Blocks:**")
        block_list = []
        for b in page_layout["text_blocks"]:
            block_list.append({
                "Block #": b["block_number"],
                "Top-Left (x0, y0)": f"({b['x0']}, {b['y0']})",
                "Bottom-Right (x1, y1)": f"({b['x1']}, {b['y1']})",
                "Snippet Preview": b["text"][:80] + "..." if len(b["text"]) > 80 else b["text"]
            })
        if block_list:
            st.dataframe(block_list, use_container_width=True)
        else:
            st.info("No text blocks detected on this page.")

    # Tab 3: Extracted Tables
    with tab_tables:
        st.subheader(f"Table Extraction (PyMuPDF find_tables) — Page {selected_page_num}")
        tables = page_tables["tables"]
        if tables:
            st.success(f"{len(tables)} table(s) detected on Page {selected_page_num}:")
            for idx, table in enumerate(tables):
                st.markdown(f"**Table {idx + 1}:**")
                st.dataframe(table, use_container_width=True)
        else:
            st.info(f"No tables detected on Page {selected_page_num}. Note: PyMuPDF detects tables by looking for explicit cell lines/borders.")

    # Tab 4: Semantic Chunks
    with tab_chunks:
        st.subheader("Generated Text Chunks")
        st.caption("Splits are generated sequentially and retain page numbers for reference in future milestones.")
        
        # Filter chunks by selected page
        page_chunks = [c for c in chunks if c["page_number"] == selected_page_num]
        
        st.markdown(f"**Showing {len(page_chunks)} chunks originating from Page {selected_page_num}:**")
        
        for c in page_chunks:
            with st.container(border=True):
                st.markdown(f"**Chunk ID: {c['chunk_id']}** (Page {c['page_number']})")
                st.caption(f"Characters: {c['char_count']} | Words: {c['word_count']}")
                st.text(c["text"])
                st.json(c["metadata"])

    # Tab 5: Chunking Analytics
    with tab_analytics:
        st.subheader("Chunking Engine Statistics")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Chunk Statistics Summary**")
            st.write(f"- **Total Chunks Created:** {chunk_stats['total_chunks']}")
            st.write(f"- **Min Chunk Size:** {chunk_stats['min_chunk_size']} chars")
            st.write(f"- **Max Chunk Size:** {chunk_stats['max_chunk_size']} chars")
            st.write(f"- **Avg Chunk Size:** {chunk_stats['avg_chunk_size']} chars")
            st.write(f"- **Total Characters Processed:** {chunk_stats['total_characters']} chars")
            st.write(f"- **Pages Covered:** {', '.join(map(str, chunk_stats['pages_covered']))}")
        
        with col_s2:
            st.markdown("**Chunks Distribution per Page**")
            for page, count in chunk_stats["chunks_per_page"].items():
                st.write(f"Page {page}: " + "🟩" * count + f" ({count} chunks)")
        
        st.markdown("---")
        st.markdown("**Interactive Parameter Comparison**")
        st.caption("Compare how other configurations would split this same document:")
        
        comp_sizes = [(500, 100), (1000, 200), (2000, 400)]
        comp_data = []
        for size, overlap in comp_sizes:
            test_chunks = parser.chunk_text(parsed_doc, chunk_size=size, chunk_overlap=overlap)
            test_stats = parser.get_chunk_statistics(test_chunks)
            comp_data.append({
                "Target Size": size,
                "Target Overlap": overlap,
                "Total Chunks": test_stats["total_chunks"],
                "Avg Chunk Size (chars)": test_stats["avg_chunk_size"],
                "Min Chunk Size": test_stats["min_chunk_size"],
                "Max Chunk Size": test_stats["max_chunk_size"]
            })
        st.table(comp_data)


if __name__ == "__main__":
    main()

