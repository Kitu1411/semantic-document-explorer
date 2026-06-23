"""
Day 2 & Day 3 Demo Script
==========================
Demonstrates the DocumentParser capabilities:
  - Day 2: PDF text extraction, layout properties, table detection
  - Day 3: Recursive text chunking with page metadata

Usage:
    .venv\\Scripts\\python demo_parser.py
    .venv\\Scripts\\python demo_parser.py path/to/your/file.pdf
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.document_parser import DocumentParser


def print_separator(title="", char="=", width=70):
    """Print a formatted section separator."""
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"\n{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def demo_pdf_parsing(parser, pdf_path):
    """Day 2: Demonstrate PDF text extraction."""

    print_separator("DAY 2: PDF TEXT PARSING")

    # Step 1: Parse the PDF
    print("\n[*] Parsing PDF file...")
    parsed = parser.parse_pdf(pdf_path)

    print(f"\n  File Name:    {parsed['file_name']}")
    print(f"  File Size:    {parsed['file_size_kb']} KB")
    print(f"  Total Pages:  {parsed['total_pages']}")

    # Step 2: Show document metadata
    print_separator("Document Metadata", "-")
    meta = parsed["metadata"]
    for key, value in meta.items():
        if value:
            print(f"  {key:20s}: {value}")

    # Step 3: Show per-page text extraction
    print_separator("Page-by-Page Extraction", "-")
    for page in parsed["pages"]:
        print(f"\n  [Page {page['page_number']}]")
        print(f"     Characters: {page['char_count']}")
        print(f"     Words:      {page['word_count']}")
        print(f"     Images:     {page['image_count']}")

        # Show first 200 chars of clean text
        preview = page["clean_text"][:200]
        if len(page["clean_text"]) > 200:
            preview += "..."
        print(f"     Preview:    {preview}")

    # Step 4: Layout properties
    print_separator("Layout Properties", "-")
    layout = parser.extract_layout_properties(pdf_path)
    for page_layout in layout:
        pg = page_layout["page_number"]
        w, h = page_layout["width"], page_layout["height"]
        blocks = len(page_layout["text_blocks"])
        fonts = page_layout["fonts_used"]
        print(f"  Page {pg}: {w}x{h} pts | {blocks} text blocks | Fonts: {', '.join(fonts)}")

    # Step 5: Table extraction
    print_separator("Table Detection", "-")
    tables = parser.extract_tables_text(pdf_path)
    for page_tables in tables:
        pg = page_tables["page_number"]
        num_tables = len(page_tables["tables"])
        if num_tables > 0:
            print(f"  Page {pg}: {num_tables} table(s) detected")
            for t_idx, table in enumerate(page_tables["tables"]):
                print(f"    Table {t_idx + 1}: {len(table)} rows")
                for row in table[:3]:
                    print(f"      {row}")
                if len(table) > 3:
                    print(f"      ... ({len(table) - 3} more rows)")
        else:
            print(f"  Page {pg}: No tables detected")

    return parsed


def demo_text_chunking(parser, parsed):
    """Day 3: Demonstrate recursive text chunking."""

    print_separator("DAY 3: TEXT CHUNKING")

    # Step 1: Generate chunks
    print(f"\n[*] Chunking with size={parser.chunk_size}, overlap={parser.chunk_overlap}")
    chunks = parser.chunk_text(parsed)

    # Step 2: Show chunk statistics
    stats = parser.get_chunk_statistics(chunks)
    print_separator("Chunk Statistics", "-")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Avg chunk size:   {stats['avg_chunk_size']} chars")
    print(f"  Min chunk size:   {stats['min_chunk_size']} chars")
    print(f"  Max chunk size:   {stats['max_chunk_size']} chars")
    print(f"  Total characters: {stats['total_characters']}")
    print(f"  Pages covered:    {stats['pages_covered']}")

    print_separator("Chunks per Page", "-")
    for page_num, count in stats["chunks_per_page"].items():
        bar = "#" * count
        print(f"  Page {page_num}: {bar} ({count} chunks)")

    # Step 3: Display all chunks
    print_separator("Chunk Details", "-")
    for chunk in chunks:
        print(f"\n  --- Chunk {chunk['chunk_id']} | Page {chunk['page_number']} | {chunk['char_count']} chars | {chunk['word_count']} words ---")
        text_preview = chunk["text"]
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."
        for line in text_preview.split("\n"):
            print(f"  | {line}")
        print(f"  {'-' * 60}")

    # Step 4: Test with different chunk sizes
    print_separator("Chunk Size Comparison", "-")
    for size, overlap in [(500, 100), (1000, 200), (2000, 400)]:
        test_chunks = parser.chunk_text(parsed, chunk_size=size, chunk_overlap=overlap)
        test_stats = parser.get_chunk_statistics(test_chunks)
        print(f"  Size={size:4d}, Overlap={overlap:3d} -> {test_stats['total_chunks']:3d} chunks "
              f"(avg {test_stats['avg_chunk_size']:.0f} chars)")


def main():
    # Determine PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Generate sample PDF if none provided
        pdf_path = "tests/sample_report.pdf"
        if not os.path.exists(pdf_path):
            print("No PDF provided. Generating sample PDF...")
            from tests.create_sample_pdf import create_sample_pdf
            create_sample_pdf(pdf_path)

    print(f"{'=' * 70}")
    print(f"  Semantic Document Explorer -- Parser Demo")
    print(f"  PDF: {pdf_path}")
    print(f"{'=' * 70}")

    # Initialize parser with default settings (1000 char chunks, 200 overlap)
    parser = DocumentParser(chunk_size=1000, chunk_overlap=200)

    # Run Day 2 demo: PDF parsing
    parsed = demo_pdf_parsing(parser, pdf_path)

    # Run Day 3 demo: Text chunking
    demo_text_chunking(parser, parsed)

    print_separator("DEMO COMPLETE")
    print("  [OK] Day 2: PDF text extraction -- WORKING")
    print("  [OK] Day 3: Recursive text chunking -- WORKING")
    print_separator()


if __name__ == "__main__":
    main()
