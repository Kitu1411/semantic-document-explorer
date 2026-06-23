"""
Generate a sample multi-page PDF for testing the DocumentParser.
Creates a realistic document with headings, paragraphs, tables, and metadata.
"""

import fitz  # PyMuPDF
import os


def create_sample_pdf(output_path: str = "tests/sample_report.pdf"):
    """Create a multi-page sample PDF with varied content for testing."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = fitz.open()

    # Set document metadata
    doc.set_metadata({
        "title": "Quarterly Financial Report Q4 2025",
        "author": "Analytics Division",
        "subject": "Financial Performance Summary",
        "creator": "Semantic Document Explorer Test Suite",
    })

    # ── Page 1: Title Page & Executive Summary ──
    page1 = doc.new_page(width=595, height=842)  # A4 size
    page1.insert_text(
        (72, 120),
        "Quarterly Financial Report",
        fontsize=24,
        fontname="helv",
        color=(0.1, 0.2, 0.5),
    )
    page1.insert_text(
        (72, 155),
        "Q4 2025 — TechNova Corporation",
        fontsize=14,
        fontname="helv",
        color=(0.3, 0.3, 0.3),
    )
    page1.insert_text(
        (72, 200),
        "Executive Summary",
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.2, 0.5),
    )

    exec_summary = (
        "TechNova Corporation delivered strong financial performance in Q4 2025, "
        "with total revenue reaching $4.2 billion, representing a 15% year-over-year "
        "increase. Operating income grew to $890 million, driven by robust demand in "
        "our cloud computing and artificial intelligence segments. Net income for the "
        "quarter was $720 million, or $3.45 per diluted share, compared to $610 million, "
        "or $2.92 per diluted share, in the prior year period.\n\n"
        "Key highlights for the quarter include the successful launch of our next-generation "
        "AI platform, NovaMind 3.0, which has already been adopted by over 2,500 enterprise "
        "customers. Our cloud infrastructure segment grew 28% year-over-year, establishing "
        "TechNova as a leading player in the hybrid cloud market. International revenue "
        "accounted for 42% of total revenue, up from 38% in Q4 2024.\n\n"
        "Looking ahead, we expect continued momentum in fiscal year 2026, with projected "
        "revenue growth of 12-18%. Capital expenditure is planned at $1.8 billion to support "
        "expansion of data center capacity and R&D initiatives in quantum computing."
    )

    # Insert wrapped text
    rect1 = fitz.Rect(72, 225, 523, 700)
    page1.insert_textbox(rect1, exec_summary, fontsize=10, fontname="helv")

    # ── Page 2: Revenue Breakdown ──
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (72, 72),
        "Revenue Breakdown by Segment",
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.2, 0.5),
    )

    revenue_text = (
        "The following table summarizes revenue by business segment for Q4 2025 "
        "compared to Q4 2024:\n\n"
        "Segment                    Q4 2025        Q4 2024        Change\n"
        "─────────────────────────────────────────────────────────────────\n"
        "Cloud Computing            $1,680M        $1,312M        +28.0%\n"
        "AI & Machine Learning      $1,050M        $830M          +26.5%\n"
        "Enterprise Software        $840M          $790M          +6.3%\n"
        "Professional Services      $420M          $410M          +2.4%\n"
        "Hardware Solutions          $210M          $310M          -32.3%\n"
        "─────────────────────────────────────────────────────────────────\n"
        "Total Revenue              $4,200M        $3,652M        +15.0%\n\n"
        "Cloud Computing remains our fastest-growing segment, driven by increased adoption "
        "of our NovaCumulus platform. The AI & Machine Learning segment benefited from "
        "the launch of NovaMind 3.0 and growing demand for generative AI solutions. "
        "Hardware Solutions experienced a planned decline as we transition customers to "
        "cloud-based offerings.\n\n"
        "Geographic distribution of revenue showed strong international growth. North "
        "America contributed $2,436M (58%), Europe contributed $924M (22%), Asia-Pacific "
        "contributed $630M (15%), and Rest of World contributed $210M (5%). The European "
        "market showed the strongest regional growth at 22% year-over-year, driven by "
        "increased regulatory demand for on-premises AI solutions."
    )

    rect2 = fitz.Rect(72, 95, 523, 750)
    page2.insert_textbox(rect2, revenue_text, fontsize=10, fontname="helv")

    # ── Page 3: R&D and Future Outlook ──
    page3 = doc.new_page(width=595, height=842)
    page3.insert_text(
        (72, 72),
        "Research & Development",
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.2, 0.5),
    )

    rd_text = (
        "R&D expenditure for Q4 2025 totaled $580 million, representing 13.8% of total "
        "revenue. This investment was allocated across three primary initiatives:\n\n"
        "1. Quantum Computing Research ($180M): Our quantum computing lab in Boulder, "
        "Colorado, achieved a breakthrough in error correction algorithms, reducing quantum "
        "bit error rates by 40%. We expect to deliver a commercially viable quantum "
        "processor by late 2027.\n\n"
        "2. NovaMind AI Platform ($250M): Development of NovaMind 4.0 is underway, "
        "featuring multimodal capabilities including text, image, and video understanding. "
        "Early benchmarks show a 35% improvement in reasoning tasks compared to version 3.0. "
        "The model architecture uses a novel mixture-of-experts approach with 1.2 trillion "
        "parameters.\n\n"
        "3. Cybersecurity Division ($150M): Launch of NovaShield, an AI-powered threat "
        "detection system that analyzes network traffic patterns in real-time. Initial "
        "deployments have shown a 94% detection rate for zero-day vulnerabilities.\n\n"
    )

    rect3a = fitz.Rect(72, 95, 523, 450)
    page3.insert_textbox(rect3a, rd_text, fontsize=10, fontname="helv")

    page3.insert_text(
        (72, 470),
        "Future Outlook & Risk Factors",
        fontsize=16,
        fontname="helv",
        color=(0.1, 0.2, 0.5),
    )

    outlook_text = (
        "Management projects fiscal year 2026 revenue between $17.5 billion and $19.2 "
        "billion. Key risk factors include:\n\n"
        "- Intensifying competition in the AI infrastructure market from established "
        "players and well-funded startups.\n"
        "- Regulatory uncertainties around AI governance in the European Union and "
        "potential export restrictions on advanced chips.\n"
        "- Macroeconomic headwinds including elevated interest rates and currency "
        "fluctuations in key international markets.\n"
        "- Supply chain constraints for specialized GPU and TPU hardware components.\n\n"
        "Despite these challenges, management remains confident in the company's strategic "
        "positioning and competitive advantages in hybrid cloud and enterprise AI solutions. "
        "The Board of Directors has approved a $2 billion share repurchase program and "
        "declared a quarterly dividend of $0.85 per share."
    )

    rect3b = fitz.Rect(72, 495, 523, 780)
    page3.insert_textbox(rect3b, outlook_text, fontsize=10, fontname="helv")

    # Save the document
    doc.save(output_path)
    doc.close()
    print(f"Sample PDF created: {output_path}")
    return output_path


if __name__ == "__main__":
    create_sample_pdf()
