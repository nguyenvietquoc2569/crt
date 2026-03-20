"""
Export all research documentation to a single Word document.

Usage:
    python3 tools/export_to_docx.py --output docs/PersonaPlex_Vietnamese_Research.docx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE


def setup_styles(doc: Document):
    """Configure document styles."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    # Code style
    if "Code" not in [s.name for s in doc.styles]:
        code_style = doc.styles.add_style("Code", WD_STYLE_TYPE.PARAGRAPH)
        code_font = code_style.font
        code_font.name = "Consolas"
        code_font.size = Pt(9)
        code_font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
        code_format = code_style.paragraph_format
        code_format.space_before = Pt(4)
        code_format.space_after = Pt(4)
        code_format.left_indent = Inches(0.3)

    if "CodeInline" not in [s.name for s in doc.styles]:
        ci = doc.styles.add_style("CodeInline", WD_STYLE_TYPE.CHARACTER)
        ci.font.name = "Consolas"
        ci.font.size = Pt(9.5)
        ci.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)


def add_markdown_to_doc(doc: Document, md_text: str):
    """Parse markdown and add formatted content to Word document."""
    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_buffer = []

    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                # End code block
                code_text = "\n".join(code_buffer)
                if code_text.strip():
                    p = doc.add_paragraph(style="Code")
                    p.paragraph_format.left_indent = Inches(0.3)
                    run = p.add_run(code_text)
                    run.font.name = "Consolas"
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x2E)
                code_buffer = []
                in_code_block = False
            else:
                in_code_block = True
                code_buffer = []
            i += 1
            continue

        if in_code_block:
            code_buffer.append(line)
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
            i += 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run("─" * 80)
            run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
            run.font.size = Pt(8)
            i += 1
            continue

        # Tables
        if "|" in line and i + 1 < len(lines) and "---" in lines[i + 1]:
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1

            # Parse table
            headers = [c.strip() for c in table_lines[0].split("|") if c.strip()]
            rows = []
            for tl in table_lines[2:]:  # Skip header separator
                cells = [c.strip() for c in tl.split("|") if c.strip()]
                if cells:
                    rows.append(cells)

            if headers and rows:
                num_cols = len(headers)
                table = doc.add_table(rows=1 + len(rows), cols=num_cols)
                table.style = "Light Grid Accent 1"

                # Headers
                for j, h in enumerate(headers):
                    if j < num_cols:
                        cell = table.rows[0].cells[j]
                        cell.text = h
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.bold = True
                                run.font.size = Pt(9)

                # Data rows
                for r_idx, row_data in enumerate(rows):
                    for c_idx, cell_text in enumerate(row_data):
                        if c_idx < num_cols:
                            cell = table.rows[r_idx + 1].cells[c_idx]
                            cell.text = cell_text
                            for paragraph in cell.paragraphs:
                                for run in paragraph.runs:
                                    run.font.size = Pt(9)

                doc.add_paragraph()  # spacing after table
            continue

        # Bullet points
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            text = line.strip()[2:]
            p = doc.add_paragraph(style="List Bullet")
            _add_formatted_text(p, text)
            i += 1
            continue

        # Numbered list
        numbered = re.match(r"^\s*(\d+)\.\s+(.*)", line)
        if numbered:
            text = numbered.group(2)
            p = doc.add_paragraph(style="List Number")
            _add_formatted_text(p, text)
            i += 1
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Regular paragraph
        p = doc.add_paragraph()
        _add_formatted_text(p, line)
        i += 1


def _add_formatted_text(paragraph, text: str):
    """Add text with inline formatting (bold, italic, code)."""
    parts = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0xC7, 0x25, 0x4E)
        else:
            paragraph.add_run(part)


def main():
    parser = argparse.ArgumentParser(description="Export docs to Word")
    parser.add_argument("--output", default="docs/PersonaPlex_Vietnamese_Research.docx")
    parser.add_argument("--docs-dir", default="docs/")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    output_path = Path(args.output)

    doc = Document()
    setup_styles(doc)

    # Title page
    title = doc.add_heading("Vietnamese PersonaPlex", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(
        "Fine-tuning Full-Duplex Conversational Speech AI for Vietnamese\n"
        "Research Documentation"
    )
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run(
        "\nBased on NVIDIA PersonaPlex (Moshi Architecture + Mimi Codec)\n"
        "Date: March 2026\n"
    )
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    doc.add_page_break()

    # Table of contents placeholder
    doc.add_heading("Table of Contents", level=1)
    toc = doc.add_paragraph()
    toc.add_run(
        "1. Architecture Overview\n"
        "2. Training Data Deep Dive\n"
        "3. Training Data Generation Guide\n"
        "4. Fisher Corpus & Vietnamese Equivalents\n"
        "5. Mimi Voice Embedding Deep Dive\n"
    )
    doc.add_page_break()

    # Process each markdown file in order
    md_files = sorted(docs_dir.glob("*.md"))
    for md_file in md_files:
        print(f"Processing: {md_file.name}")
        md_content = md_file.read_text(encoding="utf-8")
        add_markdown_to_doc(doc, md_content)
        doc.add_page_break()

    # References section
    doc.add_heading("References", level=1)
    refs = [
        "PersonaPlex Paper: https://arxiv.org/abs/2602.06053",
        "PersonaPlex GitHub: https://github.com/NVIDIA/personaplex",
        "PersonaPlex Weights: https://huggingface.co/nvidia/personaplex-7b-v1",
        "Moshi Paper: https://arxiv.org/abs/2410.00037",
        "Moshi GitHub: https://github.com/kyutai-labs/moshi",
        "Mimi Model Card: https://huggingface.co/kyutai/mimi",
        "Kyutai Codec Explainer: https://kyutai.org/codec-explainer",
        "Fisher Corpus Part 1: https://catalog.ldc.upenn.edu/LDC2004S13",
        "VLSP Resources: https://vlsp.org.vn/resources",
        "VIVOS Dataset: https://huggingface.co/datasets/vivos",
        "Chatterbox TTS: https://github.com/resemble-ai/chatterbox",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(ref).font.size = Pt(10)

    doc.save(str(output_path))
    print(f"\nWord document saved to: {output_path}")
    print(f"Total pages: ~{len(md_files) * 8} (estimated)")


if __name__ == "__main__":
    main()
