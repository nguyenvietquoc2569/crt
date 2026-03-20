"""
Generate a comprehensive Word document teaching Mimi neural audio codec.
Designed for beginners with extensive diagrams.
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

OUTPUT_PATH = "docs/Mimi_Neural_Audio_Codec_Lecture.docx"


def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_diagram_table(doc, lines, title=None, bg_color="F5F5F5"):
    """Add a monospace 'diagram' as a single-cell table with background."""
    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, bg_color)

    # Set cell margins
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        '  <w:top w:w="100" w:type="dxa"/>'
        '  <w:left w:w="150" w:type="dxa"/>'
        '  <w:bottom w:w="100" w:type="dxa"/>'
        '  <w:right w:w="150" w:type="dxa"/>'
        '</w:tcMar>'
    )
    tcPr.append(tcMar)

    # Add monospace text
    for i, line in enumerate(lines):
        if i == 0:
            para = cell.paragraphs[0]
        else:
            para = cell.add_paragraph()
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = Pt(13)
        run = para.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return table


def add_info_box(doc, text, box_type="info"):
    """Add a colored info/warning box."""
    colors = {
        "info": ("E8F4FD", "1565C0"),
        "tip": ("E8F5E9", "2E7D32"),
        "warn": ("FFF3E0", "E65100"),
        "key": ("F3E5F5", "6A1B9A"),
    }
    bg, fg = colors.get(box_type, colors["info"])

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, bg)

    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        '  <w:top w:w="120" w:type="dxa"/>'
        '  <w:left w:w="200" w:type="dxa"/>'
        '  <w:bottom w:w="120" w:type="dxa"/>'
        '  <w:right w:w="200" w:type="dxa"/>'
        '</w:tcMar>'
    )
    tcPr.append(tcMar)

    para = cell.paragraphs[0]
    run = para.add_run(text)
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(
        int(fg[0:2], 16), int(fg[2:4], 16), int(fg[4:6], 16)
    )
    doc.add_paragraph()  # spacer


def add_number_table(doc, headers, rows, col_widths=None):
    """Add a formatted table with header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, "2C3E50")
        para = cell.paragraphs[0]
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.name = "Calibri"

    # Data rows
    for r_idx, row in enumerate(rows):
        bg = "FFFFFF" if r_idx % 2 == 0 else "F8F9FA"
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            set_cell_shading(cell, bg)
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(str(val))
            run.font.size = Pt(9)
            run.font.name = "Calibri"

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph()  # spacer
    return table


def build_document():
    doc = Document()

    # ── Page setup ──
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ── Styles ──
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = Pt(16)

    for level in range(1, 4):
        sname = f"Heading {level}"
        s = doc.styles[sname]
        s.font.name = "Calibri"
        s.font.color.rgb = RGBColor(0x1A, 0x47, 0x80)

    # ════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ════════════════════════════════════════════════════════════════
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("MIMI Neural Audio Codec")
    run.bold = True
    run.font.size = Pt(32)
    run.font.color.rgb = RGBColor(0x1A, 0x47, 0x80)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("A Complete Beginner's Guide")
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    doc.add_paragraph()

    subtitle2 = doc.add_paragraph()
    subtitle2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle2.add_run("How sound becomes numbers and numbers become sound again")
    run.italic = True
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    for _ in range(4):
        doc.add_paragraph()

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run("Developed by Kyutai Labs (Paris, France)\n")
    run.font.size = Pt(11)
    run = meta.add_run("Used in: Moshi (Speech-Text Foundation Model)\n")
    run.font.size = Pt(11)
    run = meta.add_run("Used in: PersonaPlex (NVIDIA, Full-Duplex Conversation)")
    run.font.size = Pt(11)

    doc.add_paragraph()
    date_p = doc.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_p.add_run("March 2026")
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("Table of Contents", level=1)
    toc_items = [
        "1. What is Mimi? (The Big Picture)",
        "2. Why Do We Need Mimi?",
        "3. The 30-Second Overview",
        "4. Step 1: The Encoder (Sound -> Numbers)",
        "   4.1 Convolutional Layers (Feature Extraction)",
        "   4.2 Transformer Layers (Adding Context)",
        "   4.3 Downsampler (Compression)",
        "5. Step 2: The Quantizer (Continuous -> Discrete)",
        "   5.1 What is Vector Quantization?",
        "   5.2 Semantic Codebook (What is Said)",
        "   5.3 Acoustic Codebooks (How it Sounds)",
        "   5.4 The Split Design",
        "6. Step 3: The Decoder (Numbers -> Sound)",
        "7. Full Architecture Diagram",
        "8. Key Numbers & Parameters",
        "9. The Training Process",
        "10. How Mimi is Used in PersonaPlex",
        "11. Summary & Quick Reference",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        if item.startswith("   "):
            p.paragraph_format.left_indent = Cm(1)
            p.runs[0].font.size = Pt(10)

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 1: WHAT IS MIMI?
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("1. What is Mimi? (The Big Picture)", level=1)

    doc.add_paragraph(
        "Mimi is a neural audio codec -- a deep learning model that compresses "
        "audio into tiny discrete tokens (numbers) and reconstructs audio back "
        "from those tokens. Think of it like a ZIP file for sound, but powered "
        "by artificial intelligence."
    )

    add_info_box(doc,
        "ANALOGY: Imagine you want to send a painting to a friend through a "
        "tiny mail slot. You can't send the whole painting. Instead, you describe "
        "it with a code: 'blue sky, green tree, red house, small dog'. Your "
        "friend reads the code and paints a new picture that looks very similar. "
        "Mimi does the same thing with sound!",
        "tip"
    )

    add_diagram_table(doc, [
        "THE BASIC IDEA:",
        "",
        "  COMPRESS                         DECOMPRESS",
        "  (Encode)                         (Decode)",
        "",
        "  Sound wave    ──►  Small tokens  ──►  Sound wave",
        "  (big data)         (tiny data)        (nearly identical)",
        "",
        "  3 seconds     ──►  8 x 38 = 304 ──►  3 seconds",
        "  = 72,000           numbers            = 72,000",
        "    numbers           (237x smaller!)      numbers",
    ], "Figure 1.1: Mimi compresses audio by ~237x")

    doc.add_paragraph()
    doc.add_paragraph(
        "Key facts about Mimi:"
    )

    facts = [
        ("Built by", "Kyutai Labs, Paris, France"),
        ("Model size", "96.2 million parameters"),
        ("Input", "24,000 Hz mono audio (24kHz sample rate)"),
        ("Output", "Discrete tokens at 12.5 Hz (12.5 token-sets per second)"),
        ("Bitrate", "1.1 kbps (extremely compressed)"),
        ("Streaming", "Yes -- can process audio in real-time, frame by frame"),
        ("Latency", "80ms (one frame = 80 milliseconds)"),
    ]
    add_number_table(doc,
        ["Property", "Value"],
        facts,
        col_widths=[4, 12]
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 2: WHY DO WE NEED MIMI?
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("2. Why Do We Need Mimi?", level=1)

    doc.add_paragraph(
        "Language models (like GPT, LLaMA, etc.) work with tokens -- discrete "
        "symbols from a vocabulary. Text is naturally made of discrete symbols "
        "(letters, words). But audio is a continuous waveform -- a smooth curve "
        "of air pressure over time."
    )

    add_diagram_table(doc, [
        "THE PROBLEM:",
        "",
        "  Text:   H-e-l-l-o       Already discrete! (5 symbols)",
        "                           LLM can process this directly.",
        "",
        "  Audio:  ~~~~\\/\\/~~\\/~    Continuous! (72,000 numbers for 3 sec)",
        "                           LLM CANNOT process this directly.",
        "                           Way too many numbers, not discrete.",
        "",
        "THE SOLUTION (Mimi):",
        "",
        "  Audio:  ~~~~\\/\\/~~\\/~  ──►  [427, 1893, 512, ...]  ──►  LLM",
        "          continuous wave       discrete tokens              can process!",
        "          72,000 numbers        304 numbers                  (like words)",
    ], "Figure 2.1: Why we need audio tokenization")

    doc.add_paragraph()
    doc.add_paragraph(
        "Mimi bridges the gap between continuous audio and discrete language models. "
        "This enables models like Moshi and PersonaPlex to have real-time voice "
        "conversations -- the AI 'reads' audio tokens just like it reads text tokens."
    )

    add_info_box(doc,
        "KEY INSIGHT: Mimi is NOT a speech-to-text system. It does not convert "
        "audio to words. It converts audio to SOUND tokens that preserve everything "
        "-- the words, the voice, the emotion, the accent, even background noise.",
        "key"
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 3: THE 30-SECOND OVERVIEW
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("3. The 30-Second Overview", level=1)

    doc.add_paragraph(
        "Before diving into details, here is the complete Mimi pipeline at a glance. "
        "Don't worry if you don't understand every part -- we will explain each one."
    )

    add_diagram_table(doc, [
        "MIMI: THE COMPLETE PIPELINE",
        "============================",
        "",
        "      ENCODER (compress)              DECODER (decompress)",
        "  ┌─────────────────────┐         ┌─────────────────────┐",
        "  │                     │         │                     │",
        "  │  Audio (24kHz)      │         │  Tokens (12.5Hz)    │",
        "  │  [1, 72000]         │         │  [8, 38]            │",
        "  │       |             │         │       |             │",
        "  │       v             │         │       v             │",
        "  │  Conv Layers ----+  │         │  Dequantize         │",
        "  │  (4 blocks)     |  │         │  (lookup codebook)  │",
        "  │  stride 960     |  │         │       |             │",
        "  │       |         |  │         │       v             │",
        "  │       v         |  │         │  Upsample (x2)      │",
        "  │  Transformer    |  │         │       |             │",
        "  │  (8 layers)     |  │         │       v             │",
        "  │       |         |  │         │  Transformer        │",
        "  │       v         |  │         │  (8 layers)         │",
        "  │  Downsample     |  │         │       |             │",
        "  │  (x2)           |  │         │       v             │",
        "  │       |         |  │         │  Deconv Layers      │",
        "  │       v         |  │         │  (4 blocks)         │",
        "  │  Quantize ------+  │         │  stride 960         │",
        "  │  (8 codebooks)     │         │       |             │",
        "  │       |             │         │       v             │",
        "  │       v             │         │  Audio (24kHz)      │",
        "  │  Tokens (12.5Hz)   │         │  [1, 72000]         │",
        "  │  [8, 38]           │         │                     │",
        "  └─────────────────────┘         └─────────────────────┘",
        "",
        "  Total parameters: 96.2 Million",
        "  Compression: 24,000 Hz  -->  12.5 Hz  (1,920x)",
        "  Bitrate: 1.1 kbps",
    ], "Figure 3.1: Complete Mimi architecture overview")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "THE THREE MAIN PARTS:",
        "",
        "  Part 1: ENCODER           Part 2: QUANTIZER         Part 3: DECODER",
        "  ==================        ==================        ==================",
        "  Continuous audio           Continuous --> Discrete   Discrete --> Audio",
        "  --> Continuous features     (lookup table)           (reverse of encoder)",
        "",
        "  Contains:                  Contains:                Contains:",
        "  - Conv layers (compress)   - 1 semantic codebook    - Upsampler",
        "  - Transformer (context)    - 7 acoustic codebooks   - Transformer",
        "  - Downsampler (shrink)     Each: 2,048 entries      - Deconv layers",
    ], "Figure 3.2: The three main components")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 4: THE ENCODER
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("4. Step 1: The Encoder (Sound -> Numbers)", level=1)

    doc.add_paragraph(
        "The encoder takes a raw audio waveform and converts it into a compact "
        "numerical representation. It has three sub-parts: convolutional layers, "
        "a Transformer, and a downsampler."
    )

    # ── 4.1 Conv Layers ──
    doc.add_heading("4.1 Convolutional Layers (Feature Extraction)", level=2)

    doc.add_paragraph(
        "The convolutional encoder is inspired by SEANet (used in SoundStream and EnCodec). "
        "It uses strided 1D convolutions to progressively downsample the audio while "
        "extracting features."
    )

    add_info_box(doc,
        "ANALOGY: Imagine looking at a painting from far away. First you see the "
        "overall shape (that's a person!). Then you zoom in -- you see the face. "
        "Zoom more -- you see the eyes. Each 'zoom level' captures different details. "
        "Conv layers do the same with sound -- each layer captures different frequency features.",
        "tip"
    )

    add_diagram_table(doc, [
        "CONVOLUTIONAL ENCODER: Layer-by-Layer",
        "(Example: 3 seconds of audio = 72,000 samples at 24kHz)",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  INPUT: Raw Audio Waveform                                  │",
        "  │  Shape: [Batch=1, Channels=1, Samples=72,000]              │",
        "  │  ~~~~\\/\\/~~\\/~~~~~\\/\\/\\/~~~/\\/\\/~~~/\\~/\\              │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Layer 0: Initial Conv1d                                    │",
        "  │  Channels: 1 --> 64    Kernel: 7    Stride: 1              │",
        "  │  Output: [1, 64, 72,000]                                   │",
        "  │  (Extract 64 different audio features, keep all samples)    │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Block 1: Downsample by 4x                                  │",
        "  │  ┌──────────────────────────────────────────────────┐       │",
        "  │  │  ResnetBlock (64ch):                              │       │",
        "  │  │    ELU -> Conv1d(64->32, k=3) -> ELU -> Conv1d(32->64)  │",
        "  │  │    + skip connection (identity)                  │       │",
        "  │  └──────────────────────────────────────────────────┘       │",
        "  │  ELU activation                                             │",
        "  │  Conv1d(64 --> 128, kernel=8, stride=4)                     │",
        "  │  Output: [1, 128, 18,000]                                  │",
        "  │  (Double channels, quarter the samples)                     │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Block 2: Downsample by 5x                                  │",
        "  │  ResnetBlock(128ch) + ELU                                   │",
        "  │  Conv1d(128 --> 256, kernel=10, stride=5)                   │",
        "  │  Output: [1, 256, 3,600]                                   │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Block 3: Downsample by 6x                                  │",
        "  │  ResnetBlock(256ch) + ELU                                   │",
        "  │  Conv1d(256 --> 512, kernel=12, stride=6)                   │",
        "  │  Output: [1, 512, 600]                                     │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Block 4: Downsample by 8x                                  │",
        "  │  ResnetBlock(512ch) + ELU                                   │",
        "  │  Conv1d(512 --> 1024, kernel=16, stride=8)                  │",
        "  │  Output: [1, 1024, 75]                                     │",
        "  └───────────────────────┬─────────────────────────────────────┘",
        "                          |",
        "                          v",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │  Final Conv: Conv1d(1024 --> 512, kernel=3, stride=1)       │",
        "  │  Output: [1, 512, 75]  <-- 25 Hz frame rate                │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  Total convolutional stride: 4 x 5 x 6 x 8 = 960",
        "  72,000 samples / 960 = 75 frames  -->  25 Hz intermediate rate",
    ], "Figure 4.1: Convolutional encoder architecture")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "HOW DOWNSAMPLING WORKS (simplified):",
        "",
        "  Original signal (stride=1):   1 2 3 4 5 6 7 8 9 10 11 12",
        "                                | | | | | | | | | |  |  |",
        "                                v v v v v v v v v v  v  v",
        "                                keep all 12 values",
        "",
        "  After stride=4:               1 2 3 4 5 6 7 8 9 10 11 12",
        "                                |       |       |        |",
        "                                v       v       v        v",
        "                                keep every 4th: 3 values",
        "",
        "  The conv kernel (size=8) slides across the input,",
        "  jumping 4 positions each time, mixing nearby values",
        "  into a single output number.",
    ], "Figure 4.2: What 'stride' means in convolution")

    doc.add_paragraph()

    # Channel progression table
    add_number_table(doc,
        ["Stage", "Channels", "Time Steps", "Frequency", "Stride"],
        [
            ("Input audio", "1", "72,000", "24,000 Hz", "-"),
            ("After initial conv", "64", "72,000", "24,000 Hz", "1"),
            ("After Block 1", "128", "18,000", "6,000 Hz", "4"),
            ("After Block 2", "256", "3,600", "1,200 Hz", "5"),
            ("After Block 3", "512", "600", "200 Hz", "6"),
            ("After Block 4", "1024", "75", "25 Hz", "8"),
            ("After final conv", "512", "75", "25 Hz", "1"),
        ],
        col_widths=[4, 2.5, 2.5, 3, 2]
    )

    add_info_box(doc,
        "KEY OBSERVATION: Notice how channels INCREASE (1 -> 64 -> 128 -> 256 -> 512 -> 1024) "
        "while time steps DECREASE (72,000 -> 18,000 -> 3,600 -> 600 -> 75). "
        "This is the trade-off: we lose time resolution but gain richer feature representations.",
        "key"
    )

    doc.add_page_break()

    # ── 4.2 Transformer ──
    doc.add_heading("4.2 Transformer Layers (Adding Context)", level=2)

    doc.add_paragraph(
        "After the conv layers produce 75 frames at 25 Hz, a Transformer adds "
        "temporal context. Each frame can 'look at' nearby frames to understand "
        "the bigger picture -- like reading a sentence instead of individual letters."
    )

    add_diagram_table(doc, [
        "ENCODER TRANSFORMER: 8 Identical Layers",
        "",
        "  Input: [Batch, 75 frames, 512 dimensions]  (from conv encoder)",
        "",
        "  ┌────────────────────────────────────────────────────┐",
        "  │  Transformer Layer (repeated 8 times)              │",
        "  │                                                     │",
        "  │  ┌──────────────────────────────────────────────┐  │",
        "  │  │  1. Layer Normalization                       │  │",
        "  │  │     Normalize each frame's 512 values         │  │",
        "  │  └──────────────────────────────────────────────┘  │",
        "  │                    |                                │",
        "  │                    v                                │",
        "  │  ┌──────────────────────────────────────────────┐  │",
        "  │  │  2. Multi-Head Self-Attention                 │  │",
        "  │  │     8 heads, each with dimension 64           │  │",
        "  │  │     (8 x 64 = 512 total dimension)           │  │",
        "  │  │                                               │  │",
        "  │  │     Each frame asks: 'What other frames       │  │",
        "  │  │     are relevant to me?' (via Q, K, V)        │  │",
        "  │  │                                               │  │",
        "  │  │     Causal mask: can ONLY look at past frames │  │",
        "  │  │     (for streaming -- can't peek at future)   │  │",
        "  │  │                                               │  │",
        "  │  │     RoPE: Rotary Position Embeddings          │  │",
        "  │  │     (tells model where each frame is in time) │  │",
        "  │  │                                               │  │",
        "  │  │     Sliding window: 250 frames (20 seconds)   │  │",
        "  │  └──────────────────────────────────────────────┘  │",
        "  │                    |                                │",
        "  │                    v                                │",
        "  │          LayerScale (x0.01) + Residual Connection  │",
        "  │                    |                                │",
        "  │                    v                                │",
        "  │  ┌──────────────────────────────────────────────┐  │",
        "  │  │  3. Layer Normalization                       │  │",
        "  │  └──────────────────────────────────────────────┘  │",
        "  │                    |                                │",
        "  │                    v                                │",
        "  │  ┌──────────────────────────────────────────────┐  │",
        "  │  │  4. Feed-Forward Network (MLP)                │  │",
        "  │  │     Linear(512 -> 2048)                       │  │",
        "  │  │     GELU activation                           │  │",
        "  │  │     Linear(2048 -> 512)                       │  │",
        "  │  └──────────────────────────────────────────────┘  │",
        "  │                    |                                │",
        "  │          LayerScale (x0.01) + Residual Connection  │",
        "  │                    |                                │",
        "  │                    v                                │",
        "  │  Output: [Batch, 75 frames, 512 dimensions]        │",
        "  │  (same shape! only the VALUES changed)             │",
        "  └────────────────────────────────────────────────────┘",
        "",
        "  This block is repeated 8 times (8 layers).",
        "  Each layer adds more context understanding.",
    ], "Figure 4.3: One Transformer layer (repeated 8 times)")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "WHY CAUSAL ATTENTION MATTERS (for streaming):",
        "",
        "  Frame:    F0   F1   F2   F3   F4   F5   F6   F7",
        "  Time:    0ms  80ms 160ms 240ms ...",
        "",
        "  What F3 can attend to:",
        "     F0  F1  F2  [F3]  F4  F5  F6  F7",
        "      v   v   v   v    x   x   x   x",
        "     YES YES YES SELF  NO  NO  NO  NO",
        "",
        "  F3 can look at F0, F1, F2 (past) and itself,",
        "  but NOT F4, F5, F6, F7 (future).",
        "  This allows real-time processing!",
    ], "Figure 4.4: Causal attention mask")

    doc.add_paragraph()

    add_number_table(doc,
        ["Parameter", "Value", "Explanation"],
        [
            ("Layers", "8", "8 identical transformer blocks stacked"),
            ("Attention Heads", "8", "8 parallel attention computations"),
            ("Head Dimension", "64", "Each head processes 64 numbers"),
            ("Model Dimension", "512", "8 heads x 64 = 512 total"),
            ("FFN Dimension", "2048", "4x expansion in feed-forward"),
            ("Position Encoding", "RoPE", "Rotary embeddings for position"),
            ("Context Window", "250 frames", "Can look back 20 seconds"),
            ("Masking", "Causal", "Can only see past frames"),
            ("LayerScale Init", "0.01", "Stabilizes deep training"),
        ],
        col_widths=[3.5, 3, 8]
    )

    doc.add_page_break()

    # ── 4.3 Downsampler ──
    doc.add_heading("4.3 Downsampler (Final Compression)", level=2)

    doc.add_paragraph(
        "After the Transformer, a learned downsampler reduces the frame rate "
        "from 25 Hz to 12.5 Hz -- cutting the number of frames in half."
    )

    add_diagram_table(doc, [
        "DOWNSAMPLER:",
        "",
        "  Before:  [Batch, 512, 75 frames]   at 25 Hz",
        "              |",
        "              v",
        "     Conv1d(512 -> 512, kernel=4, stride=2)",
        "     (depthwise: each of 512 channels processed independently)",
        "              |",
        "              v",
        "  After:   [Batch, 512, 37 frames]   at 12.5 Hz",
        "",
        "  Visualization:",
        "  25 Hz: |F0|F1|F2|F3|F4|F5|...  (75 frames in 3 seconds)",
        "              \\/ \\/ \\/ \\/",
        "  12.5Hz: |F0 |F1 |F2 |...        (37 frames in 3 seconds)",
        "  (pairs of frames are merged into one)",
    ], "Figure 4.5: The learned downsampler")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "COMPLETE ENCODER SUMMARY (3 seconds of audio):",
        "",
        "  [1, 1, 72,000]  Raw audio at 24,000 Hz",
        "        |",
        "        |  Conv Encoder (4 blocks, stride 4x5x6x8 = 960)",
        "        v",
        "  [1, 512, 75]    Features at 25 Hz",
        "        |",
        "        |  Transpose to [1, 75, 512]",
        "        |  8-layer Transformer (causal, RoPE)",
        "        |  Transpose back to [1, 512, 75]",
        "        v",
        "  [1, 512, 75]    Contextualized features at 25 Hz",
        "        |",
        "        |  Downsampler (Conv1d stride 2)",
        "        v",
        "  [1, 512, 37]    Final embeddings at 12.5 Hz",
        "        |",
        "        |  Ready for quantization!",
        "        v",
    ], "Figure 4.6: Complete encoder data flow")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 5: THE QUANTIZER
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("5. Step 2: The Quantizer (Continuous -> Discrete)", level=1)

    doc.add_paragraph(
        "The quantizer is the heart of Mimi. It converts continuous embeddings "
        "(floating-point numbers that can be anything) into discrete tokens "
        "(integers from a fixed set). This is what makes it possible for a "
        "language model to process audio."
    )

    # ── 5.1 What is VQ ──
    doc.add_heading("5.1 What is Vector Quantization?", level=2)

    add_info_box(doc,
        "ANALOGY: Imagine you have a box of exactly 2,048 colored crayons, each with "
        "a unique number (0 to 2047). Someone shows you a color (any shade imaginable). "
        "You pick the CLOSEST crayon. Instead of describing the exact color, you just "
        "say the crayon number. That's Vector Quantization!",
        "tip"
    )

    add_diagram_table(doc, [
        "VECTOR QUANTIZATION (VQ) - How it works:",
        "",
        "  Step 1: You have a CODEBOOK (lookup table)",
        "  ┌─────────────────────────────────────────────────┐",
        "  │  Codebook: 2,048 vectors, each 256 dimensions   │",
        "  │                                                   │",
        "  │  Entry 0:    [0.23, -0.15, 0.87, ..., 0.42]     │",
        "  │  Entry 1:    [0.51, 0.33, -0.62, ..., 0.19]     │",
        "  │  Entry 2:    [-0.44, 0.78, 0.11, ..., -0.33]    │",
        "  │  ...                                              │",
        "  │  Entry 2047: [0.17, -0.92, 0.45, ..., 0.66]     │",
        "  └─────────────────────────────────────────────────┘",
        "",
        "  Step 2: For each frame's embedding, find the NEAREST entry",
        "",
        "  Input embedding: [0.25, -0.12, 0.85, ..., 0.40]",
        "                          |",
        "                          | Compare distance to ALL 2,048 entries",
        "                          |",
        "                          v",
        "  Closest match: Entry 0  [0.23, -0.15, 0.87, ..., 0.42]",
        "                          (distance = 0.05, smallest!)",
        "",
        "  Output token: 0  (just the index number!)",
        "",
        "  Step 3: To DECODE, just look up the entry",
        "  Token 0  -->  look up Entry 0  -->  [0.23, -0.15, 0.87, ..., 0.42]",
    ], "Figure 5.1: Vector Quantization explained")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "RESIDUAL VECTOR QUANTIZATION (RVQ) - Multiple rounds:",
        "",
        "  The idea: one codebook can't capture everything.",
        "  So we use MULTIPLE codebooks, each fixing the error of the previous.",
        "",
        "  Input:    [0.25, -0.12, 0.85]    (original embedding)",
        "",
        "  Round 1 (Codebook 1):",
        "    Nearest: Entry 42 = [0.20, -0.10, 0.80]   --> Token: 42",
        "    Error:   [0.05, -0.02, 0.05]               (what's left over)",
        "",
        "  Round 2 (Codebook 2):",
        "    Input: the ERROR from Round 1 = [0.05, -0.02, 0.05]",
        "    Nearest: Entry 789 = [0.04, -0.03, 0.06]   --> Token: 789",
        "    Error:   [0.01, 0.01, -0.01]                (even smaller!)",
        "",
        "  Round 3 (Codebook 3):",
        "    Input: the ERROR from Round 2 = [0.01, 0.01, -0.01]",
        "    Nearest: Entry 1500 = [0.01, 0.01, -0.01]  --> Token: 1500",
        "    Error:   [0.00, 0.00, 0.00]                 (almost zero!)",
        "",
        "  DECODE: Entry_42 + Entry_789 + Entry_1500",
        "        = [0.20, -0.10, 0.80]",
        "        + [0.04, -0.03, 0.06]",
        "        + [0.01, 0.01, -0.01]",
        "        = [0.25, -0.12, 0.85]  <-- very close to original!",
        "",
        "  More codebooks = more accuracy = better sound quality",
    ], "Figure 5.2: Residual Vector Quantization (RVQ)")

    doc.add_page_break()

    # ── 5.2 Semantic Codebook ──
    doc.add_heading("5.2 The Semantic Codebook (Codebook 0): What is Said", level=2)

    doc.add_paragraph(
        "Mimi's first codebook is special -- it captures WHAT is being said "
        "(the meaning/content of speech). It is trained using knowledge "
        "distillation from WavLM, a pre-trained speech understanding model."
    )

    add_diagram_table(doc, [
        "SEMANTIC CODEBOOK TRAINING (WavLM Distillation):",
        "",
        "               Same audio",
        "              /          \\",
        "             v            v",
        "     ┌──────────┐    ┌──────────┐",
        "     │  WavLM   │    │  Mimi    │",
        "     │ (teacher) │    │(student) │",
        "     │ pre-      │    │ being    │",
        "     │ trained   │    │ trained  │",
        "     └─────┬─────┘    └─────┬────┘",
        "           |                |",
        "           v                v",
        "     Teacher's          Mimi's Codebook 0",
        "     embeddings         output vectors",
        "     (understand        ",
        "      speech well)      ",
        "           |                |",
        "           └───── Loss ─────┘",
        "                  |",
        "        cosine_similarity(teacher, student)",
        "        'Make them match!'",
        "",
        "  Result: Codebook 0 learns to capture speech CONTENT",
        "    - Phonemes (sounds like 'b', 'a', 'k')",
        "    - Words (token patterns for 'hello', 'goodbye')",
        "    - Prosody structure (question vs statement)",
        "    - Largely IGNORES speaker identity",
    ], "Figure 5.3: How the semantic codebook is trained")

    doc.add_paragraph()

    add_info_box(doc,
        "IMPORTANT FOR VIETNAMESE: WavLM was trained mainly on English speech. "
        "English has no lexical tones. Vietnamese has 6 tones that change word meaning "
        "(ma/ma\\u0300/ma\\u0301/ma\\u0309/ma\\u0303/ma\\u0323). The semantic codebook may NOT "
        "distinguish Vietnamese tones correctly. This is a key research question!",
        "warn"
    )

    doc.add_page_break()

    # ── 5.3 Acoustic Codebooks ──
    doc.add_heading("5.3 The Acoustic Codebooks (1-7): How it Sounds", level=2)

    doc.add_paragraph(
        "The remaining 7 codebooks capture acoustic details -- the voice quality, "
        "pitch, timbre, and fine details that make speech sound natural."
    )

    add_diagram_table(doc, [
        "ACOUSTIC CODEBOOKS - What each one captures:",
        "",
        "  Codebook 1:  Coarse acoustic structure",
        "               (overall pitch contour, energy pattern)",
        "               |||||||||||||||||",
        "",
        "  Codebook 2:  Speaker timbre",
        "               (voice quality - deep/high, breathy/clear)",
        "               ||||||||||||",
        "",
        "  Codebook 3:  Formant details",
        "               (vowel shapes, resonance)",
        "               ||||||||",
        "",
        "  Codebook 4:  Fine spectral details",
        "               (harmonics, noise characteristics)",
        "               |||||",
        "",
        "  Codebook 5:  Micro-prosodic features",
        "               (subtle timing, emphasis)",
        "               |||",
        "",
        "  Codebook 6:  Residual acoustic texture",
        "               (breathiness, roughness)",
        "               ||",
        "",
        "  Codebook 7:  Finest reconstruction details",
        "               (last bit of quality improvement)",
        "               |",
        "",
        "  The bars (|) show relative importance.",
        "  Earlier codebooks carry more information.",
        "  Each codebook refines what the previous ones missed.",
    ], "Figure 5.4: What each acoustic codebook captures")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "PROGRESSIVE DECODING (adding codebooks one by one):",
        "",
        "  1 codebook  (semantic only):",
        "     Sounds like a robot reading text",
        "     [intelligible but very metallic]",
        "",
        "  + 1 acoustic  (2 total):",
        "     Sounds like AM radio",
        "     [pitch and energy restore]",
        "",
        "  + 2 acoustic  (3 total):",
        "     Sounds like phone call",
        "     [voice identity emerging]",
        "",
        "  + 3 acoustic  (4 total):",
        "     Sounds like FM radio",
        "     [quite natural now]",
        "",
        "  + 4 acoustic  (5 total):",
        "     Sounds like CD quality",
        "     [hard to tell from original]",
        "",
        "  All 8 codebooks:  MUSHRA score 81.0 / 100",
        "     (very close to human perception of original)",
    ], "Figure 5.5: Quality improves with more codebooks")

    doc.add_page_break()

    # ── 5.4 The Split Design ──
    doc.add_heading("5.4 The Split Design (Mimi's Key Innovation)", level=2)

    doc.add_paragraph(
        "What makes Mimi special is the SPLIT design: the semantic codebook and "
        "acoustic codebooks operate INDEPENDENTLY on the same input. They are "
        "NOT sequential -- this is different from standard RVQ!"
    )

    add_diagram_table(doc, [
        "SPLIT RVQ vs STANDARD RVQ:",
        "",
        "  STANDARD RVQ (like EnCodec):",
        "  ================================",
        "  embedding --> CB1 --> residual --> CB2 --> residual --> CB3 ...",
        "               (each codebook operates on the PREVIOUS error)",
        "               (all codebooks are entangled)",
        "",
        "",
        "  SPLIT RVQ (Mimi's approach):",
        "  ================================",
        "                    ┌───────────────────┐",
        "                    │  Input embedding   │",
        "                    │  [512 dimensions]  │",
        "                    └────────┬──────────┘",
        "                    project to 256 dims",
        "                             |",
        "               ┌─────────────┴────────────┐",
        "               |                          |",
        "               v                          v",
        "     ┌─────────────────┐      ┌─────────────────────┐",
        "     │ SEMANTIC VQ     │      │ ACOUSTIC RVQ        │",
        "     │ (1 codebook)    │      │ (7 codebooks)       │",
        "     │                 │      │                     │",
        "     │ Same input!     │      │ Same input!         │",
        "     │ Captures WHAT   │      │ CB1 -> residual     │",
        "     │ is said         │      │ CB2 -> residual     │",
        "     │                 │      │ ...                  │",
        "     │ Trained via     │      │ CB7                  │",
        "     │ WavLM loss      │      │ Captures HOW        │",
        "     │                 │      │ it sounds            │",
        "     └────────┬────────┘      └──────────┬──────────┘",
        "              |                          |",
        "              └──────────┬───────────────┘",
        "                         |",
        "                    ADD together",
        "                         |",
        "                         v",
        "              ┌──────────────────┐",
        "              │ Combined output   │",
        "              │ project to 512    │",
        "              └──────────────────┘",
        "",
        "  WHY THIS MATTERS:",
        "  - You can manipulate semantic and acoustic INDEPENDENTLY",
        "  - Change the voice (acoustic) without changing words (semantic)",
        "  - Change words without changing the voice",
        "  - Much better for voice-conditioned generation (PersonaPlex)",
    ], "Figure 5.6: Split RVQ -- Mimi's key innovation")

    doc.add_paragraph()

    add_number_table(doc,
        ["Component", "Codebooks", "Entries per CB", "Dim", "Total Params"],
        [
            ("Semantic VQ", "1", "2,048", "256", "524,288"),
            ("Acoustic RVQ", "7", "2,048 each", "256", "3,670,016"),
            ("Input projection", "1", "-", "512->256", "131,072"),
            ("Output projection", "1", "-", "256->512", "131,072"),
            ("TOTAL", "8", "-", "-", "~4.5M"),
        ],
        col_widths=[3.5, 2.5, 3, 2.5, 3]
    )

    doc.add_page_break()

    add_diagram_table(doc, [
        "QUANTIZER DATA FLOW (for one frame):",
        "",
        "  Encoder output:  [512 float values]",
        "                           |",
        "                   Linear(512 -> 256)",
        "                           |",
        "                   [256 float values]",
        "                      /          \\",
        "                     /            \\",
        "                    v              v",
        "           Semantic VQ       Acoustic RVQ",
        "           (find nearest     (7 rounds of",
        "            among 2048)       find-nearest)",
        "                |                  |",
        "                v                  v",
        "           1 integer          7 integers",
        "           [0..2047]         [0..2047] each",
        "                \\                /",
        "                 \\              /",
        "                  v            v",
        "          Output: [CB0, CB1, CB2, CB3, CB4, CB5, CB6, CB7]",
        "                  [427, 1893, 512, 88, 2001, 7, 1456, 903]",
        "",
        "          8 integers per frame!",
        "          Each integer is 11 bits (log2(2048) = 11)",
        "",
        "          Bitrate = 8 codebooks x 11 bits x 12.5 Hz",
        "                  = 1,100 bits/second = 1.1 kbps",
    ], "Figure 5.7: Quantizer data flow for a single frame")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 6: THE DECODER
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("6. Step 3: The Decoder (Numbers -> Sound)", level=1)

    doc.add_paragraph(
        "The decoder is the mirror image of the encoder. It takes the discrete "
        "tokens and reconstructs the original audio waveform."
    )

    add_diagram_table(doc, [
        "DECODER: COMPLETE ARCHITECTURE",
        "",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  INPUT: Codes [Batch, 8 codebooks, 37 frames]               │",
        "  │  Example: [[427,1893,512,...], [88,2001,7,...], ...]         │",
        "  └───────────────────────────┬──────────────────────────────────┘",
        "                              |",
        "                              v",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  DEQUANTIZE: Look up codebook vectors + add them            │",
        "  │  semantic_vec = semantic_CB[token_0]     (256-dim vector)    │",
        "  │  acoustic_vec = acoustic_CB1[token_1]                       │",
        "  │               + acoustic_CB2[token_2]                       │",
        "  │               + ... + acoustic_CB7[token_7]                 │",
        "  │  combined = semantic_vec + acoustic_vec                      │",
        "  │  Project: Linear(256 -> 512)                                │",
        "  │  Output: [Batch, 512, 37 frames]                            │",
        "  └───────────────────────────┬──────────────────────────────────┘",
        "                              |",
        "                              v",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  UPSAMPLER: ConvTranspose1d(512->512, k=4, stride=2)        │",
        "  │  (depthwise, groups=512)                                    │",
        "  │  Output: [Batch, 512, 75 frames]  (12.5Hz -> 25Hz)         │",
        "  └───────────────────────────┬──────────────────────────────────┘",
        "                              |",
        "                              v",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  DECODER TRANSFORMER: 8 layers (same config as encoder)     │",
        "  │  Input:  [Batch, 75, 512]                                   │",
        "  │  Output: [Batch, 75, 512]  (adds temporal context)          │",
        "  └───────────────────────────┬──────────────────────────────────┘",
        "                              |",
        "                              v",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  TRANSPOSED CONV DECODER (mirror of encoder):               │",
        "  │                                                              │",
        "  │  Initial: Conv1d(512 -> 1024, k=3, stride=1)               │",
        "  │                                                              │",
        "  │  Upsample Block 1: ConvT(1024->512,  k=16, stride=8)      │",
        "  │    + ResnetBlock(512)                                        │",
        "  │    [1, 512, 600]                                            │",
        "  │                                                              │",
        "  │  Upsample Block 2: ConvT(512->256,   k=12, stride=6)      │",
        "  │    + ResnetBlock(256)                                        │",
        "  │    [1, 256, 3600]                                           │",
        "  │                                                              │",
        "  │  Upsample Block 3: ConvT(256->128,   k=10, stride=5)      │",
        "  │    + ResnetBlock(128)                                        │",
        "  │    [1, 128, 18000]                                          │",
        "  │                                                              │",
        "  │  Upsample Block 4: ConvT(128->64,    k=8,  stride=4)      │",
        "  │    + ResnetBlock(64)                                         │",
        "  │    [1, 64, 72000]                                           │",
        "  │                                                              │",
        "  │  Final: Conv1d(64 -> 1, k=3, stride=1)                     │",
        "  │    [1, 1, 72000]                                            │",
        "  └───────────────────────────┬──────────────────────────────────┘",
        "                              |",
        "                              v",
        "  ┌──────────────────────────────────────────────────────────────┐",
        "  │  OUTPUT: Reconstructed Audio [Batch, 1, 72,000]             │",
        "  │  ~~~~\\/\\/~~\\/~~~~~\\/\\/\\/~~~/\\/\\/~~~/\\~/\\  at 24kHz   │",
        "  └──────────────────────────────────────────────────────────────┘",
    ], "Figure 6.1: Complete decoder architecture")

    doc.add_paragraph()

    add_number_table(doc,
        ["Decoder Stage", "Channels", "Time Steps", "Frequency"],
        [
            ("Dequantized tokens", "512", "37", "12.5 Hz"),
            ("After upsampler", "512", "75", "25 Hz"),
            ("After Transformer", "512", "75", "25 Hz"),
            ("After upsample block 1 (x8)", "512", "600", "200 Hz"),
            ("After upsample block 2 (x6)", "256", "3,600", "1,200 Hz"),
            ("After upsample block 3 (x5)", "128", "18,000", "6,000 Hz"),
            ("After upsample block 4 (x4)", "64", "72,000", "24,000 Hz"),
            ("Final output", "1", "72,000", "24,000 Hz"),
        ],
        col_widths=[5.5, 2, 2.5, 3]
    )

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 7: FULL ARCHITECTURE DIAGRAM
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("7. Full Architecture Diagram", level=1)

    doc.add_paragraph(
        "Here is the complete end-to-end architecture showing every component "
        "and tensor shape transformation."
    )

    add_diagram_table(doc, [
        "MIMI COMPLETE ARCHITECTURE (end-to-end)",
        "=========================================",
        "",
        "  INPUT AUDIO",
        "  [1, 1, 72000]  (3 seconds at 24kHz)",
        "       |",
        "  ═════╪══════════════════════ ENCODER ═══════════════════════",
        "       |",
        "       |  Conv1d(1->64, k=7, s=1)",
        "  [1, 64, 72000]",
        "       |  ResnetBlock(64) + Conv1d(64->128, k=8, s=4)",
        "  [1, 128, 18000]",
        "       |  ResnetBlock(128) + Conv1d(128->256, k=10, s=5)",
        "  [1, 256, 3600]",
        "       |  ResnetBlock(256) + Conv1d(256->512, k=12, s=6)",
        "  [1, 512, 600]",
        "       |  ResnetBlock(512) + Conv1d(512->1024, k=16, s=8)",
        "  [1, 1024, 75]",
        "       |  Conv1d(1024->512, k=3, s=1)",
        "  [1, 512, 75]   <-- 25 Hz, 512 dims",
        "       |",
        "       |  Transpose to [1, 75, 512]",
        "       |  8x Transformer Layer:",
        "       |    LayerNorm -> Attn(8 heads, dim 64, RoPE) -> LayerScale",
        "       |    LayerNorm -> FFN(512->2048->512, GELU) -> LayerScale",
        "       |  Transpose to [1, 512, 75]",
        "  [1, 512, 75]   <-- 25 Hz, contextualized",
        "       |",
        "       |  Conv1d(512->512, k=4, s=2)  [downsampler]",
        "  [1, 512, 37]   <-- 12.5 Hz",
        "       |",
        "  ═════╪════════════════════ QUANTIZER ══════════════════════",
        "       |",
        "       |  Linear(512->256)",
        "  [1, 256, 37]",
        "       |",
        "       +---> Semantic VQ (1 CB x 2048 entries x 256 dim)",
        "       |        => codes[0]:  [1, 37] integers",
        "       |",
        "       +---> Acoustic RVQ (7 CB x 2048 entries x 256 dim)",
        "       |        => codes[1:8]: [7, 1, 37] integers",
        "       |",
        "  codes: [1, 8, 37]  <-- 8 tokens per frame, 37 frames",
        "       |",
        "  ═════╪═════════════════════ DECODER ═══════════════════════",
        "       |",
        "       |  Dequantize: lookup + add semantic & acoustic",
        "       |  Linear(256->512)",
        "  [1, 512, 37]   <-- 12.5 Hz",
        "       |",
        "       |  ConvT(512->512, k=4, s=2, groups=512) [upsampler]",
        "  [1, 512, 75]   <-- 25 Hz",
        "       |",
        "       |  8x Transformer Layer (same config as encoder)",
        "  [1, 512, 75]   <-- 25 Hz, contextualized",
        "       |",
        "       |  Conv1d(512->1024, k=3, s=1)",
        "  [1, 1024, 75]",
        "       |  ConvT(1024->512, k=16, s=8) + ResnetBlock",
        "  [1, 512, 600]",
        "       |  ConvT(512->256, k=12, s=6) + ResnetBlock",
        "  [1, 256, 3600]",
        "       |  ConvT(256->128, k=10, s=5) + ResnetBlock",
        "  [1, 128, 18000]",
        "       |  ConvT(128->64, k=8, s=4) + ResnetBlock",
        "  [1, 64, 72000]",
        "       |  Conv1d(64->1, k=3, s=1)",
        "  [1, 1, 72000]  <-- reconstructed 24kHz audio",
        "       |",
        "  OUTPUT AUDIO",
    ], "Figure 7.1: Complete Mimi architecture with all tensor shapes")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 8: KEY NUMBERS
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("8. Key Numbers & Parameters", level=1)

    doc.add_paragraph("A reference of all important numbers in one place.")

    add_number_table(doc,
        ["Category", "Parameter", "Value"],
        [
            ("Audio", "Sample rate", "24,000 Hz"),
            ("Audio", "Channels", "1 (mono)"),
            ("Audio", "Samples per frame", "1,920"),
            ("Audio", "Frame duration", "80 ms"),
            ("Tokens", "Token frame rate", "12.5 Hz"),
            ("Tokens", "Codebooks per frame", "8"),
            ("Tokens", "Codebook size", "2,048 entries"),
            ("Tokens", "Bits per codebook", "11 (log2 2048)"),
            ("Tokens", "Bitrate", "1,100 bps (1.1 kbps)"),
            ("Encoder", "Conv strides", "[4, 5, 6, 8]"),
            ("Encoder", "Conv total stride", "960"),
            ("Encoder", "Intermediate rate", "25 Hz"),
            ("Encoder", "Downsampler stride", "2"),
            ("Encoder", "Final rate", "12.5 Hz"),
            ("Encoder", "Total stride", "1,920"),
            ("Transformer", "Layers (per block)", "8"),
            ("Transformer", "Attention heads", "8"),
            ("Transformer", "Head dimension", "64"),
            ("Transformer", "Model dimension", "512"),
            ("Transformer", "FFN dimension", "2,048"),
            ("Transformer", "Context window", "250 frames (20s)"),
            ("Transformer", "Position encoding", "RoPE (theta=10000)"),
            ("Quantizer", "Semantic codebooks", "1"),
            ("Quantizer", "Acoustic codebooks", "7"),
            ("Quantizer", "Codebook dimension", "256"),
            ("Quantizer", "Projection", "512 <-> 256"),
            ("Model", "Total parameters", "96.2 M"),
            ("Model", "Latency", "80 ms (1 frame)"),
            ("Model", "Streaming", "Yes (fully causal)"),
        ],
        col_widths=[3, 4, 5.5]
    )

    doc.add_page_break()

    doc.add_heading("Parameter Count Breakdown", level=2)

    add_diagram_table(doc, [
        "WHERE ARE THE 96.2 MILLION PARAMETERS?",
        "",
        "  Component                          Parameters    % of Total",
        "  ──────────────────────────────────  ──────────   ──────────",
        "  Conv Encoder (4 blocks + convs)       ~15.0 M      15.6%",
        "  Encoder Transformer (8 layers)        ~25.2 M      26.2%",
        "  Downsampler (Conv1d)                   ~1.0 M       1.0%",
        "  Quantizer (8 codebooks + projections)  ~4.5 M       4.7%",
        "  Upsampler (ConvTranspose1d)            ~1.0 M       1.0%",
        "  Decoder Transformer (8 layers)        ~25.2 M      26.2%",
        "  Conv Decoder (4 blocks + convs)       ~15.0 M      15.6%",
        "  Other (norms, biases, etc.)            ~9.3 M       9.7%",
        "  ──────────────────────────────────  ──────────   ──────────",
        "  TOTAL                                 ~96.2 M     100.0%",
        "",
        "  Visual bar chart:",
        "  Conv Encoder:      ████████████████   15.6%",
        "  Enc Transformer:   ██████████████████████████   26.2%",
        "  Downsampler:       █   1.0%",
        "  Quantizer:         █████   4.7%",
        "  Upsampler:         █   1.0%",
        "  Dec Transformer:   ██████████████████████████   26.2%",
        "  Conv Decoder:      ████████████████   15.6%",
        "",
        "  The two Transformers together = 52.4% of all parameters!",
    ], "Figure 8.1: Parameter distribution across components")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 9: TRAINING PROCESS
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("9. The Training Process", level=1)

    doc.add_paragraph(
        "Understanding how Mimi is trained helps you understand its strengths "
        "and limitations -- important for fine-tuning it for Vietnamese."
    )

    doc.add_heading("9.1 Training Configuration", level=2)

    add_number_table(doc,
        ["Parameter", "Value"],
        [
            ("Optimizer", "AdamW"),
            ("Learning rate", "8e-4"),
            ("Weight decay", "5e-2 (Transformer only)"),
            ("Batch size", "128"),
            ("Training steps", "4,000,000 (4M)"),
            ("Audio crop length", "12 seconds (random)"),
            ("Training codebooks", "Up to 32 (with dropout)"),
            ("Inference codebooks", "8"),
        ],
        col_widths=[5, 8]
    )

    doc.add_heading("9.2 Loss Functions", level=2)

    add_diagram_table(doc, [
        "MIMI TRAINING LOSSES:",
        "",
        "  Original Audio ──────────────────────────────────┐",
        "       |                                           |",
        "       v                                           |",
        "   [Mimi Encoder] --> [Quantizer] --> [Decoder]    |",
        "       |                   |              |        |",
        "       |                   |              v        v",
        "       |                   |     Reconstructed   Original",
        "       |                   |         Audio        Audio",
        "       |                   |              |        |",
        "       |                   |              v        v",
        "       |                   |     ┌──────────────────────┐",
        "       |                   |     │ Loss 1: ADVERSARIAL  │",
        "       |                   |     │ (GAN discriminator)  │",
        "       |                   |     │ 'Does it sound real?'│",
        "       |                   |     └──────────────────────┘",
        "       |                   |              |        |",
        "       |                   |              v        v",
        "       |                   |     ┌──────────────────────┐",
        "       |                   |     │ Loss 2: FEATURE      │",
        "       |                   |     │ MATCHING              │",
        "       |                   |     │ (match disc features) │",
        "       |                   |     └──────────────────────┘",
        "       |                   |",
        "       v                   v",
        "  [Mimi CB0]         [WavLM]",
        "   output              embeddings",
        "       |                   |",
        "       v                   v",
        "  ┌──────────────────────────────┐",
        "  │ Loss 3: DISTILLATION         │",
        "  │ cosine_similarity(CB0, WavLM)│",
        "  │ 'Match the teacher!'         │",
        "  └──────────────────────────────┘",
        "",
        "  Also:",
        "  Loss 4: COMMITMENT (standard VQ: stay close to codebook)",
    ], "Figure 9.1: Training loss functions")

    doc.add_paragraph()

    add_info_box(doc,
        "KEY FINDING: The Moshi paper found that PURE adversarial training "
        "(removing the reconstruction loss entirely) produces much better "
        "perceived quality: MUSHRA score 81.0 vs 58.8. The discriminator "
        "teaches the model what 'real audio' sounds like, which is more "
        "effective than pixel-level matching.",
        "key"
    )

    doc.add_heading("9.3 RVQ Dropout (Bitrate Scalability)", level=2)

    add_diagram_table(doc, [
        "RVQ DROPOUT DURING TRAINING:",
        "",
        "  During each training step, randomly choose how many",
        "  codebooks to use (1 to 32):",
        "",
        "  Step 100:  Use 5 codebooks   CB0 CB1 CB2 CB3 CB4 ---- ----",
        "  Step 101:  Use 2 codebooks   CB0 CB1 ---- ---- ---- ---- ----",
        "  Step 102:  Use 8 codebooks   CB0 CB1 CB2 CB3 CB4 CB5 CB6 CB7",
        "  Step 103:  Use 1 codebook    CB0 ---- ---- ---- ---- ---- ----",
        "  Step 104:  Use 32 codebooks  CB0 CB1 CB2 ... CB31",
        "",
        "  Also: 50% chance of SKIPPING quantization entirely",
        "  (continuous embeddings pass through directly)",
        "",
        "  Result: Mimi works well with ANY number of codebooks",
        "          from 1 (very compressed) to 32 (high quality)",
    ], "Figure 9.2: RVQ dropout for robustness")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 10: MIMI IN PERSONAPLEX
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("10. How Mimi is Used in PersonaPlex", level=1)

    doc.add_paragraph(
        "PersonaPlex (NVIDIA) uses Mimi as the audio tokenizer in a full-duplex "
        "conversational AI system. Here's how Mimi fits into the bigger picture."
    )

    add_diagram_table(doc, [
        "PERSONAPLEX SYSTEM ARCHITECTURE:",
        "",
        "  User speaks                          System speaks",
        "       |                                    ^",
        "       v                                    |",
        "  ┌─────────┐                          ┌─────────┐",
        "  │  Mimi    │                          │  Mimi    │",
        "  │ ENCODER  │                          │ DECODER  │",
        "  │          │                          │          │",
        "  │ audio -> │                          │ tokens ->│",
        "  │ tokens   │                          │ audio    │",
        "  └────┬─────┘                          └────^─────┘",
        "       |                                     |",
        "       | User audio tokens                   | System audio tokens",
        "       | [8 codebooks, T]                    | [8 codebooks, T]",
        "       |                                     |",
        "       v                                     |",
        "  ┌──────────────────────────────────────────────┐",
        "  │            MOSHI / PERSONAPLEX LM             │",
        "  │            (7 Billion parameters)              │",
        "  │                                                │",
        "  │  Input streams:           Output streams:      │",
        "  │  - User audio (8 tokens)  - System audio      │",
        "  │  - Text token             - Text token         │",
        "  │  Total: 9 token streams   Total: 9 streams     │",
        "  │                                                │",
        "  │  + Voice Prompt  (who to sound like)           │",
        "  │  + Text Prompt   (how to behave)               │",
        "  └────────────────────────────────────────────────┘",
        "",
        "  Mimi is the 'ears' and 'mouth' of PersonaPlex.",
        "  The LM is the 'brain'.",
    ], "Figure 10.1: Mimi's role in PersonaPlex")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "THE 9-TOKEN STREAM FORMAT:",
        "",
        "  At each time step (12.5 Hz = every 80ms), PersonaPlex processes:",
        "",
        "  Stream 0:  Text token        (from/to text vocabulary)",
        "  Stream 1:  Audio codebook 0  (semantic - what is said)",
        "  Stream 2:  Audio codebook 1  (acoustic coarse)",
        "  Stream 3:  Audio codebook 2  (acoustic detail)",
        "  Stream 4:  Audio codebook 3  (acoustic detail)",
        "  Stream 5:  Audio codebook 4  (acoustic detail)",
        "  Stream 6:  Audio codebook 5  (acoustic detail)",
        "  Stream 7:  Audio codebook 6  (acoustic detail)",
        "  Stream 8:  Audio codebook 7  (acoustic finest)",
        "",
        "  Training tensor shape: codes[Batch, 9, Time]",
        "",
        "  The text stream allows the model to 'think in text'",
        "  while 'speaking in audio' -- this is key to intelligence!",
    ], "Figure 10.2: The 9-token stream format")

    doc.add_paragraph()

    add_diagram_table(doc, [
        "VOICE PROMPT: How Mimi Creates a Voice Identity",
        "",
        "  Reference audio (3-10 seconds of target voice)",
        "       |",
        "       v",
        "  ┌─────────────────────────────────────────────┐",
        "  │  1. Mimi ENCODES reference audio             │",
        "  │     audio -> [8 tokens per frame]            │",
        "  │     These tokens capture the voice identity  │",
        "  └─────────────────────┬───────────────────────┘",
        "                        |",
        "                        v",
        "  ┌─────────────────────────────────────────────┐",
        "  │  2. Moshi LM PROCESSES the Mimi tokens       │",
        "  │     Builds internal representation of voice  │",
        "  │     Stores in Transformer KV cache            │",
        "  └─────────────────────┬───────────────────────┘",
        "                        |",
        "                        v",
        "  ┌─────────────────────────────────────────────┐",
        "  │  3. Saved as .pt file:                       │",
        "  │     embeddings: [N_frames, 4096]             │",
        "  │     cache: Transformer KV cache state        │",
        "  │                                               │",
        "  │  This is NOT a simple vector!                 │",
        "  │  It's a full sequence + memory state.         │",
        "  └─────────────────────────────────────────────┘",
        "",
        "  During conversation, the LM 'remembers' this voice",
        "  and generates all speech tokens to match it.",
    ], "Figure 10.3: Voice prompt creation pipeline")

    doc.add_page_break()

    # ════════════════════════════════════════════════════════════════
    # CHAPTER 11: SUMMARY
    # ════════════════════════════════════════════════════════════════
    doc.add_heading("11. Summary & Quick Reference", level=1)

    doc.add_paragraph(
        "Here's everything you need to remember about Mimi in one page."
    )

    add_diagram_table(doc, [
        "MIMI IN ONE PICTURE:",
        "",
        "                24kHz audio",
        "                    |",
        "        ┌───────────┴───────────┐",
        "        |      ENCODER          |",
        "        |  Conv (stride 960)    |",
        "        |  Transformer (8L)     |",
        "        |  Downsample (x2)      |",
        "        └───────────┬───────────┘",
        "                    |",
        "              512-dim @ 12.5Hz",
        "                    |",
        "        ┌───────────┴───────────┐",
        "        |     QUANTIZER         |",
        "        |  ┌────────┬────────┐  |",
        "        |  |Semantic|Acoustic|  |",
        "        |  | 1 CB   | 7 CBs  |  |",
        "        |  | (what) | (how)  |  |",
        "        |  └────────┴────────┘  |",
        "        └───────────┬───────────┘",
        "                    |",
        "         8 integers @ 12.5Hz",
        "             = 1.1 kbps",
        "                    |",
        "        ┌───────────┴───────────┐",
        "        |      DECODER          |",
        "        |  Dequantize + add     |",
        "        |  Upsample (x2)       |",
        "        |  Transformer (8L)     |",
        "        |  DeConv (stride 960)  |",
        "        └───────────┬───────────┘",
        "                    |",
        "                24kHz audio",
        "           (reconstructed)",
    ], "Figure 11.1: Mimi architecture summary")

    doc.add_paragraph()

    add_info_box(doc,
        "REMEMBER THESE KEY NUMBERS:\n"
        "- 24,000 Hz input/output sample rate\n"
        "- 12.5 Hz token rate (80ms per frame)\n"
        "- 8 codebooks (1 semantic + 7 acoustic)\n"
        "- 2,048 entries per codebook\n"
        "- 512 dimensions (hidden size)\n"
        "- 1,920 samples per token (= 24000/12.5)\n"
        "- 1.1 kbps bitrate\n"
        "- 96.2M parameters\n"
        "- 8 Transformer layers in encoder AND decoder",
        "key"
    )

    doc.add_paragraph()

    doc.add_heading("Comparison with Other Codecs", level=2)

    add_number_table(doc,
        ["Codec", "Developer", "Bitrate", "Quality (MUSHRA)", "Semantic?"],
        [
            ("MP3", "Fraunhofer", "128 kbps", "~95", "No"),
            ("Opus", "IETF", "6-510 kbps", "~90 at 32kbps", "No"),
            ("EnCodec", "Meta", "1.5 kbps", "~70", "No"),
            ("Mimi", "Kyutai", "1.1 kbps", "81.0", "Yes (Split RVQ)"),
        ],
        col_widths=[2.5, 2.5, 2.5, 3, 3.5]
    )

    doc.add_paragraph(
        "Mimi achieves better quality than EnCodec at a LOWER bitrate, "
        "and its semantic tokens enable language model integration. "
        "Traditional codecs like MP3 and Opus are much higher bitrate "
        "but don't produce tokens suitable for language models."
    )

    doc.add_paragraph()
    doc.add_paragraph()

    # End note
    end = doc.add_paragraph()
    end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = end.add_run("--- End of Document ---")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    run.italic = True

    doc.add_paragraph()
    ref = doc.add_paragraph()
    ref.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ref.add_run(
        "References: Moshi Paper (arXiv 2410.00037), "
        "HuggingFace kyutai/mimi model card, "
        "Kyutai Labs codec-explainer"
    )
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    return doc


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc = build_document()
    doc.save(OUTPUT_PATH)
    print(f"Document saved to: {OUTPUT_PATH}")

    # Print stats
    para_count = len(doc.paragraphs)
    table_count = len(doc.tables)
    print(f"  Paragraphs: {para_count}")
    print(f"  Tables/Diagrams: {table_count}")
    print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")
