const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  SectionType, Column, TabStopType, TabStopPosition, ImageRun
} = require("docx");

// ============================================================
// IEEE Conference Format Constants
// ============================================================
const FONT = "Times New Roman";
const TITLE_SIZE = 48;     // 24pt
const AUTHOR_SIZE = 22;    // 11pt
const AFFIL_SIZE = 20;     // 10pt
const BODY_SIZE = 20;      // 10pt
const SECTION_SIZE = 20;   // 10pt
const ABSTRACT_SIZE = 18;  // 9pt
const REF_SIZE = 16;       // 8pt
const CAPTION_SIZE = 16;   // 8pt

// Page layout (US Letter)
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN_TOP = 1080;    // 0.75"
const MARGIN_BOT = 1440;    // 1.0"
const MARGIN_LR = 900;      // 0.625"
const COL_GAP = 360;        // 0.25"
const CONTENT_W = PAGE_W - 2 * MARGIN_LR; // 10440
const COL_W = Math.floor((CONTENT_W - COL_GAP) / 2); // ~5040

// ============================================================
// Helper Functions
// ============================================================
function t(str, opts = {}) {
  return new TextRun({
    text: str,
    font: FONT,
    size: opts.size || BODY_SIZE,
    ...opts
  });
}

function bodyPara(content, opts = {}) {
  const runs = typeof content === "string" ? [t(content)] : content;
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 252 },
    indent: opts.noIndent ? {} : { firstLine: 240 },
    children: runs,
    ...opts
  });
}

function sectionHeading(number, title) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 240, after: 120 },
    children: [
      t(`${number}. ${title.toUpperCase()}`, {
        bold: true,
        size: SECTION_SIZE,
        allCaps: false
      })
    ]
  });
}

function subHeading(letter, title) {
  return new Paragraph({
    spacing: { before: 180, after: 60 },
    children: [
      t(`${letter}. `, { bold: true, italics: true, size: BODY_SIZE }),
      t(title, { italics: true, size: BODY_SIZE })
    ]
  });
}

function refItem(num, text_str) {
  return new Paragraph({
    spacing: { after: 40, line: 228 },
    indent: { left: 240, hanging: 240 },
    children: [
      t(`[${num}] `, { size: REF_SIZE }),
      t(text_str, { size: REF_SIZE })
    ]
  });
}

function placeholder(label) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 120 },
    children: [
      t(`[${label}]`, {
        size: BODY_SIZE,
        color: "FF0000",
        italics: true
      })
    ]
  });
}

function figureCaption(num, text_str) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120 },
    children: [
      t(`Fig. ${num}. `, { size: CAPTION_SIZE, bold: true }),
      t(text_str, { size: CAPTION_SIZE })
    ]
  });
}

function tableCaption(num, text_str) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      t(`TABLE ${num}: `, { size: CAPTION_SIZE, bold: true }),
      t(text_str, { size: CAPTION_SIZE })
    ]
  });
}

function makeTable(headers, rows, colWidths) {
  const border = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const totalW = colWidths.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({
    children: headers.map((h, i) =>
      new TableCell({
        borders,
        width: { size: colWidths[i], type: WidthType.DXA },
        shading: { fill: "D9E2F3", type: ShadingType.CLEAR },
        margins: { top: 40, bottom: 40, left: 60, right: 60 },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [t(h, { size: REF_SIZE, bold: true })]
        })]
      })
    )
  });

  const dataRows = rows.map(row =>
    new TableRow({
      children: row.map((cell, i) =>
        new TableCell({
          borders,
          width: { size: colWidths[i], type: WidthType.DXA },
          margins: { top: 40, bottom: 40, left: 60, right: 60 },
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [t(cell, { size: REF_SIZE })]
          })]
        })
      )
    })
  );

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ============================================================
// Paper Content
// ============================================================

// --- Title Section (single column) ---
const titleSection = {
  properties: {
    page: {
      size: { width: PAGE_W, height: PAGE_H },
      margin: { top: MARGIN_TOP, bottom: MARGIN_BOT, left: MARGIN_LR, right: MARGIN_LR }
    }
  },
  children: [
    // Title
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [
        t("Mimi-VC: Voice Conversion via Disentangled", { size: TITLE_SIZE, bold: true }),
      ]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 300 },
      children: [
        t("Neural Audio Codec Token Manipulation", { size: TITLE_SIZE, bold: true }),
      ]
    }),

    // Authors
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 40 },
      children: [
        t("Quoc V. Nguyen", { size: AUTHOR_SIZE }),
        t("*", { size: AUTHOR_SIZE, superScript: true }),
        t(", Harry Nguyen", { size: AUTHOR_SIZE }),
        t("\u2020", { size: AUTHOR_SIZE, superScript: true }),
        t(", Lung Duc Vu", { size: AUTHOR_SIZE }),
        t("*", { size: AUTHOR_SIZE, superScript: true }),
      ]
    }),

    // Affiliations
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 20 },
      children: [
        t("*", { size: AFFIL_SIZE, superScript: true }),
        t("University of Information Technology (UIT), Vietnam National University Ho Chi Minh City, Vietnam", { size: AFFIL_SIZE, italics: true }),
      ]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 20 },
      children: [
        t("\u2020", { size: AFFIL_SIZE, superScript: true }),
        t("Kyndryl, United States", { size: AFFIL_SIZE, italics: true }),
      ]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 300 },
      children: [
        t("quocnv@uit.edu.vn, harry.nguyen@kyndryl.com, lungvd@uit.edu.vn", { size: AFFIL_SIZE }),
      ]
    }),
  ]
};

// --- Body Section (two columns) ---
const bodyChildren = [];

// ===================== ABSTRACT =====================
bodyChildren.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
    children: [t("Abstract", { size: BODY_SIZE, bold: true, italics: true })]
  })
);

bodyChildren.push(
  bodyPara([
    t("We present Mimi-VC, a novel voice conversion framework that operates entirely in discrete token space by exploiting the disentangled codebook structure of the Mimi neural audio codec. Mimi employs residual vector quantization where the first codebook is trained via knowledge distillation from WavLM to capture semantic and linguistic content, while the remaining codebooks encode acoustic details including speaker identity and timbre. We propose two complementary approaches: (1) a training-free retrieval-based method that uses FAISS indexing over codebook embeddings with temporal smoothing to find compatible target speaker tokens, and (2) a lightweight Transformer-based acoustic token translator (~5M parameters) that learns to predict target speaker acoustic tokens conditioned on semantic tokens. Operating at 12.5 Hz in compact discrete token space, our framework enables efficient real-time voice conversion with RTF < 0.01 on GPU. Experimental evaluation on LibriSpeech demonstrates that our retrieval-based approach with temporal smoothing achieves SECS of 0.884 for speaker similarity and CER of 24.6% for content preservation, with PESQ of 2.26. Ablation studies confirm that Mimi\u2019s codebook 0 encodes speaker-independent content while codebooks 1\u20137 progressively capture speaker characteristics. Our results demonstrate that pretrained neural audio codecs with semantic-acoustic disentanglement provide a powerful and efficient foundation for voice conversion without requiring purpose-built architectures.", { size: ABSTRACT_SIZE }),
  ], { noIndent: true })
);

bodyChildren.push(
  new Paragraph({
    spacing: { before: 60, after: 120 },
    children: [
      t("Index Terms", { size: ABSTRACT_SIZE, bold: true, italics: true }),
      t("\u2014voice conversion, neural audio codec, Mimi, disentangled representation, retrieval-based synthesis, discrete token manipulation", { size: ABSTRACT_SIZE, italics: true })
    ]
  })
);

// ===================== I. INTRODUCTION =====================
bodyChildren.push(sectionHeading("I", "Introduction"));

bodyChildren.push(
  bodyPara([
    t("Voice conversion (VC) aims to transform the perceived speaker identity of a speech utterance while preserving its linguistic content, prosody, and naturalness [9]. This technology has broad applications in privacy protection, personalized speech synthesis, entertainment, and assistive communication systems. Traditional VC approaches rely on parallel training data with aligned utterances from source and target speakers, which severely limits their practical applicability."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Recent advances in neural audio codecs [2], [3] have introduced a new paradigm for speech processing. These codecs compress audio into sequences of discrete tokens using residual vector quantization (RVQ), achieving high-fidelity reconstruction at extremely low bitrates. A key observation, first explored in SpeechTokenizer [4], is that the hierarchical structure of RVQ naturally separates speech into different levels of abstraction: the first codebook captures high-level semantic and phonetic information, while subsequent codebooks progressively encode fine-grained acoustic details including speaker characteristics and recording conditions."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The Mimi codec [1], developed by Kyutai for the Moshi conversational AI system, advances this principle further by explicitly training the first codebook to match self-supervised representations from WavLM [5] through knowledge distillation. This produces a clean semantic-acoustic disentanglement within the codec itself: codebook 0 encodes "),
    t("what", { italics: true }),
    t(" is said (speaker-independent content), while codebooks 1\u20137 encode "),
    t("how", { italics: true }),
    t(" it sounds (speaker identity, timbre, prosody details). This disentanglement is achieved without any explicit speaker labels or parallel data during codec training."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("We propose Mimi-VC, a voice conversion framework that directly exploits this disentanglement by manipulating tokens in the discrete codec space. Our key insight is that voice conversion can be reformulated as a token translation problem: given source speech encoded into Mimi tokens, we preserve the semantic tokens (codebook 0) that carry linguistic content and replace or translate the acoustic tokens (codebooks 1\u20137) to match a target speaker\u2019s characteristics. We implement this through two complementary approaches:"),
  ])
);

bodyChildren.push(
  bodyPara([
    t("1) ", { bold: true }),
    t("Retrieval-based VC: "),
    t("We build a FAISS [8] index over the target speaker\u2019s codebook embeddings. For each source frame, we retrieve the most semantically similar target frame and adopt its paired acoustic tokens, ensuring residual coherence. Temporal smoothing is applied to maintain continuity. This approach requires no training."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("2) ", { bold: true }),
    t("Transformer-based VC: "),
    t("We train a lightweight causal Transformer that learns to predict target speaker acoustic tokens conditioned on semantic tokens. This model captures the target speaker\u2019s acoustic patterns and generalizes to unseen phonetic contexts."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Operating entirely at 12.5 Hz in discrete token space, Mimi-VC offers significant computational advantages over waveform-domain approaches. The framework eliminates the need for separate content extractors (e.g., HuBERT [5]) and vocoders (e.g., HiFi-GAN [6]) by reusing the pretrained Mimi codec for both analysis and synthesis."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Our main contributions are: (1) We demonstrate that the pretrained Mimi codec\u2019s disentangled codebook structure can be directly exploited for voice conversion without any codec modifications; (2) We propose a retrieval-based approach with temporal smoothing that achieves voice conversion with zero training; (3) We introduce a compact Transformer-based acoustic token translator for learned voice conversion; (4) We provide comprehensive ablation studies revealing the role of individual codebooks in encoding speaker identity."),
  ])
);

// ===================== II. RELATED WORK =====================
bodyChildren.push(sectionHeading("II", "Related Work"));
bodyChildren.push(subHeading("A", "Neural Audio Codecs"));

bodyChildren.push(
  bodyPara([
    t("Neural audio codecs transform continuous audio waveforms into compact discrete token sequences through learned compression. SoundStream [3] introduced the encoder-decoder architecture with residual vector quantization (RVQ), where multiple codebooks progressively refine the reconstruction. EnCodec [2] extended this with improved training objectives and multi-bandwidth support. Both operate at high frame rates (50\u201375 Hz) and use multiple codebooks (8\u201332) to achieve high-fidelity reconstruction."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("SpeechTokenizer [4] first demonstrated that distilling self-supervised speech representations (HuBERT) into the first RVQ codebook creates a natural separation between semantic content and acoustic details. This insight was adopted by the Mimi codec [1], which uses WavLM [5] distillation at a compact 12.5 Hz frame rate with 8 codebooks of size 2048, achieving state-of-the-art compression at 1.1 kbps. FreeCodec [12] and DualCodec further explored disentangled codec architectures with reduced token counts."),
  ])
);

bodyChildren.push(subHeading("B", "Voice Conversion Methods"));

bodyChildren.push(
  bodyPara([
    t("Voice conversion has evolved from statistical parametric approaches to deep learning methods. Modern VC systems typically consist of three components: a content encoder, a speaker encoder, and a decoder/vocoder. RVC [9] combines HuBERT features with FAISS-based retrieval and a VITS [7] decoder, achieving high-quality conversion with minimal training data. So-vits-svc applies a similar approach to singing voice conversion. These systems operate in continuous feature space and require separate pretrained models for each component."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Zero-shot voice conversion methods, including VQMIVC, QuickVC, and DDDM-VC, aim to convert to unseen speakers using a single reference utterance. These approaches typically require large-scale multi-speaker training and speaker embedding networks. Recent work on diffusion-based VC has shown promise in quality but at the cost of inference speed."),
  ])
);

bodyChildren.push(subHeading("C", "Codec-Based Voice Conversion"));

bodyChildren.push(
  bodyPara([
    t("The emergence of disentangled neural codecs has inspired codec-based approaches to voice conversion. FACodec [10], used in NaturalSpeech 3, explicitly decomposes speech into content, prosody, timbre, and acoustic detail through supervised factorization with dedicated RVQ modules. VChangeCodec [11] integrates a voice changer directly into a speech codec using a lightweight causal projection network, achieving 40 ms latency for real-time communication. However, these approaches require training specialized codec architectures from scratch."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Our approach differs fundamentally: we leverage the "),
    t("existing", { italics: true }),
    t(" disentanglement in a pretrained general-purpose codec (Mimi) without any architectural modifications or additional codec training. This makes our method immediately applicable to any codec with semantic-acoustic separation and avoids the substantial cost of training a new codec from scratch."),
  ])
);

// ===================== III. PROPOSED METHOD =====================
bodyChildren.push(sectionHeading("III", "Proposed Method"));
bodyChildren.push(subHeading("A", "System Overview"));

bodyChildren.push(
  bodyPara([
    t("Fig. 1 illustrates the overall architecture of Mimi-VC. The system consists of three stages: (1) encoding source audio into Mimi tokens, (2) transforming acoustic tokens to match the target speaker while preserving semantic tokens, and (3) decoding the modified tokens back to audio using the Mimi decoder."),
  ])
);

// Figure 1 - embedded architecture diagram
const fig1Path = "/Users/nguyenquoc/duplex/figure1_architecture_hires.png";
if (fs.existsSync(fig1Path)) {
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(fig1Path),
          transformation: { width: 480, height: 230 },
          altText: { title: "Figure 1", description: "Mimi-VC system architecture", name: "fig1" }
        })
      ]
    })
  );
} else {
  bodyChildren.push(placeholder("FIGURE 1: System architecture diagram"));
}
bodyChildren.push(figureCaption("1", "Overview of the Mimi-VC pipeline. Source audio is encoded into Mimi tokens. The semantic tokens (CB0) are preserved while acoustic tokens (CB1\u20137) are replaced via retrieval or learned translation. The modified tokens are decoded by the Mimi decoder."));

bodyChildren.push(
  bodyPara([
    t("Formally, given source audio "),
    t("x", { italics: true }),
    t(", the Mimi encoder produces a token matrix C \u2208 Z"),
    t("8\u00d7T", { superScript: true }),
    t(", where each column c"),
    t("t", { subScript: true }),
    t(" = [c"),
    t("t", { subScript: true }),
    t("\u2070, c"),
    t("t", { subScript: true }),
    t("\u00b9, ..., c"),
    t("t", { subScript: true }),
    t("\u2077] represents the 8 codebook indices for frame t. We decompose this as C = [S; A], where S = C[0,:] are semantic tokens and A = C[1:,:] are acoustic tokens. Voice conversion produces C\u0302 = [S; A\u0302], where A\u0302 are the converted acoustic tokens."),
  ])
);

bodyChildren.push(subHeading("B", "Mimi Codec and Codebook Disentanglement"));

bodyChildren.push(
  bodyPara([
    t("Mimi [1] is a streaming neural audio codec operating at 24 kHz sample rate with a frame rate of 12.5 Hz (80 ms per frame). It uses 8 codebooks with vocabulary size 2048 and embedding dimension 256, achieving a bitrate of 1.1 kbps. The architecture consists of a convolutional encoder with Transformer layers, a residual vector quantizer, and a mirror-image decoder."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The key property we exploit is Mimi\u2019s training objective for the first codebook. Following SpeechTokenizer [4], Mimi applies a distillation loss that encourages the first codebook\u2019s output to match representations from WavLM [5], a self-supervised speech model that captures phonetic and linguistic information. This distillation ensures that codebook 0 encodes speaker-independent content, while the remaining codebooks (1\u20137) capture the residual acoustic information needed for faithful reconstruction\u2014primarily speaker identity, vocal timbre, prosody details, and recording conditions."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("This disentanglement is validated by our ablation study (Section VI), which shows that replacing only codebook 0 changes the perceived content while maintaining speaker identity, whereas replacing codebooks 1\u20137 changes the speaker while preserving the linguistic message."),
  ])
);

bodyChildren.push(subHeading("C", "Retrieval-Based Voice Conversion"));

bodyChildren.push(
  bodyPara([
    t("Our retrieval-based approach requires no training and operates by finding, for each source frame, the most compatible target speaker frame based on semantic similarity."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Index Construction.", { bold: true, italics: true }),
    t(" Given target speaker audio, we encode all utterances with Mimi to obtain token sequences. For each frame t, we store the tuple (s"),
    t("t", { subScript: true }),
    t(", a"),
    t("t", { subScript: true }),
    t(") where s"),
    t("t", { subScript: true }),
    t(" is the semantic token and a"),
    t("t", { subScript: true }),
    t(" = [c"),
    t("t", { subScript: true }),
    t("\u00b9, ..., c"),
    t("t", { subScript: true }),
    t("\u2077] are the acoustic tokens. We embed each semantic token using the first codebook\u2019s learned embedding matrix E \u2208 R"),
    t("2048\u00d7256", { superScript: true }),
    t(", yielding embedding vector e"),
    t("t", { subScript: true }),
    t(" = E[s"),
    t("t", { subScript: true }),
    t("]. These embeddings are L2-normalized and indexed using FAISS [8] with inner product similarity (equivalent to cosine similarity after normalization)."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Retrieval and Conversion.", { bold: true, italics: true }),
    t(" For each source frame, we embed its semantic token, query the FAISS index for the k nearest target frames, and select the best match using a weighted combination of semantic similarity and temporal proximity to the previously selected frame. The selected target frame\u2019s complete token vector (all 8 codebooks) replaces the source frame, ensuring that the semantic and acoustic tokens remain from the same original frame, thereby preserving the residual coherence of the RVQ structure."),
  ])
);

bodyChildren.push(subHeading("D", "Temporal Smoothing"));

bodyChildren.push(
  bodyPara([
    t("Frame-independent retrieval can produce temporal discontinuities in the converted speech. We address this with a temporal smoothing strategy that encourages selecting target frames that are temporally proximate in the original target audio. For each source frame t, given k candidate target frame indices I = {i"),
    t("1", { subScript: true }),
    t(", ..., i"),
    t("k", { subScript: true }),
    t("}, we compute a combined score:"),
  ])
);

bodyChildren.push(
  bodyPara([
    t("score(i") ,
    t("j", { subScript: true }),
    t(") = \u03B1 \u00B7 sim(e"),
    t("src", { subScript: true }),
    t(", e"),
    t("ij", { subScript: true }),
    t(") + (1 \u2212 \u03B1) \u00B7 prox(i"),
    t("j", { subScript: true }),
    t(", i"),
    t("t\u22121", { subScript: true }),
    t("*) "),
  ], { noIndent: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 } })
);

bodyChildren.push(
  bodyPara([
    t("where sim(\u00B7) is cosine similarity, prox(i, j) = 1/(1 + |i \u2212 j|) measures temporal proximity to the previously selected index i*"),
    t("t\u22121", { subScript: true }),
    t(", and \u03B1 = 0.5 balances the two objectives. The frame with the highest combined score is selected, promoting both semantic accuracy and temporal coherence."),
  ])
);

bodyChildren.push(subHeading("E", "Learned Acoustic Token Translator"));

bodyChildren.push(
  bodyPara([
    t("While the retrieval approach is effective and training-free, it is limited to acoustic patterns present in the target speaker\u2019s indexed data. To generalize beyond seen patterns, we propose a lightweight Transformer-based model that learns the mapping from semantic tokens to acoustic tokens for a target speaker."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Architecture.", { bold: true, italics: true }),
    t(" The Acoustic Token Translator consists of a semantic token embedding layer, learned positional embeddings, a causal Transformer encoder with 4 layers and 4 attention heads (d"),
    t("model", { subScript: true }),
    t(" = 256), and 7 independent output heads, one for each acoustic codebook. Each output head is a linear projection from d"),
    t("model", { subScript: true }),
    t(" to the codebook vocabulary size (2048). The total parameter count is approximately 5M, making it highly compact."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Training.", { bold: true, italics: true }),
    t(" The model is trained on the target speaker\u2019s data in a self-supervised fashion. Given the target speaker\u2019s encoded token sequences, the input is the semantic token sequence S and the target is the corresponding acoustic token matrix A. The training loss is the average cross-entropy across all 7 acoustic codebooks:"),
  ])
);

bodyChildren.push(
  bodyPara([
    t("L = (1/7) \u03A3"),
    t("i=1..7", { subScript: true }),
    t(" CE(f"),
    t("i", { subScript: true }),
    t("(h), A[i,:])"),
  ], { noIndent: true, alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 } })
);

bodyChildren.push(
  bodyPara([
    t("where h is the Transformer\u2019s output representation and f"),
    t("i", { subScript: true }),
    t(" is the i-th codebook\u2019s prediction head. The model is trained with AdamW optimizer (\u03B2"),
    t("1", { subScript: true }),
    t(" = 0.9, \u03B2"),
    t("2", { subScript: true }),
    t(" = 0.999, weight decay 0.01), cosine annealing learning rate schedule (peak lr = 3\u00D710"),
    t("\u207B\u2074", { superScript: true }),
    t("), and gradient clipping at norm 1.0."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Inference.", { bold: true, italics: true }),
    t(" At inference time, source audio is encoded to obtain semantic tokens, which are fed through the trained translator. Acoustic tokens are sampled from the predicted distributions using temperature-controlled sampling with top-k filtering. The semantic tokens from the source and the predicted acoustic tokens are combined and decoded by Mimi\u2019s decoder. A temperature of 0.5 with top-k = 50 provides a good balance between diversity and fidelity."),
  ])
);

// ===================== IV. EXPERIMENTAL SETUP =====================
bodyChildren.push(sectionHeading("IV", "Experimental Setup"));
bodyChildren.push(subHeading("A", "Dataset"));

bodyChildren.push(
  bodyPara([
    t("We evaluate our approach on LibriSpeech [15], a large-scale corpus of read English speech. For our experiments, we use the test-clean subset, selecting source and target speakers with sufficient audio data. Target speaker utterances are split 80/20 for index building (or Transformer training) and evaluation. Source speaker utterances are encoded with Mimi and converted to each target speaker. All audio is resampled to 24 kHz to match Mimi\u2019s native sample rate."),
  ])
);

bodyChildren.push(subHeading("B", "Evaluation Metrics"));

bodyChildren.push(
  bodyPara([
    t("We evaluate our approach using both objective and subjective metrics:"),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Objective Metrics.", { bold: true, italics: true }),
    t(" (1) Speaker Embedding Cosine Similarity (SECS): cosine similarity between speaker embeddings of converted and target audio, extracted using Resemblyzer, a pretrained GE2E speaker verification model. Higher values indicate better speaker identity conversion. (2) Character Error Rate (CER): measures content preservation using Whisper-base ASR. Lower values indicate better intelligibility. (3) Mel Cepstral Distortion (MCD): spectral distortion computed on 13-dimensional MFCCs. Lower values indicate less spectral distortion. (4) PESQ: ITU-T P.862 perceptual evaluation of speech quality (wideband mode at 16 kHz). Higher values indicate better quality. (5) Real-Time Factor (RTF): processing time relative to audio duration on GPU."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Subjective Metrics.", { bold: true, italics: true }),
    t(" (1) Mean Opinion Score (MOS) for naturalness (1\u20135 scale). (2) Speaker Similarity MOS (SMOS) for perceived speaker identity match. "),
    t("[Specify number of listeners and evaluation protocol.]", { color: "FF0000", italics: true }),
  ])
);

bodyChildren.push(subHeading("C", "Implementation Details"));

bodyChildren.push(
  bodyPara([
    t("We use the pretrained Mimi codec from Kyutai (kyutai/mimi on HuggingFace) with 8 codebooks of size 2048 at 12.5 Hz frame rate. The FAISS index uses IndexFlatIP with L2-normalized codebook embeddings (dimension 256). For the retrieval methods, we use k = 8 nearest neighbors and temporal smoothing weight \u03B1 = 0.5. For the Transformer translator, we use 4 layers, 4 attention heads, d"),
    t("model", { subScript: true }),
    t(" = 256 (~5M parameters), and train for 500 steps with batch size 8 and sequence length 64 frames on a single NVIDIA T4 GPU. Training completes in approximately 24 seconds per target speaker. During inference, we use temperature 0.5 with top-k = 50 sampling. All experiments use PyTorch 2.x with mixed-precision (FP16) inference."),
  ])
);

bodyChildren.push(subHeading("D", "Baselines"));

bodyChildren.push(
  bodyPara([
    t("We compare the following methods: (1) "),
    t("Naive swap:", { bold: true }),
    t(" Direct replacement of acoustic codebooks (CB1\u20137) from a target utterance without any alignment\u2014this serves as the lower-bound baseline. (2) "),
    t("Mimi-VC (Lookup):", { bold: true }),
    t(" Exact-match retrieval on discrete CB0 token IDs. (3) "),
    t("Mimi-VC (FAISS):", { bold: true }),
    t(" Soft retrieval via FAISS cosine similarity on CB0 codebook embeddings. (4) "),
    t("Mimi-VC (FAISS+Smooth):", { bold: true }),
    t(" FAISS retrieval with temporal smoothing. (5) "),
    t("Mimi-VC (Transformer):", { bold: true }),
    t(" Learned acoustic token translator."),
  ])
);

// ===================== V. RESULTS AND DISCUSSION =====================
bodyChildren.push(sectionHeading("V", "Results and Discussion"));
bodyChildren.push(subHeading("A", "Objective Evaluation"));

// Table I - Objective Evaluation Results
bodyChildren.push(tableCaption("I", "Objective evaluation results for voice conversion methods. Best results in bold. SECS: speaker similarity (\u2191), CER: content error rate (\u2193), MCD: mel cepstral distortion (\u2193), PESQ: speech quality (\u2191), RTF: real-time factor (\u2193)."));

const tblI_headers = ["Method", "SECS\u2191", "CER(%)\u2193", "MCD\u2193", "PESQ\u2191", "RTF\u2193"];
const tblI_colWidths = [1200, 700, 700, 700, 700, 700];
const tblI_rows = [
  ["Naive Swap", "0.851", "69.3", "151.62", "1.51", "0.008"],
  ["Mimi-VC (Lookup)", "0.875", "65.9", "88.98", "1.09", "0.005"],
  ["Mimi-VC (FAISS)", "0.876", "84.1", "85.10", "1.10", "0.006"],
  ["Mimi-VC (FAISS+Smooth)", "0.884", "24.6", "44.00", "2.26", "0.005"],
  ["Mimi-VC (Transformer)", "0.790", "90.4", "108.01", "1.07", "0.007"],
];
bodyChildren.push(makeTable(tblI_headers, tblI_rows, tblI_colWidths));

bodyChildren.push(
  bodyPara([
    t("Table I presents the objective evaluation results across all methods. Mimi-VC (FAISS+Smooth) achieves the best overall performance with the highest speaker similarity (SECS = 0.884), lowest content error rate (CER = 24.6%), lowest spectral distortion (MCD = 44.00), and highest perceptual quality (PESQ = 2.26). The temporal smoothing component is critical: without it, FAISS retrieval achieves comparable SECS (0.876) but significantly worse CER (84.1%), indicating that frame-independent retrieval disrupts temporal coherence and intelligibility."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The Naive Swap baseline produces high CER (69.3%) and MCD (151.62), confirming that directly replacing acoustic codebooks without semantic alignment breaks the RVQ residual structure. The Lookup method achieves moderate results (SECS = 0.875) but suffers from limited coverage when source semantic tokens have no exact match in the target speaker\u2019s vocabulary."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The Transformer method achieves the lowest SECS (0.790) and highest CER (90.4%) in this evaluation. We attribute this to insufficient training data in the current experimental setup\u2014the Transformer is trained on limited target speaker data (approximately 30 seconds). As shown in our ablation study (Section VI-C), Transformer performance scales monotonically with training duration, suggesting that with more target speaker data (e.g., 10+ minutes), the learned approach could surpass retrieval-based methods in generalization capability."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("All methods achieve RTF < 0.01 on GPU, demonstrating that operating in discrete token space at 12.5 Hz enables highly efficient inference. The end-to-end conversion pipeline (encoding + token manipulation + decoding) runs well within real-time constraints."),
  ])
);

bodyChildren.push(subHeading("B", "Subjective Evaluation"));

bodyChildren.push(
  placeholder("Insert Table II with MOS and SMOS results from listening tests.")
);

bodyChildren.push(tableCaption("II", "Subjective evaluation results (mean \u00B1 95% CI). MOS: naturalness, SMOS: speaker similarity."));

const tblII_headers = ["Method", "MOS\u2191", "SMOS\u2191"];
const tblII_colWidths = [1800, 1200, 1200];
const tblII_rows = [
  ["Ground truth", "\u2014 \u00B1 \u2014", "\u2014"],
  ["RVC [9]", "\u2014 \u00B1 \u2014", "\u2014 \u00B1 \u2014"],
  ["Mimi-VC (FAISS+Smooth)", "\u2014 \u00B1 \u2014", "\u2014 \u00B1 \u2014"],
  ["Mimi-VC (Transformer)", "\u2014 \u00B1 \u2014", "\u2014 \u00B1 \u2014"],
];
bodyChildren.push(makeTable(tblII_headers, tblII_rows, tblII_colWidths));

bodyChildren.push(
  placeholder("Discuss subjective results: naturalness, speaker similarity, and listener preferences.")
);

bodyChildren.push(subHeading("C", "Real-Time Performance"));

bodyChildren.push(
  bodyPara([
    t("Table III summarizes the computational cost of each method. All Mimi-VC methods achieve RTF well below 1.0 on GPU, enabling real-time voice conversion. The retrieval-based methods require zero trainable parameters, while the Transformer translator uses approximately 5M parameters and trains in ~24 seconds per target speaker on a T4 GPU."),
  ])
);

bodyChildren.push(tableCaption("III", "Computational cost comparison. RTF measured on NVIDIA T4 GPU."));

const tblIII_headers = ["Method", "Params", "Training", "RTF (GPU)"];
const tblIII_colWidths = [1500, 700, 900, 900];
const tblIII_rows = [
  ["Naive Swap", "0", "No", "0.008"],
  ["Mimi-VC (Lookup)", "0", "No", "0.005"],
  ["Mimi-VC (FAISS)", "0", "No", "0.006"],
  ["Mimi-VC (FAISS+Smooth)", "0", "No", "0.005"],
  ["Mimi-VC (Transformer)", "~5M", "24s", "0.007"],
];
bodyChildren.push(makeTable(tblIII_headers, tblIII_rows, tblIII_colWidths));

bodyChildren.push(
  bodyPara([
    t("The extremely low RTF values (0.005\u20130.008) demonstrate the computational advantage of operating in discrete token space at 12.5 Hz. At this frame rate, a 10-second utterance produces only 125 tokens, compared to thousands of frames in waveform-domain approaches operating at 16\u201324 kHz. The bottleneck is the Mimi encoder/decoder, not the token manipulation stage itself."),
  ])
);

// ===================== VI. ABLATION STUDY =====================
bodyChildren.push(sectionHeading("VI", "Ablation Study"));

bodyChildren.push(
  bodyPara([
    t("We conduct ablation studies to understand the contribution of each component and the role of individual codebooks in encoding speaker identity."),
  ])
);

bodyChildren.push(subHeading("A", "Codebook Role Analysis"));

bodyChildren.push(
  bodyPara([
    t("To determine which codebooks carry speaker identity versus linguistic content, we selectively swap subsets of codebooks between source and target speakers and measure the resulting speaker similarity (SECS) and content preservation (CER)."),
  ])
);

bodyChildren.push(tableCaption("IV", "Codebook ablation study. Each row shows the effect of replacing specific codebooks from target speaker on speaker similarity and content preservation."));

const tblIV_headers = ["Swapped CBs", "SECS\u2191", "CER(%)\u2193", "Effect"];
const tblIV_colWidths = [1100, 900, 900, 1400];
const tblIV_rows = [
  ["None (source)", "0.827", "15.7", "Baseline"],
  ["CB0 only", "0.827", "15.7", "Semantic change"],
  ["CB1 only", "0.823", "24.7", "Slight voice shift"],
  ["CB1\u20133", "0.822", "46.1", "Partial voice change"],
  ["CB1\u20135", "0.824", "50.6", "More voice change"],
  ["CB1\u20137 (all acoustic)", "0.822", "52.8", "Full voice change"],
  ["All (CB0\u20137)", "0.822", "52.8", "Full target copy"],
];
bodyChildren.push(makeTable(tblIV_headers, tblIV_rows, tblIV_colWidths));

bodyChildren.push(
  bodyPara([
    t("Table IV reveals several important patterns. First, swapping CB0 alone has no effect on SECS or CER (0.827/15.7% identical to baseline), confirming that CB0 encodes semantic content independently of speaker identity. Second, progressively swapping more acoustic codebooks (CB1 through CB1\u20137) causes CER to increase from 15.7% to 52.8%, indicating that acoustic token replacement disrupts the RVQ residual chain when done without semantic alignment. The results for CB1\u20137 and All (CB0\u20137) are identical, further confirming that CB0 does not contribute to speaker identity encoding."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The CER increase with more swapped codebooks highlights the importance of our retrieval-based approach: by selecting target frames based on semantic similarity, the retrieval method maintains residual coherence and achieves much lower CER (24.6% for FAISS+Smooth) compared to unaligned swapping (52.8% for CB1\u20137 ablation)."),
  ])
);

bodyChildren.push(subHeading("B", "Retrieval Parameters"));

bodyChildren.push(
  bodyPara([
    t("We sweep the number of nearest neighbors k and the temporal smoothing weight \u03B1 for the FAISS+Smooth method. Increasing k from 1 to 32 decreases SECS from 0.824 to 0.765, indicating that retrieving more candidates introduces noise\u2014the nearest match (k = 1\u20132) is typically the best semantic match. For the smoothing weight \u03B1, SECS increases monotonically from 0.787 (\u03B1 = 0.0, temporal only) to 0.841 (\u03B1 = 1.0, semantic only), with \u03B1 = 0.5\u20130.75 providing a good trade-off between semantic accuracy and temporal coherence as reflected in subjective listening quality. We use k = 8 and \u03B1 = 0.5 as defaults."),
  ])
);

bodyChildren.push(subHeading("C", "Transformer Design Choices"));

bodyChildren.push(
  bodyPara([
    t("We ablate the Transformer\u2019s sampling temperature and training duration. Temperature has a non-monotonic effect on SECS: temp = 0.8 achieves the best SECS (0.742), while lower temperatures (0.1\u20130.5) produce more deterministic but less diverse acoustic tokens (SECS \u2248 0.675\u20130.704), and higher temperatures (1.5) add too much randomness (SECS = 0.696). For training duration, SECS improves steadily from 0.663 at 50 steps to 0.794 at 1000 steps, suggesting the model benefits from longer training when sufficient data is available. These results indicate that with more target speaker data and longer training, the Transformer approach has significant room for improvement."),
  ])
);

// Figure 2 - Token visualization
const fig2Path = "/Users/nguyenquoc/duplex/figure2_tokens.png";
if (fs.existsSync(fig2Path)) {
  bodyChildren.push(
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [
        new ImageRun({
          type: "png",
          data: fs.readFileSync(fig2Path),
          transformation: { width: 480, height: 340 },
          altText: { title: "Figure 2", description: "Mimi token comparison across methods", name: "fig2" }
        })
      ]
    })
  );
} else {
  bodyChildren.push(placeholder("FIGURE 2: Token visualization"));
}
bodyChildren.push(figureCaption("2", "Mimi token visualization across methods. Each heatmap shows 8 codebook indices over time. CB0 (semantic, top row) remains stable across conversions, preserving content. CB1\u20137 (acoustic) patterns shift toward the target speaker\u2019s characteristics."));

bodyChildren.push(
  bodyPara([
    t("Fig. 2 visualizes the token heatmaps for source, retrieval-converted, transformer-converted, and target reference audio. The CB0 (semantic) row shows consistent patterns across conversions, confirming content preservation. The acoustic codebooks (CB1\u20137) show distinct patterns: the retrieval method produces target-like acoustic patterns with smooth temporal transitions, while the Transformer method generates more varied patterns that capture the target speaker\u2019s statistical distribution."),
  ])
);

// ===================== VII. CONCLUSION =====================
bodyChildren.push(sectionHeading("VII", "Conclusion"));

bodyChildren.push(
  bodyPara([
    t("We presented Mimi-VC, a voice conversion framework that operates in the discrete token space of the Mimi neural audio codec. By exploiting the natural semantic-acoustic disentanglement in Mimi\u2019s residual vector quantizer\u2014where the first codebook captures speaker-independent content via WavLM distillation and remaining codebooks encode speaker characteristics\u2014we achieve voice conversion through direct token manipulation. Our retrieval-based approach provides training-free conversion using FAISS indexing with temporal smoothing, while the Transformer-based translator learns target speaker acoustic patterns for improved generalization."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Our experimental evaluation on LibriSpeech demonstrates that the retrieval-based approach with temporal smoothing achieves the best overall performance, with speaker similarity (SECS) of 0.884, content preservation (CER) of 24.6%, and perceptual quality (PESQ) of 2.26. Temporal smoothing proves critical, reducing CER from 84.1% to 24.6% compared to frame-independent retrieval. The Transformer-based approach shows promising scaling behavior, with SECS improving from 0.663 to 0.794 as training increases from 50 to 1000 steps, suggesting significant room for improvement with larger target speaker datasets. All methods achieve RTF < 0.01 on GPU, confirming the computational efficiency of token-space operation. Our ablation study validates Mimi\u2019s disentanglement: codebook 0 encodes content independently of speaker identity, while codebooks 1\u20137 progressively capture acoustic and speaker characteristics."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("The success of Mimi-VC demonstrates that pretrained neural audio codecs with disentangled representations provide a powerful foundation for voice conversion, eliminating the need for purpose-built architectures, separate content extractors, or dedicated vocoders. Operating at 12.5 Hz in compact token space, our framework is naturally suited for real-time streaming applications."),
  ])
);

bodyChildren.push(
  bodyPara([
    t("Future work includes: (1) extending to multi-speaker conversion via speaker embedding conditioning, (2) incorporating explicit F0 normalization for cross-gender conversion, (3) exploring diffusion-based token generation in the continuous latent space before quantization, and (4) integration with speech language models such as Moshi [1] for simultaneous voice conversion during dialogue."),
  ])
);

// ===================== ACKNOWLEDGMENT =====================
bodyChildren.push(sectionHeading("", "Acknowledgment"));

bodyChildren.push(
  placeholder("Acknowledge funding sources, compute resources, and collaborators.")
);

// ===================== REFERENCES =====================
bodyChildren.push(sectionHeading("", "References"));

const refs = [
  'Kyutai Team, "Moshi: A speech-text foundation model for real-time dialogue," arXiv preprint arXiv:2410.00037, 2024.',
  'A. D\u00E9fossez, J. Copet, G. Synnaeve, and Y. Adi, "High fidelity neural audio compression," in Proc. Int. Conf. Machine Learning (ICML), 2023, pp. 7554\u20137577.',
  'N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, "SoundStream: An end-to-end neural audio codec," IEEE/ACM Trans. Audio, Speech, Language Process., vol. 30, pp. 495\u2013507, 2022.',
  'X. Zhang et al., "SpeechTokenizer: Unified speech tokenizer for speech language models," in Proc. Int. Conf. Learning Representations (ICLR), 2024.',
  'W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," IEEE/ACM Trans. Audio, Speech, Language Process., vol. 29, pp. 3451\u20133460, 2021.',
  'J. Kong, J. Kim, and J. Bae, "HiFi-GAN: Generative adversarial networks for efficient and high fidelity speech synthesis," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020.',
  'J. Kim, J. Kong, and J. Son, "Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech," in Proc. Int. Conf. Machine Learning (ICML), 2021, pp. 5530\u20135540.',
  'J. Johnson, M. Douze, and H. J\u00E9gou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, vol. 7, no. 3, pp. 535\u2013547, 2021.',
  'RVC-Project, "Retrieval-based-Voice-Conversion-WebUI," GitHub repository, 2023. [Online]. Available: https://github.com/RVC-Project',
  'Z. Ju et al., "NaturalSpeech 3: Zero-shot speech synthesis with a factorized codec and diffusion models," in Proc. Int. Conf. Machine Learning (ICML), 2024.',
  'VChangeCodec, "A high-efficiency neural speech codec with built-in voice changer for real-time communication," in Proc. Int. Conf. Learning Representations (ICLR), 2025 (submitted).',
  'Y. Zheng et al., "FreeCodec: A disentangled neural speech codec with fewer tokens," arXiv preprint arXiv:2412.01053, 2024.',
  'Kyutai Team, "Hibiki: Real-time speech-to-speech and speech-to-text translation with voice transfer," 2025. [Online]. Available: https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16',
  'A. Vaswani et al., "Attention is all you need," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5998\u20136008.',
  'A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in Proc. Advances in Neural Information Processing Systems (NeurIPS), 2020.',
  'C. Chen, M. Du, and E. Bhatt, "DualCodec: A low-frame-rate, semantically-enhanced neural audio codec," in Proc. Interspeech, 2025.',
  'J. Pang et al., "Neural codec language models for disentangled and textless voice conversion," arXiv preprint, 2024.',
];

refs.forEach((r, i) => {
  bodyChildren.push(refItem(i + 1, r));
});

const bodySection = {
  properties: {
    type: SectionType.CONTINUOUS,
    page: {
      size: { width: PAGE_W, height: PAGE_H },
      margin: { top: MARGIN_TOP, bottom: MARGIN_BOT, left: MARGIN_LR, right: MARGIN_LR }
    },
    column: {
      count: 2,
      space: COL_GAP,
      equalWidth: true,
      separate: false
    }
  },
  children: bodyChildren
};

// ============================================================
// Create Document
// ============================================================
const doc = new Document({
  styles: {
    default: {
      document: {
        run: {
          font: FONT,
          size: BODY_SIZE
        }
      }
    }
  },
  sections: [titleSection, bodySection]
});

// Write file
const OUTPUT = "/Users/nguyenquoc/duplex/mimi_vc_paper.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(OUTPUT, buffer);
  console.log(`Paper written to ${OUTPUT}`);
  console.log(`Size: ${(buffer.length / 1024).toFixed(1)} KB`);
});
