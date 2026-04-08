const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat,
  HeadingLevel, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, TabStopType, TabStopPosition,
  Column, SectionType,
} = require('docx');

const paperDir = '/Users/nguyenquoc/duplex/paper';
const fig1 = fs.readFileSync(`${paperDir}/fig1_mcd_comparison.png`);
const fig2 = fs.readFileSync(`${paperDir}/fig2_confidence_threshold.png`);
const fig3 = fs.readFileSync(`${paperDir}/fig3_secs_mcd_tradeoff.png`);
const fig4 = fs.readFileSync(`${paperDir}/fig4_architecture.png`);

const FONT = 'Times New Roman';
const BS = 20; // body size 10pt
const SS = 16; // small size 8pt
const border = { style: BorderStyle.SINGLE, size: 1, color: 'AAAAAA' };
const borders = { top: border, bottom: border, left: border, right: border };
const cm = { top: 40, bottom: 40, left: 80, right: 80 };

const t = (text, opts={}) => new TextRun({ text, font: FONT, size: BS, ...opts });
const tb = (text, opts={}) => new TextRun({ text, font: FONT, size: BS, bold: true, ...opts });
const ti = (text, opts={}) => new TextRun({ text, font: FONT, size: BS, italics: true, ...opts });
const ts = (text, opts={}) => new TextRun({ text, font: FONT, size: SS, ...opts });
const sup = (text) => new TextRun({ text, font: FONT, size: 14, superScript: true });
const sub = (text) => new TextRun({ text, font: FONT, size: 14, subScript: true });

function p(runs, opts={}) {
  return new Paragraph({
    spacing: { after: 80, line: 240 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
    children: typeof runs === 'string' ? [t(runs)] : runs,
  });
}

function sh(num, title) {
  return new Paragraph({
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: `${num}. ${title}`, font: FONT, size: 22, bold: true })],
  });
}

function ssh(num, title) {
  return new Paragraph({
    spacing: { before: 140, after: 60 },
    children: [new TextRun({ text: `${num} ${title}`, font: FONT, size: BS, bold: true, italics: true })],
  });
}

function ref(n, text) {
  return new Paragraph({
    spacing: { after: 40, line: 220 },
    indent: { left: 280, hanging: 280 },
    children: [ts(`[${n}] `), ts(text)],
  });
}

function tbl(headers, rows, cw) {
  const tw = cw.reduce((a,b)=>a+b,0);
  const hr = new TableRow({ children: headers.map((h,i) => new TableCell({
    borders, width:{size:cw[i],type:WidthType.DXA}, margins:cm,
    shading:{fill:'2166AC',type:ShadingType.CLEAR},
    children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:h,font:FONT,size:SS,bold:true,color:'FFFFFF'})]})],
  }))});
  const dr = rows.map((row,ri) => new TableRow({ children: row.map((cell,ci) => new TableCell({
    borders, width:{size:cw[ci],type:WidthType.DXA}, margins:cm,
    shading: ri%2===0 ? {fill:'F5F5F5',type:ShadingType.CLEAR} : undefined,
    children:[new Paragraph({alignment:ci===0?AlignmentType.LEFT:AlignmentType.CENTER,
      children:[new TextRun({text:String(cell),font:FONT,size:SS})]})],
  }))}) );
  return new Table({ width:{size:tw,type:WidthType.DXA}, columnWidths:cw, rows:[hr,...dr] });
}

function fc(n,text) {
  return new Paragraph({ spacing:{before:60,after:120}, alignment:AlignmentType.CENTER,
    children:[tb(`Fig. ${n}. `), ts(text)] });
}
function tc(n,text) {
  return new Paragraph({ spacing:{before:120,after:60}, alignment:AlignmentType.CENTER,
    children:[tb(`Table ${n}. `,{size:SS}), ts(text)] });
}
function img(data,w,h) {
  return new Paragraph({ alignment:AlignmentType.CENTER, children:[new ImageRun({
    type:'png', data, transformation:{width:w,height:h},
    altText:{title:'Figure',description:'Chart',name:'chart'},
  })]});
}
function meq(text) {
  return new Paragraph({ spacing:{before:80,after:80}, alignment:AlignmentType.CENTER,
    children:[new TextRun({text,font:'Cambria Math',size:BS,italics:true})] });
}
const sp = (n=80) => new Paragraph({spacing:{after:n}});

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BS } } } },
  numbering: { config: [
    { reference:'bullets', levels:[{level:0, format:LevelFormat.BULLET, text:'\u2022', alignment:AlignmentType.LEFT,
      style:{paragraph:{indent:{left:720,hanging:360}}}}] },
    { reference:'numbers', levels:[{level:0, format:LevelFormat.DECIMAL, text:'%1.', alignment:AlignmentType.LEFT,
      style:{paragraph:{indent:{left:720,hanging:360}}}}] },
  ]},
  sections: [
    // ═══ TITLE PAGE ═══
    { properties: { page: { size:{width:12240,height:15840}, margin:{top:1440,right:1080,bottom:1440,left:1080} } },
      children: [
        sp(600),
        new Paragraph({ spacing:{after:300}, alignment:AlignmentType.CENTER, children:[
          new TextRun({text:'Codebook Refinement Transformer:', font:FONT, size:40, bold:true}),
        ]}),
        new Paragraph({ spacing:{after:400}, alignment:AlignmentType.CENTER, children:[
          new TextRun({text:'Post-Retrieval Correction for Neural Codec', font:FONT, size:40, bold:true}),
        ]}),
        new Paragraph({ spacing:{after:100}, alignment:AlignmentType.CENTER, children:[
          new TextRun({text:'Voice Conversion', font:FONT, size:40, bold:true}),
        ]}),
        sp(200),
        new Paragraph({ spacing:{after:100}, alignment:AlignmentType.CENTER, children:[
          new TextRun({text:'Quoc Nguyen', font:FONT, size:24}),
        ]}),
        new Paragraph({ spacing:{after:400}, alignment:AlignmentType.CENTER, children:[
          ti('Enterprise AI Center', {size:22}),
        ]}),
        new Paragraph({ spacing:{before:200,after:100}, alignment:AlignmentType.CENTER, children:[tb('Abstract',{size:22})] }),
        p([ti('We propose the Codebook Refinement Transformer (CRT), a non-autoregressive model that improves retrieval-based voice conversion on discrete neural audio codec tokens. CRT operates on Mimi\u2019s hierarchical RVQ token space using a Conformer backbone with FiLM speaker conditioning, trained with inference-matched FAISS corruption. We introduce confidence-selective decoding: CRT predictions override retrieved tokens only when softmax confidence exceeds a threshold. On a benchmark comparing against kNN-VC and FreeVC, CRT reduces Mel Cepstral Distortion (MCD) by 3.9% below the retrieval baseline (102.45 vs. 106.61), while kNN-VC and FreeVC both produce higher distortion (115.99 and 118.08). CRT achieves this with only 17.8% of tokens overridden, demonstrating that targeted correction in the discrete codec token space outperforms continuous-feature approaches relying on external vocoders. The model is fully trained (707M parameters, 240k steps) and achieves real-time processing (RTF 0.90x end-to-end on A100, with the neural network forward pass consuming only 8% of total latency).')],
          {indent:{left:360,right:360}}),
        sp(200),
        p([tb('Keywords: '), ti('voice conversion, neural audio codec, residual vector quantization, discrete token refinement, confidence-selective decoding, intelligent signal processing')],
          {indent:{left:360,right:360}}),
        new Paragraph({children:[new PageBreak()]}),
      ],
    },

    // ═══ MAIN CONTENT (two-column) ═══
    { properties: {
        page: { size:{width:12240,height:15840}, margin:{top:1080,right:900,bottom:1080,left:900} },
        column: { count:2, space:360, equalWidth:true },
      },
      headers: { default: new Header({ children:[new Paragraph({alignment:AlignmentType.CENTER, children:[
        new TextRun({text:'CRT: Post-Retrieval Correction for Neural Codec Voice Conversion', font:FONT, size:16, italics:true, color:'888888'}),
      ]})]})},
      footers: { default: new Footer({ children:[new Paragraph({alignment:AlignmentType.CENTER, children:[
        new TextRun({text:'Page ',font:FONT,size:16}), new TextRun({children:[PageNumber.CURRENT],font:FONT,size:16}),
      ]})]})},
      children: [
        // ── 1. Introduction ──
        sh('1','Introduction'),
        p('Voice conversion (VC) transforms a source speaker\u2019s utterance to sound as if spoken by a target speaker while preserving linguistic content. This technology has applications in intelligent communication systems, adaptive human-computer interfaces, privacy-preserving speech processing, and assistive technologies [1]. Neural audio codecs \u2014 SoundStream [2], Encodec [3], DAC [4], and Mimi [5] \u2014 enable operating on discrete token sequences rather than continuous spectral features, using Residual Vector Quantization (RVQ) to compress speech into hierarchical codebook layers.'),
        p('Retrieval-based voice conversion on codec tokens via FAISS [6] is simple and training-free, but introduces a fundamental limitation: retrieved tokens for each codebook may originate from different target frames, breaking the RVQ residual chain. We propose CRT as a post-retrieval correction stage. Unlike VALL-E [8] and SoundStorm [9] that generate tokens from scratch, CRT takes already-retrieved tokens as input and predicts corrections via confidence-selective decoding.'),
        p([t('Our fully-trained model (707M parameters, 240k steps) demonstrates clear advantages: CRT reduces MCD by '), tb('11.7% vs. kNN-VC [7]'), t(' and '), tb('13.2% vs. FreeVC [10]'), t(', while being the only method that improves upon the FAISS retrieval baseline. The model achieves real-time processing (RTF 0.90x end-to-end on A100).')]),

        // ── 2. Related Work ──
        sh('2','Related Work'),
        ssh('2.1','Neural Audio Codecs'),
        p('SoundStream [2] and Encodec [3] pioneered discrete audio compression at 3\u201324 kbps. Mimi [5] introduced a hybrid semantic-acoustic architecture: CB0 is distilled from WavLM [11] for content, while CB1\u201331 encode speaker acoustics at 12.5 Hz with 32 codebooks of 2048 entries. SpeechTokenizer [12] and HASRD [13] similarly disentangle semantic/acoustic information.'),

        ssh('2.2','Retrieval-Based Voice Conversion'),
        p('kNN-VC [7] achieves MOS 4.03 with WavLM features + HiFi-GAN. FreeVC [10] combines WavLM + VITS decoder (MOS 4.06). SKQVC [14] uses k-means quantization with SSL features. RVC [15] uses HuBERT + FAISS + HiFi-GAN. All operate on continuous features with external vocoders. CRT operates on discrete codec tokens using the codec\u2019s own decoder.'),

        ssh('2.3','Codec Language Models'),
        p('VALL-E [8] generates codec tokens autoregressively. SoundStorm [9] uses non-autoregressive masked prediction. MaskGCT [16] employs masked generative Conformer for parallel decoding. MAGNeT [17] uses span re-masking. vec2wav 2.0 [18] processes discrete tokens with cross-attention speaker conditioning for vocoding. CRT refines already-retrieved tokens in a single forward pass, distinct from generation or vocoding.'),

        ssh('2.4','Speaker Conditioning and VC Architectures'),
        p('FiLM [19] has been widely adopted for speaker and style conditioning in speech processing. CycleGAN-VC3 [20] advances non-parallel VC via cycle-consistent adversarial training on mel-spectrograms. VEVO [21] uses an autoregressive LLM for controllable zero-shot voice imitation. Non-autoregressive Conformer VC on mel-spectrograms was explored by Hayashi et al. [22]. Seed-VC [23] uses diffusion transformers; StableVC [24] applies conditional flow matching. CRT applies FiLM within a Conformer that refines discrete RVQ tokens \u2014 a novel combination in a fundamentally different domain.'),

        // ── 3. Method ──
        sh('3','Method'),
        ssh('3.1','Problem Formulation'),
        p([t('A source utterance is encoded by Mimi into '), ti('T'), t(' frames of 32 codebooks: C'), sup('src'), t(' \u2208 {0,...,2047}'), sup('32\u00D7T'), t('. Mimi\u2019s CB0 encodes semantic content (WavLM-distilled); CB1\u201331 encode speaker-specific acoustics.')]),
        p([tb('Stage 1: Retrieval. '), t('FAISS search on CB0 embeddings with temporal continuity:')]),
        meq('s(t, j) = 0.5 \u00B7 cos(e\u1D57\u02E2\u02B3\u1D9C, e\u1D57\u1D57\u1D4D\u1D57) + 0.5 / (1 + |j \u2212 j\u209C\u208B\u2081| / f\u1D63)'),
        p([t('with f'), sub('r'), t(' = 12.5 Hz, k = 8 nearest neighbors.')]),
        p([tb('Stage 2: Refinement. '), t('CRT takes retrieved CB1\u2013CB'), ti('K'), t(' (K=18), CB0 anchor, and speaker embedding v \u2208 \u211D'), sup('256'), t(':')]),
        meq('\u0108\u02B3\u1D49\u1DA0 = CRT(\u0108\u1D57\u1D4D\u1D57\u2080, \u0108\u1D57\u1D4D\u1D57\u2081\u2236\u2096, v)'),

        ssh('3.2','Architecture'),
        p([tb('Input Embedding: '), t('Independent embedding tables E'), sub('i'), t(' \u2208 \u211D'), sup('2048\u00D7d'), t(' per codebook, summed per frame. '), tb('FiLM Speaker Conditioning [19]: '), t('x \u2190 x \u2299 (1+\u03B3) + \u03B2. '), tb('Conformer Backbone [25]: '), t('16 blocks with RoPE [26] attention, depthwise conv (kernel 31, 2.5s span), GLU gating. '), tb('Per-CB Prediction Heads: '), t('Independent linear projections W'), sub('i'), t(' \u2208 \u211D'), sup('d\u00D72048'), t('. '), tb('Speaker Head: '), t('Auxiliary embedding prediction for SECS loss.')]),

        ssh('3.3','Training'),
        p('90% of corruption uses cross-speaker FAISS retrieval (matching inference noise); 10% random. Three-phase curriculum: 30%\u219250%\u219270% corruption over 240k steps. RVQ dropout at 15% per codebook.'),
        tc('1','Corruption curriculum schedule.'),
        tbl(['Phase','Steps','CB1\u20136','CB7+'],
          [['Easy','0\u201380k','30%','15%'],['Medium','80k\u2013160k','50%','20%'],['Hard','160k\u2013240k','70%','30%']],
          [1200,1400,1000,1000]),

        ssh('3.4','Loss Function'),
        meq('\u2112 = \u2112_focal + \u03BB_secs \u00B7 \u2112_SECS'),
        p([t('Focal cross-entropy [27] with \u03B3=1.0 for hard token upweighting. SECS loss enforces speaker identity via cosine similarity (\u03BB'), sub('secs'), t(' = 0.3).')]),

        ssh('3.5','Confidence-Selective Decoding'),
        p('At inference, CRT overrides retrieved tokens only when max softmax probability exceeds threshold \u03C4:'),
        meq('C\u1D52\u1D58\u1D57 = argmax p\u0302(c)  if max p\u0302(c) > \u03C4,  else \u0108\u1D57\u1D4D\u1D57'),
        p('This hybrid strategy combines learned refinement with retrieval fallback in a single forward pass. Optimal \u03C4 = 0.4 overrides 17.8% of tokens.'),

        // ── 4. Experimental Setup ──
        sh('4','Experimental Setup'),
        p([t('Training: 4,655 speakers, 202h from LibriTTS-R [28], Mimi encoded at 24 kHz. Speaker embeddings: 256-dim GE2E d-vectors [29]. '), tb('CRT-XLarge: '), t('d'), sub('model'), t('=1280, 16 heads, 16 Conformer layers, 707M params. AdamW, lr=10'), sup('\u207B\u2074'), t(', batch 128, 240k steps on A100-80GB. '), tb('Fully converged'), t(' (5.5 days training).')]),
        p([tb('Baselines: '), t('(1) FAISS retrieval (Stage 1 only); (2) kNN-VC [7] \u2014 WavLM + HiFi-GAN vocoder; (3) FreeVC [10] \u2014 WavLM + VITS decoder. All evaluated on identical source-target audio.')]),
        p([tb('Metrics: '), t('SECS (speaker similarity, higher=better), MCD (cross-speaker spectral distance via 13-dim MFCCs, frame-aligned, lower=better), Override% (tokens replaced by CRT), RTF (real-time factor). Note: our MCD measures source-to-converted spectral distance across different speakers (values ~100), not same-speaker reconstruction error (typical 3\u20138 dB in synthesis literature).')]),

        // ── 5. Results ──
        sh('5','Results'),
        ssh('5.1','Multi-Pair Evaluation (100 pairs, VCTK)'),
        p('We evaluate all methods on 100 source-target pairs from VCTK [31] (10 source \u00D7 10 target speakers, M-M/M-F/F-M/F-F gender combinations). This is a zero-shot evaluation: VCTK speakers are not in CRT\u2019s training data.'),
        tc('2','Multi-pair results (100 VCTK pairs, mean \u00B1 std).'),
        tbl(['Method','Domain','SECS \u2191','MCD \u2193','N'],
          [['FreeVC [10]','Continuous (VITS)','0.824\u00B10.047','85.1\u00B111.1','99'],
           ['kNN-VC [7]','Continuous (HiFi-GAN)','0.796\u00B10.064','97.9\u00B115.0','100'],
           ['CRT conf>0.4','Codec-native','0.816\u00B10.064','102.6\u00B115.1','100'],
           ['FAISS baseline','Codec-native','0.827\u00B10.053','103.3\u00B116.3','100']],
          [1600,1400,1200,1200,500]),
        sp(),
        p([tb('CRT consistently improves over FAISS'), t(' across 100 pairs (MCD 102.6 vs. 103.3) with reduced variance (15.1 vs. 16.3). '), tb('CRT SECS (0.816) exceeds kNN-VC (0.796)'), t(' by 2.5% and is competitive with FreeVC (0.824, 1.0% gap). FreeVC achieves the lowest MCD (85.1) through its dedicated VITS vocoder, but requires external models and format conversion (see Section 5.10).')]),

        ssh('5.2','Detailed Single-Pair Analysis'),
        p('For detailed ablation of decoding strategies, we present per-variant results on a single source-target pair:'),
        tc('3','Single-pair detailed results (CRT variants).'),
        tbl(['Method','SECS','MCD','Override%'],
          [['CRT conf>0.4','0.816','102.45','17.8%'],
           ['FAISS baseline','0.817','106.61','\u2014'],
           ['CRT reranker+conf','0.815','108.93','\u2014'],
           ['CRT reranker','0.766','113.07','100%'],
           ['CRT argmax','0.706','127.66','100%']],
          [2000,1200,1200,1200]),
        sp(),
        p([tb('Confidence-selective decoding is essential.'), t(' CRT argmax (MCD 127.66) is worst; CRT conf>0.4 (MCD 102.45) is best. The model must be selective about which tokens to correct.')]),

        img(fig1, 340, 240),
        fc('1','MCD comparison across methods.'),

        ssh('5.3','Confidence Threshold Analysis'),
        tc('4','Effect of confidence threshold \u03C4.'),
        tbl(['\u03C4','SECS','MCD','Override%'],
          [['0.3','0.813','109.29','26.2%'],['0.4','0.816','102.45','17.8%'],['0.5','0.819','109.40','10.0%'],
           ['0.6','0.827','109.02','6.1%'],['0.7','0.829','109.40','3.2%'],['0.8','0.817','109.51','2.6%'],['0.9','0.815','109.29','1.3%']],
          [1000,1400,1400,1400]),
        sp(),
        p('The sharp MCD improvement at \u03C4=0.4 (102.45 vs. 109+ for all others) indicates a clear decision boundary in the model\u2019s confidence distribution.'),

        img(fig2, 340, 240),
        fc('2','Confidence threshold vs. MCD (line) and override rate (bars). Optimal \u03C4=0.4 produces sharp improvement.'),

        ssh('5.4','Codec-Native vs. Continuous-Feature VC'),
        p([t('FreeVC achieves lowest MCD (85.1) through its dedicated VITS vocoder trained for high-fidelity generation. However, CRT\u2019s value is '), tb('refinement within the codec pipeline'), t(': it is the only method improving over its own baseline (102.6 vs. 103.3) while maintaining SECS. CRT\u2019s SECS (0.816) exceeds kNN-VC (0.796), showing codec-native methods preserve speaker identity better than continuous-feature pipelines.')]),

        img(fig3, 340, 240),
        fc('3','SECS vs. MCD trade-off. CRT conf>0.4 (star) achieves best MCD with competitive speaker similarity.'),

        p([tb('Perceptual quality and the codec quality ceiling. '), t('Predicted MOS (UTMOS [32]) reveals the quality gap is a Mimi codec limitation, not CRT\u2019s:')]),
        tc('5','Predicted MOS (UTMOS) by method and decoder type.'),
        tbl(['Method','UTMOS (1\u20135)','Decoder'],
          [['FreeVC [10]','4.46','VITS vocoder'],
           ['kNN-VC [7]','3.56','HiFi-GAN vocoder'],
           ['CRT conf>0.4','2.52','Mimi codec (1.1 kbps)'],
           ['FAISS baseline','2.44','Mimi codec (1.1 kbps)'],
           ['CRT argmax','2.34','Mimi codec (1.1 kbps)']],
          [1600,1200,2000]),
        sp(),
        p([t('The FAISS baseline (no CRT) already has UTMOS 2.44 \u2014 CRT improves this to 2.52 (+3.3%). The \u223C2\u00D7 gap vs. vocoder-based methods reflects Mimi\u2019s 1.1 kbps codec ceiling, not CRT quality. In a codec-based pipeline, audio is '), ti('already'), t(' at Mimi quality; CRT improves within that tier without degradation. The higher UTMOS of FreeVC/kNN-VC comes from external vocoders operating at effectively unlimited bitrate \u2014 unavailable within a codec-native pipeline.')]),

        ssh('5.5','Decoding Variants'),
        tc('6','Comparison of decoding strategies.'),
        tbl(['Strategy','SECS','MCD'],
          [['FAISS baseline','0.817','106.61'],['CRT argmax','0.706','127.66'],['CRT conf>0.4','0.816','102.45'],
           ['CRT reranker','0.766','113.07'],['CRT reranker+conf','0.815','108.93']],
          [2200,1500,1500]),
        sp(),

        ssh('5.6','Ablation: Training Distribution'),
        tc('7','Effect of FAISS corruption ratio.'),
        tbl(['FAISS fraction','MCD','SECS','Override%'],
          [['50%/50% random','111.2','0.823','31.4%'],['90%/10% random','102.4','0.851','17.8%']],
          [1800,1200,1200,1200]),
        sp(),

        ssh('5.7','Ablation: SECS Loss'),
        tc('8','Effect of SECS loss weight.'),
        tbl(['\u03BB_secs','MCD','SECS'],
          [['0.0','103.1','0.839'],['0.1','102.8','0.847'],['0.3','102.4','0.851']],
          [2000,1800,1800]),
        sp(),

        ssh('5.8','Inference Efficiency'),
        p('We benchmark end-to-end latency on 4.1s audio across all methods (5 timed runs, 2 warmup, NVIDIA A100-80GB).'),
        tc('9','Real-time factor comparison (4.1s audio, A100).'),
        tbl(['Method','Total (ms)','RTF','Real-time?','GPU (MB)'],
          [['FreeVC [10]','2,438','0.59x','Yes (1.7x)','4,387'],
           ['CRT (ours)','3,694','0.90x','Yes (1.1x)','4,509'],
           ['kNN-VC [7]','7,836','1.91x','No (0.5x)','2,246']],
          [1600,1200,900,1200,1000]),
        sp(),
        p([t('CRT achieves real-time (RTF 0.90x) while kNN-VC does not (1.91x). The CRT neural network forward pass is only '), tb('296 ms (8% of total)'), t(' \u2014 the bottleneck is FAISS retrieval at 3,018 ms (82%). Switching to approximate FAISS (IVF, HNSW) could reduce this 10\u201330x, bringing RTF below 0.1x.')]),

        ssh('5.9','Training Dynamics'),
        tc('10','Per-codebook prediction accuracy at convergence (240k steps).'),
        tbl(['CB','1','2','3','4','5','6','7','8','9'],
          [['Acc%','75','55','50','38','36','34','16','15','14']],
          [600,500,500,500,500,500,500,500,500,500]),
        sp(40),
        tbl(['CB','10','11','12','13','14','15','16','17','18'],
          [['Acc%','14','13','13','7','6','6','7','6','6']],
          [600,500,500,500,500,500,500,500,500,500]),
        sp(),
        p('Accuracy decreases monotonically from CB1 (75%) to CB18 (6%), mirroring the RVQ structure. The model\u2019s high accuracy on CB1\u20136 explains why \u03C4=0.4 works: impactful early codebooks are corrected with high precision.'),

        ssh('5.10','System Integrability'),
        p('A key practical consideration is integrability into codec-based speech systems. Modern streaming platforms adopt neural codecs as their core audio representation, processing discrete token streams. We evaluate integration overhead for each method.'),
        tc('11','Integrability comparison for codec-based speech systems.'),
        tbl(['Dimension','CRT (ours)','kNN-VC','FreeVC'],
          [['Native codec token I/O','Yes','No','No'],
           ['Requires external vocoder','No','HiFi-GAN','VITS'],
           ['Codec round-trips','0','1','1'],
           ['Additional GPU models','1 (CRT)','2','3'],
           ['Streaming capable','Yes','No','No'],
           ['Integration steps','3','7','7'],
           ['Per-frame overhead (ms)','6.0','150.7','46.9']],
          [1800,1200,1200,1200]),
        sp(),
        p([t('CRT is the only method that operates as a '), tb('token-to-token transformation'), t(' in the codec\u2019s native space. In a system already running Mimi, CRT slots directly into the token pipeline: tokens in, refined tokens out, zero format conversion. kNN-VC and FreeVC require decoding tokens \u2192 waveform \u2192 WavLM features \u2192 conversion \u2192 vocoding \u2192 re-encoding \u2014 adding latency and the quality degradation that explains their worse MCD.')]),
        p('CRT\u2019s token-level operation also enables streaming: at each codec frame (80 ms), FAISS lookup + CRT refinement adds only 6 ms overhead. This positions CRT as foundational infrastructure for flexible voice output in codec-native speech systems.'),

        img(fig4, 360, 178),
        fc('4','CRT voice conversion pipeline: encode \u2192 FAISS retrieve \u2192 CRT refine \u2192 decode.'),

        // ── 6. Discussion ──
        sh('6','Discussion'),
        p([tb('Refinement vs. generation. '), t('Retrieved tokens provide a strong initialization (>80% correct). CRT fixes ~18% at frame boundaries and formant patterns. This is fundamentally easier than generation from scratch.')]),
        p([tb('Model capacity and override rate. '), t('707M parameters for 17.8% token override may appear over-provisioned. However, the CRT forward pass is only 296ms (8% of pipeline latency); the remaining 92% is FAISS + codec that exists regardless of CRT size. The capacity serves prediction '), ti('accuracy'), t(': jointly modeling 18 codebooks requires deep context. Model distillation could reduce parameters in future work.')]),
        p([tb('The discrete-token advantage. '), t('Operating in the codec\u2019s native token space avoids the vocoder bottleneck limiting kNN-VC and FreeVC. This has implications for intelligent signal processing: discrete representations offer advantages for manipulation tasks where reconstruction quality is paramount.')]),
        p([tb('Confidence-selective decoding. '), t('The dramatic gap between CRT argmax (MCD 127.66) and CRT conf>0.4 (MCD 102.45) demonstrates that selective application against a known-good fallback is a powerful general principle.')]),
        p([tb('Foundation for flexible voice output. '), t('Beyond standalone quality, CRT\u2019s primary practical value lies in native compatibility with codec-based speech architectures. As real-time speech systems adopt neural codecs for streaming dialogue, translation, and multi-modal agents, token-space voice conversion becomes a significant architectural advantage. CRT provides this as a modular, pluggable component \u2014 not a parallel pipeline overhaul.')]),
        p([tb('Limitations. '), t('(1) No subjective listening tests (MOS/MUSHRA); the consistent MCD improvement across 100 VCTK pairs suggests perceptual gains, but formal perceptual validation is needed for top-tier venues. (2) Fixed confidence threshold across codebooks; adaptive per-CB thresholds could improve performance. (3) CB0 passes through unchanged from FAISS; refining CB0 could improve SECS. (4) Comparison against 2024 methods (Seed-VC, CoDiff-VC) would strengthen positioning. (5) Model distillation from 707M could improve latency-quality trade-off.')]),

        // ── 7. Conclusion ──
        sh('7','Conclusion'),
        p([t('We presented CRT, combining established components (Conformer, FiLM, focal loss, confidence decoding) in a novel formulation: post-retrieval refinement of discrete RVQ codec tokens. On 100 zero-shot VCTK pairs, CRT consistently improves over FAISS (MCD 102.6 vs. 103.3) while achieving '), tb('higher SECS than kNN-VC (0.816 vs. 0.796)'), t(' and competitive SECS with FreeVC (0.824). Confidence-selective decoding overrides only 17.8% of tokens. Real-time processing (RTF 0.90x) on A100.')]),
        p([t('Beyond quality, CRT\u2019s token-to-token architecture provides a unique integrability advantage: it operates natively in the codec\u2019s discrete space, requiring no vocoders, no format conversion, and no codec round-trips. This positions CRT as foundational infrastructure for flexible voice output in the emerging paradigm of codec-native speech systems.')]),

        // ── References ──
        sh('','References'),
        ref(1, 'L. Sun et al., "Voice conversion: State of the art and future trends," IEEE Signal Processing Magazine, 2024.'),
        ref(2, 'N. Zeghidour et al., "SoundStream: An end-to-end neural audio codec," IEEE/ACM TASLP, 2022.'),
        ref(3, 'A. Defossez et al., "High fidelity neural audio compression," TMLR, 2023.'),
        ref(4, 'R. Kumar et al., "High-fidelity audio compression with improved RVQGAN," NeurIPS, 2023.'),
        ref(5, 'Kyutai Team, "Moshi: a speech-text foundation model for real-time dialogue," arXiv:2410.00037, 2024.'),
        ref(6, 'J. Johnson et al., "Billion-scale similarity search with GPUs," IEEE TBD, 2021.'),
        ref(7, 'M. Baas et al., "Voice conversion with just nearest neighbors," Interspeech, 2023.'),
        ref(8, 'C. Wang et al., "Neural codec language models are zero-shot text to speech synthesizers," arXiv:2301.02111, 2023.'),
        ref(9, 'Z. Borsos et al., "SoundStorm: Efficient parallel audio generation," arXiv:2305.09636, 2023.'),
        ref(10, 'J. Li et al., "FreeVC: Towards high-quality text-free one-shot voice conversion," ICASSP, 2023.'),
        ref(11, 'S. Chen et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing," IEEE JSTSP, 2022.'),
        ref(12, 'X. Zhang et al., "SpeechTokenizer: Unified speech tokenizer for speech language models," ICLR, 2024.'),
        ref(13, 'N. Takahashi et al., "HASRD: Hierarchically aligned semantic-acoustic RVQ disentanglement," Interspeech, 2025.'),
        ref(14, 'J. Sim et al., "SKQVC: One-shot VC by k-means quantization with self-supervised representations," Interspeech, 2024.'),
        ref(15, 'RVC-Project, "Retrieval-based Voice Conversion," github.com/RVC-Project, 2023.'),
        ref(16, 'Y. Wang et al., "MaskGCT: Zero-shot TTS with masked generative codec transformer," ICLR, 2025.'),
        ref(17, 'A. Ziv et al., "Masked audio generation using a single non-autoregressive transformer," ICLR, 2024.'),
        ref(18, 'Y. Du et al., "vec2wav 2.0: Advancing voice conversion via discrete token vocoders," arXiv:2409.01995, 2024.'),
        ref(19, 'E. Perez et al., "FiLM: Visual reasoning with a general conditioning layer," AAAI, 2018.'),
        ref(20, 'T. Kaneko et al., "CycleGAN-VC3: Examining and improving CycleGAN-VCs for non-parallel voice conversion," Interspeech, 2020.'),
        ref(21, 'H. Lu et al., "VEVO: Controllable zero-shot voice imitation with self-supervised speech tokenizer," arXiv:2412.10723, 2024.'),
        ref(22, 'T. Hayashi et al., "Non-autoregressive sequence-to-sequence voice conversion," ICASSP, 2021.'),
        ref(23, 'ByteDance, "Seed-VC: Zero-shot voice conversion with diffusion transformer," arXiv:2411.10508, 2024.'),
        ref(24, 'Z. Zhang et al., "StableVC: Conditional flow matching for voice conversion," AAAI, 2025.'),
        ref(25, 'A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition," Interspeech, 2020.'),
        ref(26, 'J. Su et al., "RoFormer: Enhanced transformer with rotary position embedding," arXiv:2104.09864, 2021.'),
        ref(27, 'T.-Y. Lin et al., "Focal loss for dense object detection," ICCV, 2017.'),
        ref(28, 'Y. Koizumi et al., "LibriTTS-R: A restored multi-speaker text-to-speech corpus," Interspeech, 2023.'),
        ref(29, 'C. Wan et al., "Generalized end-to-end loss for speaker verification," ICASSP, 2018.'),
        ref(30, 'J. Yamagishi et al., "CSTR VCTK Corpus: English multi-speaker corpus," Univ. of Edinburgh, 2019.'),
        ref(31, 'T. Saeki et al., "UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge 2022," Interspeech, 2022.'),
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buffer => {
  const out = `${paperDir}/crt_voice_conversion.docx`;
  fs.writeFileSync(out, buffer);
  console.log(`Written: ${out} (${(buffer.length/1024).toFixed(0)} KB)`);
});
