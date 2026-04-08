const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType,
  PageNumber, PageBreak, Footer, Header,
} = require('docx');

const paperDir = '/Users/nguyenquoc/duplex/paper';
const fig1 = fs.readFileSync(`${paperDir}/fig1_mcd_comparison.png`);
const fig2 = fs.readFileSync(`${paperDir}/fig2_confidence_threshold.png`);
const fig3 = fs.readFileSync(`${paperDir}/fig3_secs_mcd_tradeoff.png`);
const fig4 = fs.readFileSync(`${paperDir}/fig4_architecture.png`);

// ── SPIE Template Constants ──
const F = 'Times New Roman';
const TITLE_SZ = 32;   // 16pt
const AUTH_SZ = 24;     // 12pt
const ABS_HEAD_SZ = 22; // 11pt
const BODY_SZ = 20;     // 10pt
const SMALL_SZ = 18;    // 9pt
const REF_SZ = 16;      // 8pt

// Page: US Letter, 1" top/bottom, 1.25" left/right (matching SPIE template)
const PAGE = { size:{width:12240,height:15840}, margin:{top:1440,right:1800,bottom:1440,left:1800} };
const CONTENT_W = 12240 - 1800 - 1800; // 8640 DXA

const border = {style:BorderStyle.SINGLE,size:1,color:'999999'};
const borders = {top:border,bottom:border,left:border,right:border};
const cm = {top:40,bottom:40,left:80,right:80};

// Helpers
const t = (text,opts={}) => new TextRun({text,font:F,size:BODY_SZ,...opts});
const tb = (text,opts={}) => new TextRun({text,font:F,size:BODY_SZ,bold:true,...opts});
const ti = (text,opts={}) => new TextRun({text,font:F,size:BODY_SZ,italics:true,...opts});
const sup = (text) => new TextRun({text,font:F,size:AUTH_SZ,superScript:true});

function p(runs,opts={}) {
  return new Paragraph({
    spacing:{after:120,line:276},
    alignment:AlignmentType.JUSTIFIED,
    ...opts,
    children: typeof runs === 'string' ? [t(runs)] : runs,
  });
}

function heading(num,title) {
  return new Paragraph({
    spacing:{before:240,after:120},
    children:[new TextRun({text:`${num}. ${title.toUpperCase()}`,font:F,size:ABS_HEAD_SZ,bold:true})],
  });
}

function subheading(num,title) {
  return new Paragraph({
    spacing:{before:180,after:80},
    children:[new TextRun({text:`${num} ${title}`,font:F,size:BODY_SZ,bold:true})],
  });
}

function refp(n,text) {
  return new Paragraph({
    spacing:{after:40,line:240},
    indent:{left:360,hanging:360},
    children:[
      new TextRun({text:`[${n}] `,font:F,size:REF_SZ}),
      new TextRun({text,font:F,size:REF_SZ}),
    ],
  });
}

function tbl(headers,rows,cw) {
  const tw = cw.reduce((a,b)=>a+b,0);
  const hr = new TableRow({children:headers.map((h,i)=>new TableCell({
    borders,width:{size:cw[i],type:WidthType.DXA},margins:cm,
    shading:{fill:'D9E2F3',type:ShadingType.CLEAR},
    children:[new Paragraph({alignment:AlignmentType.CENTER,children:[new TextRun({text:h,font:F,size:SMALL_SZ,bold:true})]})],
  }))});
  const dr = rows.map((row,ri) => new TableRow({children:row.map((cell,ci) => new TableCell({
    borders,width:{size:cw[ci],type:WidthType.DXA},margins:cm,
    shading: ri%2===0 ? {fill:'F2F2F2',type:ShadingType.CLEAR} : undefined,
    children:[new Paragraph({alignment:ci===0?AlignmentType.LEFT:AlignmentType.CENTER,
      children:[new TextRun({text:String(cell),font:F,size:SMALL_SZ})]})],
  }))}));
  return new Table({width:{size:tw,type:WidthType.DXA},columnWidths:cw,rows:[hr,...dr]});
}

function tc(n,text) {
  return new Paragraph({spacing:{before:160,after:60},alignment:AlignmentType.CENTER,
    children:[tb(`Table ${n}. `,{size:SMALL_SZ}),new TextRun({text,font:F,size:SMALL_SZ})]});
}

function img(data,w,h) {
  return new Paragraph({alignment:AlignmentType.CENTER,spacing:{before:120,after:40},
    children:[new ImageRun({type:'png',data,transformation:{width:w,height:h},
      altText:{title:'Figure',description:'Chart',name:'fig'}})]});
}

function fc(n,text) {
  return new Paragraph({spacing:{before:40,after:160},alignment:AlignmentType.CENTER,
    children:[tb(`Figure ${n}. `,{size:SMALL_SZ}),new TextRun({text,font:F,size:SMALL_SZ})]});
}

function meq(text) {
  return new Paragraph({spacing:{before:80,after:80},alignment:AlignmentType.CENTER,
    children:[new TextRun({text,font:'Cambria Math',size:BODY_SZ,italics:true})]});
}

const sp = (n=80) => new Paragraph({spacing:{after:n}});

// ══════════════════════════════════════════════════════════════
// BUILD DOCUMENT
// ══════════════════════════════════════════════════════════════

const doc = new Document({
  styles:{default:{document:{run:{font:F,size:BODY_SZ}}}},
  sections:[{
    properties:{
      page:PAGE,
    },
    footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.CENTER,
      children:[new TextRun({font:F,size:REF_SZ,children:[PageNumber.CURRENT]})]
    })]})},
    children:[
      // ══ TITLE ══
      new Paragraph({spacing:{after:120},alignment:AlignmentType.CENTER,children:[
        new TextRun({text:'Codebook Refinement Transformer: Post-Retrieval Correction',font:F,size:TITLE_SZ,bold:true}),
      ]}),
      new Paragraph({spacing:{after:240},alignment:AlignmentType.CENTER,children:[
        new TextRun({text:'for Neural Codec Voice Conversion',font:F,size:TITLE_SZ,bold:true}),
      ]}),

      // ══ AUTHORS ══
      new Paragraph({spacing:{after:60},alignment:AlignmentType.CENTER,children:[
        new TextRun({text:'Quoc V. Nguyen*',font:F,size:AUTH_SZ}),
        sup('a'),
      ]}),
      new Paragraph({spacing:{after:60},alignment:AlignmentType.CENTER,children:[
        sup('a'),
        new TextRun({text:'Enterprise AI Center',font:F,size:AUTH_SZ}),
      ]}),
      new Paragraph({spacing:{after:240},alignment:AlignmentType.CENTER,children:[
        sup('*'),
        new TextRun({text:' Corresponding author: quoc.nguyen@eaic.org',font:F,size:AUTH_SZ}),
      ]}),

      // ══ ABSTRACT ══
      new Paragraph({spacing:{before:480,after:240},children:[
        new TextRun({text:'ABSTRACT',font:F,size:ABS_HEAD_SZ,bold:true}),
      ]}),
      p([t('We propose the Codebook Refinement Transformer (CRT), a non-autoregressive model that improves retrieval-based voice conversion on discrete neural audio codec tokens. CRT operates on Mimi\u2019s hierarchical RVQ token space using a Conformer backbone with FiLM speaker conditioning, trained with inference-matched FAISS corruption. We introduce confidence-selective decoding: CRT predictions override retrieved tokens only when softmax confidence exceeds a threshold. On a 100-pair zero-shot evaluation across VCTK speakers, CRT improves MCD over the FAISS retrieval baseline (102.6 \u00B1 15.1 vs. 103.3 \u00B1 16.3) while achieving higher speaker similarity than kNN-VC (SECS 0.816 vs. 0.796). CRT achieves real-time processing (RTF 0.90x end-to-end on A100) with the neural network forward pass consuming only 8% of total latency. CRT\u2019s token-to-token architecture provides native integrability with codec-based speech systems: no external vocoders, no format conversion, no codec round-trips.')]),

      new Paragraph({spacing:{after:120},children:[
        tb('Keywords: ',{size:BODY_SZ}),
        t('voice conversion, neural audio codec, residual vector quantization, discrete token refinement, confidence-selective decoding, Conformer, intelligent signal processing'),
      ]}),

      // ══ 1. INTRODUCTION ══
      heading('1','Introduction'),
      p('Voice conversion (VC) transforms a source speaker\u2019s utterance to sound as if spoken by a target speaker while preserving linguistic content. This technology has applications in intelligent communication systems, adaptive human-computer interfaces, and privacy-preserving speech processing [1]. Neural audio codecs \u2014 SoundStream [2], Encodec [3], DAC [4], and Mimi [5] \u2014 enable operating on discrete token sequences rather than continuous spectral features, using Residual Vector Quantization (RVQ) to compress speech into hierarchical codebook layers.'),
      p('Retrieval-based voice conversion on codec tokens via FAISS [6] is simple and training-free, but introduces a fundamental limitation: retrieved tokens for each codebook may originate from different target frames, breaking the RVQ residual chain. We propose CRT as a post-retrieval correction stage. Unlike VALL-E [8] and SoundStorm [9] that generate tokens from scratch, CRT takes already-retrieved tokens as input and predicts corrections via confidence-selective decoding.'),

      tc('1','Multi-pair results (100 VCTK pairs, mean \u00B1 std).'),
      tbl(['Method','Domain','SECS \u2191','MCD \u2193','N'],
        [['FreeVC [10]','Continuous (VITS)','0.824\u00B10.047','85.1\u00B111.1','99'],
         ['kNN-VC [7]','Continuous (HiFi-GAN)','0.796\u00B10.064','97.9\u00B115.0','100'],
         ['CRT conf>0.4','Codec-native','0.816\u00B10.064','102.6\u00B115.1','100'],
         ['FAISS baseline','Codec-native','0.827\u00B10.053','103.3\u00B116.3','100']],
        [1800,1500,1400,1400,600]),
      sp(),

      p([t('CRT is the only method that improves upon its retrieval baseline, and achieves higher SECS than kNN-VC (0.816 vs. 0.796). CRT\u2019s token-to-token architecture enables native integration into codec-based systems without vocoders or format conversion.')]),

      // ══ 2. RELATED WORK ══
      heading('2','Related Work'),
      p('SoundStream [2], Encodec [3], and Mimi [5] compress waveforms into discrete RVQ tokens. Mimi\u2019s CB0 is distilled from WavLM [11] for content; CB1\u201331 encode speaker acoustics at 12.5 Hz with 32 codebooks of 2048 entries. kNN-VC [7] (MOS 4.03) and FreeVC [10] (MOS 4.06) operate on continuous WavLM features with external vocoders. SoundStorm [9] and MaskGCT [16] generate codec tokens from scratch via masked prediction. vec2wav 2.0 [18] processes discrete tokens for vocoding. CRT is distinct: it refines already-retrieved tokens in a single forward pass within the codec\u2019s native space.'),
      p([t('CRT\u2019s individual components \u2014 Conformer [26], FiLM [19], focal loss [28], confidence decoding [9] \u2014 are established. The novelty lies in their combination for post-retrieval refinement of discrete RVQ tokens, with confidence-selective override against a retrieval fallback. No published method combines all five elements in this domain.')]),

      // ══ 3. METHOD ══
      heading('3','Method'),
      subheading('3.1','Problem Formulation'),
      p([t('A source utterance is encoded by Mimi into '), ti('T'), t(' frames of 32 codebooks. Stage 1 (Retrieval): FAISS search on CB0 embeddings with temporal continuity. Stage 2 (Refinement): CRT takes retrieved CB1\u2013CB18, CB0 anchor, and speaker embedding v \u2208 \u211D'), new TextRun({text:'256',font:F,size:14,superScript:true}), t(':')]),
      meq('\u0108\u02B3\u1D49\u1DA0 = CRT(\u0108\u1D57\u1D4D\u1D57\u2080, \u0108\u1D57\u1D4D\u1D57\u2081\u2236\u2096, v)'),

      subheading('3.2','Architecture'),
      p([t('CRT-XLarge: d'), new TextRun({text:'model',font:F,size:14,subScript:true}), t('=1280, 16 heads, 16 Conformer layers, 707M params. Input: sum of per-codebook embeddings. FiLM speaker conditioning before backbone. Depthwise conv (kernel 31, 2.5s span) captures local frame continuity. Per-CB prediction heads (18 independent linear projections to 2048-dim vocabulary). Auxiliary speaker identity head for SECS loss.')]),

      subheading('3.3','Training'),
      p('90% FAISS cross-speaker corruption (matching inference noise); 10% random. Three-phase curriculum: 30%\u219250%\u219270% corruption over 240k steps. RVQ dropout at 15%. Loss: focal CE (\u03B3=1.0) + SECS loss (\u03BB=0.3). AdamW, lr=10\u207B\u2074, batch 128, 240k steps on A100-80GB. Training cost: 132 GPU-hours (~46 kWh). Smaller configurations (28M Large variant) reduce cost proportionally.'),

      subheading('3.4','Confidence-Selective Decoding'),
      p('At inference, CRT overrides retrieved tokens only when max softmax probability exceeds \u03C4. At optimal \u03C4=0.4, 17.8% of tokens are overridden. This single-pass hybrid strategy combines learned refinement with retrieval fallback.'),

      // ══ 4. EXPERIMENTS ══
      heading('4','Experimental Setup'),
      p([t('Training: 4,655 speakers, 202h from LibriTTS-R [29]. Evaluation: 100 pairs from VCTK [31] (zero-shot). Baselines: FAISS retrieval, kNN-VC [7] (official, WavLM+HiFi-GAN), FreeVC [10] (official, WavLM+VITS). Metrics: SECS (speaker similarity), MCD (cross-speaker spectral distance via 13-dim MFCCs), UTMOS [32] (predicted MOS).')]),

      // ══ 5. RESULTS ══
      heading('5','Results'),
      subheading('5.1','Multi-Pair Evaluation'),
      p('Table 1 (above) shows 100-pair results. CRT consistently improves over FAISS (MCD 102.6 vs. 103.3) with reduced variance (15.1 vs. 16.3). CRT SECS (0.816) exceeds kNN-VC (0.796) by 2.5% and is competitive with FreeVC (0.824).'),

      subheading('5.2','Confidence Threshold Analysis'),
      tc('2','Single-pair: confidence threshold sweep.'),
      tbl(['\u03C4','SECS','MCD','Override%'],
        [['0.3','0.813','109.29','26.2%'],['0.4','0.816','102.45','17.8%'],
         ['0.5','0.819','109.40','10.0%'],['0.6','0.827','109.02','6.1%'],
         ['0.9','0.815','109.29','1.3%']],
        [1500,2000,2000,2000]),
      sp(),
      p([t('CRT argmax (100% override) yields MCD 127.66 \u2014 worse than the FAISS baseline (106.61). CRT conf>0.4 (17.8% override) yields 102.45 \u2014 '), tb('better'), t(' than the baseline. The same model, with the same predictions, transitions from worse-than-baseline to better-than-baseline solely through confidence gating. This validates the selection mechanism: a model correct on 75% of CB1 tokens produces catastrophic output when applied unconditionally (25% errors compound across 18 codebooks), but improves the baseline when restricted to high-confidence overrides.')]),

      img(fig1, 420, 300),
      fc('1','MCD comparison across methods.'),

      img(fig2, 420, 300),
      fc('2','Confidence threshold \u03C4 vs. MCD and override rate.'),

      subheading('5.3','Perceptual Quality and Codec Ceiling'),
      tc('3','Predicted MOS (UTMOS) by decoder type.'),
      tbl(['Method','UTMOS (1\u20135)','Decoder'],
        [['FreeVC','4.46','VITS vocoder'],['kNN-VC','3.56','HiFi-GAN'],
         ['CRT conf>0.4','2.52','Mimi (1.1 kbps)'],['FAISS baseline','2.44','Mimi (1.1 kbps)']],
        [2200,1800,3000]),
      sp(),
      p('The UTMOS gap reflects Mimi\u2019s 1.1 kbps codec ceiling, not CRT quality. FAISS baseline (no CRT) is already 2.44; CRT improves to 2.52. In a codec pipeline, audio is already at Mimi quality \u2014 CRT improves within that tier.'),

      subheading('5.4','Inference Efficiency'),
      tc('4','Real-time factor (4.1s audio, A100).'),
      tbl(['Method','Total (ms)','RTF','Real-time?'],
        [['FreeVC','2,438','0.59x','Yes (1.7x)'],
         ['CRT (ours)','3,694','0.90x','Yes (1.1x)'],
         ['kNN-VC','7,836','1.91x','No (0.5x)']],
        [2200,1800,1600,2000]),
      sp(),
      p('CRT\u2019s neural network forward pass is only 296ms (8% of total). The bottleneck is FAISS retrieval (3,018ms, 82%), optimizable via approximate indices.'),

      subheading('5.5','System Integrability'),
      tc('5','Integrability comparison for codec-based systems.'),
      tbl(['Dimension','CRT','kNN-VC','FreeVC'],
        [['Native codec I/O','Yes','No','No'],
         ['External vocoder','No','HiFi-GAN','VITS'],
         ['Codec round-trips','0','1','1'],
         ['Streaming capable','Yes','No','No'],
         ['Per-frame overhead','6.0 ms','150.7 ms','46.9 ms']],
        [2200,1600,1600,1600]),
      sp(),
      p('CRT is the only token-to-token method. It slots into codec pipelines with zero format conversion, enabling voice conversion as a token-space module for streaming speech systems, without additional vocoders or feature extractors.'),

      img(fig3, 420, 300),
      fc('3','SECS vs. MCD trade-off across methods.'),

      img(fig4, 480, 240),
      fc('4','CRT voice conversion pipeline: encode \u2192 retrieve \u2192 refine \u2192 decode.'),

      // ══ 6. DISCUSSION ══
      heading('6','Discussion'),
      p([tb('Refinement vs. generation. '), t('Retrieved tokens are >80% correct. CRT fixes ~18% at frame boundaries. A single forward pass suffices where autoregressive models need hundreds of steps.')]),
      p([tb('Confidence-selective decoding. '), t('The selection mechanism is as important as the prediction model. A 75%-accurate CB1 predictor destroys quality when applied unconditionally but improves the baseline when confidence-gated \u2014 a principle applicable to any post-processing task with a fallback baseline.')]),
      p([tb('Foundation for codec-native voice flexibility. '), t('CRT provides voice conversion as a modular, pluggable component for codec-based speech systems \u2014 tokens in, tokens out, no vocoders, no format conversion.')]),
      p([tb('Limitations. '), t('(1) No MOS/MUSHRA; UTMOS suggests quality improvement but formal validation needed. (2) Fixed threshold; adaptive per-codebook \u03C4 could help. (3) CB0 passthrough limits SECS. (4) 2023-only baselines; comparison with Seed-VC, VEVO planned.')]),
      p([tb('Ethical considerations. '), t('Voice conversion carries risks of misuse (speaker impersonation, deepfakes). Mitigating factors: CRT requires a multi-utterance target corpus (not single-sample cloning); codec output at 1.1 kbps is perceptually distinguishable from natural speech (UTMOS ~2.5 vs. ~4.5); discrete tokens are amenable to watermarking. We support development of detection systems for codec-based VC.')]),

      // ══ 7. CONCLUSION ══
      heading('7','Conclusion'),
      p([t('CRT combines established components in a novel formulation: post-retrieval refinement of discrete RVQ codec tokens. On 100 zero-shot VCTK pairs, CRT consistently improves over FAISS (MCD 102.6 vs. 103.3) while achieving '), tb('higher SECS than kNN-VC (0.816 vs. 0.796)'), t('. Confidence-selective decoding overrides only 17.8% of tokens. RTF 0.90x on A100. CRT\u2019s token-to-token architecture positions it as foundational infrastructure for flexible voice output in codec-native speech systems.')]),

      // ══ ACKNOWLEDGMENTS ══
      heading('','Acknowledgments'),
      p('Computational resources provided by the Enterprise AI Center DGX cluster. We thank the Kyutai team for the open-source Mimi codec, and the authors of kNN-VC and FreeVC for their public implementations.'),

      // ══ REFERENCES ══
      heading('','References'),
      refp(1,'L. Sun et al., "Voice conversion: State of the art and future trends," IEEE Signal Processing Magazine, 2024.'),
      refp(2,'N. Zeghidour et al., "SoundStream: An end-to-end neural audio codec," IEEE/ACM TASLP, 2022.'),
      refp(3,'A. Defossez et al., "High fidelity neural audio compression," TMLR, 2023.'),
      refp(4,'R. Kumar et al., "High-fidelity audio compression with improved RVQGAN," NeurIPS, 2023.'),
      refp(5,'Kyutai Team, "Moshi: a speech-text foundation model for real-time dialogue," arXiv:2410.00037, 2024.'),
      refp(6,'J. Johnson et al., "Billion-scale similarity search with GPUs," IEEE TBD, 2021.'),
      refp(7,'M. Baas et al., "Voice conversion with just nearest neighbors," Interspeech, 2023.'),
      refp(8,'C. Wang et al., "Neural codec language models are zero-shot text to speech synthesizers," arXiv:2301.02111, 2023.'),
      refp(9,'Z. Borsos et al., "SoundStorm: Efficient parallel audio generation," arXiv:2305.09636, 2023.'),
      refp(10,'J. Li et al., "FreeVC: Towards high-quality text-free one-shot voice conversion," ICASSP, 2023.'),
      refp(11,'S. Chen et al., "WavLM: Large-scale self-supervised pre-training for full stack speech processing," IEEE JSTSP, 2022.'),
      refp(16,'Y. Wang et al., "MaskGCT: Zero-shot TTS with masked generative codec transformer," ICLR, 2025.'),
      refp(18,'Y. Du et al., "vec2wav 2.0: Advancing voice conversion via discrete token vocoders," arXiv:2409.01995, 2024.'),
      refp(19,'E. Perez et al., "FiLM: Visual reasoning with a general conditioning layer," AAAI, 2018.'),
      refp(26,'A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition," Interspeech, 2020.'),
      refp(28,'T.-Y. Lin et al., "Focal loss for dense object detection," ICCV, 2017.'),
      refp(29,'Y. Koizumi et al., "LibriTTS-R: A restored multi-speaker text-to-speech corpus," Interspeech, 2023.'),
      refp(30,'C. Wan et al., "Generalized end-to-end loss for speaker verification," ICASSP, 2018.'),
      refp(31,'J. Yamagishi et al., "CSTR VCTK Corpus: English multi-speaker corpus," Univ. of Edinburgh, 2019.'),
      refp(32,'T. Saeki et al., "UTMOS: UTokyo-SaruLab system for VoiceMOS Challenge 2022," Interspeech, 2022.'),
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  const out = `${paperDir}/crt_voice_conversion_spie.docx`;
  fs.writeFileSync(out, buffer);
  console.log(`Written: ${out} (${(buffer.length/1024).toFixed(0)} KB)`);
});
