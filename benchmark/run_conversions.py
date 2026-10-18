#!/usr/bin/env python3
"""
H3 — conversion harness for the CRT journal extension (ANALYSIS_PLAN.md §2: S1 FAISS, S2/S3 CRT τ, S4 CRT argmax;
ANALYSIS_PLAN_v1.md §2.5: the Mimi Q = 32 copy-synthesis controls).

Inference is the published SPIE pipeline (run_multipair_benchmark.py init_crt + run_crt_pair), refactored into
encode / retrieve / crt_logits / gate / decode with the same calls and numerics:
  Mimi       transformers MimiModel 'kyutai/mimi' (HF cache), 24 kHz, 12.5 Hz, 32 codebooks, float16 on CUDA as
             published. On CPU it runs in float32: torch 2.5.1 has no Half replication_pad1d on CPU, so the published
             code cannot run there at all (a CPU run is a numerical, not an algorithmic, deviation).
  Reference  make_pairs.load_reference(R[budget], 24000) + check_reference_len (new sets), or the old target_files
             concatenated at full length (--compat_spie). Encoded in 30 s chunks; chunks of <= 1 s are dropped.
  Retrieval  per-target FAISS IndexFlatIP over the L2-normalised CB0 codebook vectors of the reference frames (the
             float16 codebook table on every device, as published); source CB0 queries, k = 8; greedy path with the
             continuity score 0.5 * sim + 0.5 / (1 + |cand - prev| / 12.5), first frame = top-1. All 32 codebooks
             are copied from the chosen frame: this is S1 FAISS.
  CRT        --variant (default v5.6.5, the published model; VARIANTS lists the lineage checkpoints of E12). The CRT
             class is imported from that variant's own train_crt.py, built from the checkpoint's config (not the
             script's MODEL_CONFIGS), n_cb / cb_size / spk_emb_dim read from the weights, load_state_dict(strict=True).
             CRTAdapter gives every variant one interface (F-u): 'anchor' classes take forward(cb0, cb_noisy, spk_emb)
             and refine CB1-n_cb (n_cb = 18, or 8 for v9.1, 12 or 18 on the checkpoints_v5 ladder); the 'joint' v10.0
             class takes forward(cb_noisy, spk_emb) and predicts CB0-n_cb-1, CB0 included; the output is either the
             stacked logits (v5 ladder, v5.3.1, v5.3.1-1024, v5.3.2) or (logits, spk_pred). Only heads that received
             a loss gradient somewhere in the variant's lineage are refined (TRAINED_HEADS; default every head): v10.0
             at step 24000 trained only its CB0 head, so it refines CB0 alone and CB1-31 pass through (its CB1-9
             heads are random initialisation). conv_cfg records trained_cbs and untrained_heads. Inputs: the retrieved
             frames' codes + the Resemblyzer embedding of the first 15 s of the reference (CPU VoiceEncoder,
             embed_utterance without preprocess_wav). Autocast float16 on CUDA only; softmax in float32.
  Gate       Each refined codebook's token (CB1-18 for v5.6.5) replaced by the CRT argmax where max-softmax > τ
             (strict); every other codebook passes through. crt_argmax replaces every refined token (trained heads only).
  Decode     Mimi decode of the 32 x T codes, written as 24 kHz PCM_16 (the soundfile default the SPIE run used).
  Copies     mimi_copy_src / mimi_copy_P (the codec ceiling): the source recording / the pair's parallel target
             recording P encoded at all 32 codebooks and decoded, with the encode and decode above (same Mimi dtype and
             device). The copies use no retrieved code and no CRT output, but the pair's reference, retrieval, CRT
             forward and dump still run (a copies-only run rewrites the dumps of an existing --out_dir). P is read only
             when its copy is to be written. A pair without P gets no mimi_copy_P row (not an error row). The rows have
             the fields of a faiss row (ref_text = the source text, which P shares after make_pairs normalisation;
             parallel_wav = P; sim_ref_wavs = the target's E); n_frames is the copied recording's.
One retrieval and one CRT forward per pair; every τ is gated from the same probabilities; identical code tensors are
decoded once. No sampling anywhere; cuDNN is left at the published defaults. FAISS runs on 1 OpenMP thread and torch
on OMP_NUM_THREADS; outputs are reproducible for a fixed device type and thread count.

Outputs under --out_dir:
  <set>/<budget>/<system>/<pair_id>.wav
  dumps/<set>/<budget>/<pair_id>.npz  src_codes, retrieved_codes (32 x T, = the FAISS output codes), ref_codes
      (T_ref x 32 int16: the reference codes as encoded in this run; cand_idx and path index its rows, so
      ref_codes[cand_idx] are the 32 codebooks of every candidate frame and no offline variant re-encodes the
      reference), path (chosen reference frame per source frame), path_rank, path_sim, cand_idx / cand_sim (T x 8, all
      FAISS candidates), crt_first_cb (codebook of row 0 of every CRT array: 1, or 0 for v10.0), crt_argmax,
      crt_conf (max softmax), crt_margin (top-1 minus top-2 prob), crt_entropy (n_cb x T, n_cb = the refined
      codebooks; 18 x T = CB1-18 for v5.6.5, 1 x T = CB0 for v10.0), topk_prob / topk_idx (n_cb x T x 16: the 16
      largest CRT probabilities and their tokens, descending), cand_logp (n_cb x 8 x T: CRT log-prob of each
      candidate frame's token), mask_tau<τ> (n_cb x T) per τ; with
      --dump_logits also crt_logits (n_cb x T x 2048 float16). Temperature-scaled gates need the full logits: they are
      exact only from a --dump_logits run (or a new CRT forward), not from the top-16.
  <set>/<budget>/manifest[.shard<i>of<n>].jsonl  one row per (pair_id, system), directly consumable by metrics.py
      (native schema): pair_id, set, budget, system, tau, wav, ref_text, parallel_wav, sim_ref_wavs (E), source_wav,
      R ({budget: the R entry used}), wer_primary, speakers, genders, dump, n_frames, n_ref_frames,
      override_coverage (M7: gate open, i.e. max-softmax > τ, the published "override %"), change_coverage (gate
      open and argmax != retrieved token; 0 tokens means the output equals FAISS), n_changed, same_codes_as,
      conv_time_s (per stage; ref_* are 0 when the reference came from the per-target cache), run_id, conv_cfg_sha.
      A failed pair gets rows with `error` for every system it still needed (wav path kept, stale file removed); they
      are retried on the next run and fail metrics.py rows instead of vanishing. Resume skips finished rows written
      under the same conv_cfg; a different conv_cfg (weights, Mimi dtype, device type, library versions, dump
      version, --dump_logits) is an error. At the end the file is compacted to the last row per (pair_id, system).
      --out_dir is made absolute, so wav and dump paths do not depend on the working directory.
  <set>/<budget>/runs.jsonl  a start and an end record per invocation: conv_cfg (variant, weights sha256, step,
      model_size, config, refined codebooks, ...), args, source checkpoint, timings, error count.
--ckpt defaults to the variant's slim weights file (models/crt_<variant>_best_weights.pt; step, model_state,
model_size, config) when it exists, else to the variant checkpoint itself (read with mmap; the optimizer and EWC
state are never read). v5.6.5's slim file is created from --src_ckpt when missing. weights_sha256 hashes the
model_state tensors (names, dtypes, shapes, bytes), so a slim file and its source checkpoint hash the same.
Output wavs and dumps do not record the variant: give each variant its own --out_dir.

--compat_spie: --pairs is the old eval_pairs.json; set as given, budget 'full', reference = every existing
target_file at full length (as the SPIE run). Each faiss / crt_tau0.4 output is compared with the saved
<compare_dir>/<pair>/<pair>_{faiss,crt04}.wav (row field repro: sample diffs; per 1920-sample frame the absolute RMS
difference, and the relative one only on speech frames, frame RMS >= -20 dB re the loudest frame; code agreement
after re-encoding both with this Mimi). The crt_tau0.4 row also gets gate_repro: the first frame where the saved
faiss and crt04 files differ (Mimi decoding is causal, so it dates the first changed code) against our first
changed frame, and the non-retrieved CRT argmax whose confidence is nearest to τ. Gate decisions near τ can flip
between CPU float32 and the published GPU float16, so only a GPU run can confirm them. Compat rows lack E and P, so
they are not metrics.py input.
"""
import argparse
import filecmp
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import time
import traceback
import warnings

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_pairs  # noqa: E402
import metrics  # noqa: E402

D = '/datastore/nvquoc/crt/mimi_vc'
MIMI_ID = 'kyutai/mimi'
FRAME = 1920          # samples per Mimi frame at 24 kHz
FRAME_RATE = 12.5     # continuity scale of the published greedy path
K = 8
TOPK = 16             # CRT top-k probabilities in the dump
SPK_S = 15            # seconds of reference for the speaker embedding
CHUNK_S = 30          # reference encoding chunk
DUMP_VERSION = 3      # 2: + ref_codes, topk_prob, topk_idx (crt_logits with --dump_logits); 3: + crt_first_cb

# E12 lineage (extension_plan F-q): variant -> (best checkpoint, the train_crt.py whose CRT class it was saved from).
_DIRS = {'v5.3.1': 'v5.3.1-CRT-parallel', 'v5.3.1-1024': 'v5.3.1-CRT-parallel-layers-14-d_model-1024',
         'v5.3.2': 'v5.3.2-CRT-parallel-layers-16-d_model-1280-curriculum-corruption',
         'v5.4.0': 'v5.4.0-CRT-parallel-SECS-reg', 'v5.6': 'v5.6-CRT-curriculum-SECS',
         'v5.6.1': 'v5.6.1-CRT-FAISS-passthrough', 'v5.6.1-datav2': 'v5.6.1-CRT-FAISS-passthrough-datav2',
         'v5.6.2': 'v5.6.2-confidence-reranker', 'v5.6.4-datav3': 'v5.6.4-datav3', 'v5.6.5': 'v5.6.5-secs-high',
         'v8.0': 'v8.0-progressive-rvq-dropout', 'v9.0': 'v9.0-ewc-progressive', 'v9.1': 'v9.1-ewc-cb8',
         'v10.0': 'v10.0-cb0-refine'}
VARIANTS = {k: {'dir': f'{D}/{d}', 'ckpt': f'{D}/{d}/checkpoints/crt_xlarge_best.pt', 'script': f'{D}/{d}/train_crt.py'}
            for k, d in _DIRS.items()}
VARIANTS.update({f'v5-{s}': {'dir': f'{D}/checkpoints_v5', 'ckpt': f'{D}/checkpoints_v5/crt_{s}_best.pt',
                             'script': f'{D}/v5/train_crt.py'} for s in ('small', 'medium', 'large', 'xlarge')})
# The last checkpoint where the best one is not the variant's own training: the v5.6.1-datav2 and v5.6.4-datav3 best
# files are their warm starts (identical to v5.6.1 saves), and v9.1's best (48k) precedes its CB2-6 phases.
VARIANTS.update({f'{k}-last': dict(VARIANTS[k], ckpt=VARIANTS[k]['ckpt'].replace('_best.pt', '.pt'))
                 for k in ('v5.6.1-datav2', 'v5.6.4-datav3', 'v9.1')})
# Heads outside the lineage's trained set (lineage_table.py) are random initialisation and are not refined. The entry is
# tied to the checkpoint step it was checked at; a different step stops the run until the lineage is re-checked.
TRAINED_HEADS = {'v10.0': {'step': 24000, 'n': 1, 'source': 'lineage_table: v10.0 trained from scratch, phase_idx 0 at '
                           'step 24000 (active 1/10); CE only over range(active) (train_crt.py:281-284)'}}
for _k, _v in VARIANTS.items():
    _v['slim'] = f'{D}/benchmark/journal/models/crt_{"v565" if _k == "v5.6.5" else _k}_best_weights.pt'
SPEECH_DB = -20       # compat per-frame diffs: speech frame = RMS within 20 dB of the loudest frame
SPIE_SUFFIX = {'faiss': 'faiss', 'crt_tau0.4': 'crt04'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='make_pairs.py JSONL (eval_pairs.json with --compat_spie)')
    p.add_argument('--set', required=True, help='must match the set field of the pair rows')
    p.add_argument('--budget', default='30', help='reference budget in s: 3, 30 or 180 (ignored with --compat_spie)')
    p.add_argument('--systems', default='faiss,crt_tau:0.4,crt_argmax',
                   help='comma list of faiss, crt_argmax, mimi_copy_src, mimi_copy_P and crt_tau:<τ>[,<τ>...], e.g. '
                        'faiss,crt_tau:0.3,0.4,crt_argmax')
    p.add_argument('--variant', default='v5.6.5', choices=list(VARIANTS), help='CRT checkpoint (VARIANTS)')
    p.add_argument('--ckpt', default='', help="weights file; default: the variant's slim file if it exists (always "
                   'for v5.6.5, made when missing), else --src_ckpt')
    p.add_argument('--src_ckpt', default='', help="default: the variant's crt_*_best.pt")
    p.add_argument('--crt_script', default='', help="default: the variant's train_crt.py")
    p.add_argument('--out_dir', required=True)
    p.add_argument('--shard', default='', help='i/n with 0 <= i < n: convert pair rows k with k %% n == i')
    p.add_argument('--max_pairs', type=int, default=0, help='first N pair rows (before sharding)')
    p.add_argument('--pair_ids', default='', help='comma list: only these pair_ids')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help='cuda fails at once when CUDA is unavailable; auto falls back to cpu')
    p.add_argument('--dump_logits', action='store_true', help='also dump the full CRT logits (float16, ~7 MB / 100 frames)')
    p.add_argument('--compat_spie', action='store_true')
    p.add_argument('--compare_dir', default=f'{D}/benchmark/results_multipair')
    return p.parse_args()


def parse_systems(spec):
    """'faiss,crt_tau:0.3,0.4,crt_argmax' -> [('faiss', None), ('crt_tau0.3', 0.3), ('crt_tau0.4', 0.4), ('crt_argmax', None)]
    mimi_copy_src and mimi_copy_P are names like faiss."""
    out, in_tau = [], False
    for tok in (t.strip() for t in spec.split(',')):
        if tok.startswith('crt_tau:'):
            in_tau, tok = True, tok[len('crt_tau:'):]
        if tok in ('faiss', 'crt_argmax', 'mimi_copy_src', 'mimi_copy_P'):
            in_tau = False
            out.append((tok, None))
        elif in_tau and re.fullmatch(r'0?\.\d+|0', tok):
            out.append((f'crt_tau{float(tok):g}', float(tok)))
        else:
            raise SystemExit(f'ERROR: bad --systems token {tok!r}')
    if len({n for n, _ in out}) != len(out):
        raise SystemExit(f'ERROR: duplicate system in {spec!r}')
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


# ── Model loading ────────────────────────────────────────────

def ensure_slim(path, src):
    """Writes {step, model_state, model_size, config} of a checkpoint (no optimizer or EWC state) to `path`."""
    if os.path.exists(path):
        return False
    import torch
    print(f'  creating slim weights {path} from {src}', flush=True)
    ck = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
    slim = {k: ck[k] for k in ('step', 'model_state', 'model_size', 'config')}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(slim, tmp)
    os.replace(tmp, path)
    return True


def state_sha256(sd):
    """sha256 over the sorted tensor names, dtypes, shapes and bytes of a state dict."""
    import torch
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(f'{k}|{t.dtype}|{tuple(t.shape)}|'.encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


def import_script(path, name='crt_variant'):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():   # train_crt.py calls warnings.filterwarnings('ignore') at import
        spec.loader.exec_module(mod)
    return mod


class CRTAdapter:
    """One interface over the lineage CRT classes (F-u): __call__(codes [B, 32, T], spk [B, D]) -> logits
    [B, n_out, T, V] of codebooks first_cb .. first_cb + n_out - 1 (n_out: the trained heads, default all n_cb), from
    the retrieved codes.
      anchor  forward(cb0, cb_noisy, spk_emb): CB0 is an input only; CB1-n_cb are the inputs and the outputs.
      joint   forward(cb_noisy, spk_emb) (v10.0): CB0-(n_cb-1) are the inputs and the outputs, so CB0 is refined too.
    `returns` is 'tensor' (the stacked logits: v5 ladder, v5.3.1, v5.3.1-1024, v5.3.2) or 'tuple' ((logits, spk_pred))."""

    def __init__(self, model, mode, n_cb, cb_size, n_out=None):
        self.model, self.mode, self.n_cb, self.cb_size = model, mode, n_cb, cb_size
        self.n_out = n_out or n_cb   # refined: the first n_out predicted codebooks (the trained heads)
        self.first_cb = {'anchor': 1, 'joint': 0}[mode]
        self.returns = None

    def __call__(self, codes, spk):
        lo, hi = self.first_cb, self.first_cb + self.n_cb
        out = self.model(codes[:, 0], codes[:, lo:hi], spk) if self.mode == 'anchor' else self.model(codes[:, lo:hi], spk)
        kind = 'tuple' if isinstance(out, tuple) else 'tensor'
        if self.returns not in (None, kind):
            raise RuntimeError(f'CRT returned a {kind}, earlier a {self.returns}')
        self.returns = kind
        logits = out[0] if kind == 'tuple' else out
        want = (codes.shape[0], self.n_cb, codes.shape[2], self.cb_size)
        if tuple(logits.shape) != want:
            raise RuntimeError(f'CRT logits {tuple(logits.shape)}, expected {want}')
        return logits[:, :self.n_out]


def cb_range(a, b):
    return f'CB{a}' if a == b else f'CB{a}-{b}'


def load_crt(ckpt_path, crt_script, device, variant):
    """The variant's CRT class, built from the checkpoint's config and the weights' dimensions; strict loading raises
    on any missing or unexpected key. Returns (CRTAdapter, info, timings)."""
    import inspect
    import torch
    tm = {}
    t = time.time()
    mod = import_script(crt_script, 'crt_' + re.sub(r'\W', '_', variant))
    tm['crt_import'] = time.time() - t
    t = time.time()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)   # own training checkpoints
    sd = ck['model_state']
    tm['crt_torch_load'] = time.time() - t
    heads = sorted(int(k.split('.')[1]) for k in sd if re.fullmatch(r'heads\.\d+\.weight', k))
    if not heads or heads != list(range(len(heads))):
        raise SystemExit(f'ERROR: {ckpt_path} has heads {heads}')
    sig = list(inspect.signature(mod.CRT.forward).parameters)[1:]
    mode = {('cb0', 'cb_noisy', 'spk_emb'): 'anchor', ('cb_noisy', 'spk_emb'): 'joint'}.get(tuple(sig))
    if mode is None or (mode == 'anchor') != ('cb0_embed.weight' in sd):
        raise SystemExit(f'ERROR: {crt_script} CRT.forward{tuple(sig)} does not fit the weights '
                         f'(cb0_embed: {"cb0_embed.weight" in sd})')
    trained = TRAINED_HEADS.get(variant)
    if trained and int(ck['step']) != trained['step']:
        raise SystemExit(f'ERROR: {ckpt_path} is step {ck["step"]}; TRAINED_HEADS[{variant!r}] was checked at step '
                         f'{trained["step"]}: re-check which heads were trained (lineage_table.py) first')
    cfg = dict(ck['config'])
    dims = {'n_cb': len(heads), 'cb_size': sd['heads.0.weight'].shape[0], 'spk_emb_dim': sd['spk_proj.0.weight'].shape[1]}
    t = time.time()
    model = mod.CRT(**dims, **cfg)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tm['crt_build_load'] = time.time() - t
    t = time.time()
    sha = state_sha256(model.state_dict())
    tm['crt_sha256'] = time.time() - t
    crt = CRTAdapter(model, mode, dims['n_cb'], dims['cb_size'], trained and trained['n'])
    lo, hi = crt.first_cb, crt.first_cb + crt.n_out - 1
    t = time.time()
    with torch.no_grad():   # 4-frame probe: fixes crt.returns before any real forward
        crt(torch.zeros(1, 32, 4, dtype=torch.long, device=device), torch.zeros(1, dims['spk_emb_dim'], device=device))
    tm['crt_probe'] = time.time() - t
    info = {'variant': variant, 'weights_sha256': sha, 'step': int(ck['step']), 'model_size': ck['model_size'],
            'config': cfg, 'config_is_script_default': cfg == getattr(mod, 'MODEL_CONFIGS', {}).get(ck['model_size']),
            **dims, 'n_params': sum(p.numel() for p in model.parameters()), 'crt_mode': mode, 'crt_returns': crt.returns,
            'first_cb': lo, 'refined_cbs': cb_range(lo, hi), 'trained_cbs': cb_range(lo, hi),
            'untrained_heads': cb_range(hi + 1, lo + dims['n_cb'] - 1) if crt.n_out < dims['n_cb'] else None,
            'trained_cbs_source': trained['source'] if trained else 'all heads (lineage_table)',
            'passthrough_cbs': 'CB0,' * (lo == 1) + f'CB{hi + 1}-31'}
    del ck, sd
    return crt, info, tm


def cb0_table(mimi):
    """CB0 codebook vectors as the published float16 Mimi gives them (codebook.embed of the fp16 model, as float32),
    whatever dtype Mimi runs in here."""
    cb = mimi.quantizer.semantic_residual_vector_quantizer.layers[0].codebook
    emb = cb.embed_sum.detach().half() / cb.cluster_usage.detach().half().clamp(min=cb.epsilon)[:, None]
    return emb.cpu().float().numpy()


class Pipeline:
    def __init__(self, args, device):
        import torch
        from resemblyzer import VoiceEncoder
        from transformers import AutoFeatureExtractor, MimiModel
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == 'cuda' else torch.float32
        self.mimi = MimiModel.from_pretrained(MIMI_ID, torch_dtype=self.dtype).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(MIMI_ID)
        self.sr = self.fe.sampling_rate
        assert self.sr == FRAME * FRAME_RATE, self.sr
        self.cb0 = cb0_table(self.mimi)
        self.spk_enc = VoiceEncoder(device='cpu')
        self.crt, self.crt_info, self.crt_load_s = load_crt(args.ckpt, args.crt_script, device, args.variant)
        self.cbs = slice(self.crt.first_cb, self.crt.first_cb + self.crt.n_out)   # refined codebooks

    def sync(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def encode(self, wav):
        """[1, 32, T] int64 codes (CPU)."""
        inp = self.fe(raw_audio=wav, sampling_rate=self.sr, return_tensors='pt')
        inp = {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else None) for k, v in inp.items()}
        with self.torch.no_grad():
            return self.mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes.cpu()

    def encode_reference(self, wav):
        """[T_ref, 32] codes of the reference, encoded in 30 s chunks; chunks of <= 1 s are dropped (published)."""
        c = int(CHUNK_S * self.sr)
        codes = self.torch.cat([self.encode(wav[i:i + c]) for i in range(0, len(wav), c)
                                if len(wav[i:i + c]) > self.sr], dim=2)
        return codes[0].numpy().T

    def build_index(self, ref_codes):
        import faiss
        emb = self.cb0[ref_codes[:, 0]].astype(np.float32).copy()
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(self.cb0.shape[1])
        index.add(emb)
        return index

    def speaker_embedding(self, wav):
        w16 = librosa.resample(wav[:self.sr * SPK_S], orig_sr=self.sr, target_sr=16000)
        return self.torch.tensor(self.spk_enc.embed_utterance(w16), dtype=self.torch.float32)

    def retrieve(self, src_codes, index):
        """Greedy path over the k = 8 FAISS candidates of each source frame. Returns path (reference frame per source
        frame), rank (its position among the candidates), cand_idx and cand_sim (T x 8)."""
        import faiss
        q = self.cb0[src_codes[0, 0].numpy()].astype(np.float32).copy()
        faiss.normalize_L2(q)
        cand_sim, cand_idx = index.search(q, K)
        if (cand_idx < 0).any():
            raise ValueError(f'reference has fewer than {K} frames')
        path, rank = np.zeros(len(q), dtype=np.int64), np.zeros(len(q), dtype=np.int64)
        path[0] = cand_idx[0, 0]
        for t in range(1, len(q)):
            cands = cand_idx[t]
            combined = 0.5 * cand_sim[t] + 0.5 / (1 + np.abs(cands - path[t - 1]) / FRAME_RATE)
            rank[t] = np.argmax(combined)
            path[t] = cands[rank[t]]
        return path, rank, cand_idx, cand_sim

    def crt_logits(self, codes, spk):
        """Float32 logits [1, n_cb, T, V] (on the device) of the refined codebooks, for the retrieved codes [1, 32, T]."""
        torch = self.torch
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=self.device == 'cuda'):
                logits = self.crt(codes.to(self.device), spk.unsqueeze(0).to(self.device))
            return logits.float()

    def crt_stats(self, logits, ref_codes, cand_idx):
        """Published argmax / max-softmax (CPU tensors, n_cb x T) plus dump-only statistics (numpy)."""
        torch = self.torch
        with torch.no_grad():
            probs = logits.softmax(-1)
            tokens = probs.argmax(-1)[0].cpu()
            conf = probs.max(-1).values[0].cpu()
            topk = probs[0].topk(TOPK, dim=-1)
            logp = logits[0].log_softmax(-1)
            ent = -(probs[0] * logp).sum(-1)
            cand_tok = torch.from_numpy(np.ascontiguousarray(ref_codes[cand_idx][:, :, self.cbs].transpose(2, 0, 1)))
            cand_logp = logp.gather(-1, cand_tok.to(self.device)).permute(0, 2, 1)   # [n_cb, 8, T]
        extra = {'crt_margin': (topk.values[..., 0] - topk.values[..., 1]).cpu().numpy(), 'crt_entropy': ent.cpu().numpy(),
                 'topk_prob': topk.values.cpu().numpy(), 'topk_idx': topk.indices.cpu().numpy().astype(np.int16),
                 'cand_logp': cand_logp.cpu().numpy()}
        return tokens, conf, extra

    def decode(self, codes):
        with self.torch.no_grad():
            return self.mimi.decode(codes.to(self.device)).audio_values.cpu().squeeze().numpy().astype(np.float32)


def gate(faiss_codes, crt_tokens, crt_conf, tau, cbs):
    """Codes of a CRT system: the refined codebooks `cbs` (CB1-18 for v5.6.5) of the retrieved frames replaced by the
    CRT argmax where max-softmax > tau (tau None: every token); every other codebook passes through.
    Returns (codes [1, 32, T], mask [n_cb, T])."""
    import torch
    sel = faiss_codes[0, cbs].clone()
    mask = torch.ones_like(crt_conf, dtype=torch.bool) if tau is None else crt_conf > tau
    sel[mask] = crt_tokens[mask]
    codes = faiss_codes.clone()
    codes[0, cbs] = sel
    return codes, mask


# ── Pairs, references ────────────────────────────────────────

def load_pairs(args):
    if args.compat_spie:
        rows = [{'pair_id': f'{x["source_speaker"]}_to_{x["target_speaker"]}', 'set': args.set,
                 'source_speaker': x['source_speaker'], 'target_speaker': x['target_speaker'],
                 'source_path': x['source_file'], 'ref_paths': x['target_files']} for x in json.load(open(args.pairs))]
    else:
        rows = [json.loads(line) for line in open(args.pairs) if line.strip()]
        bad = {r['set'] for r in rows} - {args.set}
        if bad:
            raise SystemExit(f'ERROR: {args.pairs} has rows of set {sorted(bad)}, not {args.set!r}')
    ids = [r['pair_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit('ERROR: duplicate pair_id in --pairs')
    if args.pair_ids:
        want = args.pair_ids.split(',')
        missing = set(want) - set(ids)
        if missing:
            raise SystemExit(f'ERROR: pair_ids not in --pairs: {sorted(missing)}')
        rows = [r for r in rows if r['pair_id'] in want]
    if args.max_pairs:
        rows = rows[:args.max_pairs]
    if args.shard:
        si, sn = args.shard
        rows = rows[si::sn]
    return rows


def parse_shard(spec):
    m = re.fullmatch(r'(\d+)/(\d+)', spec)
    if not m or not 0 <= int(m[1]) < int(m[2]):
        raise SystemExit(f'ERROR: --shard {spec!r} is not i/n with 0 <= i < n')
    return int(m[1]), int(m[2])


def get_reference(pipe, row, args, cache):
    """Codes, FAISS index and speaker embedding of a pair's reference, cached per target reference."""
    if args.compat_spie:
        paths = [f for f in row['ref_paths'] if os.path.exists(f)]   # as published
        if not paths:
            raise ValueError('no target file found')
        key = tuple(paths)
    else:
        entry = row['R'][args.budget]
        key = (row['target_speaker'], args.budget, tuple(entry['paths']), tuple(entry['keep_s']))
    tm = {k: 0.0 for k in ('ref_load', 'ref_encode', 'ref_index', 'ref_spk')}
    if key in cache:
        return cache[key], dict(tm, ref_cached=True)
    t = time.time()
    if args.compat_spie:
        wav = np.concatenate([librosa.load(f, sr=pipe.sr, mono=True)[0].astype(np.float32) for f in paths])
    else:
        wav = make_pairs.load_reference(entry, pipe.sr)
    tm['ref_load'] = time.time() - t
    t = time.time()
    codes = pipe.encode_reference(wav)
    tm['ref_encode'] = time.time() - t
    if not args.compat_spie:
        make_pairs.check_reference_len(len(codes) * FRAME, pipe.sr, entry['budget_s'], hop=FRAME)
    t = time.time()
    index = pipe.build_index(codes)
    tm['ref_index'] = time.time() - t
    t = time.time()
    spk = pipe.speaker_embedding(wav)
    tm['ref_spk'] = time.time() - t
    cache[key] = {'codes': codes, 'index': index, 'spk': spk, 'ref_s': len(wav) / pipe.sr}
    return cache[key], dict(tm, ref_cached=False)


# ── Outputs ──────────────────────────────────────────────────

def paths_for(args, pair_id, name):
    return (os.path.join(args.out_dir, args.set, args.budget, name, f'{pair_id}.wav'),
            os.path.join(args.out_dir, 'dumps', args.set, args.budget, f'{pair_id}.npz'))


def base_row(args, row, name, tau):
    wav, dump = paths_for(args, row['pair_id'], name)
    r = {'pair_id': row['pair_id'], 'set': args.set, 'budget': args.budget if args.compat_spie else int(args.budget),
         'system': name, 'tau': tau, 'wav': wav}
    if args.compat_spie:
        r.update(source_wav=row['source_path'], ref_paths=row['ref_paths'])
    else:
        r.update(ref_text=row['source_text'], parallel_wav=row['P'], sim_ref_wavs=row['E']['paths'],
                 source_wav=row['source_path'], R={args.budget: row['R'][args.budget]}, wer_primary=row['wer_primary'],
                 source_is_shared_text=row['source_is_shared_text'], genders=row['genders'])
    r.update(source_speaker=row['source_speaker'], target_speaker=row['target_speaker'], dump=dump)
    return r


def write_wav(path, wav, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path[:-4]}.tmp{os.getpid()}.wav'
    sf.write(tmp, wav, sr, subtype='PCM_16')
    os.replace(tmp, path)


def write_dump(path, arrays, conf, taus):
    """npz dump; masks for the requested τ plus any τ already in an older dump, all from the current conf."""
    if os.path.exists(path):
        with np.load(path) as z:
            taus = set(taus) | {float(k[len('mask_tau'):]) for k in z.files if k.startswith('mask_tau')}
    masks = {f'mask_tau{t:g}': (conf > t).numpy() for t in sorted(taus)}   # the gate's comparison
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path[:-4]}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, **arrays, **masks)
    os.replace(tmp, path)


def compare_saved(pipe, ours, saved, our_codes):
    """Our output vs the saved SPIE wav: sample diffs, and per-codebook agreement of the two files re-encoded with
    this Mimi (roundtrip = our file re-encoded vs the codes it was decoded from, the re-encoding noise floor)."""
    a, sa = sf.read(saved, dtype='float32')
    b, sb = sf.read(ours, dtype='float32')
    assert sa == sb == pipe.sr, (sa, sb)
    n = min(len(a), len(b))
    ca, cb = pipe.encode(a)[0].numpy(), pipe.encode(b)[0].numpy()
    m = min(ca.shape[1], cb.shape[1], our_codes.shape[1])

    def agree(x, y):
        e = (x[:, :m] == y[:, :m]).mean(axis=1)
        return {'cb0': round(float(e[0]), 4), 'cb1_18': round(float(e[1:19].mean()), 4), 'cb19_31': round(float(e[19:].mean()), 4)}
    k = n // FRAME * FRAME
    f_rms = np.sqrt((a[:k].reshape(-1, FRAME) ** 2).mean(1))
    f_diff = np.sqrt(((a[:k] - b[:k]).reshape(-1, FRAME) ** 2).mean(1))
    speech = f_rms >= f_rms.max() * 10 ** (SPEECH_DB / 20)
    rel = f_diff[speech] / f_rms[speech]
    return {'saved': saved, 'bytes_identical': filecmp.cmp(saved, ours, shallow=False), 'n_saved': len(a),
            'n_ours': len(b), 'max_abs_diff': float(np.abs(a[:n] - b[:n]).max()),
            'rms_saved': float(np.sqrt(np.mean(a ** 2))), 'rms_diff': float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2))),
            'corr': float(np.corrcoef(a[:n], b[:n])[0, 1]),
            'frame_diff': {'n_frames': len(f_rms), 'n_speech': int(speech.sum()),
                           'abs_rms_median': round(float(np.median(f_diff)), 5),
                           'abs_rms_max': round(float(f_diff.max()), 5), 'abs_rms_argmax': int(f_diff.argmax()),
                           'speech_rel_median': round(float(np.median(rel)), 4), 'speech_rel_max': round(float(rel.max()), 4),
                           'speech_frames_rel_over_0.2': np.nonzero(speech)[0][rel > 0.2].tolist()},
            'reenc_agree_saved_vs_ours': agree(ca, cb), 'reenc_roundtrip_ours': agree(cb, our_codes)}


def gate_repro(compare_dir, pair_id, changed, conf, tokens, retrieved, tau, first_cb):
    """Saved SPIE faiss vs crt04: first differing frame (Mimi decoding is causal), against our first frame with a
    changed token; plus the CB1-18 token with argmax != retrieved whose confidence is nearest to τ."""
    p = [os.path.join(compare_dir, pair_id, f'{pair_id}_{s}.wav') for s in ('faiss', 'crt04')]
    if not all(map(os.path.exists, p)):
        return {'spie_first_diff_frame': None, 'saved': None}
    a, b = (sf.read(x, dtype='int16')[0] for x in p)
    n = min(len(a), len(b))
    d = np.nonzero(a[:n] != b[:n])[0]
    spie = int(d[0] // FRAME) if len(d) else None
    fr = np.nonzero(changed.any(0).numpy())[0]
    ours = int(fr[0]) if len(fr) else None
    out = {'spie_first_diff_frame': spie, 'our_first_changed_frame': ours,
           'identity_match': (spie is None) == (ours is None), 'first_frame_match': spie == ours}
    c = np.where((tokens != retrieved).numpy(), conf.numpy(), np.nan)
    if not np.isnan(c).all():
        cb, t = np.unravel_index(np.nanargmin(np.abs(c - tau)), c.shape)
        out['nearest_to_tau'] = {'cb': int(cb) + first_cb, 'frame': int(t), 'conf': round(float(c[cb, t]), 4)}
    return out


def convert_pair(pipe, row, args, systems, need, cache):
    """Encode, retrieve and run CRT once, then gate every τ and decode the systems in `need`. Returns rows."""
    torch = pipe.torch
    tm = {}
    t = time.time()
    src_wav = librosa.load(row['source_path'], sr=pipe.sr, mono=True)[0].astype(np.float32)
    tm['load_src'] = time.time() - t
    t = time.time()
    src_codes = pipe.encode(src_wav)
    pipe.sync()
    tm['encode_src'] = time.time() - t
    p_codes = None
    if row.get('P') and any(name == 'mimi_copy_P' for name, _ in need):   # P is read only for its copy
        t = time.time()
        p_wav = librosa.load(row['P'], sr=pipe.sr, mono=True)[0].astype(np.float32)
        tm['load_P'] = time.time() - t
        t = time.time()
        p_codes = pipe.encode(p_wav)
        pipe.sync()
        tm['encode_P'] = time.time() - t
    ref, tm_ref = get_reference(pipe, row, args, cache)
    tm.update(tm_ref)
    t = time.time()
    path, rank, cand_idx, cand_sim = pipe.retrieve(src_codes, ref['index'])
    faiss_codes = torch.from_numpy(np.ascontiguousarray(ref['codes'][path].T))[None]
    assert faiss_codes.dtype == src_codes.dtype == torch.long and faiss_codes.shape == src_codes.shape
    tm['retrieve'] = time.time() - t
    t = time.time()
    logits = pipe.crt_logits(faiss_codes, ref['spk'])
    pipe.sync()
    tm['crt_forward'] = time.time() - t
    t = time.time()
    tokens, conf, extra = pipe.crt_stats(logits, ref['codes'], cand_idx)
    if args.dump_logits:
        extra['crt_logits'] = logits[0].half().cpu().numpy()
    del logits
    tm['crt_stats'] = time.time() - t

    t = time.time()
    retrieved, lo = faiss_codes[0, pipe.cbs], pipe.crt.first_cb
    variants = {}
    for name, tau in systems:
        if name == 'faiss':
            variants[name] = (faiss_codes, None)
        elif name == 'mimi_copy_src':
            variants[name] = (src_codes, None)
        elif name == 'mimi_copy_P':
            if p_codes is not None:
                variants[name] = (p_codes, None)
        else:
            variants[name] = gate(faiss_codes, tokens, conf, tau, pipe.cbs)
    tm['gate'] = time.time() - t
    T = faiss_codes.shape[2]
    _, dump = paths_for(args, row['pair_id'], 'faiss')
    t = time.time()
    arrays = {'src_codes': src_codes[0].numpy().astype(np.int16), 'retrieved_codes': faiss_codes[0].numpy().astype(np.int16),
              'ref_codes': ref['codes'].astype(np.int16), 'path': path.astype(np.int32), 'path_rank': rank.astype(np.int8),
              'path_sim': cand_sim[np.arange(T), rank].astype(np.float32), 'cand_idx': cand_idx.astype(np.int32),
              'cand_sim': cand_sim.astype(np.float32), 'crt_first_cb': np.int8(lo), 'crt_argmax': tokens.numpy().astype(np.int16),
              'crt_conf': conf.numpy().astype(np.float32), **extra, 'n_ref_frames': np.int32(len(ref['codes']))}
    write_dump(dump, arrays, conf, [tau for _, tau in systems if tau is not None])
    tm['write_dump'] = time.time() - t

    rows, decoded, first = [], {}, {}
    for name, tau in systems:
        if name not in variants:   # mimi_copy_P of a pair without P, or already finished
            continue
        codes, mask = variants[name]
        key = codes.numpy().tobytes()
        first.setdefault(key, name)
        if name not in {n for n, _ in need}:
            continue
        r = base_row(args, row, name, tau)
        r.update(n_frames=codes.shape[2], n_ref_frames=len(ref['codes']), ref_s=round(ref['ref_s'], 3),
                 same_codes_as=first[key] if first[key] != name else None)
        if mask is not None:
            changed = mask & (tokens != retrieved)
            r.update(override_coverage=metrics.override_coverage(mask.numpy(), first_cb=lo),
                     change_coverage=metrics.override_coverage(changed.numpy(), first_cb=lo), n_changed=int(changed.sum()),
                     override_counts=mask.sum(1).tolist(), change_counts=changed.sum(1).tolist())
            if args.compat_spie and name in SPIE_SUFFIX:
                r['gate_repro'] = gate_repro(args.compare_dir, row['pair_id'], changed, conf, tokens, retrieved, tau, lo)
        ts = dict(tm)
        if key not in decoded:
            t = time.time()
            decoded[key] = pipe.decode(codes)
            pipe.sync()
            ts['decode'] = time.time() - t
        t = time.time()
        write_wav(r['wav'], decoded[key], pipe.sr)
        ts['write_wav'] = time.time() - t
        if args.compat_spie and name in SPIE_SUFFIX:
            t = time.time()
            saved = os.path.join(args.compare_dir, row['pair_id'], f'{row["pair_id"]}_{SPIE_SUFFIX[name]}.wav')
            r['repro'] = compare_saved(pipe, r['wav'], saved, codes[0].numpy()) if os.path.exists(saved) else {'saved': None}
            ts['compare'] = time.time() - t
        r['conv_time_s'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ts.items()}
        rows.append(r)
    return rows


def load_manifest(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r['pair_id'], r['system'])] = r
    return out


def main():
    args = parse_args()
    systems = parse_systems(args.systems)
    args.out_dir, args.compare_dir = os.path.abspath(args.out_dir), os.path.abspath(args.compare_dir)
    if args.shard:
        args.shard = parse_shard(args.shard)
    if args.compat_spie:
        args.budget = 'full'
    elif args.budget not in ('3', '30', '180'):
        raise SystemExit(f'ERROR: --budget {args.budget} not in 3, 30, 180')
    t_start = time.time()
    import faiss
    import torch
    import transformers
    n_threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    torch.set_num_threads(n_threads)
    faiss.omp_set_num_threads(1)
    torch.backends.cudnn.benchmark = False   # the default, as in the published run
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('ERROR: --device cuda, but CUDA is not available')

    rows = load_pairs(args)
    run_dir = os.path.join(args.out_dir, args.set, args.budget)
    os.makedirs(run_dir, exist_ok=True)
    manifest = os.path.join(run_dir, 'manifest.jsonl')
    if args.shard:
        manifest = os.path.join(run_dir, 'manifest.shard{}of{}.jsonl'.format(*args.shard))

    v = VARIANTS[args.variant]
    args.src_ckpt = args.src_ckpt or v['ckpt']
    args.crt_script = args.crt_script or v['script']
    if not args.ckpt:
        args.ckpt = v['slim'] if os.path.exists(v['slim']) or args.variant == 'v5.6.5' else args.src_ckpt
    tl = {}
    t = time.time()
    slim_created = args.ckpt != args.src_ckpt and ensure_slim(args.ckpt, args.src_ckpt)
    tl['make_slim'] = time.time() - t
    t = time.time()
    pipe = Pipeline(args, device)
    tl['load_models'] = time.time() - t
    tl.update(pipe.crt_load_s)
    weights_sha = pipe.crt_info['weights_sha256']
    cfg = {'weights': os.path.abspath(args.ckpt), **pipe.crt_info,
           'crt_script_sha256': sha256_file(args.crt_script), 'mimi': MIMI_ID,
           'mimi_commit': getattr(pipe.mimi.config, '_commit_hash', None), 'mimi_dtype': str(pipe.dtype),
           'device_type': device, 'k': K, 'spk_s': SPK_S, 'chunk_s': CHUNK_S, 'wav_subtype': 'PCM_16',
           'dump_version': DUMP_VERSION, 'topk': TOPK, 'dump_logits': args.dump_logits,
           'reference': 'spie_target_files' if args.compat_spie else 'make_pairs.load_reference',
           'versions': {'torch': torch.__version__, 'transformers': transformers.__version__, 'faiss': faiss.__version__,
                        'numpy': np.__version__, 'librosa': librosa.__version__}}
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    run_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{socket.gethostname()}-{os.getpid()}'
    src_stat = os.stat(args.src_ckpt) if os.path.exists(args.src_ckpt) else None
    with open(os.path.join(run_dir, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps({'run_id': run_id, 'event': 'start', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'conv_cfg_sha': cfg_sha, 'conv_cfg': cfg, 'args': vars(args), 'torch_threads': n_threads,
                            'src_ckpt': {'path': args.src_ckpt, 'size': src_stat.st_size if src_stat else None,
                                         'mtime': src_stat.st_mtime if src_stat else None},
                            'slim_created': slim_created, 'load_s': {k: round(v, 2) for k, v in tl.items()}}) + '\n')
    print(f'run_conversions {run_id}: {args.set} budget {args.budget}, {len(rows)} pairs, systems '
          f'{[n for n, _ in systems]}, device {device} (Mimi {pipe.dtype}), CRT {args.variant} step {cfg["step"]} '
          f'{cfg["n_params"] / 1e6:.1f}M refines {cfg["refined_cbs"]} ({cfg["crt_mode"]}, returns {cfg["crt_returns"]}), '
          f'weights sha256 {weights_sha[:16]}..., conv_cfg {cfg_sha}, '
          f'load {json.dumps({k: round(x, 2) for k, x in tl.items()})} -> {manifest}', flush=True)

    prev = load_manifest(manifest)
    other = {r.get('conv_cfg_sha') for r in prev.values()} - {cfg_sha}
    if other:
        raise SystemExit(f'ERROR: {manifest} holds rows of conv_cfg {sorted(map(str, other))}, now {cfg_sha}; '
                         f'use another --out_dir')

    def finished(r):
        return r is not None and 'error' not in r and os.path.exists(r['wav']) and os.path.exists(r['dump'])
    todo = [(row, [s for s in systems if not finished(prev.get((row['pair_id'], s[0])))
                   and (s[0] != 'mimi_copy_P' or row.get('P'))]) for row in rows]   # without P: no mimi_copy_P row
    todo = [(row, need) for row, need in todo if need]
    print(f'  {len(todo)} pairs to convert ({len(rows) - len(todo)} already complete)', flush=True)

    cache, n_err, n_short, stage = {}, 0, 0, {}
    with open(manifest, 'a') as out:
        for i, (row, need) in enumerate(todo):
            t0 = time.time()
            try:
                new = convert_pair(pipe, row, args, systems, need, cache)
            except Exception as e:
                msg = f'{type(e).__name__}: {e}'
                short = msg.startswith('ValueError: reference shortfall')
                n_err += not short
                n_short += short
                print(f'  ERROR {row["pair_id"]}: {msg}\n{traceback.format_exc()}', flush=True)
                new = []
                for name, tau in need:
                    r = base_row(args, row, name, tau)
                    if os.path.exists(r['wav']):
                        os.remove(r['wav'])
                    new.append(dict(r, error=msg, shortfall=short))
            for r in new:
                r.update(run_id=run_id, conv_cfg_sha=cfg_sha)
                out.write(json.dumps(r) + '\n')
            out.flush()
            if new and 'error' not in new[0]:
                ts = new[0]['conv_time_s']
                for k, v in ts.items():
                    if k != 'ref_cached' and not (k.startswith('ref_') and ts['ref_cached']):
                        stage.setdefault(k, []).append(v)
                for r in new[1:]:
                    for k in ('decode', 'write_wav', 'compare'):
                        if k in r['conv_time_s']:
                            stage.setdefault(k, []).append(r['conv_time_s'][k])
                cov = ' '.join(f'{r["system"]}:open={r["override_coverage"]["overall"]:.3f}/chg={r["n_changed"]}'
                               + (f'(={r["same_codes_as"]})' if r['same_codes_as'] else '')
                               for r in new if 'override_coverage' in r)
                print(f'  [{i + 1}/{len(todo)}] {row["pair_id"]} T={new[0]["n_frames"]} ref={new[0]["n_ref_frames"]}fr'
                      f'{" (cached)" if ts["ref_cached"] else ""} {cov} | '
                      + ' '.join(f'{k}={v:.2f}' for k, v in ts.items() if isinstance(v, float) and v > 0)
                      + f' | {time.time() - t0:.1f}s', flush=True)
                for r in new:
                    if 'repro' in r:
                        print(f'    repro {r["system"]}: {json.dumps(r["repro"])}', flush=True)
                    if args.compat_spie and r['system'] == 'crt_tau0.4':
                        print(f'    tau0.4 per-CB gate-open {r["override_counts"]} changed {r["change_counts"]} '
                              f'(of {r["n_frames"]} frames) gate_repro {json.dumps(r["gate_repro"])}', flush=True)

    kept = load_manifest(manifest)
    with open(manifest + '.tmp', 'w') as f:
        for r in kept.values():
            f.write(json.dumps(r) + '\n')
    os.replace(manifest + '.tmp', manifest)
    summary = {k: {'n': len(v), 'mean_s': round(float(np.mean(v)), 4), 'max_s': round(float(np.max(v)), 4)}
               for k, v in stage.items()}
    gr = [r['gate_repro'] for r in kept.values() if 'identity_match' in r.get('gate_repro', {})]
    repro = {'n_pairs': len(gr), 'identity_match': sum(g['identity_match'] for g in gr),
             'first_frame_match': sum(g['first_frame_match'] for g in gr)} if gr else None
    with open(os.path.join(run_dir, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps({'run_id': run_id, 'event': 'end', 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'n_pairs_done': len(todo) - n_err - n_short, 'n_errors': n_err, 'n_shortfall': n_short,
                            'stage_s': summary, 'gate_repro': repro, 'total_s': round(time.time() - t_start, 1)}) + '\n')
    if repro:
        print(f'  SPIE gate reproduction over {repro["n_pairs"]} pairs (manifest): identical-vs-differing status '
              f'{repro["identity_match"]}/{repro["n_pairs"]}, first changed frame {repro["first_frame_match"]}/{repro["n_pairs"]}')
    print('  per-stage seconds (ref_* per target reference, decode/write_wav per output, the rest per pair):')
    for k, v in summary.items():
        print(f'    {k:<12} n={v["n"]:<4} mean={v["mean_s"]:.3f} max={v["max_s"]:.3f}')
    print(f'done in {time.time() - t_start:.0f}s: {len(todo) - n_err - n_short} pairs converted, {n_err} errors, '
          f'{n_short} reference shortfalls -> {manifest} ({len(kept)} rows)', flush=True)
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
