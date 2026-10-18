#!/usr/bin/env python3
"""
Objective metrics for the CRT journal extension, ANALYSIS_PLAN.md §3 (M1-M7). Every signal is resampled to 16 kHz.

  M1 mcd_db      dB-MCD: pyworld (harvest, cheaptrick, 5 ms) -> pysptk.sp2mc order 24, alpha = mcepalpha(16000),
                 c0 excluded, DTW on c1..c24, per-frame 10*sqrt(2)/ln10 * ||c_ref - c_syn||, mean over the path.
                 Silence rule (sprocket/s3prl-vc `extfrm`): a frame's power is the power of its WORLD spectral
                 envelope, normalised by the utterance's mean frame power (npow, dB). Frames with npow <= `npow_db`
                 (-20) are removed from EACH utterance BEFORE DTW; M1 and M2 use only the remaining frames
                 (mcd_frames = DTW path length).
  M2 lf0_rmse    speaker-normalised log-F0 RMSE (natural-log units): each signal's log-F0 minus the mean log-F0 of
                 its own voiced non-silent frames (per utterance, mean only), RMSE over jointly voiced path pairs.
     vuv_err     fraction of path pairs (non-silent frames) whose voicing decisions differ.
     f0_pcc      Pearson correlation of log-F0 over the jointly voiced path pairs (same DTW path as M1).
     f0_pcc_src  same, output vs source (own DTW path on the cross-speaker mel-cepstra), only with `source_wav`.
  M3 wer_<m>_*   Whisper <m>, EnglishTextNormalizer on reference and hypothesis, jiwer edit counts. Rows store hyp,
                 S/D/I and N (reference length) for words and characters (spaces count as characters) and
                 len_ratio = hypothesis words / reference words (flags repetition loops). Decoding: WHISPER_DECODE,
                 batched (--asr_batch). corpus_wer() sums the counts and raises if any row lacks the key
                 (primary_only=True keeps wer_primary rows: plan M3, VCTK primary WER); paired() aligns two systems
                 on the same pair_ids. A requested checkpoint that is not fully cached is an error; there is no
                 fallback, so wer_large_v3 always comes from large-v3. large-v3 is GPU-only in practice (measured
                 177-262 s per output on the login-node CPU).
  M4 sim_wavlm   WavLMForXVector (microsoft/wavlm-base-plus-sv, local safetensors copy): cosine between the output's
                 x-vector and the mean of the L2-normalised x-vectors of the reference wavs (E).
     secs_resemblyzer  Resemblyzer cosine to the mean embedding of E (secondary; CRT was trained against it).
                 Rows also store emb_<enc> (the output's embedding) and ref_emb_<enc> (the E centroid) as base64
                 float32 (unpack with emb()), so M5 needs no re-embedding.
  M5 eer()       equal error rate from genuine / impostor score lists. sv_trials(rows) builds them for one system
                 of one set: every output against the E centroid of every target in `rows`; genuine = its own
                 target, impostor = each other target (all genders).
  M6 utmos       UTMOS22-strong (torch.hub tarepan/SpeechMOS:v1.2.0).
  M7 override_coverage()  fraction of overridden tokens per codebook from a boolean mask.

CLI: score rows of a JSONL (pair_id, system, wav, ref_text, parallel_wav|null, sim_ref_wavs[, source_wav,
copy_control]), or make_pairs.py rows plus system and wav (P -> parallel_wav, E.paths -> sim_ref_wavs,
source_path -> source_wav, source_text -> ref_text; wer_primary, set and target_speaker are carried through, R is
checked and then dropped). Every input row is checked first (ValueError): wav, parallel_wav and source_wav are not in
sim_ref_wavs (E) or R, R, P and E are mutually disjoint, no R recording has the source's text, and
parallel_wav == wav only with copy_control. One JSONL row per input row (input fields, outputs, per-metric seconds
in time_s, cfg). Resume: a row in --out_jsonl is reused when its input fields match; only the requested keys it
lacks are computed. A different cfg (f0_method, npow_db, WavLM model, Whisper decoding) is an error. A failing row
gets an `error` field and is retried next run; the file is compacted to the last row per (pair_id, system) at the
end. Read results with load_results(). --shard i/n scores every n-th row into its own file.

Protocol choices not in ANALYSIS_PLAN v0 (to record in v1): the npow_db = -20 silence rule above; per-utterance
mean-only log-F0 normalisation; V/UV error over the non-silent path pairs; harvest (not dio) F0; CER counts spaces;
WHISPER_DECODE (greedy, no temperature fallback); f0_pcc_src on the cross-speaker DTW path; the sv_trials() trial
list; large-v3 WER is scored on the GPU node only (if it stays unavailable: a dated deviation, never Whisper-base).
"""
import argparse
import base64
import json
import os
import re
import time

import librosa
import numpy as np

SR = 16000
FRAME_MS = 5.0
MCEP_ORDER = 24
MCD_CONST = 10.0 * np.sqrt(2.0) / np.log(10.0)
ALL_METRICS = ('mcd', 'wer', 'sim', 'secs', 'utmos')
WHISPER_DECODE = ('greedy, temperature 0 without fallback, language=en, without_timestamps, whisper default token '
                  'suppression, pad_or_trim to 30 s, batched; inputs > 30 s: whisper.transcribe(temperature=0, '
                  'condition_on_previous_text=False) one at a time')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--in_jsonl', required=True)
    p.add_argument('--out_jsonl', required=True)
    p.add_argument('--metrics', default=','.join(ALL_METRICS),
                   help='comma list of mcd (M1+M2), wer (M3), sim (M4 WavLM), secs (M4 Resemblyzer), utmos (M6)')
    p.add_argument('--whisper_models', default='large-v3,base')
    p.add_argument('--asr_batch', type=int, default=16, help='Whisper batch size (also the write granularity)')
    p.add_argument('--wavlm_model', default='/datastore/nvquoc/crt/mimi_vc/benchmark/journal/models/wavlm-base-plus-sv',
                   help='microsoft/wavlm-base-plus-sv re-saved as safetensors (transformers 5 refuses .bin with torch < 2.6)')
    p.add_argument('--f0_method', default='harvest', choices=['harvest', 'dio'])
    p.add_argument('--npow_db', type=float, default=-20.0,
                   help='M1/M2 silence rule: frames with power <= this many dB re the utterance mean are removed')
    p.add_argument('--shard', default='', help='i/n: score input rows k with k %% n == i into <out_jsonl>.shard<i>of<n>')
    p.add_argument('--device', default='auto')
    p.add_argument('--max_rows', type=int, default=0)
    return p.parse_args()


def load_wav(path):
    w, _ = librosa.load(path, sr=SR, mono=True)
    return w.astype(np.float32)


# ── M1 / M2: WORLD features, DTW-MCD, F0 ────────────────────

def world_features(wav, f0_method='harvest'):
    """Returns dict with f0 (Hz, 0 = unvoiced), mcep c1..c24 (T x 24) and npow (frame power of the WORLD spectral
    envelope in dB re the utterance's mean frame power, as sprocket spc2npow), on a 5 ms grid."""
    import pysptk
    import pyworld as pw
    x = np.ascontiguousarray(wav, dtype=np.float64)
    if f0_method == 'harvest':
        f0, t = pw.harvest(x, SR, frame_period=FRAME_MS)
    else:
        f0, t = pw.dio(x, SR, frame_period=FRAME_MS)
        f0 = pw.stonemask(x, f0, t, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    mc = pysptk.sp2mc(sp, order=MCEP_ORDER, alpha=pysptk.util.mcepalpha(SR))[:, 1:]
    power = (sp[:, 0] + sp[:, -1] + 2.0 * sp[:, 1:-1].sum(axis=1)) / (2 * (sp.shape[1] - 1))
    return {'f0': f0, 'mcep': mc, 'npow': 10.0 * np.log10(power / power.mean())}


def dtw_path(a, b):
    """DTW on mel-cepstra (T x D each), euclidean, steps (1,1),(1,0),(0,1). Returns (i, j) index arrays, start->end."""
    _, wp = librosa.sequence.dtw(X=a.T, Y=b.T, metric='euclidean')
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1]


def _norm_lf0(f0):
    v = f0 > 0
    lf0 = np.zeros_like(f0)
    lf0[v] = np.log(f0[v])
    if v.any():
        lf0[v] -= lf0[v].mean()
    return lf0


def _pcc(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _align(ref_feat, syn_feat, npow_db):
    """Removes each utterance's silent frames (npow <= npow_db), adds per-utterance normalised log-F0 (lf0n), runs
    DTW and returns the two feature dicts indexed along the path, or (None, None) if either has no frame left."""
    a = {k: v[ref_feat['npow'] > npow_db] for k, v in ref_feat.items()}
    b = {k: v[syn_feat['npow'] > npow_db] for k, v in syn_feat.items()}
    if not len(a['f0']) or not len(b['f0']):
        return None, None
    a['lf0n'], b['lf0n'] = _norm_lf0(a['f0']), _norm_lf0(b['f0'])
    i, j = dtw_path(a['mcep'], b['mcep'])
    return {k: v[i] for k, v in a.items()}, {k: v[j] for k, v in b.items()}


def mcd_f0(ref_feat, syn_feat, npow_db=-20.0):
    """M1 + M2 of `syn` against its parallel `ref` (features from world_features), silent frames removed before DTW."""
    r, s = _align(ref_feat, syn_feat, npow_db)
    if r is None:
        return {'mcd_db': float('nan'), 'mcd_frames': 0}
    d = MCD_CONST * np.sqrt(np.sum((r['mcep'] - s['mcep']) ** 2, axis=1))
    vr, vs = r['f0'] > 0, s['f0'] > 0
    both = vr & vs
    return {'mcd_db': float(d.mean()), 'mcd_frames': int(len(d)),
            'lf0_rmse': float(np.sqrt(np.mean((r['lf0n'][both] - s['lf0n'][both]) ** 2))) if both.any() else float('nan'),
            'vuv_err': float(np.mean(vr != vs)),
            'f0_pcc': _pcc(np.log(r['f0'][both]), np.log(s['f0'][both])), 'f0_voiced_frames': int(both.sum())}


def f0_pcc(ref_feat, syn_feat, npow_db=-20.0):
    """log-F0 Pearson correlation after the same silence removal and DTW (used for output vs source)."""
    r, s = _align(ref_feat, syn_feat, npow_db)
    if r is None:
        return float('nan')
    both = (r['f0'] > 0) & (s['f0'] > 0)
    return _pcc(np.log(r['f0'][both]), np.log(s['f0'][both]))


# ── M3: WER / CER ────────────────────────────────────────────

def edit_counts(ref_text, hyp_text, normalizer):
    import jiwer
    r, h = normalizer(ref_text).strip(), normalizer(hyp_text).strip()
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    return {'ref_norm': r, 'hyp_norm': h, 'len_ratio': len(h.split()) / len(r.split()) if r else float('nan'),
            'S': w.substitutions, 'D': w.deletions, 'I': w.insertions, 'N': w.substitutions + w.deletions + w.hits,
            'cS': c.substitutions, 'cD': c.deletions, 'cI': c.insertions, 'cN': c.substitutions + c.deletions + c.hits}


def load_results(paths):
    """Rows of one or more out_jsonl files (e.g. shards), keeping the last row per (pair_id, system)."""
    out = {}
    for path in [paths] if isinstance(paths, str) else paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[(r['pair_id'], r['system'])] = r
    return list(out.values())


def _missing(v):
    return v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v))


def _require(rows, keys):
    bad = [(r.get('pair_id'), r.get('system')) for r in rows
           if 'error' in r or any(_missing(r.get(k)) for k in keys)]
    if bad:
        raise ValueError(f'{len(bad)} of {len(rows)} rows have an error or lack a finite {list(keys)}, e.g. {bad[:3]}')


def paired(rows, sys_a, sys_b, keys=()):
    """Rows of two systems aligned by pair_id. Raises unless both systems cover the same pair_ids once each and every
    row has all `keys` (so paired deltas and corpus WERs are always over the same conversions)."""
    by = {}
    for s in (sys_a, sys_b):
        rs = [r for r in rows if r['system'] == s]
        by[s] = {r['pair_id']: r for r in rs}
        if len(by[s]) != len(rs):
            raise ValueError(f'{s}: duplicate pair_ids (read rows with load_results)')
        _require(rs, keys)
    a, b = by[sys_a], by[sys_b]
    if set(a) != set(b):
        raise ValueError(f'{sys_a} ({len(a)}) and {sys_b} ({len(b)}) differ on {len(set(a) ^ set(b))} pair_ids')
    ids = sorted(a)
    return [a[i] for i in ids], [b[i] for i in ids]


def corpus_wer(rows, key='wer_large_v3', primary_only=False):
    """Corpus WER and CER (%) from per-row edit counts stored under `key` (e.g. rows[k]['wer_base']).
    Raises if any row has an error or lacks `key`; compare systems on rows from paired(). primary_only=True keeps
    the rows with wer_primary (plan M3: VCTK primary WER and the Dev WER of the τ rule) and raises if a row lacks it."""
    if primary_only:
        _require(rows, ['wer_primary'])
        rows = [r for r in rows if r['wer_primary']]
    _require(rows, [key])
    e = [r[key] for r in rows]
    n, cn = sum(x['N'] for x in e), sum(x['cN'] for x in e)
    return {'n_utts': len(e),
            'wer': 100.0 * sum(x['S'] + x['D'] + x['I'] for x in e) / n if n else float('nan'),
            'cer': 100.0 * sum(x['cS'] + x['cD'] + x['cI'] for x in e) / cn if cn else float('nan')}


def whisper_root():
    return os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'whisper')


def check_whisper_cached(name):
    """Exits unless the Whisper checkpoint `name` is fully cached (SHA256 matches); there is no fallback model."""
    import hashlib
    import whisper
    url = whisper._MODELS[name]
    path = os.path.join(whisper_root(), os.path.basename(url))
    if os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 24), b''):
                h.update(chunk)
        if h.hexdigest() == url.split('/')[-2]:
            return
    raise SystemExit(f'ERROR: Whisper {name} not (fully) cached in {whisper_root()}')


class WhisperASR:
    def __init__(self, name, device):
        import torch
        import whisper
        from whisper.normalizers import EnglishTextNormalizer
        print(f'  Loading Whisper ({name})...', flush=True)
        self.torch, self.whisper, self.name, self.device = torch, whisper, name, device
        self.model = whisper.load_model(name, device=device, download_root=whisper_root())
        self.key = 'wer_' + name.replace('-', '_')
        self.normalizer = EnglishTextNormalizer()

    def transcribe(self, wavs):
        """WHISPER_DECODE: inputs <= 30 s are decoded as one batch; longer ones one at a time."""
        w = self.whisper
        fp16 = self.device == 'cuda'
        out = [None] * len(wavs)
        short = [k for k, x in enumerate(wavs) if len(x) <= 30 * SR]
        if short:
            mel = self.torch.stack([w.log_mel_spectrogram(w.pad_or_trim(wavs[k]), n_mels=self.model.dims.n_mels)
                                    for k in short]).to(self.model.device)
            res = w.decode(self.model, mel, w.DecodingOptions(language='en', without_timestamps=True, fp16=fp16))
            for k, r in zip(short, res):
                out[k] = r.text.strip()
        for k in range(len(wavs)):
            if out[k] is None:
                out[k] = w.transcribe(self.model, wavs[k], language='en', temperature=0.0, fp16=fp16,
                                      condition_on_previous_text=False)['text'].strip()
        return out

    def score(self, wavs, ref_texts):
        return [dict(hyp=h, **edit_counts(t, h, self.normalizer)) for h, t in zip(self.transcribe(wavs), ref_texts)]


# ── M4: speaker similarity ───────────────────────────────────

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pack(v):
    return base64.b64encode(np.asarray(v, '<f4').tobytes()).decode('ascii')


def emb(row, key):
    """A stored embedding (emb_wavlm, ref_emb_wavlm, emb_resemblyzer, ref_emb_resemblyzer) as a float32 vector."""
    return np.frombuffer(base64.b64decode(row[key]), '<f4')


class WavLMSV:
    def __init__(self, name, device):
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector
        print(f'  Loading {name}...', flush=True)
        self.torch, self.device = torch, device
        self.fe = AutoFeatureExtractor.from_pretrained(name)
        self.model = WavLMForXVector.from_pretrained(name).to(device).eval()

    def embed(self, wav16):
        inp = self.fe(wav16, sampling_rate=SR, return_tensors='pt')
        with self.torch.no_grad():
            e = self.model(**{k: v.to(self.device) for k, v in inp.items()}).embeddings[0]
        e = self.torch.nn.functional.normalize(e, dim=-1)
        return e.cpu().numpy()


class ResemblyzerSV:
    def __init__(self, device):
        from resemblyzer import VoiceEncoder, preprocess_wav
        print('  Loading Resemblyzer VoiceEncoder...', flush=True)
        self.pre = preprocess_wav
        self.enc = VoiceEncoder(device=device)

    def embed(self, wav16):
        return self.enc.embed_utterance(self.pre(wav16, source_sr=SR))


# ── M5: EER ──────────────────────────────────────────────────

def eer(genuine, impostor):
    """Equal error rate (fraction) with the threshold, linearly interpolated where FNR crosses FPR."""
    g, im = np.sort(np.asarray(genuine, float)), np.sort(np.asarray(impostor, float))
    th = np.concatenate([[-np.inf], np.unique(np.concatenate([g, im])), [np.inf]])
    fnr = np.searchsorted(g, th, side='left') / len(g)             # genuine rejected: score < th
    fpr = 1.0 - np.searchsorted(im, th, side='left') / len(im)     # impostor accepted: score >= th
    d = fnr - fpr                                                  # non-decreasing in th
    k = int(np.argmax(d >= 0))
    if k == 0 or d[k] == 0:
        return {'eer': float((fnr[k] + fpr[k]) / 2), 'threshold': float(th[k])}
    a = d[k - 1] / (d[k - 1] - d[k])                               # d[k-1] < 0 < d[k]
    e0, e1 = fnr[k - 1] + a * (fnr[k] - fnr[k - 1]), fpr[k - 1] + a * (fpr[k] - fpr[k - 1])
    return {'eer': float((e0 + e1) / 2), 'threshold': float(th[k])}


def sv_trials(rows, enc='wavlm'):
    """Genuine / impostor cosine lists for eer(), from the stored embeddings of ONE system on ONE set. Each output is
    scored against the E centroid of every target present in `rows`: its own target is a genuine trial, every other
    target an impostor trial. Target identity is target_speaker, else the E path list."""
    if len({r['system'] for r in rows}) != 1 or len({r.get('set') for r in rows}) != 1:
        raise ValueError('sv_trials takes the rows of one system on one set')
    _require(rows, [f'emb_{enc}', f'ref_emb_{enc}'])

    def tid(r):
        return r.get('target_speaker') or tuple(r['sim_ref_wavs'])
    cents = {}
    for r in rows:
        c = emb(r, f'ref_emb_{enc}')
        if tid(r) in cents and not np.allclose(cents[tid(r)], c, atol=1e-6):
            raise ValueError(f'target {tid(r)} has two different E centroids')
        cents[tid(r)] = c
    if len(cents) < 2:
        raise ValueError('sv_trials needs at least two targets')
    gen, imp = [], []
    for r in rows:
        e = emb(r, f'emb_{enc}')
        for t, c in cents.items():
            (gen if t == tid(r) else imp).append(cosine(e, c))
    return gen, imp


# ── M6: UTMOS ────────────────────────────────────────────────

class UTMOS:
    def __init__(self, device):
        import torch
        print('  Loading UTMOS22-strong...', flush=True)
        self.torch, self.device = torch, device
        self.model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True).to(device).eval()

    def score(self, wav16):
        t = self.torch.from_numpy(wav16.astype(np.float32)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return float(self.model(t, SR).item())


# ── M7: override coverage ────────────────────────────────────

def override_coverage(mask, first_cb=1):
    """mask: bool array (n_cb, T), True where CRT overrode the retrieved token; row 0 is codebook `first_cb`.
    Pass a list of masks to pool frames over conversions."""
    masks = mask if isinstance(mask, (list, tuple)) else [mask]
    m = np.concatenate([np.asarray(x, bool) for x in masks], axis=1)
    per_cb = m.mean(axis=1)
    return {'per_cb': {f'cb{first_cb + c}': float(v) for c, v in enumerate(per_cb)},
            'overall': float(m.mean()), 'n_frames': int(m.shape[1])}


# ── CLI ──────────────────────────────────────────────────────

def as_metric_row(r):
    """Native rows pass through. make_pairs.py rows (source_path, source_text, P, E) plus system/wav are mapped to
    ref_text, parallel_wav, sim_ref_wavs, source_wav (explicit native fields win); R is dropped (pass it to
    check_row first)."""
    row = {k: v for k, v in r.items() if k != 'R'}
    if 'source_path' in r:
        row.setdefault('ref_text', r['source_text'])
        row.setdefault('parallel_wav', r['P'])
        row.setdefault('sim_ref_wavs', r['E']['paths'])
        row.setdefault('source_wav', r['source_path'])
    return row


_TEXT = {}


def _norm_text(s):
    return ' '.join(re.sub(r'[^a-z0-9\s]', ' ', s.lower()).split())


def text_of(path):
    """Normalised transcript of a VCTK (txt/pXXX/pXXX_NNN.txt) or LibriTTS-R (<stem>.normalized.txt) recording."""
    if path not in _TEXT:
        if '/wav48_silence_trimmed/' in path:
            spk, utt = os.path.basename(path).split('_')[:2]
            t = os.path.join(path.split('/wav48_silence_trimmed/')[0], 'txt', spk, f'{spk}_{utt}.txt')
        else:
            t = os.path.splitext(path)[0] + '.normalized.txt'
        if not os.path.isfile(t):
            raise ValueError(f'no transcript for {path} (looked for {t})')
        _TEXT[path] = _norm_text(open(t).read())
    return _TEXT[path]


def check_row(r, R=None):
    """Plan §1 reference rule at scoring time (ValueError). wav, parallel_wav and source_wav are never in
    sim_ref_wavs (E); parallel_wav != wav unless the row is marked copy_control (natural target control). With the
    make_pairs.py reference pool R ({budget: {'paths': [...]}}): R, P and E are mutually disjoint, wav and
    source_wav are not in R, and no R recording has the source's text."""
    def fail(msg):
        raise ValueError(f'{r["pair_id"]} {r["system"]}: {msg}')
    n = os.path.normpath
    e = {n(x) for x in r.get('sim_ref_wavs') or []}
    for k in ('wav', 'parallel_wav', 'source_wav'):
        if r.get(k) and n(r[k]) in e:
            fail(f'{k} is in sim_ref_wavs')
    if r.get('parallel_wav') and not r.get('copy_control') and n(r['parallel_wav']) == n(r['wav']):
        fail('parallel_wav == wav without copy_control')
    if R:
        rp = {n(x) for b in R.values() for x in b['paths']}
        if rp & e:
            fail(f'R and E share {sorted(rp & e)[:2]}')
        for k in ('parallel_wav', 'wav', 'source_wav'):
            if r.get(k) and n(r[k]) in rp:
                fail(f'{k} is in R')
        src = _norm_text(r['ref_text'])
        same = [x for x in rp if text_of(x) == src]
        if same:
            fail(f'R has a recording of the source sentence: {same[:2]}')


def expected_keys(r, metrics, asr_keys):
    """Output keys a finished row must have for the requested metrics."""
    k = []
    if 'mcd' in metrics:
        k += ['mcd_db'] if r.get('parallel_wav') else []
        k += ['f0_pcc_src'] if r.get('source_wav') else []
    k += asr_keys if 'wer' in metrics else []
    k += [key for m, key in (('sim', 'sim_wavlm'), ('secs', 'secs_resemblyzer'), ('utmos', 'utmos')) if m in metrics]
    return k


def main():
    args = parse_args()
    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    metrics = set(args.metrics.split(','))
    if not metrics <= set(ALL_METRICS):
        raise SystemExit(f'ERROR: unknown metrics {metrics - set(ALL_METRICS)}')
    rows = []
    for line in open(args.in_jsonl):
        if line.strip():
            raw = json.loads(line)
            rows.append(as_metric_row(raw))
            check_row(rows[-1], raw.get('R'))
    if args.max_rows:
        rows = rows[:args.max_rows]
    out_path = args.out_jsonl
    if args.shard:
        si, sn = map(int, args.shard.split('/'))
        rows = rows[si::sn]
        root, ext = os.path.splitext(args.out_jsonl)
        out_path = f'{root}.shard{si}of{sn}{ext}'
    keys = [(r['pair_id'], r['system']) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit('ERROR: duplicate (pair_id, system) in input')

    names = list(dict.fromkeys(args.whisper_models.split(','))) if 'wer' in metrics else []
    for m in names:
        check_whisper_cached(m)
    if 'large-v3' in names and device == 'cpu':
        print('  WARN: Whisper large-v3 on CPU took 177-262 s per output on the login node; score it on the GPU node',
              flush=True)
    asr_keys = ['wer_' + m.replace('-', '_') for m in names]
    cfg = {'f0_method': args.f0_method, 'npow_db': args.npow_db, 'wavlm_model': args.wavlm_model,
           'whisper_decode': WHISPER_DECODE}

    # Resume: a row is reused only if its inputs and cfg match; it is finished when it has every requested key.
    prev = {(r['pair_id'], r['system']): r for r in load_results(out_path)} if os.path.exists(out_path) else {}
    for p in prev.values():
        if p.get('cfg', cfg) != cfg:
            raise SystemExit(f'ERROR: {out_path} was written with cfg {p["cfg"]}, now {cfg}; use another --out_jsonl')
    todo, n_done = [], 0
    for r in rows:
        p = prev.get((r['pair_id'], r['system']))
        same = p is not None and all(p.get(k) == v for k, v in r.items())
        base = {k: v for k, v in p.items() if k != 'error'} if same else r
        missing = [k for k in expected_keys(r, metrics, asr_keys) if k not in base]
        if missing:
            todo.append((r, base, set(missing)))
        else:
            n_done += 1
    print(f'metrics.py: {len(todo)} rows to score ({n_done} already complete), metrics={sorted(metrics)}, '
          f'whisper={names}, device={device} -> {out_path}', flush=True)

    t0 = time.time()
    need = set().union(*(m for _, _, m in todo)) if todo else set()
    asrs = [WhisperASR(m, device) for m, k in zip(names, asr_keys) if k in need]
    sv = WavLMSV(args.wavlm_model, device) if 'sim_wavlm' in need else None
    rz = ResemblyzerSV(device) if 'secs_resemblyzer' in need else None
    mos = UTMOS(device) if 'utmos' in need else None
    print(f'  models ready in {time.time() - t0:.0f}s', flush=True)

    feat_cache, ref_cache = {}, {}

    def feats(path):
        if path not in feat_cache:
            feat_cache[path] = world_features(load_wav(path), args.f0_method)
        return feat_cache[path]

    def ref_emb(name, enc, paths):
        k = (name, tuple(paths))
        if k not in ref_cache:
            es = [enc.embed(load_wav(p)) for p in paths]
            ref_cache[k] = np.mean([e / np.linalg.norm(e) for e in es], axis=0)
        return ref_cache[k]

    def score_row(r, base, miss, wav, asr_res):
        row, tm = dict(base), dict(base.get('time_s', {}))
        row['dur_s'] = len(wav) / SR
        if miss & {'mcd_db', 'f0_pcc_src'}:
            t = time.time()
            syn = world_features(wav, args.f0_method)
            if r.get('parallel_wav'):
                row.update(mcd_f0(feats(r['parallel_wav']), syn, args.npow_db))
            if r.get('source_wav'):
                row['f0_pcc_src'] = f0_pcc(feats(r['source_wav']), syn, args.npow_db)
            tm['mcd_f0'] = time.time() - t
        for key, (res, sec) in asr_res.items():
            row[key], tm[key] = res, sec
        for key, name, enc in (('sim_wavlm', 'wavlm', sv), ('secs_resemblyzer', 'resemblyzer', rz)):
            if key in miss:
                t = time.time()
                e, c = enc.embed(wav), ref_emb(name, enc, r['sim_ref_wavs'])
                row[key], row[f'emb_{name}'], row[f'ref_emb_{name}'] = cosine(e, c), pack(e), pack(c)
                tm[key] = time.time() - t
        if 'utmos' in miss:
            t = time.time()
            row['utmos'] = mos.score(wav)
            tm['utmos'] = time.time() - t
        row['time_s'], row['cfg'] = tm, cfg
        return row

    def err(e):
        return f'{type(e).__name__}: {e}'

    out = open(out_path, 'a')
    n_err = 0
    for c0 in range(0, len(todo), args.asr_batch):
        chunk = todo[c0:c0 + args.asr_batch]
        wavs, errs, asr_res = {}, {}, {}
        for k, (r, _, _) in enumerate(chunk):
            try:
                wavs[k] = load_wav(r['wav'])
            except Exception as e:
                errs[k] = err(e)
        for asr in asrs:   # one batched decode per model per chunk; the time is split evenly over its rows
            ks = [k for k in wavs if asr.key in chunk[k][2]]
            if not ks:
                continue
            t = time.time()
            try:
                res = asr.score([wavs[k] for k in ks], [chunk[k][0]['ref_text'] for k in ks])
            except Exception as e:
                errs.update({k: err(e) for k in ks})
                continue
            for k, x in zip(ks, res):
                asr_res.setdefault(k, {})[asr.key] = (x, (time.time() - t) / len(ks))
        for k, (r, base, miss) in enumerate(chunk):
            if k in errs:
                row = dict(base, error=errs[k])
            else:
                try:
                    row = score_row(r, base, miss, wavs[k], asr_res.get(k, {}))
                except Exception as e:
                    row = dict(base, error=err(e))
            if 'error' in row:
                n_err += 1
                print(f'  ERROR {r["pair_id"]} {r["system"]}: {row["error"]}', flush=True)
            out.write(json.dumps(row) + '\n')
            out.flush()
            print(f'  [{c0 + k + 1}/{len(todo)}] {r["pair_id"]} {r["system"]} ({time.time() - t0:.0f}s)', flush=True)
    out.close()
    # Compact: keep the last row per (pair_id, system), so retried error rows do not stay next to their success.
    kept = load_results(out_path)
    with open(out_path + '.tmp', 'w') as f:
        for row in kept:
            f.write(json.dumps(row) + '\n')
    os.replace(out_path + '.tmp', out_path)
    print(f'done in {time.time() - t0:.0f}s, {n_err} row errors -> {out_path} ({len(kept)} rows)', flush=True)


if __name__ == '__main__':
    main()
