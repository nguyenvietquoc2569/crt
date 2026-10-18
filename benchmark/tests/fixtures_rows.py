"""Synthetic inputs shared by the tests of lab_rows, crt_anchor_lab, run_conversions and check_d0. Nothing here reads
the real pair lists, the results/ folder or any audio file, and no model is loaded:
  - make_pairs.py pair rows with the fields of pairs/{dev,vctk2k,ltc2k}.jsonl, and three tiny lists: Dev-like (every
    source speaker is a target), VCTK-2k-like (no source speaker is a target) and LTC-2k-like (one target short of
    180 s of reference);
  - run_conversions.py manifest rows (the base_row fields, then the fields convert_pair and main add) and the rows
    crt_decode_lab.py main writes (lab_<system>);
  - FT, a numpy array with the torch.Tensor methods the harness calls, a fake torch / faiss / transformers / librosa,
    and FakePipe, a deterministic stand-in for run_conversions.Pipeline (codes, retrieval, CRT logits and audio are
    functions of their inputs, so equal inputs give equal outputs, as on the GPU).
"""
import json
import os
import re
import sys
import types
import zlib

import numpy as np

ROOT = '/data/vctk/wav48_silence_trimmed'      # the VCTK layout of the real rows
FRAME, SR = 1920, 24000                        # Mimi samples per frame, sample rate
V = 32                                         # vocabulary of the fake CRT (> run_conversions.TOPK = 16)
HOP = 240                                      # samples per frame of the fake decoder (short WAVs)


# ── pair rows (make_pairs.py) ────────────────────────────────

def wav(spk, n):
    """Recording n of speaker spk, VCTK style."""
    return f'{ROOT}/{spk}/{spk}_{n:03d}_mic1.flac'


def gender(spk):
    return 'MF'[int(re.sub(r'\D', '', spk)) % 2 == 0]


def pair_row(set_, src, tgt, utt, parallel=True, e_paths=None, shortfall=()):
    """A make_pairs.py row: utterance `utt` of `src` converted to `tgt`. parallel: shared-text pair with P (the target's
    recording of the sentence, not wer_primary); else speaker-specific, P null, wer_primary. E: `e_paths` or three
    recordings of the target; R: 1 / 3 / 6 recordings of the target at 3 / 30 / 180 s, `shortfall` the budgets the
    target cannot fill. E, R, P and the source recordings are disjoint (E 300-302, R 400-405, sentences < 300)."""
    R = {}
    for b, n in (('3', 1), ('30', 3), ('180', 6)):
        short = b in shortfall
        total = round(int(b) * (0.6 if short else 1.0), 3)
        each = round(total / n, 3)
        R[b] = {'budget_s': int(b), 'paths': [wav(tgt, 400 + i) for i in range(n)], 'durs': [each] * n,
                'keep_s': [each] * n, 'total_s': total, 'uncut_total_s': total if short else total + 0.5,
                'shortfall': short}
    e = [wav(tgt, 300 + i) for i in range(3)] if e_paths is None else list(e_paths)
    text = f'Shared sentence number {utt}.' if parallel else f'Sentence {utt} of speaker {src}.'
    return {'pair_id': f'{set_}__{src}_{utt:03d}__{tgt}', 'set': set_, 'source_speaker': src,
            'source_gender': gender(src), 'source_utt': f'{src}_{utt:03d}', 'source_path': wav(src, utt),
            'source_text': text, 'source_dur': 3.1, 'source_is_shared_text': parallel,
            'shared_sentence_id': f'{utt:03d}' if parallel else None, 'wer_primary': not parallel,
            'target_speaker': tgt, 'target_gender': gender(tgt), 'genders': f'{gender(src)}2{gender(tgt)}',
            'P': wav(tgt, utt) if parallel else None, 'P_text': text if parallel else None,
            'P_dur': 3.3 if parallel else None, 'R': R,
            'E': {'paths': e, 'durs': [3.0] * len(e), 'total_s': 3.0 * len(e)}}


DEV_SPEAKERS = ('p225', 'p226', 'p227', 'p228')
VCTK_SOURCES, VCTK_TARGETS = ('p301', 'p302', 'p303'), ('p311', 'p312')
LTC_SPEAKERS, LTC_SHORT = ('1089', '2830', '6930', '908'), '6930'


def dev_pairs(speakers=DEV_SPEAKERS, utts=(1, 2)):
    """Dev-like: every speaker is a source and a target; per source the shared sentence utts[0] (with P) and the
    speaker-specific ones after it (without P), each converted to every other speaker (4 x 2 x 3 = 24 rows)."""
    return [pair_row('dev', s, t, u, parallel=(u == utts[0])) for s in speakers for u in utts
            for t in speakers if t != s]


def vctk_pairs():
    """VCTK-2k-like: the sources p301-p303 are never targets (p311, p312): 3 x 2 x 2 = 12 rows."""
    return [pair_row('vctk2k', s, t, u, parallel=(u == 1)) for s in VCTK_SOURCES for u in (1, 2)
            for t in VCTK_TARGETS]


def ltc_pairs():
    """LTC-2k-like: every speaker is a source and a target, no P; target 6930 has a 180 s shortfall (12 rows)."""
    return [pair_row('ltc2k', s, t, 7, parallel=False, shortfall=('180',) if t == LTC_SHORT else ())
            for s in LTC_SPEAKERS for t in LTC_SPEAKERS if t != s]


# ── manifest rows ────────────────────────────────────────────

def conv_row(pair, system, out_dir, budget='30', tau=None, **extra):
    """A run_conversions.py manifest row of `pair`: the base_row fields in their order, then the fields of
    convert_pair and main (gate fields only for the CRT systems)."""
    pid, s = pair['pair_id'], pair['set']
    r = {'pair_id': pid, 'set': s, 'budget': int(budget), 'system': system, 'tau': tau,
         'wav': f'{out_dir}/{s}/{budget}/{system}/{pid}.wav', 'ref_text': pair['source_text'],
         'parallel_wav': pair['P'], 'sim_ref_wavs': pair['E']['paths'], 'source_wav': pair['source_path'],
         'R': {budget: pair['R'][budget]}, 'wer_primary': pair['wer_primary'],
         'source_is_shared_text': pair['source_is_shared_text'], 'genders': pair['genders'],
         'source_speaker': pair['source_speaker'], 'target_speaker': pair['target_speaker'],
         'dump': f'{out_dir}/dumps/{s}/{budget}/{pid}.npz', 'n_frames': 61, 'n_ref_frames': 375, 'ref_s': 29.999,
         'same_codes_as': None}
    if system.startswith('crt_'):
        r.update(override_coverage={'overall': 0.25}, change_coverage={'overall': 0.1}, n_changed=12,
                 override_counts=[3] * 18, change_counts=[1] * 18)
    r.update(conv_time_s={'decode': 0.1, 'ref_cached': True}, run_id='20261018-010203-node-1',
             conv_cfg_sha='0123456789abcdef')
    r.update(extra)
    return r


LAB_DROP = ('override_coverage', 'change_coverage', 'n_changed', 'same_codes_as', 'conv_time_s', 'tau', 'error')


def lab_row(faiss_row, system, out_dir):
    """crt_decode_lab.py main's row of `system`: the pair's faiss row without the gate fields, named lab_<system>."""
    row = {k: v for k, v in faiss_row.items() if k not in LAB_DROP}
    row.update(system='lab_' + system, wav=os.path.join(str(out_dir), system, faiss_row['pair_id'] + '.wav'))
    return row


def write_jsonl(path, rows):
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── torch, faiss, transformers, librosa stand-ins ────────────

class FT(np.ndarray):
    """A numpy array with the torch.Tensor methods the harness calls."""

    def clone(self):
        return self.copy()

    def numpy(self):
        return np.asarray(self)

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def half(self):
        return self.astype(np.float16)

    def float(self):
        return self.astype(np.float32)

    def exp(self):
        return np.exp(self)

    def logaddexp(self, other):
        return np.logaddexp(self, other)

    def log_softmax(self, dim):
        x = self - self.max(dim, keepdims=True)
        return x - np.log(np.exp(x).sum(dim, keepdims=True))


def tensor(a):
    return np.asarray(a).view(FT)


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


TORCH = _module('torch', __version__='stub-torch', long=np.int64, bool=np.bool_, float32=np.float32,
                from_numpy=tensor,
                as_tensor=lambda x, device=None, dtype=None: np.asarray(x, dtype=dtype).view(FT),
                ones_like=lambda x, dtype=None: np.ones_like(np.asarray(x), dtype=dtype).view(FT),
                set_num_threads=lambda n: None,
                backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=None)),
                cuda=types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None))
FAISS = _module('faiss', __version__='stub-faiss', omp_set_num_threads=lambda n: None)
TRANSFORMERS = _module('transformers', __version__='stub-transformers')


def seed(*parts):
    return zlib.crc32(b''.join(np.ascontiguousarray(p).tobytes() for p in parts))


def fake_audio(path):
    """The fake recording at `path`: 8-19 Mimi frames of noise, a function of the path."""
    h = zlib.crc32(path.encode())
    return np.random.default_rng(h).uniform(-0.5, 0.5, FRAME * (8 + h % 12)).astype(np.float32)


def fake_load_reference(entry, sr, allow_shortfall=False):
    """make_pairs.load_reference on fake recordings (the same shortfall error, cut to the budget)."""
    if entry['shortfall'] and not allow_shortfall:
        raise ValueError(f'reference shortfall: {entry["uncut_total_s"]} s available for budget {entry["budget_s"]} s')
    return np.concatenate([fake_audio(p) for p in entry['paths']])[:int(round(entry['budget_s'] * sr))]


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class FakePipe:
    """run_conversions.Pipeline without Mimi or CRT; `log` records the CRT forwards (codes, speaker embedding)."""
    sr = SR

    def __init__(self, args=None, device='cpu', log=None):
        self.torch, self.device, self.dtype = TORCH, device, 'stub.float32'
        self.log = log if log is not None else types.SimpleNamespace(crt=[])
        self.mimi = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash='stub-commit'))
        self.crt = types.SimpleNamespace(first_cb=1, n_out=18)
        self.cbs = slice(1, 19)
        self.crt_info = {'variant': 'v5.6.5', 'weights_sha256': 'b26d6f94' * 8, 'step': 1, 'model_size': 'xlarge',
                         'config': {'d_model': 8}, 'n_cb': 18, 'cb_size': 2048, 'spk_emb_dim': 8, 'n_params': 1e6,
                         'crt_mode': 'anchor', 'crt_returns': 'tensor', 'first_cb': 1, 'refined_cbs': 'CB1-18'}
        self.crt_load_s = {}

    def sync(self):
        pass

    @staticmethod
    def encode(wav_):
        """[1, 32, T] int64 codes, T = len // FRAME."""
        return tensor(np.random.default_rng(seed(wav_)).integers(0, 2048, (1, 32, max(1, len(wav_) // FRAME))))

    def encode_reference(self, wav_):
        return self.encode(wav_)[0].numpy().T

    def build_index(self, codes):
        return {'n': len(codes), 'seed': seed(codes)}

    def speaker_embedding(self, wav_):
        return tensor(np.random.default_rng(seed(wav_)).standard_normal(8).astype(np.float32))

    def retrieve(self, src_codes, index):
        t = src_codes.shape[2]
        rng = np.random.default_rng(index['seed'] ^ seed(src_codes))
        cand_idx = rng.integers(0, index['n'], (t, 8))
        cand_sim = -np.sort(-rng.random((t, 8)), 1).astype(np.float32)
        rank = rng.integers(0, 8, t)
        return cand_idx[np.arange(t), rank], rank, cand_idx, cand_sim

    def crt_logits(self, codes, spk):
        self.log.crt.append((seed(codes), seed(spk)))
        rng = np.random.default_rng(seed(codes, spk))
        return tensor((3 * rng.standard_normal((1, 18, codes.shape[2], V))).astype(np.float32))

    def crt_stats(self, logits, ref_codes, cand_idx):
        p = softmax(np.asarray(logits[0], np.float64))
        logp = np.log(p)
        order = np.argsort(-p, -1, kind='stable')[..., :16]
        top = np.take_along_axis(p, order, -1)
        cand_tok = ref_codes[cand_idx][:, :, self.cbs].transpose(2, 0, 1) % V
        extra = {'crt_margin': top[..., 0] - top[..., 1], 'crt_entropy': -(p * logp).sum(-1), 'topk_prob': top,
                 'topk_idx': order.astype(np.int16),
                 'cand_logp': np.take_along_axis(logp, cand_tok, -1).transpose(0, 2, 1)}
        return tensor(p.argmax(-1)), tensor(p.max(-1).astype(np.float32)), extra

    @staticmethod
    def decode(codes):
        c = np.asarray(codes)[0]
        return np.repeat(((c.sum(0) % 1000) / 1000.0 - 0.5).astype(np.float32), HOP)


def fake_pm_rq(lab, retr, logits):
    """crt_anchor_lab.pm_rq without the latent algebra: CB1-18 := the logits' argmax, CB19-31 changed (re-quantised)."""
    c = retr.clone()
    c[1:19] = np.asarray(logits).argmax(-1)
    c[19:] = (c[19:] + 7) % 2048
    return c


def stub_pipeline(monkeypatch, rc, fail=()):
    """Replaces, for one test, torch / faiss / transformers (imported inside main and gate), run_conversions' librosa
    (fake recordings; a path in `fail` raises), make_pairs.load_reference and run_conversions.Pipeline. Every reference
    fetch goes through the real run_conversions.get_reference. Returns the log: refs [(pair_id, target speaker)] of
    the rows whose reference was fetched, loads [path] read through librosa, crt [CRT forward inputs]."""
    log = types.SimpleNamespace(refs=[], loads=[], crt=[])
    for name, mod in (('torch', TORCH), ('faiss', FAISS), ('transformers', TRANSFORMERS)):
        monkeypatch.setitem(sys.modules, name, mod)

    def load(path, sr=None, mono=True, duration=None):
        log.loads.append(path)
        if path in fail:
            raise RuntimeError(f'cannot read {path}')
        return fake_audio(path), sr

    monkeypatch.setattr(rc, 'librosa', _module('librosa', __version__='stub-librosa', load=load))
    monkeypatch.setattr(rc.make_pairs, 'load_reference', fake_load_reference)
    monkeypatch.setattr(rc, 'Pipeline', lambda args, device: FakePipe(args, device, log))
    real = rc.get_reference

    def get_reference(pipe, row, args, cache):
        log.refs.append((row['pair_id'], row['target_speaker']))
        return real(pipe, row, args, cache)

    monkeypatch.setattr(rc, 'get_reference', get_reference)
    return log


def stub_anchor_lab(monkeypatch, cal, fail=()):
    """stub_pipeline plus crt_anchor_lab's own librosa, crt_decode_lab.Lab and pm_rq."""
    log = stub_pipeline(monkeypatch, cal.rc, fail)
    monkeypatch.setattr(cal, 'librosa', cal.rc.librosa)
    monkeypatch.setattr(cal.cdl, 'Lab', lambda device: types.SimpleNamespace(device=device))
    monkeypatch.setattr(cal, 'pm_rq', fake_pm_rq)
    return log
