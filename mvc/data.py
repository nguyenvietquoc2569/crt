"""Training data for Mimi-VC models, read from the shards written by encode_mimi.py.

Corpus(roots) indexes every shard under the given roots (one per subset: <out>/<subset>) and memory-maps the arrays.
A training example is a speaker-consistent pair:
  target  a crop of one utterance (latents [T, 512] and codes [T, 32]),
  prompt  a crop of a DIFFERENT utterance of the same speaker (the reference the model must take the voice from),
plus optional extra per-frame streams registered with add_stream (e.g. the codes of perturbed audio, aligned frame by
frame), for both the target and the prompt. Speakers with a single utterance are skipped, and so are the utterance ids
in `exclude` (the audit_data.py exclusion list). split_speakers() holds out whole speakers.
"""
import json
from pathlib import Path

import numpy as np


class Corpus:
    def __init__(self, roots, min_frames=25, speakers=None, exclude=None):
        self.shards, self.utts, self.n_excluded = [], [], 0
        for root in [roots] if isinstance(roots, (str, Path)) else roots:
            for js in sorted(Path(root).glob('shard_*.json')):
                base = str(js)[:-5]
                rows = json.load(open(js))
                si = len(self.shards)
                self.shards.append(dict(base=base, lat=None, codes=None, streams={}))
                for ri, r in enumerate(rows):
                    if exclude and r['id'] in exclude:
                        self.n_excluded += 1
                    elif r['frames'] >= min_frames and (speakers is None or r['speaker'] in speakers):
                        self.utts.append(dict(r, shard=si, row=ri))
        if not self.utts:
            raise ValueError(f'no utterances of at least {min_frames} frames under {roots}')
        by = {}
        for i, u in enumerate(self.utts):
            by.setdefault(u['speaker'], []).append(i)
        self.by_speaker = {s: v for s, v in by.items() if len(v) >= 2}
        self.speakers = sorted(self.by_speaker)
        self.stream_names = []
        self.utt_streams = {}

    def _arr(self, si, key):
        sh = self.shards[si]
        if sh[key] is None:
            sh[key] = np.load(sh['base'] + ('.lat.npy' if key == 'lat' else '.codes.npy'), mmap_mode='r')
        return sh[key]

    def add_stream(self, name, suffix):
        """Register a per-frame array stored beside each shard as <base><suffix> (same frame layout as .codes.npy)."""
        for sh in self.shards:
            p = sh['base'] + suffix
            if not Path(p).exists():
                raise FileNotFoundError(p)
            sh['streams'][name] = p
        self.stream_names.append(name)

    def add_utt_stream(self, name, suffix):
        """Register a per-utterance array <base><suffix> [rows in shard, D] (e.g. speaker embeddings)."""
        arrs = []
        for sh in self.shards:
            p = sh['base'] + suffix
            if not Path(p).exists():
                raise FileNotFoundError(p)
            arrs.append(np.load(p, mmap_mode='r'))
        self.utt_streams[name] = arrs

    def utt_value(self, ui, name):
        u = self.utts[ui]
        return np.asarray(self.utt_streams[name][u['shard']][u['row']], dtype=np.float32)

    def _stream(self, si, name):
        v = self.shards[si]['streams'][name]
        if isinstance(v, str):
            v = self.shards[si]['streams'][name] = np.load(v, mmap_mode='r')
        return v

    def frames_of(self, ui, start, n, streams=()):
        u = self.utts[ui]
        a, b = u['offset'] + start, u['offset'] + start + n
        out = dict(lat=np.asarray(self._arr(u['shard'], 'lat')[a:b], dtype=np.float32),
                   codes=np.asarray(self._arr(u['shard'], 'codes')[a:b], dtype=np.int64))
        for s in streams:
            out[s] = np.asarray(self._stream(u['shard'], s)[a:b])
        return out

    def hours(self):
        return sum(u['frames'] for u in self.utts) / 12.5 / 3600


def split_speakers(speakers, n_val, seed=0):
    rng = np.random.default_rng(seed)
    s = sorted(speakers)
    val = set(rng.choice(s, size=min(n_val, len(s)), replace=False).tolist()) if n_val else set()
    return [x for x in s if x not in val], sorted(val)


def crop(rng, frames, max_len):
    n = min(frames, max_len)
    return int(rng.integers(0, frames - n + 1)), n


class PairSampler:
    """Draws (target, prompt) pairs: speaker uniformly, then two distinct utterances of that speaker."""

    def __init__(self, corpus, max_target=125, prompt_range=(38, 125), streams=(), seed=0, speakers=None):
        self.c, self.max_target, self.prompt_range = corpus, max_target, prompt_range
        self.streams = tuple(streams)
        self.rng = np.random.default_rng(seed)
        self.speakers = [s for s in (speakers or corpus.speakers) if s in corpus.by_speaker]
        if not self.speakers:
            raise ValueError('no speaker with two utterances')

    def one(self):
        rng = self.rng
        spk = self.speakers[rng.integers(len(self.speakers))]
        ti, pi = rng.choice(self.c.by_speaker[spk], size=2, replace=False)
        t0, tn = crop(rng, self.c.utts[ti]['frames'], self.max_target)
        plen = int(rng.integers(self.prompt_range[0], self.prompt_range[1] + 1))
        p0, pn = crop(rng, self.c.utts[pi]['frames'], plen)
        tgt = self.c.frames_of(ti, t0, tn, self.streams)
        prm = self.c.frames_of(pi, p0, pn, self.streams)
        utt = {k: self.c.utt_value(pi, k) for k in self.c.utt_streams}
        return dict(speaker=spk, target_id=self.c.utts[ti]['id'], prompt_id=self.c.utts[pi]['id'], target=tgt, prompt=prm,
                    prompt_utt=utt)

    def batch(self, n):
        return collate([self.one() for _ in range(n)])


def _pad(arrs, dtype):
    L = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), L) + arrs[0].shape[1:], dtype=dtype)
    mask = np.zeros((len(arrs), L), dtype=bool)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
        mask[i, :len(a)] = True
    return out, mask


def collate(items):
    """Right-padded numpy batch with masks: target_lat [B, T, 512], target_codes [B, T, 32], target_mask [B, T],
    prompt_lat [B, P, 512], prompt_codes, prompt_mask, and every extra stream as target_<name> / prompt_<name>."""
    b = {}
    b['target_lat'], b['target_mask'] = _pad([x['target']['lat'] for x in items], np.float32)
    b['target_codes'], _ = _pad([x['target']['codes'] for x in items], np.int64)
    b['prompt_lat'], b['prompt_mask'] = _pad([x['prompt']['lat'] for x in items], np.float32)
    b['prompt_codes'], _ = _pad([x['prompt']['codes'] for x in items], np.int64)
    for side in ('target', 'prompt'):
        for k in items[0][side]:
            if k not in ('lat', 'codes'):
                b[f'{side}_{k}'], _ = _pad([x[side][k] for x in items], items[0][side][k].dtype)
    if 'prompt_seg' in b:
        b['prompt_seg'][~b['prompt_mask']] = -1
    for k in items[0].get('prompt_utt', {}):
        b[f'prompt_{k}'] = np.stack([x['prompt_utt'][k] for x in items])
    b['speaker'] = [x['speaker'] for x in items]
    return b


MIX = ((0.2, 38, 62), (0.6, 63, 375), (0.2, 376, 750))     # prompt-length mixture of the v1 plan (frames)


class PromptSampler(PairSampler):
    """v1 examples (PLAN.md 2.5): the target is a crop of at most max_target frames of utterance u; the prompt is a
    concatenation of other utterances of the same speaker (never u), drawn from another chapter with probability
    p_cross when the speaker has two or more, of a total length drawn from the mixture `mix`, cut into segments of at
    most seg_len frames (segment ids 1..n per prompt frame). The prompt's global speaker vector is the renormalised mean
    of its utterances' L2-normalised embeddings (utterance stream 'spk', when registered)."""

    def __init__(self, corpus, max_target=125, mix=MIX, seg_len=125, p_cross=0.5, streams=(), seed=0, speakers=None):
        super().__init__(corpus, max_target, (mix[0][1], mix[-1][2]), streams, seed, speakers)
        self.mix, self.seg_len, self.p_cross = mix, seg_len, p_cross
        w = np.array([m[0] for m in mix], dtype=np.float64)
        self.mix_p = w / w.sum()

    def prompt_length(self):
        _, lo, hi = self.mix[self.rng.choice(len(self.mix), p=self.mix_p)]
        return int(self.rng.integers(lo, hi + 1))

    def one(self):
        rng, c = self.rng, self.c
        spk = self.speakers[rng.integers(len(self.speakers))]
        utts = c.by_speaker[spk]
        ti = int(rng.choice(utts))
        t0, tn = crop(rng, c.utts[ti]['frames'], self.max_target)
        others = [u for u in utts if u != ti]
        chap = c.utts[ti]['chapter']
        cross = [u for u in others if c.utts[u]['chapter'] != chap]
        pool = cross if cross and rng.random() < self.p_cross else others
        want = self.prompt_length()
        order = list(rng.permutation(pool))
        parts, total, used = [], 0, []
        while total < want and order:
            ui = int(order.pop())
            n = min(c.utts[ui]['frames'], want - total)
            s0 = int(rng.integers(0, c.utts[ui]['frames'] - n + 1))
            parts.append(c.frames_of(ui, s0, n, self.streams))
            used.append(ui)
            total += n
        prm = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        seg, sid = [], 1
        for p in parts:                                  # segments never cross an utterance boundary
            n = len(p['lat'])
            for a in range(0, n, self.seg_len):
                seg.append(np.full(min(self.seg_len, n - a), sid, dtype=np.int64))
                sid += 1
        prm['seg'] = np.concatenate(seg)
        utt = {}
        for k in c.utt_streams:
            v = np.stack([c.utt_value(u, k) for u in used])
            v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
            m = v.mean(0)
            utt[k] = (m / max(np.linalg.norm(m), 1e-8)).astype(np.float32)
        tgt = c.frames_of(ti, t0, tn, self.streams)
        return dict(speaker=spk, target_id=c.utts[ti]['id'], prompt_id='+'.join(c.utts[u]['id'] for u in used),
                    target=tgt, prompt=prm, prompt_utt=utt)

    def batch_tokens(self, budget, pool=None):
        """A length-bucketed batch: draw `pool` examples, sort by prompt + target length, and return the run of
        neighbours (from a random start) whose padded size B x max_len stays within `budget` tokens."""
        items = [self.one() for _ in range(pool or max(8, budget // 300))]
        items.sort(key=lambda x: len(x['prompt']['lat']) + len(x['target']['lat']))
        start = int(self.rng.integers(0, len(items)))
        out, mx = [], 0
        for x in items[start:] + items[:start][::-1]:
            L = len(x['prompt']['lat']) + len(x['target']['lat'])
            if out and (len(out) + 1) * max(mx, L) > budget:
                break
            out.append(x)
            mx = max(mx, L)
        return collate(out)
