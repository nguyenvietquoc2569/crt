#!/usr/bin/env python3
"""Probe P2 (PLAN.md section 4): the content ceiling at 12.5 Hz. How much of the transcript can a small model read
from each Mimi content stream, and is soft CB0 worth having?

A character CTC probe of about 5M parameters (input projection -> 4-layer pre-LN Transformer, d=256, 4 heads ->
ConvTranspose1d stride 4 to 50 Hz, since read speech runs at about 15 characters/s -> linear to 28 characters + blank)
is trained from scratch for each stream on --hours of train-clean-100 shards (encode_mimi.py, encode_perturbed.py),
then scored by greedy-CTC CER on --n_eval dev-clean utterances. Streams (--streams):
  cb0        clean CB0 index -> learned 256-d embedding
  pcb0       CB0 of perturbed draw --draw (<base>.pcodes<k>.npy), same embedding
  soft_T<T>  s_t = sum_k softmax(-d2_tk / (T sigma0^2)) Emb[idx_tk] over the 8 CB0 codewords nearest to the perturbed
             semantic latent (<base>.psoft<k>.idx/.d2.npy); sigma0^2 = the median top-1 d2 over the training frames,
             computed once
  lat        the clean latent, standardised per dimension with training statistics (512-d linear input; upper bound)
Characters: a-z, space and apostrophe of the lowercased LibriTTS-R normalized text; every other character becomes a
space and runs of spaces are collapsed. Utterances over --max_sec are skipped, and so are those CTC cannot align (the
4T output frames must cover the label plus one blank between equal neighbours); both are counted.

Also, on all dev-clean utterances, the P0-style chapter-disjoint speaker-ID (probe_speaker_info.evaluate, P0's split)
of each soft stream as a 2048-d soft bag: every frame spreads its softmax weights over its 8 codeword indices, and the
bag takes the idf of the pcb0 bag of the same draw (its own document frequency would count all 8 indices of every
frame, since no softmax weight is 0), so as T -> 0 it becomes the pcb0 feature and only the soft weighting differs.
cb0, pcb0 (tf-idf) and the transcript control 'text' are scored beside it; excess = acc - acc(text). With
--vctk_pairs, also the CER on up to --vctk_n distinct wer_primary sources of the VCTK Dev pair list, Mimi-encoded
clean (cb0, lat) and after one perturbation seeded by (--perturb_seed, utterance id, --draw) (pcb0, soft). Each source
is padded with TRIM frames of silence before encoding and perturbing, so its stored frames (all but the last TRIM, as
in the shards) cover all of its audio: trimmed VCTK audio may end in speech, and a source encoded alone has no batch
padding to guard against. The VCTK inputs and every array the streams need are checked before any training.

Outputs in --out:
  p2_content.json      config, data counts, sigma0^2, per stream CER / n_eval / n_skipped / train time (and VCTK CER),
                       speaker-ID, and the P2 gates: CB0 content GO if CER(pcb0) <= 1.5 CER(lat) and <= 15 %; start
                       F1 if CER(pcb0) > 1.5 CER(lat); a soft stream is a P6 arm if CER(soft) <= 0.85 CER(pcb0) and its
                       speaker-ID excess <= excess(pcb0) + 0.02. Rewritten after every stream.
  p2_ckpt_<stream>.pt  resume checkpoint, at the end the trained probe.
A rerun with the same config skips finished streams, resumes an interrupted one at its last checkpoint and keeps the
speaker-ID of soft streams scored earlier, so the streams may be split across runs.

    python -u probe_content_ctc.py --out $WORK_DIR/mvc/results/p2 --vctk_pairs $J/pairs/dev.jsonl
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import encode_mimi as em
import encode_perturbed as ep
import probe_speaker_info as ps

MVC = '/datastore/ceai/crt/mimi_vc/data_storage/mvc'
CHARS = " 'abcdefghijklmnopqrstuvwxyz"          # symbol i + 1; 0 is the CTC blank
UP = 4                                          # 12.5 Hz -> 50 Hz
VOCAB = 2048
STREAM_SUFFIX = {'cb0': '.codes.npy', 'pcb0': '.pcodes{k}.npy', 'idx': '.psoft{k}.idx.npy', 'd2': '.psoft{k}.d2.npy',
                 'lat': '.lat.npy'}


def normalise(text):
    t = ''.join(c if c in CHARS else ' ' for c in text.lower())
    return ' '.join(t.split())


def encode_text(text):
    return np.array([CHARS.index(c) + 1 for c in text], np.int64)


def ctc_frames_needed(label):
    """Output frames CTC needs for a label: one per symbol plus a blank between equal neighbours."""
    label = np.asarray(label)
    return int(len(label) + np.sum(label[1:] == label[:-1]))


def greedy(ids):
    ids = np.asarray(ids)
    if len(ids):
        ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    return ''.join(CHARS[i - 1] for i in ids if i)


def edit_distance(ref, hyp):
    """Levenshtein distance of two strings, one numpy pass per reference character."""
    h = np.array([ord(c) for c in hyp], np.int64)
    j = np.arange(len(h) + 1)
    d = j.copy()
    for i, c in enumerate(ref, 1):
        cur = np.empty_like(d)
        cur[0] = i
        cur[1:] = np.minimum(d[1:] + 1, d[:-1] + (h != ord(c)))
        d = np.minimum.accumulate(cur - j) + j
    return int(d[-1])


def cer(refs, hyps):
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / max(1, sum(len(r) for r in refs))


def kind(stream):
    if stream in ('cb0', 'pcb0'):
        return 'tok'
    if stream == 'lat':
        return 'lat'
    if stream.startswith('soft_T'):
        temperature(stream)
        return 'soft'
    raise ValueError(f'unknown stream {stream}')


def temperature(stream):
    return float(stream[len('soft_T'):])


def soft_weights(d2, T, sigma0_sq):
    """[n, 8] squared distances -> softmax(-d2 / (T sigma0^2)) per frame."""
    z = -np.asarray(d2, np.float32) / (T * sigma0_sq)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def soft_bag(idx, w, vocab=VOCAB):
    return np.bincount(np.asarray(idx, np.int64).ravel(), weights=np.asarray(w, np.float64).ravel(), minlength=vocab)


def make_input(stream, get, stats):
    """Probe input of one utterance; get(name) gives its frames of 'cb0', 'pcb0' [n], 'idx', 'd2' [n, 8] or
    'lat' [n, 512]."""
    k = kind(stream)
    if k == 'tok':
        return dict(tok=np.asarray(get(stream), np.int64))
    if k == 'lat':
        return dict(x=(np.asarray(get('lat'), np.float32) - stats['mean']) / stats['std'])
    w = soft_weights(get('d2'), temperature(stream), stats['sigma0_sq'])
    return dict(idx=np.asarray(get('idx'), np.int64), w=w)


class Shards:
    """The utterances of shard directories, with lazy memory maps of the arrays beside each shard."""

    def __init__(self, dirs, draw=0):
        self.bases, self.rows, self.draw, self._mm = [], [], draw, {}
        for d in dirs:
            for js in sorted(Path(d).glob('shard_*.json')):
                self.rows += [dict(r, shard=len(self.bases)) for r in json.load(open(js))]
                self.bases.append(str(js)[:-5])
        if not self.rows:
            raise SystemExit(f'no shards under {dirs}')

    def missing(self, names):
        paths = [b + STREAM_SUFFIX[n].format(k=self.draw) for b in self.bases for n in sorted(names)]
        return [p for p in paths if not os.path.exists(p)]

    def frames(self, i, name):
        r = self.rows[i]
        key = (r['shard'], name)
        if key not in self._mm:
            self._mm[key] = np.load(self.bases[r['shard']] + STREAM_SUFFIX[name].format(k=self.draw), mmap_mode='r')
        x = self._mm[key][r['offset']:r['offset'] + r['frames']]
        return x[:, 0] if name in ('cb0', 'pcb0') else x


def needed(streams):
    """Names of the arrays the probe inputs of streams read."""
    out = set()
    for s in streams:
        k = kind(s)
        out |= {s} if k == 'tok' else {'idx', 'd2'} if k == 'soft' else {'lat'}
    return out


def why_skip(r, max_frames):
    text = normalise(r.get('text', ''))
    if not text:
        return 'empty'
    if r['frames'] > max_frames:
        return 'too_long'
    if UP * r['frames'] < ctc_frames_needed(encode_text(text)):
        return 'ctc'
    return None


def select(rows, max_frames, seed, hours=None, n=None):
    """A seeded random subset of the usable rows, up to `hours` of audio or `n` utterances, and the skip counts over
    all rows."""
    skipped, ok = Counter(), []
    for i, r in enumerate(rows):
        why = why_skip(r, max_frames)
        if why:
            skipped[why] += 1
        else:
            ok.append(i)
    keep, sec = [], 0.0
    for i in np.random.default_rng(seed).permutation(ok):
        if (n is not None and len(keep) >= n) or (hours is not None and sec >= hours * 3600):
            break
        keep.append(int(i))
        sec += rows[i]['samples'] / em.SR
    return keep, dict(n_rows=len(rows), n_used=len(keep), hours=round(sec / 3600, 3), skipped=dict(skipped),
                      n_skipped=sum(skipped.values()))


def sigma0_sq(sh, idx):
    """Median squared distance to the nearest CB0 codeword over all frames of utterances idx."""
    return float(np.median(np.concatenate([np.asarray(sh.frames(i, 'd2')[:, 0], np.float32) for i in idx])))


def lat_stats(sh, idx):
    s, s2, n = 0.0, 0.0, 0
    for i in idx:
        x = np.asarray(sh.frames(i, 'lat'), np.float64)
        s, s2, n = s + x.sum(0), s2 + (x * x).sum(0), n + len(x)
    mean = s / n
    return mean.astype(np.float32), np.maximum(np.sqrt(np.maximum(s2 / n - mean ** 2, 0)), 1e-4).astype(np.float32)


def item(sh, i, stream, stats):
    r = sh.rows[i]
    text = normalise(r['text'])
    x = make_input(stream, lambda name: sh.frames(i, name), stats)
    return dict(id=r['id'], text=text, label=encode_text(text), x=x)


def sinusoid(T, d, device):
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    ang = pos / 10000 ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
    return torch.stack([ang.sin(), ang.cos()], -1).reshape(T, d)


class CTCProbe(torch.nn.Module):
    def __init__(self, kind, d=256, layers=4, heads=4, ff=1536, dropout=0.1, emb_dim=256, in_dim=512):
        super().__init__()
        self.kind = kind
        if kind == 'lat':
            self.inp = torch.nn.Linear(in_dim, d)
        else:
            self.emb = torch.nn.Embedding(VOCAB, emb_dim)
            self.inp = torch.nn.Linear(emb_dim, d)
        self.inp_norm = torch.nn.LayerNorm(d)              # every stream enters at unit scale, beside the positions
        layer = torch.nn.TransformerEncoderLayer(d, heads, ff, dropout, activation='gelu', batch_first=True,
                                                 norm_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, layers, norm=torch.nn.LayerNorm(d), enable_nested_tensor=False)
        self.up = torch.nn.ConvTranspose1d(d, d, UP, stride=UP)
        self.norm = torch.nn.LayerNorm(d)
        self.out = torch.nn.Linear(d, len(CHARS) + 1)

    def forward(self, b):
        if self.kind == 'tok':
            h = self.emb(b['tok'])
        elif self.kind == 'soft':
            h = (self.emb(b['idx']) * b['w'][..., None]).sum(-2)
        else:
            h = b['x']
        h = self.inp_norm(self.inp(h))
        h = h + sinusoid(h.shape[1], h.shape[2], h.device)
        h = self.enc(h, src_key_padding_mask=~b['mask'])
        h = self.up(h.transpose(1, 2)).transpose(1, 2)
        return self.out(self.norm(F.gelu(h)))                            # [B, UP T, 29]


def collate(items, device):
    n = [len(next(iter(it['x'].values()))) for it in items]
    B, T = len(items), max(n)
    b = {}
    for k, v in items[0]['x'].items():
        arr = np.zeros((B, T) + v.shape[1:], v.dtype)
        for i, it in enumerate(items):
            arr[i, :n[i]] = it['x'][k]
        b[k] = torch.as_tensor(arr, device=device)
    b['mask'] = torch.as_tensor(np.arange(T)[None] < np.array(n)[:, None], device=device)
    b['lens'] = torch.as_tensor(n, device=device)
    b['targets'] = torch.as_tensor(np.concatenate([it['label'] for it in items]), device=device)
    b['tgt_lens'] = torch.as_tensor([len(it['label']) for it in items], device=device)
    return b


def ctc_loss(logits, b):
    lp = logits.float().log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(lp, b['targets'], UP * b['lens'], b['tgt_lens'], blank=0, zero_infinity=True)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, a.steps - a.warmup)))


def train_batch(sh, idx, stream, stats, step, a, device):
    """The batch of a step: a function of (seed, step) only, so a resumed run sees the same batches."""
    pick = np.random.default_rng([a.seed, step]).choice(len(idx), size=min(a.batch, len(idx)), replace=False)
    return collate([item(sh, idx[j], stream, stats) for j in pick], device)


def save_atomic(obj, path):
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')


def train_stream(stream, sh, idx, stats, a, cfg, ckpt, log=print):
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    model = CTCProbe(kind(stream), a.d, a.layers, a.heads, a.ff, a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    step, train_s, tail = 0, 0.0, []
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        if ck['config'] == cfg:
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            step, train_s, tail = ck['step'], ck['train_s'], ck['tail']
            if step < a.steps:
                log(f'{stream}: resumed at step {step}')
    model.train()
    t0 = time.time()
    while step < a.steps:
        b = train_batch(sh, idx, stream, stats, step, a, dev)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a)
        torch.manual_seed(a.seed * 1000003 + step)                       # dropout masks also a function of the step
        with autocast(dev):
            logits = model(b)
        loss = ctc_loss(logits, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tail = (tail + [loss.item()])[-100:]
        if step % a.log_every == 0:
            log(f'{stream} step {step}/{a.steps}: ctc {np.mean(tail):.3f}, lr {lr_at(step - 1, a):.2e}, '
                f'{train_s + time.time() - t0:.0f}s')
        if step % a.ckpt_every == 0 or step == a.steps:
            save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, tail=tail, config=cfg,
                             train_s=train_s + time.time() - t0), ckpt)
    train_s += time.time() - t0
    return model, dict(steps=step, train_s=round(train_s, 1), final_ctc=float(np.mean(tail)) if tail else None,
                       n_params=sum(p.numel() for p in model.parameters()))


def evaluate(model, items, device, bs=32, n_examples=3):
    dev = torch.device(device)
    model.eval()
    hyps = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            b = collate(items[i:i + bs], dev)
            with autocast(dev):
                ids = model(b).float().argmax(-1).cpu().numpy()
            hyps += [greedy(ids[k, :UP * n]) for k, n in enumerate(b['lens'].tolist())]
    model.train()
    return dict(cer=cer([it['text'] for it in items], hyps), n_eval=len(items),
                examples=[dict(id=it['id'], ref=it['text'], hyp=h) for it, h in zip(items[:n_examples], hyps)])


def vctk_sources(path, n):
    """The first n distinct sources of the wer_primary rows of a pair list."""
    out, seen = [], set()
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r.get('wer_primary') and r['source_path'] not in seen:
                seen.add(r['source_path'])
                out.append(r)
    return out[:n]


def vctk_inputs(sources, enc, max_frames, seed, draw, perturb_fn, reader=em.read_wav):
    """Mimi frames of every source, clean (cb0, lat) and after one seeded perturbation (pcb0, idx, d2), cast to the
    shard dtypes. The source is padded with TRIM frames of silence, so the stored frames are all frames of its audio."""
    items, skipped = [], Counter()
    for r in sources:
        text = normalise(r['source_text'])
        uid = r.get('source_utt') or Path(r['source_path']).stem
        w = np.pad(reader(r['source_path']), (0, em.TRIM * em.HOP))
        n = em.stored_frames(len(w))
        why = why_skip(dict(text=text, frames=n), max_frames)
        if why:
            skipped[why] += 1
            continue
        lat, cod = enc([w])[0]
        plat, pcod = enc([perturb_fn(w, em.SR, np.random.default_rng(ep.utt_seed(seed, uid, draw)))])[0]
        sidx, sd2 = ep.soft_cb0(enc, plat[:n])
        fr = dict(cb0=cod[:n, 0].astype(np.int16), pcb0=pcod[:n, 0].astype(np.int16), idx=sidx.astype(np.int16),
                  d2=sd2.astype(np.float16), lat=lat[:n].astype(np.float16))
        items.append(dict(id=uid, text=text, label=encode_text(text), frames=fr))
    return items, dict(n_sources=len(sources), n_eval=len(items), skipped=dict(skipped),
                       n_skipped=sum(skipped.values()))


def speaker_id(sh, soft_streams, s0):
    """P0-style speaker-ID on all utterances of sh: tf-idf bags of cb0 and pcb0, the soft bags weighted by the idf of
    the pcb0 bag, and the text control."""
    bags = {s: [] for s in ['cb0', 'pcb0'] + soft_streams}
    for i in range(len(sh.rows)):
        bags['cb0'].append(ps.bag(sh.frames(i, 'cb0')))
        bags['pcb0'].append(ps.bag(sh.frames(i, 'pcb0')))
        if soft_streams:
            idx, d2 = sh.frames(i, 'idx'), sh.frames(i, 'd2')
            for s in soft_streams:
                bags[s].append(soft_bag(idx, soft_weights(d2, temperature(s), s0)))
    X = {s: np.stack(v) for s, v in bags.items()}
    Fs = {s: ps.tfidf(X[s]) for s in ('cb0', 'pcb0')}
    idf = np.log((1 + len(X['pcb0'])) / (1 + (X['pcb0'] > 0).sum(0))) + 1
    for s in soft_streams:
        Y = X[s] * idf
        Fs[s] = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    Fs['text'] = ps.tfidf(ps.text_bag([r.get('text', '') for r in sh.rows]))
    spk = [r['speaker'] for r in sh.rows]
    chap = [f"{r['speaker']}_{r.get('chapter', '')}" for r in sh.rows]
    res = {s: ps.evaluate(f, spk, chapters=chap) for s, f in Fs.items()}
    for v in res.values():
        v['excess'] = v['acc'] - res['text']['acc']
    return res


def gates(streams, spk=None):
    cer_ = {s: e['cer'] for s, e in streams.items() if 'cer' in e}
    g = {}
    if 'pcb0' in cer_ and 'lat' in cer_:
        b, e = cer_['pcb0'], cer_['lat']
        g['cb0_content_go'] = bool(b <= 1.5 * e and b <= 0.15)
        g['start_f1'] = bool(b > 1.5 * e)
    soft = {}
    for s in cer_:
        if kind(s) == 'soft' and 'pcb0' in cer_:
            d = dict(cer_ok=bool(cer_[s] <= 0.85 * cer_['pcb0']))
            if spk and s in spk:
                d['speaker_ok'] = bool(spk[s]['excess'] <= spk['pcb0']['excess'] + 0.02)
            d['p6_arm'] = d['cer_ok'] and d.get('speaker_ok', False)
            soft[s] = d
    if soft:
        arms = sorted((cer_[s], s) for s, d in soft.items() if d['p6_arm'])
        g.update(soft=soft, soft_p6_arm=arms[0][1] if arms else None, soft_T=temperature(arms[0][1]) if arms else None)
    return g


def config(a):
    keys = ('draw', 'hours', 'max_sec', 'n_eval', 'steps', 'batch', 'lr', 'wd', 'warmup', 'd', 'layers', 'heads', 'ff',
            'dropout', 'stats_utts', 'seed')
    cfg = {k: getattr(a, k) for k in keys}
    cfg.update(train=[os.path.abspath(p) for p in a.train.split(',')],
               dev=[os.path.abspath(p) for p in a.dev.split(',')])
    return cfg


def load_results(path, cfg):
    if os.path.exists(path):
        r = json.load(open(path))
        if r.get('config') == cfg:
            return r
    return dict(probe='P2 content ceiling at 12.5 Hz', config=cfg, streams={})


def summary(res, log=print):
    d = res['data']
    log(f"train {d['train']['n_used']} utts ({d['train']['hours']} h, skipped {d['train']['skipped']}), dev "
        f"{d['dev']['n_used']} utts (skipped {d['dev']['skipped']})")
    for s, e in res['streams'].items():
        v = f", VCTK CER {100 * e['vctk']['cer']:.1f} % ({e['vctk']['n_eval']})" if 'vctk' in e else ''
        log(f"{s:10s} CER {100 * e['cer']:.1f} % ({e['n_eval']} utts){v}, {e['steps']} steps in {e['train_s']:.0f}s")
    for s, r in res.get('speaker_id', {}).items():
        log(f"{s:10s} speaker-ID acc {r['acc']:.3f} (chance {r['chance']:.3f}), excess {r['excess']:+.3f}, "
            f"EER {r['eer']:.3f}")
    log(f"gates {json.dumps(res['gates'])}")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', default=f'{MVC}/mimi/train-clean-100', help='comma-separated training shard dirs')
    ap.add_argument('--dev', default=f'{MVC}/mimi/dev-clean', help='comma-separated evaluation shard dirs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--streams', default='cb0,pcb0,soft_T0.5,soft_T1,lat')
    ap.add_argument('--draw', type=int, default=0, help='perturbed draw k (.pcodes<k>, .psoft<k>)')
    ap.add_argument('--hours', type=float, default=50.0)
    ap.add_argument('--max_sec', type=float, default=20.0)
    ap.add_argument('--n_eval', type=int, default=300)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ff', type=int, default=1536)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--stats_utts', type=int, default=3000, help='training utterances for the latent mean and std')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vctk_pairs', default=None, help='VCTK Dev pair list: also the CER on its wer_primary sources')
    ap.add_argument('--vctk_n', type=int, default=100)
    ap.add_argument('--perturb_seed', type=int, default=20261025)
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--ckpt_every', type=int, default=1000)
    ap.add_argument('--device', default='cuda')
    return ap.parse_args(argv)


def main(argv=None, log=lambda m: print(m, flush=True)):
    a = parse(argv)
    streams = a.streams.split(',')
    soft = [s for s in streams if kind(s) == 'soft']
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, dv = Shards(a.train.split(','), a.draw), Shards(a.dev.split(','), a.draw)
    shared = {r['speaker'] for r in tr.rows} & {r['speaker'] for r in dv.rows}
    if shared:
        raise SystemExit(f'training and evaluation shards share {len(shared)} speakers')
    names = needed(streams)
    miss = tr.missing(names) + dv.missing(names | ({'cb0', 'pcb0', 'idx', 'd2'} if soft else set()))
    if miss:
        raise SystemExit(f"{len(miss)} arrays missing, e.g. {', '.join(miss[:3])}")
    max_frames = int(a.max_sec * em.FRAME_RATE)
    tr_idx, tr_info = select(tr.rows, max_frames, a.seed, hours=a.hours)
    dv_idx, dv_info = select(dv.rows, max_frames, a.seed, n=a.n_eval)
    stats = {}
    if 'lat' in streams:
        stats['mean'], stats['std'] = lat_stats(tr, tr_idx[:a.stats_utts])
    if soft:
        stats['sigma0_sq'] = sigma0_sq(tr, tr_idx)
    cfg = config(a)
    path = out / 'p2_content.json'
    res = load_results(path, cfg)
    res.setdefault('data', {}).update(train=tr_info, dev=dv_info)
    if soft:
        res['sigma0_sq'] = stats['sigma0_sq']
    vkey = json.dumps([os.path.abspath(a.vctk_pairs), a.vctk_n, a.perturb_seed]) if a.vctk_pairs else None
    vitems = None
    if vkey and any(res['streams'].get(s, {}).get('vctk', {}).get('key') != vkey for s in streams):
        vitems, res['data']['vctk'] = vctk_inputs(vctk_sources(a.vctk_pairs, a.vctk_n), em.MimiEncoder(a.device),
                                                  max_frames, a.perturb_seed, a.draw, ep.default_perturb)
    for s in streams:
        e = res['streams'].get(s, {})
        need_dev = 'cer' not in e
        need_vctk = vkey is not None and e.get('vctk', {}).get('key') != vkey
        if not (need_dev or need_vctk):
            log(f'{s}: finished earlier, kept')
            continue
        model, info = train_stream(s, tr, tr_idx, stats, a, cfg, str(out / f'p2_ckpt_{s}.pt'), log)
        e.update(info)
        if need_dev:
            e.update(evaluate(model, [item(dv, i, s, stats) for i in dv_idx], a.device), n_skipped=dv_info['n_skipped'])
        if need_vctk:
            its = [dict(it, x=make_input(s, it['frames'].__getitem__, stats)) for it in vitems]
            e['vctk'] = dict(key=vkey, n_skipped=res['data']['vctk']['n_skipped'], **evaluate(model, its, a.device))
        res['streams'][s] = e
        write_json(res, path)
        v = f", VCTK {100 * e['vctk']['cer']:.2f} %" if 'vctk' in e else ''
        log(f"{s}: dev-clean CER {100 * e['cer']:.2f} %{v}")
    if soft:
        res['speaker_id'] = dict(res.get('speaker_id', {}), **speaker_id(dv, soft, stats['sigma0_sq']))
    res['gates'] = gates(res['streams'], res.get('speaker_id'))
    write_json(res, path)
    summary(res, log)
    return 0


if __name__ == '__main__':
    sys.exit(main())
