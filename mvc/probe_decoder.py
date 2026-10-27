#!/usr/bin/env python3
"""Probe P1 (PLAN.md §4): decoder tolerance. Which decode route and precision, and how accurate must the generator be?

Inputs: up to --n_libri dev-clean utterances (the shards of encode_mimi.py for their ids and transcripts, the raw
LibriTTS-R wavs under --raw), spread round-robin over the speakers in a seeded order, plus the sources of the first
--n_dev rows of the Dev screen shard 0/20 (rows k % 20 == 0 of --pairs). Each is encoded once in fp32
(encode_mimi.MimiEncoder) to lat [T, 512] and codes [T, 32] and decoded under every condition:
  codes32_fp16     MimiModel.decode of the fp32-encoded 32 codes by a float16 copy of Mimi: the reference (fp16 decode
                   as in the published controls; the encode here is fp32)
  codes32_fp32     the same in float32 (mimi_io.decode_codes)
  cont_fp32        the unquantised path, mimi_io.decode_latent: x_dec = sem.out(sem.in(lat)) + ac.out(ac.in(lat))
  cont_fp16        the same path in the float16 model
  q8, q16          the float16 MimiModel.decode of the first 8 / 16 codebooks (HF's split RVQ sums only the levels it
                   is given, so the missing acoustic levels contribute nothing)
  noise_eps<e>     z = (lat - mean) / std plus e * N(0, I), decoded on the fp32 unquantised path
  dec_noise_eps<e> x_dec plus s * e * N(0, I), decoded by the fp32 decoder, where s^2 = ||M diag(std)||_F^2 / 512 and M
                   is the linear map lat -> x_dec: the noise has the expected squared norm of the decoder-input image of
                   noise_eps<e>, but is isotropic in decoder-input space (s is in the result file)
  smooth3          a centred 3-frame moving average of lat (2 frames at the ends), fp32 unquantised path
The noise draw N(0, I) is seeded per utterance and shared by every e and both spaces. mean / std come from --stats (an
npz with mean and std, e.g. a run's stats.npz) or from the probed latents (as train.corpus_stats).

Outputs in --out: <condition>/<key>.wav (24 kHz PCM_16, cut to the source length); rows_<condition>.jsonl and
rows_all.jsonl, input rows for the journal's metrics.py (pair_id = key, system = condition, ref_text = the transcript,
sim_ref_wavs = up to 10 other recordings of the same speaker, never the utterance itself: dev-clean utterances of at
least --min_sec, unprobed ones first, or the speaker's E recordings in the pair list, the journal's SIM centroid (its
other source recordings only if it has no E); source_wav = the original recording, parallel_wav null, wer_primary
true, no R), checked by the probe rule and metrics.check_row (when --journal is importable) before anything is encoded;
stats.npz; probe_cfg.json; probe_decoder.json (counts, the noise scale, and for the I/O decision the free space of the
--local_disk paths and of every local mount in /proc/mounts, and the cgroup memory); an empty m/ for the metrics.py
outputs. Existing wavs are kept, so a rerun resumes (a new --eps level only adds its conditions); a rerun with other
utterances or source paths, seed, stats, --device or wav-making code (its hash) is refused.

summarize reads the metrics.py outputs (--metrics wer,sim,utmos; any files or directories of jsonl, rows grouped by
system) on the utterances complete in every condition: capped WER = sum min(S+D+I, N) / sum N, mean SIM, mean UTMOS,
paired deltas against codes32_fp16, and the P1 decisions: the continuous route is allowed if cont_fp32 has
|dUTMOS| <= 0.05, dWER <= 0.5 points and dSIM >= -0.01; eps_0.2, the noise level at which UTMOS falls 0.2 below
cont_fp32 (linear between levels, from 0 at e = 0), in both spaces; R5 early and NFE 32 if eps_0.2 < 0.1; the K6 arm
if decoder-input noise is at least 2x as harmful per unit MSE at equal decoder-input energy, (eps_d(lat) /
eps_d(dec))^2 >= 2 at the drop d = 0.2, or at the largest drop both spaces reach if either never reaches 0.2
(k6_drop_level); and whether cont_fp16 and cont_fp32 differ by more than 0.02 UTMOS. With the noise scale s of
probe_decoder.json (--probe_json, default <dir of --metrics>/../probe_decoder.json) it also reports the other reading
of PLAN's "same MSE", each space per unit of its own metric (e^2 standardised, s^2 e^2 decoder-input):
k6_harm_ratio_own_metric = k6_harm_ratio / s^2, and eps_0.2(dec) in decoder-input units, s * eps_0.2(dec); k6_arm
stays on the first reading.

    python probe_decoder.py run --shards_dev $MVC/mimi/dev-clean --raw $RAW --pairs $J/pairs/dev.jsonl --journal $J \
        --out $MVC/probes/p1
    python $J/metrics.py --in_jsonl $MVC/probes/p1/rows_all.jsonl --out_jsonl $MVC/probes/p1/m/all.jsonl \
        --metrics wer,sim,utmos --whisper_models large-v3 --wavlm_model $J/models/wavlm-base-plus-sv --device cuda
    python probe_decoder.py summarize --metrics $MVC/probes/p1/m --json $MVC/probes/p1/p1_summary.json
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep

REF, CONT = 'codes32_fp16', 'cont_fp32'
N_REFS = 10
WORK_DIR = '/datastore/ceai/crt/mimi_vc'
SKIP_FS = {'nfs', 'nfs4', 'lustre', 'gpfs', 'beegfs', 'cifs', 'smb3', 'ceph', 'glusterfs', 'autofs', 'proc', 'sysfs',
             'devtmpfs', 'overlay'}


def conditions(eps):
    return (['codes32_fp16', 'codes32_fp32', 'cont_fp32', 'cont_fp16', 'q8', 'q16'] + [f'noise_eps{e:g}' for e in eps]
            + [f'dec_noise_eps{e:g}' for e in eps] + ['smooth3'])


def path_mapper(spec):
    """'OLD=NEW' -> a function replacing the prefix OLD of a path by NEW ('' -> identity)."""
    if not spec:
        return lambda p: p
    old, new = spec.split('=', 1)
    return lambda p: new + p[len(old):] if p.startswith(old) else p


def refs_of(pool, path, k=N_REFS):
    n = os.path.normpath(path)
    return [p for p in pool if os.path.normpath(p) != n][:k]


def select_libri(rows, n, seed, min_sec=1.0):
    """Up to n utterances with a transcript and at least min_sec, round-robin over the speakers in a seeded order per
    speaker; and per speaker its SIM reference pool: its utterances of at least min_sec in the same order, the probed
    ones last."""
    rng = np.random.default_rng(seed)
    order = {}
    for r in sorted(rows, key=lambda r: r['id']):
        order.setdefault(r['speaker'], []).append(r)
    order = {s: [v[i] for i in rng.permutation(len(v))] for s, v in sorted(order.items())}
    long = {s: [r for r in v if r['samples'] >= min_sec * em.SR] for s, v in order.items()}
    ok = {s: [r for r in v if r['text'].strip()] for s, v in long.items()}
    pick, i = [], 0
    while len(pick) < n and any(i < len(v) for v in ok.values()):
        pick += [v[i] for v in ok.values() if i < len(v)][:n - len(pick)]
        i += 1
    chosen = {r['id'] for r in pick}
    pool = {s: [r for r in v if r['id'] not in chosen] + [r for r in v if r['id'] in chosen] for s, v in long.items()}
    return pick, pool


def libri_items(shard_dirs, raw, n, seed, min_sec):
    rows = [r for d in shard_dirs for js in sorted(Path(d).glob('shard_*.json')) for r in json.load(open(js))]
    pick, pool = select_libri(rows, n, seed, min_sec)
    items = []
    for r in pick:
        path = ep.wav_path(raw, r)
        items.append(dict(key=f"libri__{r['id']}", part='dev-clean', speaker=r['speaker'], path=path, text=r['text'],
                          refs=refs_of([ep.wav_path(raw, x) for x in pool[r['speaker']]], path)))
    return items


def dev_items(pair_rows, n, pmap):
    """Sources of the first n rows k % 20 == 0 (the screen shard 0/20); SIM references from the speaker's E paths
    (the rows that have the speaker as target), the journal's SIM_src centroid; its other source recordings only if
    it has no E."""
    E, src = {}, {}
    for r in pair_rows:
        E.setdefault(r['target_speaker'], []).extend(pmap(p) for p in r['E']['paths'])
        src.setdefault(r['source_speaker'], []).append(pmap(r['source_path']))
    pool = {s: list(dict.fromkeys(E.get(s) or v)) for s, v in src.items()}
    items, seen = [], set()
    for r in pair_rows[::20]:
        if len(items) >= n:
            break
        if r['source_utt'] in seen:
            continue
        seen.add(r['source_utt'])
        path = pmap(r['source_path'])
        items.append(dict(key=f"vctk__{r['source_utt']}", part='vctk-dev', speaker=r['source_speaker'], path=path,
                          text=r['source_text'], refs=refs_of(pool[r['source_speaker']], path)))
    return items


def build_items(a):
    items = libri_items(a.shards_dev.split(','), a.raw, a.n_libri, a.seed, a.min_sec) if a.shards_dev else []
    if a.pairs and a.n_dev:
        rows = [json.loads(line) for line in open(a.pairs) if line.strip()]
        items += dev_items(rows, a.n_dev, path_mapper(a.path_map))
    keys = [i['key'] for i in items]
    if len(set(keys)) != len(keys) or not items:
        raise SystemExit(f'{len(items)} utterances, {len(keys) - len(set(keys))} duplicate keys')
    return items


class Half:
    """The decode routes of mimi_io.MimiIO on a 16-bit copy of Mimi (inputs cast to the model's dtype)."""

    def __init__(self, m, device):
        import torch
        self.torch, self.m, self.device = torch, m, device
        self.dtype = next(m.parameters()).dtype
        q = m.quantizer
        self.sem, self.ac = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer

    def decode_latent(self, x):
        torch, m = self.torch, self.m
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(x, np.float32), device=self.device).T[None].to(self.dtype)
            emb = m.upsample(self.sem.output_proj(self.sem.input_proj(z)) + self.ac.output_proj(self.ac.input_proj(z)))
            h = m.decoder_transformer(emb.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
            return m.decoder(h).float().cpu().squeeze().numpy()

    def decode_codes(self, codes):
        torch = self.torch
        with torch.no_grad():
            c = torch.as_tensor(np.asarray(codes), dtype=torch.long, device=self.device).T[None]
            return self.m.decode(c).audio_values.float().cpu().squeeze().numpy()


def smooth3(lat):
    p = np.pad(lat, ((1, 1), (0, 0)))
    c = np.pad(np.ones(len(lat), lat.dtype), 1)
    return (p[:-2] + p[1:-1] + p[2:]) / (c[:-2] + c[1:-1] + c[2:])[:, None]


def dec_noise_scale(io, std):
    """s with s^2 = ||M diag(std)||_F^2 / 512 (M: lat -> x_dec, linear): the per-dimension decoder-input noise std
    whose expected squared norm equals that of M applied to latent noise std * N(0, I)."""
    with io.torch.no_grad():
        X = io.decoder_input(np.diag(std).astype(np.float32))
    return float(np.sqrt((X.double() ** 2).sum().item() / len(std)))


def routes(io, half, lat, codes, std, s_dec, eps, xi):
    """{condition: () -> wav} for one utterance: lat [T, 512] and codes [T, 32] of its fp32 encode, xi [T, 512] its
    N(0, I) draw."""
    torch = io.torch

    def dec_noise(e):
        with torch.no_grad():
            x = io.decoder_input(lat)
            return io._decode_input(x + s_dec * e * torch.as_tensor(xi.T[None], device=x.device))

    r = {'codes32_fp16': lambda: half.decode_codes(codes), 'codes32_fp32': lambda: io.decode_codes(codes),
         'cont_fp32': lambda: io.decode_latent(lat), 'cont_fp16': lambda: half.decode_latent(lat),
         'q8': lambda: half.decode_codes(codes[:, :8]), 'q16': lambda: half.decode_codes(codes[:, :16]),
         'smooth3': lambda: io.decode_latent(smooth3(lat))}
    for e in eps:
        r[f'noise_eps{e:g}'] = lambda e=e: io.decode_latent(lat + std * e * xi)
        r[f'dec_noise_eps{e:g}'] = lambda e=e: dec_noise(e)
    return r


def wav_of(out, cond, item):
    return os.path.join(out, cond, item['key'] + '.wav')


def write_wav(path, wav):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path[:-4]}.tmp{os.getpid()}.wav'
    sf.write(tmp, wav, em.SR, subtype='PCM_16')
    os.replace(tmp, path)


def write_text(path, text):
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def make_row(item, cond, out):
    return dict(pair_id=item['key'], system=cond, set='p1', part=item['part'], wav=wav_of(out, cond, item),
                ref_text=item['text'], parallel_wav=None, sim_ref_wavs=item['refs'], source_wav=item['path'],
                wer_primary=True, source_speaker=item['speaker'], target_speaker=item['speaker'])


def check_row(r):
    """The probe's reference rule: 1-10 distinct SIM references, never the utterance itself or the output."""
    refs = [os.path.normpath(p) for p in r['sim_ref_wavs']]
    if (not 1 <= len(refs) <= N_REFS or len(set(refs)) != len(refs)
            or {os.path.normpath(r['source_wav']), os.path.normpath(r['wav'])} & set(refs)):
        raise ValueError(f"{r['pair_id']} {r['system']}: invalid sim_ref_wavs {r['sim_ref_wavs'][:3]}")


def journal_check(journal, log=print):
    """metrics.check_row of the journal tooling, or None if it cannot be imported here (metrics.py checks every row
    again when it scores them)."""
    path = os.path.join(journal or '', 'metrics.py')
    if not journal or not os.path.isfile(path):
        log(f'no {path}: rows checked by the probe rule only')
        return None
    spec = importlib.util.spec_from_file_location('journal_metrics', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        log(f'{path} not importable ({e}): rows checked by the probe rule only')
        return None
    return mod.check_row


def cgroup_memory(proc='/proc/self/cgroup', root='/sys/fs'):
    """(limit, used) bytes of the tightest memory limit over this process's cgroups and their parents (v2 or v1)."""
    if not os.path.exists(proc):
        return None
    best = None
    for line in open(proc).read().splitlines():
        _, ctrl, rel = line.split(':', 2)
        if ctrl == '':
            base, lim, cur = Path(root, 'cgroup'), 'memory.max', 'memory.current'
        elif 'memory' in ctrl.split(','):
            base, lim, cur = Path(root, 'cgroup', 'memory'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        d = Path(str(base) + rel.rstrip('/'))
        while True:
            if (d / lim).exists() and (d / cur).exists():
                v = (d / lim).read_text().strip()
                if v != 'max' and (best is None or int(v) < best[0]):
                    best = (int(v), int((d / cur).read_text()))
            if d == base or d == d.parent:
                break
            d = d.parent
    return best


def local_mounts(mounts='/proc/mounts'):
    """{mountpoint: (fstype, free GB)} of the mounted filesystems that are neither network nor pseudo filesystems."""
    if not os.path.exists(mounts):
        return {}
    out = {}
    for line in open(mounts).read().splitlines():
        f = line.split()
        if len(f) < 3 or f[2] in SKIP_FS or f[2].startswith(('fuse', 'cgroup')):
            continue
        try:
            u = shutil.disk_usage(f[1])
        except OSError:
            continue
        if u.total:
            out[f[1]] = (f[2], round(u.free / 1e9, 1))
    return out


def host_info(paths, mounts='/proc/mounts'):
    h = {'disk_free_gb': {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in paths if os.path.isdir(p)},
         'local_mounts': local_mounts(mounts)}
    cg = cgroup_memory()
    if cg:
        h.update(cgroup_mem_limit_gb=round(cg[0] / 1e9, 1), cgroup_mem_free_gb=round((cg[0] - cg[1]) / 1e9, 1))
    return h


def code_hash():
    """Hash of the source of the code that makes the wavs: the encode, the decode routes and the wav writer."""
    import mimi_io
    src = [inspect.getsource(f) for f in (em.MimiEncoder, em.read_wav, mimi_io.MimiIO, Half, smooth3, dec_noise_scale,
                                          routes, write_wav)]
    return hashlib.sha256('\n'.join(src).encode()).hexdigest()[:16]


def run(a, io, half, reader=em.read_wav, log=print):
    out = os.path.abspath(a.out)
    eps = [float(e) for e in a.eps.split(',')]
    conds = conditions(eps)
    items = build_items(a)
    check = journal_check(a.journal, log)
    rows = {c: [make_row(i, c, out) for i in items] for c in conds}
    for r in (r for rs in rows.values() for r in rs):
        check_row(r)
        if check:
            check(r, None)
    cfg = dict(items=[[i['key'], i['path']] for i in items], seed=a.seed,
               stats=os.path.abspath(a.stats) if a.stats else 'probed', device=a.device, code=code_hash())
    cfg_path = os.path.join(out, 'probe_cfg.json')
    os.makedirs(os.path.join(out, 'm'), exist_ok=True)
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        diff = sorted(k for k in set(cfg) | set(old) if cfg.get(k) != old.get(k))
        if diff:
            raise SystemExit(f'{cfg_path} holds other {", ".join(diff)}; use another --out')
    else:
        write_text(cfg_path, json.dumps(cfg, indent=1))
    t0 = time.time()
    enc = []
    for item in items:
        w = reader(item['path'])
        enc.append(io.encode(w) + (len(w),))
    sp = os.path.join(out, 'stats.npz')
    if a.stats or os.path.exists(sp):
        st = np.load(a.stats or sp)
        mean, std = st['mean'].astype(np.float32), st['std'].astype(np.float32)
    else:
        x = np.concatenate([e[0] for e in enc]).astype(np.float64)
        mean, std = x.mean(0).astype(np.float32), np.maximum(x.std(0), 1e-4).astype(np.float32)
    if not os.path.exists(sp):
        with open(sp + '.tmp', 'wb') as f:
            np.savez(f, mean=mean, std=std)
        os.replace(sp + '.tmp', sp)
    s_dec = dec_noise_scale(io, std)
    ss = nn = 0
    with io.torch.no_grad():
        for e in enc:
            x = io.decoder_input(e[0]).double()
            ss, nn = ss + float((x ** 2).sum()), nn + x.numel()
    rms = float(np.sqrt(ss / nn))
    t1, n_new = time.time(), 0
    for k, (item, (lat, codes, n)) in enumerate(zip(items, enc)):
        todo = [c for c in conds if not os.path.exists(wav_of(out, c, item))]
        if todo:
            rng = np.random.default_rng(ep.utt_seed(a.seed, item['key'], 0))
            xi = rng.standard_normal(lat.shape).astype(np.float32)
            fns = routes(io, half, lat, codes, std, s_dec, eps, xi)
            for c in todo:
                write_wav(wav_of(out, c, item), fns[c]()[:n])
            n_new += len(todo)
        if (k + 1) % 50 == 0:
            log(f'  [{k + 1}/{len(items)}] {n_new} wavs written, {time.time() - t1:.0f}s')
    for c, rs in rows.items():
        write_text(os.path.join(out, f'rows_{c}.jsonl'), ''.join(json.dumps(r) + '\n' for r in rs))
    write_text(os.path.join(out, 'rows_all.jsonl'), ''.join(json.dumps(r) + '\n' for c in conds for r in rows[c]))
    parts = {p: sum(i['part'] == p for i in items) for p in sorted({i['part'] for i in items})}
    res = dict(n_utts=len(items), parts=parts, conditions=conds, eps=eps, seed=a.seed, stats=cfg['stats'],
               dec_noise_scale=s_dec, dec_input_rms=rms, lat_std_mean=float(std.mean()),
               dec_noise_std={f'{e:g}': s_dec * e for e in eps}, n_rows=len(items) * len(conds), new_wavs=n_new,
               checked_with='metrics.check_row + probe rule' if check else 'probe rule', encode_s=round(t1 - t0, 1),
               decode_s=round(time.time() - t1, 1), host=host_info(a.local_disk.split(',')))
    write_text(os.path.join(out, 'probe_decoder.json'), json.dumps(res, indent=1))
    log(f'P1: {len(items)} utterances {parts}, {len(conds)} conditions, {n_new} new wavs in {time.time() - t0:.0f}s; '
        f'decoder-input noise scale s={s_dec:.4f} (x_dec rms {rms:.4f}); rows -> {out}/rows_all.jsonl; '
        f'host {res["host"]}')
    return res


# ── summarize ────────────────────────────────────────────────

def load_rows(specs):
    """Last row per (pair_id, system) of the jsonl files given, or of every *.jsonl in a directory given."""
    out = {}
    for s in specs.split(','):
        for f in sorted(Path(s).glob('*.jsonl')) if os.path.isdir(s) else [Path(s)]:
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    out[(r['pair_id'], r['system'])] = r
    return out


def complete(r, wer_key):
    v = [r.get('sim_wavlm'), r.get('utmos')]
    return ('error' not in r and isinstance(r.get(wer_key), dict)
            and all(isinstance(x, (int, float)) and np.isfinite(x) for x in v))


def score(rs, wer_key):
    w = [r[wer_key] for r in rs]
    n = sum(x['N'] for x in w)
    return dict(n=len(rs), wer_capped=100.0 * sum(min(x['S'] + x['D'] + x['I'], x['N']) for x in w) / n if n else None,
                sim=float(np.mean([r['sim_wavlm'] for r in rs])), utmos=float(np.mean([r['utmos'] for r in rs])))


def table(by, conds, pids, ref, wer_key):
    t = {}
    for c in conds:
        rs = [by[c][p] for p in pids]
        t[c] = score(rs, wer_key)
        if ref in by:
            rr = [by[ref][p] for p in pids]
            w = score(rr, wer_key)['wer_capped']
            t[c]['d_wer'] = t[c]['wer_capped'] - w if w is not None else None
            for k, key in (('utmos', 'utmos'), ('sim', 'sim_wavlm')):
                d = np.array([x[key] - y[key] for x, y in zip(rs, rr)])
                t[c][f'd_{k}'] = float(d.mean())
                t[c][f'se_d_{k}'] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    return t


def eps_at_drop(points, level=0.2):
    """Smallest noise level at which the UTMOS drop reaches `level`, linear between the measured (e, drop) points and
    from (0, 0); None if it is not reached."""
    pe, pd = 0.0, 0.0
    for e, d in sorted(points):
        if d >= level:
            return pe + (level - pd) * (e - pe) / (d - pd)
        pe, pd = e, d
    return None


def decisions(t, eps, ref=REF, cont=CONT, s_dec=None):
    d = {}
    if ref in t and cont in t:
        c = t[cont]
        d['continuous_allowed'] = bool(abs(c['d_utmos']) <= 0.05 and c['d_wer'] is not None and c['d_wer'] <= 0.5
                                       and c['d_sim'] >= -0.01)
    if 'cont_fp16' in t and cont in t:
        diff = abs(t['cont_fp16']['utmos'] - t[cont]['utmos'])
        d.update(cont_fp16_vs_fp32_utmos=diff, cont_fp16_differs=bool(diff > 0.02))
    if cont not in t:
        return d
    drops = {s: [(e, t[cont]['utmos'] - t[f'{s}_eps{e:g}']['utmos']) for e in eps if f'{s}_eps{e:g}' in t]
             for s in ('noise', 'dec_noise')}
    e02 = {s: eps_at_drop(p) for s, p in drops.items()}
    lv = min([0.2] + [max(v for _, v in p) for p in drops.values()]) if all(drops.values()) else 0.0
    lv = lv if lv > 0 else None
    ratio, how = None, 'no drop reached by both spaces'
    if lv:
        ratio = (eps_at_drop(drops['noise'], lv) / eps_at_drop(drops['dec_noise'], lv)) ** 2
        how = 'MSE ratio at the 0.2 drop' if lv == 0.2 else 'MSE ratio at the common drop k6_drop_level'
    d.update(drop_base=cont, utmos_drop={s: {f'{e:g}': v for e, v in p} for s, p in drops.items()}, eps_0_2=e02,
             r5_early_nfe32=bool(e02['noise'] is not None and e02['noise'] < 0.1), k6_harm_ratio=ratio,
             k6_drop_level=lv, k6_harm_ratio_method=how,
             k6_arm=bool(ratio is not None and ratio >= 2))
    if s_dec:
        own = ratio / s_dec ** 2 if ratio is not None else None
        d.update(dec_noise_scale=s_dec, k6_harm_ratio_own_metric=own,
                 k6_arm_own_metric=bool(own is not None and own >= 2),
                 eps_0_2_dec_input_units=s_dec * e02['dec_noise'] if e02['dec_noise'] is not None else None)
    return d


def summarize(rows, ref=REF, wer_key='wer_large_v3', s_dec=None):
    by = {}
    for (pid, s), r in rows.items():
        if complete(r, wer_key):
            by.setdefault(s, {})[pid] = r
    eps = sorted({float(s.rsplit('_eps', 1)[1]) for s in by if '_eps' in s})
    conds = [c for c in conditions(eps) if c in by] + sorted(set(by) - set(conditions(eps)))
    if not conds:
        raise SystemExit('no complete rows')
    pids = sorted(set.intersection(*(set(by[c]) for c in conds)))
    n_all = len({p for c in conds for p in by[c]})
    t = table(by, conds, pids, ref, wer_key)
    parts = {}
    for p in pids:
        parts.setdefault(by[conds[0]][p].get('part', ''), []).append(p)
    return dict(n_common=len(pids), n_dropped=n_all - len(pids), ref=ref, wer_key=wer_key, table=t,
                parts={k: table(by, conds, v, ref, wer_key) for k, v in parts.items()},
                decisions=decisions(t, eps, ref, s_dec=s_dec))


def fmt(v, f, sign=False):
    return '-' if v is None else format(v, ('+' if sign else '') + f)


def print_summary(s, log=print):
    log(f"P1 on {s['n_common']} utterances complete in every condition ({s['n_dropped']} dropped), vs {s['ref']}")
    log(f"{'condition':18s} {'n':>4s} {'WERc':>6s} {'dWER':>6s} {'SIM':>7s} {'dSIM':>8s} {'UTMOS':>6s} {'dUTMOS':>7s}")
    for c, v in s['table'].items():
        log(f"{c:18s} {v['n']:4d} {fmt(v['wer_capped'], '6.2f')} {fmt(v.get('d_wer'), '6.2f', 1)} {v['sim']:7.4f} "
            f"{fmt(v.get('d_sim'), '8.4f', 1)} {v['utmos']:6.3f} {fmt(v.get('d_utmos'), '7.3f', 1)}")
    d = s['decisions']
    if 'continuous_allowed' in d:
        log(f"continuous route {'ALLOWED' if d['continuous_allowed'] else 'NOT allowed: Q32 is the only output'} "
            f"(cont_fp32: |dUTMOS| <= 0.05, dWER <= 0.5, dSIM >= -0.01)")
    if 'cont_fp16_differs' in d:
        log(f"cont fp16 vs fp32 UTMOS {d['cont_fp16_vs_fp32_utmos']:.3f}"
            + (' > 0.02: decode continuous in fp32 and log it' if d['cont_fp16_differs'] else ''))
    if 'eps_0_2' in d:
        e = d['eps_0_2']
        log(f"eps_0.2 (UTMOS -0.2 vs {d['drop_base']}): latent {fmt(e['noise'], '.3f')}, decoder-input "
            f"{fmt(e['dec_noise'], '.3f')}; R5 early + NFE 32: {d['r5_early_nfe32']}; K6 harm ratio "
            f"{fmt(d['k6_harm_ratio'], '.2f')} at drop {fmt(d['k6_drop_level'], '.3f')} ({d['k6_harm_ratio_method']}, "
            f"equal decoder-input energy), K6 arm: {d['k6_arm']}")
    if 'k6_harm_ratio_own_metric' in d:
        log(f"  per unit of each space's own MSE (s={d['dec_noise_scale']:.4f}): K6 harm ratio "
            f"{fmt(d['k6_harm_ratio_own_metric'], '.2f')}, arm {d['k6_arm_own_metric']}; eps_0.2 decoder-input in "
            f"decoder-input units {fmt(d['eps_0_2_dec_input_units'], '.4f')}")


def rounded(x):
    if isinstance(x, float):
        return round(x, 5)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    return x


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--shards_dev', default='', help='comma-separated dev-clean shard directories (encode_mimi.py)')
    r.add_argument('--raw', default=f'{WORK_DIR}/data_storage/dataset_raw/LibriTTS_R')
    r.add_argument('--n_libri', type=int, default=300)
    r.add_argument('--min_sec', type=float, default=1.0)
    r.add_argument('--pairs', default='', help='Dev pair list (make_pairs.py rows)')
    r.add_argument('--n_dev', type=int, default=50)
    r.add_argument('--path_map', default='/datastore/nvquoc/crt=/datastore/ceai/crt',
                   help='OLD=NEW prefix rewrite of the pair-list paths ("" for none)')
    r.add_argument('--journal', default=f'{WORK_DIR}/benchmark/journal', help='for metrics.check_row')
    r.add_argument('--stats', default='', help='npz with the latent mean and std (default: the probed latents)')
    r.add_argument('--eps', default='0.05,0.1,0.2,0.4')
    r.add_argument('--seed', type=int, default=20261025)
    r.add_argument('--local_disk', default='/tmp', help='comma-separated node-local paths whose free space is logged')
    r.add_argument('--out', required=True)
    r.add_argument('--device', default='cuda')
    s = sub.add_parser('summarize')
    s.add_argument('--metrics', required=True, help='comma-separated metrics.py outputs (files or directories)')
    s.add_argument('--json', required=True)
    s.add_argument('--ref', default=REF)
    s.add_argument('--wer_key', default='wer_large_v3')
    s.add_argument('--probe_json', default=None,
                   help='probe_decoder.json of the run, for the noise scale s (default: <dir of --metrics>/../'
                        'probe_decoder.json if it exists; "" for none)')
    return ap.parse_args(argv)


def probe_json_of(metrics):
    """<dir of the first --metrics entry>/../probe_decoder.json, or '' if there is none."""
    m = metrics.split(',')[0]
    p = os.path.normpath(os.path.join(m if os.path.isdir(m) else os.path.dirname(m), '..', 'probe_decoder.json'))
    return p if os.path.isfile(p) else ''


def main(argv=None):
    a = parse(argv)
    if a.cmd == 'summarize':
        pj = probe_json_of(a.metrics) if a.probe_json is None else a.probe_json
        s_dec = json.load(open(pj))['dec_noise_scale'] if pj else None
        res = dict(summarize(load_rows(a.metrics), a.ref, a.wer_key, s_dec), probe_json=pj or None)
        print_summary(res, log=lambda m: print(m, flush=True))
        write_text(a.json, json.dumps(rounded(res), indent=1))
        return 0
    import torch
    import mimi_io
    from transformers import MimiModel
    io = mimi_io.MimiIO(a.device)
    m16 = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=torch.float16).to(a.device).eval()
    run(a, io, Half(m16, a.device), log=lambda m: print(m, flush=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
