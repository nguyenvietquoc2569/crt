#!/usr/bin/env python3
"""Train MimiFlow-VC (model.py) on encoded shards (encode_mimi.py, encode_perturbed.py).

Each step draws speaker-consistent (target, prompt) pairs (data.PairSampler), standardises the latents with corpus
statistics, takes the content tokens from the chosen stream (cb0: the clean CB0; pcb0: the CB0 of a randomly chosen
perturbed copy; mix: either, per example), packs [prompt | target] and minimises the flow loss on target frames.

Run directory: args.json, stats.npz (latent mean / std), train.log (jsonl), last.pt (resume point: model, EMA,
optimiser, step, sampler RNG state is not kept), ema_<step>.pt snapshots, samples/<step>/ decoded conversions of fixed
held-out pairs. Stops after --steps or --max_hours (checkpoint first), so a job that hits its limit resumes.
"""
import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

import data as dt
import model as mdl


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train_dirs', required=True)
    ap.add_argument('--val_dirs', required=True)
    ap.add_argument('--init_ckpt', default='',
                    help='start from the EMA weights of this checkpoint (and its run\'s stats.npz): model and EMA '
                         'loaded, fresh optimiser, step 0; ignored when the run already has last.pt')
    ap.add_argument('--null_noise_prompt', type=float, default=0.0,
                    help='share of condition-dropped rows whose prompt frames become pure noise (N1)')
    ap.add_argument('--train_speakers', type=int, default=0,
                    help='train on this many seeded-random training speakers only (capacity/memorisation probe); 0 = all')
    ap.add_argument('--out', required=True)
    ap.add_argument('--content', choices=('cb0', 'pcb0', 'mix'), default='pcb0')
    ap.add_argument('--K', type=int, default=2, help='perturbed copies per utterance (pcodes0..K-1)')
    ap.add_argument('--d', type=int, default=768)
    ap.add_argument('--layers', type=int, default=12)
    ap.add_argument('--heads', type=int, default=12)
    ap.add_argument('--content_dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--spk', type=int, default=0, help='global speaker-embedding dim (192 = CAM++ .spk.npy), 0 = none')
    ap.add_argument('--pos', choices=('restart', 'continuous', 'segment_random'), default='restart',
                    help='segment_random (v1): per-segment random RoPE offsets in training, fixed offsets at inference')
    ap.add_argument('--sampler', choices=('pair', 'prompt'), default='pair',
                    help='pair: one prompt utterance (scaffold); prompt (v1): multi-utterance 3-60 s prompts, segments')
    ap.add_argument('--tokens', type=int, default=0, help='v1 token budget per batch (prompt + target); 0 = --batch')
    ap.add_argument('--seg_len', type=int, default=125)
    ap.add_argument('--p_cross', type=float, default=0.5)
    ap.add_argument('--sched', choices=('cosine', 'const_tail'), default='cosine')
    ap.add_argument('--p_drop_spk', type=float, default=0.0, help='extra dropout of the speaker vector alone')
    ap.add_argument('--ada_single', action='store_true')
    ap.add_argument('--exclude_utts', default='', help='comma list of utterance-id files (audit_data.py exclude.txt) '
                    'whose utterances are left out of training and validation')
    ap.add_argument('--exclude_pairs', default='', help='comma list of evaluation pair lists (make_pairs.py jsonl); '
                    'training refuses to start if any training or validation speaker appears in them')
    ap.add_argument('--corrupt', type=float, default=0.1,
                    help='share of target content tokens replaced by one of their 8 nearest CB0 codewords')
    ap.add_argument('--tau_dist', choices=('uniform', 'logit_normal'), default='logit_normal')
    ap.add_argument('--whisper', type=int, default=0,
                    help='K of the perturbed Whisper stream (shard.pwhisper<k>.npy, 768-d) to add as content; 0 = none')
    ap.add_argument('--content_init', choices=('random', 'mimi'), default='mimi',
                    help="mimi: the content embedding starts from Mimi's CB0 codewords (needs content_dim 256)")
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--max_target', type=int, default=125)
    ap.add_argument('--prompt_min', type=int, default=38)
    ap.add_argument('--prompt_max', type=int, default=250)
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--ema', type=float, default=0.9995)
    ap.add_argument('--beta2', type=float, default=0.98, help='AdamW beta2 (PLAN 2.6: 0.99; 0.98 kept for older runs)')
    ap.add_argument('--rewarm', type=int, default=0,
                    help='steps of linear learning-rate re-warm from --rewarm_from (continuing a finished run longer)')
    ap.add_argument('--rewarm_from', type=int, default=0)
    ap.add_argument('--p_drop_cond', type=float, default=0.1)
    ap.add_argument('--p_drop_content', type=float, default=0.05)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--ckpt_every', type=int, default=2000)
    ap.add_argument('--snap_every', type=int, default=20000)
    ap.add_argument('--sample_every', type=int, default=10000)
    ap.add_argument('--n_samples', type=int, default=4)
    ap.add_argument('--max_hours', type=float, default=1e9)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--stats_frames', type=int, default=2_000_000)
    return ap.parse_args(argv)


def lr_at(step, a):
    lr = _lr_sched(step, a)
    if getattr(a, 'rewarm', 0) and step >= a.rewarm_from:          # linear re-warm after resuming a longer run
        lr *= min(1.0, (step - a.rewarm_from + 1) / a.rewarm)
    return lr


def _lr_sched(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if getattr(a, 'sched', 'cosine') == 'const_tail':    # constant until 80 % of the run, then cosine to 10 %
        tail0 = int(0.8 * a.steps)
        if step < tail0:
            return a.lr
        p = min(1.0, (step - tail0) / max(1, a.steps - tail0))
    else:
        p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def cb0_neighbours(E0, k=8):
    """[2048, k] indices of each CB0 codeword's k nearest other codewords (Euclidean)."""
    E = np.asarray(E0, dtype=np.float64)
    d = (E ** 2).sum(1)[:, None] + (E ** 2).sum(1)[None] - 2 * E @ E.T
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k].astype(np.int64)


def corpus_stats(c, n_frames, seed=0):
    """Per-dimension mean and std of the latent over up to n_frames frames drawn evenly across utterances."""
    rng = np.random.default_rng(seed)
    per = max(1, n_frames // len(c.utts))
    xs = []
    for i, u in enumerate(c.utts):
        k = min(per, u['frames'])
        idx = np.sort(rng.choice(u['frames'], size=k, replace=False))
        xs.append(c.frames_of(i, 0, u['frames'])['lat'][idx])
    x = np.concatenate(xs).astype(np.float64)
    return x.mean(0).astype(np.float32), np.maximum(x.std(0), 1e-4).astype(np.float32)


def content_tokens(b, side, mode, K, rng, nbr=None, corrupt=0.0):
    """[B, L] content tokens of one side of a numpy batch. The prompt always gets its clean CB0 (as at inference,
    where the reference is natural speech); the target gets the stream of `mode`, then neighbour corruption: a share
    `corrupt` of its tokens replaced by one of their 8 nearest CB0 codewords (nbr [2048, 8])."""
    clean = b[f'{side}_codes'][:, :, 0]
    if side == 'prompt':
        return clean
    tok = _target_tokens(b, clean, mode, K, rng)
    if nbr is not None and corrupt > 0:
        hit = rng.random(tok.shape) < corrupt
        tok = np.where(hit, nbr[tok, rng.integers(0, nbr.shape[1], tok.shape)], tok)
    return tok


def _target_tokens(b, clean, mode, K, rng):
    if mode == 'cb0' or K == 0:
        return clean
    ks = rng.integers(0, K, len(clean))
    pert = np.stack([b[f'target_pcodes{k}'][i, :, 0] for i, k in enumerate(ks)]).astype(np.int64)
    if mode == 'pcb0':
        return pert
    use = rng.random(len(clean)) < 0.5
    return np.where(use[:, None], pert, clean)


def to_device(b, mode, K, rng, mean, std, device, nbr=None, corrupt=0.0):
    """dict(lat, content, region, mask, is_tgt, g, cc, seg): the packed batch, the prompt's speaker embedding (or
    None), the continuous content stream on target frames (one random perturbed Whisper draw per example, or None) and
    the packed segment ids (or None, without a v1 prompt sampler)."""
    t = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
    pl = (b['prompt_lat'] - mean) / std * b['prompt_mask'][..., None]
    tl = (b['target_lat'] - mean) / std * b['target_mask'][..., None]
    pc = content_tokens(b, 'prompt', mode, K, rng)
    tc = content_tokens(b, 'target', mode, K, rng, nbr, corrupt)
    g = t(b['prompt_spk']) if 'prompt_spk' in b else None
    packed = mdl.pack(t(pl), t(b['prompt_mask'], torch.bool), t(tl), t(b['target_mask'], torch.bool),
                      t(pc, torch.long), t(tc, torch.long))
    ws = sorted(k for k in b if k.startswith('target_pwhisper'))
    cc = None
    if ws:
        pick = rng.integers(0, len(ws), len(b['target_mask']))
        arr = np.stack([b[ws[k]][i] for i, k in enumerate(pick)]).astype(np.float32)
        cc = mdl.pack_target_stream(t(arr), packed[4])
    seg = mdl.pack_seg(t(b['prompt_seg'], torch.long), packed[3] & (packed[2] == 0), t(b['target_mask'], torch.bool)) \
        if 'prompt_seg' in b else None
    return dict(zip(('lat', 'content', 'region', 'mask', 'is_tgt'), packed), g=g, cc=cc, seg=seg)


def positions_of(x, mode, train=True):
    """RoPE positions of a packed batch dict; segment_random becomes the fixed 'segment' layout outside training."""
    if mode == 'segment_random' and x['seg'] is not None:
        return mdl.positions(x['region'], x['mask'], mode if train else 'segment', seg=x['seg'])
    return mdl.positions(x['region'], x['mask'], 'restart' if mode == 'segment_random' else mode)


def pos_of(batch, mode):
    """Rotary positions of a packed batch (lat, content, region, mask, is_tgt, g)."""
    return mdl.positions(batch[2], batch[3], mode)


class Prefetch:
    def __init__(self, sampler, batch, depth=4):
        self.q = queue.Queue(depth)
        self.stop = False
        self.th = threading.Thread(target=self._run, args=(sampler, batch), daemon=True)
        self.th.start()

    def _run(self, sampler, batch):
        while not self.stop:
            self.q.put(batch(sampler) if callable(batch) else sampler.batch(batch))

    def get(self):
        return self.q.get()


def save_atomic(obj, path):
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def val_loss(model, batches, device, taus=(0.1, 0.3, 0.5, 0.7, 0.9), pos_mode='restart'):
    """Flow loss on fixed held-out batches at fixed tau with fixed noise: a comparable curve across steps."""
    model.eval()
    tot, n = 0.0, 0
    for x_ in batches:
        lat, content, region, mask, is_tgt, gg, cc = (x_[k] for k in ('lat', 'content', 'region', 'mask', 'is_tgt', 'g', 'cc'))
        pos = positions_of(x_, pos_mode, train=False)
        g = torch.Generator(device=device).manual_seed(1234)
        for tau in taus:
            noise = torch.randn(lat.shape, generator=g, device=device)
            tt = torch.full((lat.shape[0],), tau, device=device)
            x = (1 - tau) * noise + tau * lat
            cond = lat * (~is_tgt & mask)[..., None]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                v = model(x, cond, content, region, mask, tt, gg, pos, cc)
            err = ((v.float() - (lat - noise)) ** 2).mean(-1)
            tot += float((err * is_tgt).sum() / is_tgt.sum())
            n += 1
    model.train()
    return tot / max(1, n)


def build(a):
    return mdl.MimiFlow(d=a.d, layers=a.layers, heads=a.heads, content_dim=a.content_dim, dropout=a.dropout,
                        spk_dim=getattr(a, 'spk', 0), cont_dim=768 if getattr(a, 'whisper', 0) else 0,
                        ada_single=getattr(a, 'ada_single', False))


ALLOWED = ('train-clean-100', 'train-clean-360', 'train-other-500', 'dev-clean', 'dev-other')


def check_no_test_data(dirs):
    """Refuse shard dirs that are not LibriTTS-R train-* / dev-* subsets: VCTK (Dev, VCTK-2k) and LibriTTS-R
    test-clean (LTC-2k) are the evaluation sets and must never be trained or validated on."""
    for d in dirs:
        name = Path(d).name
        if name not in ALLOWED or 'vctk' in str(d).lower() or 'test' in name:
            raise SystemExit(f'refusing {d}: only LibriTTS-R {ALLOWED} may be used for training or validation')


def check_speakers(tr, va, exclude_pairs, log=print):
    """No evaluation speaker (source or target of any listed pair) may be a training or validation speaker, and
    training and validation speakers must be disjoint."""
    trs = {u['speaker'] for u in tr.utts}
    vas = {u['speaker'] for u in va.utts}
    if trs & vas:
        raise SystemExit(f'training and validation share speakers {sorted(trs & vas)[:5]}')
    ev = set()
    for p in [x for x in exclude_pairs.split(',') if x]:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                ev |= {str(r['source_speaker']), str(r['target_speaker'])}
    bad = (trs | vas) & ev
    if bad:
        raise SystemExit(f'evaluation speakers in the training data: {sorted(bad)[:10]}')
    if ev:
        log(f'contamination check: {len(trs)} training / {len(vas)} validation speakers, none of {len(ev)} evaluation speakers')


def train(a, mimi_io=None, log=print):
    check_no_test_data(a.train_dirs.split(',') + a.val_dirs.split(','))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device
    excl, audited = set(), set()
    for p in [x for x in a.exclude_utts.split(',') if x]:
        b = Path(p).read_bytes()
        excl |= {i for i in b.decode().split() if i}
        audited |= set(json.load(open(Path(p).parent / 'audit_summary.json'))['audited_shards'])
        log(f'exclusion list {p}: sha256 {hashlib.sha256(b).hexdigest()[:16]}')
    tr = dt.Corpus(a.train_dirs.split(','), exclude=excl)
    if a.train_speakers:
        keep = sorted(np.random.default_rng(a.seed).choice(tr.speakers, a.train_speakers, replace=False).tolist())
        tr = dt.Corpus(a.train_dirs.split(','), exclude=excl, speakers=set(keep))
        log(f'training speakers limited to {a.train_speakers}: {" ".join(keep)}')
    va = dt.Corpus(a.val_dirs.split(','), exclude=excl)
    if a.exclude_utts:
        for c in (tr, va):
            for sh in c.shards:
                key = f"{Path(sh['base']).parent.name}/{Path(sh['base']).name}"
                if key not in audited:
                    raise SystemExit(f'shard {key} is not covered by the audit behind {a.exclude_utts}')
        log(f'excluded by the audit: {tr.n_excluded} training and {va.n_excluded} validation utterances')
    streams = [f'pcodes{k}' for k in range(a.K)] if a.content != 'cb0' else []
    streams += [f'pwhisper{k}' for k in range(a.whisper)]
    for c in (tr, va):
        for s in streams:
            c.add_stream(s, f'.{s}.npy')
        if a.spk:
            c.add_utt_stream('spk', '.spk.npy')
    check_speakers(tr, va, a.exclude_pairs, log)
    log(f'train: {len(tr.utts)} utts, {len(tr.speakers)} speakers, {tr.hours():.1f} h; val: {len(va.utts)} utts, '
        f'{len(va.speakers)} speakers')
    sp = out / 'stats.npz'
    if a.init_ckpt and not sp.exists():                  # keep the standardisation the initial weights were trained on
        import shutil
        shutil.copy(Path(a.init_ckpt).parent / 'stats.npz', sp)
    if sp.exists():
        st = np.load(sp)
        mean, std = st['mean'], st['std']
    else:
        mean, std = corpus_stats(tr, a.stats_frames, a.seed)
        np.savez(sp, mean=mean, std=std)
    model = build(a).to(dev)
    nbr = None
    if mimi_io is not None:
        E0 = mimi_io.cb0_table()
        nbr = cb0_neighbours(E0)
        if a.content_init == 'mimi' and a.content_dim == 256 and not (out / 'last.pt').exists():
            with torch.no_grad():
                model.content.weight[:mdl.VOCAB] = torch.as_tensor(E0, device=dev)
            log("content embedding initialised from Mimi's CB0 codewords")
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, getattr(a, 'beta2', 0.98)), weight_decay=a.wd)
    ema = mdl.EMA(model, a.ema)
    step = 0
    last = out / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        ema.shadow = {k: v.to(dev) for k, v in ck['ema'].items()}
        step = ck['step']
        log(f'resumed from step {step}')
    elif a.init_ckpt:
        ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ck['ema'].items()})
        ema.shadow = {k: v.detach().clone().float().to(dev) for k, v in ck['ema'].items()}
        log(f'initialised from {a.init_ckpt} (its step {ck.get("step")}), fresh optimiser at step 0')
    log(f'model: {n_par / 1e6:.1f}M parameters, content {a.content}, device {dev}')
    if a.sampler == 'prompt':
        mk = lambda c, seed: dt.PromptSampler(c, a.max_target, seg_len=a.seg_len, p_cross=a.p_cross,  # noqa: E731
                                              streams=streams, seed=seed)
    else:
        mk = lambda c, seed: dt.PairSampler(c, a.max_target, (a.prompt_min, a.prompt_max), streams, seed=seed)  # noqa: E731
    smp, vsmp = mk(tr, a.seed + step), mk(va, 12345)
    vrng = np.random.default_rng(54321)
    vbatches = [to_device(vsmp.batch(16), a.content, a.K, vrng, mean, std, dev) for _ in range(4)]
    pairs = fixed_pairs(va, a.n_samples, seed=7)
    pf = Prefetch(smp, (lambda sm: sm.batch_tokens(a.tokens)) if a.tokens and a.sampler == 'prompt' else a.batch)
    t0, tl, nl, frames = time.time(), 0.0, 0, 0
    wall0 = time.time()
    logf = open(out / 'train.log', 'a')

    def checkpoint():
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), ema=ema.shadow, step=step, args=vars(a)), last)

    model.train()
    wait = 0.0
    while step < a.steps:
        tw = time.time()
        b = pf.get()
        wait += time.time() - tw
        x_ = to_device(b, a.content, a.K, rng, mean, std, dev, nbr, a.corrupt)
        lat, content, region, mask, is_tgt, gs, cc = (x_[k] for k in ('lat', 'content', 'region', 'mask', 'is_tgt', 'g', 'cc'))
        pos = positions_of(x_, a.pos, train=True)
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step, a)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            loss = mdl.flow_loss(model, lat, content, region, mask, is_tgt, a.p_drop_cond, a.p_drop_content, g=gs, pos=pos,
                                 tau_dist=a.tau_dist, cc=cc, p_drop_spk=a.p_drop_spk,
                                 null_noise_prompt=getattr(a, 'null_noise_prompt', 0.0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        lv = float(loss.detach())
        if not math.isfinite(lv) or not math.isfinite(gn):
            raise SystemExit(f'non-finite loss or gradient at step {step}: loss {lv}, grad norm {gn}')
        opt.step()
        ema.update(model)
        step += 1
        tl += lv
        nl += 1
        frames += int(is_tgt.sum())
        if step % a.log_every == 0:
            dt_s = time.time() - t0
            rec = dict(step=step, loss=tl / nl, lr=lr_at(step, a), grad_norm=gn, steps_per_s=nl / dt_s,
                       target_frames_per_s=frames / dt_s, hours=(time.time() - wall0) / 3600, batch=int(lat.shape[0]),
                       seq=int(lat.shape[1]), data_wait_s=round(wait, 3))
            if dev == 'cuda':
                rec['peak_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            wait = 0.0
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            log(json.dumps(rec))
            t0, tl, nl, frames = time.time(), 0.0, 0, 0
        if step % a.val_every == 0 or step == a.steps:
            ev = build(a).to(dev)
            ema.copy_to(ev)
            rec = dict(step=step, val_loss_ema=val_loss(ev, vbatches, dev, pos_mode=a.pos),
                       val_loss=val_loss(model, vbatches, dev, pos_mode=a.pos))
            logf.write(json.dumps(rec) + '\n')
            logf.flush()
            log(json.dumps(rec))
            if mimi_io is not None and (step % a.sample_every == 0 or step == a.steps):
                write_samples(ev, mimi_io, pairs, va, mean, std, out / 'samples' / f'{step:07d}', dev, a.pos)
            del ev
        if step % a.ckpt_every == 0 or step == a.steps:
            checkpoint()
        if step % a.snap_every == 0:
            save_atomic(dict(ema=ema.shadow, step=step, args=vars(a)), out / f'ema_{step:07d}.pt')
        if (time.time() - wall0) / 3600 > a.max_hours:
            checkpoint()
            log(f'time budget reached at step {step}')
            break
    pf.stop = True
    checkpoint()
    save_atomic(dict(ema=ema.shadow, step=step, args=vars(a)), out / 'ema_last.pt')
    log(f'done at step {step}')
    return step


def fixed_pairs(c, n, seed=7):
    """n cross-speaker (source utterance, prompt utterance) pairs of the held-out corpus."""
    rng = np.random.default_rng(seed)
    spk = c.speakers
    out = []
    for i in range(n):
        a, b = rng.choice(len(spk), size=2, replace=False)
        out.append((int(rng.choice(c.by_speaker[spk[a]])), int(rng.choice(c.by_speaker[spk[b]]))))
    return out


@torch.no_grad()
def convert_latents(model, src_codes, prompt_lat, prompt_codes, mean, std, device, steps=16, cfg=0.7, sway=-1.0,
                    seed=0, spk=None, pos_mode='restart', src_cont=None, seg_len=125, chunks=None, solver='euler',
                    temp=1.0, cfg_rescale=0.0, sharpen=0.0, gv=0.0, return_std=False, init_lat=None, t_start=0.0,
                    cfg_branch='std', neg_lat=None, neg_codes=None, neg_spk=None, w_neg=0.0, neg_until=1.0):
    """Content from src_codes[:, 0], voice from the prompt (lat, codes) and the speaker embedding spk (when the model
    uses one) -> converted latent [T, 512] (un-standardised). chunks: a list of (lat, codes) prompt chunks whose
    velocities are averaged at every step (chunk ensemble); by default the single chunk (prompt_lat, prompt_codes).
    Prompt frames get segment ids of seg_len frames for the 'segment' positions of v1 models. solver, temp and
    cfg_rescale go to the sampler; sharpen and gv post-process the standardised output (model.postprocess, gv toward
    the prompt frames). return_std: also return the standardised output (for diagnostics). init_lat [T, 512]
    (un-standardised) with t_start > 0: start the target from init_lat noised to t_start (SDEdit probe).
    w_neg > 0 (source-negative guidance): neg_lat / neg_codes (the source's own latent and codes, as a one-chunk prompt) and
    neg_spk (its speaker vector) form the negative branch of model.sample_ensemble, applied while t < neg_until."""
    prompts = chunks or [(prompt_lat, prompt_codes)]
    E, T = len(prompts), len(src_codes)
    P = max(len(pl) for pl, _ in prompts)
    pl = np.zeros((E, P, mdl.LAT), np.float32)
    pm = np.zeros((E, P), bool)
    pc = np.zeros((E, P), np.int64)
    ps = np.full((E, P), -1, np.int64)
    for e, (l, c) in enumerate(prompts):
        n = len(l)
        pl[e, :n] = (np.asarray(l) - mean) / std
        pm[e, :n] = True
        pc[e, :n] = np.asarray(c)[:, 0]
        ps[e, :n] = np.arange(n) // seg_len + 1
    t = lambda x, dtype=torch.float32: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)  # noqa: E731
    tm = torch.ones(E, T, dtype=torch.bool, device=device)
    tc = t(np.repeat(np.asarray(src_codes)[None, :, 0], E, 0), torch.long)
    tl = torch.zeros(E, T, mdl.LAT, device=device) if init_lat is None else \
        t(np.repeat(((np.asarray(init_lat) - mean) / std)[None], E, 0))
    lat, content, region, mask, is_tgt = mdl.pack(t(pl), t(pm, torch.bool), tl, tm, t(pc, torch.long), tc)
    seg = mdl.pack_seg(t(ps, torch.long), t(pm, torch.bool), tm)
    pos = mdl.positions(region, mask, 'segment', seg=seg) if pos_mode == 'segment_random' \
        else mdl.positions(region, mask, pos_mode)
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        gg = None if spk is None else t(np.repeat(np.asarray(spk, dtype=np.float32)[None], E, 0))
        cc = None if src_cont is None else mdl.pack_target_stream(
            t(np.repeat(np.asarray(src_cont, dtype=np.float32)[None], E, 0)), is_tgt)
        neg = None
        if w_neg:
            n = len(neg_lat)
            nl = t(((np.asarray(neg_lat) - mean) / std)[None])
            nm = torch.ones(1, n, dtype=torch.bool, device=device)
            pk = mdl.pack(nl, nm, torch.zeros(1, T, mdl.LAT, device=device), tm[:1], t(np.asarray(neg_codes)[None, :, 0], torch.long),
                          tc[:1])
            nseg = mdl.pack_seg(t((np.arange(n) // seg_len + 1)[None], torch.long), nm, tm[:1])
            npos = mdl.positions(pk[2], pk[3], 'segment', seg=nseg) if pos_mode == 'segment_random' \
                else mdl.positions(pk[2], pk[3], pos_mode)
            neg = dict(cond_lat=pk[0], content=pk[1], region=pk[2], mask=pk[3], is_tgt=pk[4], pos=npos,
                       g=None if gg is None else t(np.asarray(neg_spk, dtype=np.float32)[None]),
                       cc=None if cc is None else mdl.pack_target_stream(cc[:1][is_tgt[:1]][None], pk[4]))
        x = mdl.sample_ensemble(model, lat, content, region, mask, is_tgt, steps=steps, cfg=cfg, sway=sway, generator=g,
                                g=gg, pos=pos, cc=cc, groups=[list(range(E))], solver=solver, temp=temp,
                                cfg_rescale=cfg_rescale, x_init=None if init_lat is None else lat, t_start=t_start,
                                cfg_branch=cfg_branch, neg=neg, w_neg=w_neg, neg_until=neg_until)
    z = x[0][is_tgt[0]].float().cpu().numpy()
    if sharpen or gv:
        z = mdl.postprocess(z, np.concatenate([pl[e, :len(l)] for e, (l, _) in enumerate(prompts)]), sharpen, gv)
    out = z * std + mean
    return (out, z) if return_std else out


def write_samples(model, mimi_io, pairs, c, mean, std, d, device, pos_mode='restart'):
    import soundfile as sf
    d.mkdir(parents=True, exist_ok=True)
    model.eval()
    wstreams = [x for x in c.stream_names if x.startswith('pwhisper')][:1]
    for i, (si, pi) in enumerate(pairs):
        s = c.frames_of(si, 0, c.utts[si]['frames'], wstreams)
        p = c.frames_of(pi, 0, min(c.utts[pi]['frames'], 250))
        spk = c.utt_value(pi, 'spk') if 'spk' in c.utt_streams else None
        x = convert_latents(model, s['codes'], p['lat'], p['codes'], mean, std, device, spk=spk, pos_mode=pos_mode,
                            src_cont=s[wstreams[0]] if wstreams else None)
        sf.write(d / f'{i}_conv.wav', mimi_io.decode_latent(x), 24000)
        sf.write(d / f'{i}_src_copy.wav', mimi_io.decode_latent(s['lat']), 24000)
        sf.write(d / f'{i}_prompt_copy.wav', mimi_io.decode_latent(p['lat']), 24000)
    (d / 'pairs.json').write_text(json.dumps([dict(source=c.utts[si]['id'], prompt=c.utts[pi]['id']) for si, pi in pairs]))


def main(argv=None):
    a = parse(argv)
    mio = None
    if a.device == 'cuda' and a.n_samples:
        import mimi_io
        mio = mimi_io.MimiIO('cuda')
    train(a, mio, log=lambda m: print(m, flush=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
