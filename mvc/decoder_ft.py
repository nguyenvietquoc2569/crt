#!/usr/bin/env python3
"""R5 (PLAN.md 2.8): fine-tune Mimi's decoder (upsample, decoder transformer, SEANet decoder) to turn the latents
MimiFlow-VC generates into natural speech, as HiFi-GAN is fine-tuned on Tacotron's ground-truth-aligned mels.

Training pairs are frame-aligned: gta_latents.py converts training utterances in reconstruction mode (the source is
the target's own utterance, the prompt other utterances of the same speaker), so the generated latent of frame t
belongs with the real audio samples [1920 t, 1920 (t + 1)). With probability p_real an example uses the real stored
latent instead, so decoding real latents does not get worse. The quantiser projections that map a latent to the
decoder input stay frozen; the stock decoder stays in use for the Q32 (codes) route.

Loss (EnCodec / Mimi recipe): multi-scale log-mel L1 (n_fft 512, 1024, 2048) + 3 * LSGAN generator loss + 3 * feature
matching, against a multi-scale STFT discriminator (5 scales). The discriminator alone trains for the first d_warmup
steps. Each example decodes window + 2 * margin frames and is scored on the central window only.

R5b options (26 Oct, after local probes showed paired losses on free-running GTA latents lower UTMOS by 0.5 because
their pitch differs from the real audio): --pair_losses real applies mel and feature matching to real-latent rows only;
--w_anchor adds, on GTA rows, a 24-band log-mel L1 to the frozen stock decoder's output for the same latent (pitch-
aligned by construction, 0 at the start); --prefix decodes from the utterance start to the window end (the decoder
is causal, so no right margin is needed, and a 4-frame left context departs from the full decode by -8.6 dB); the
SEANet decoder then runs on the window plus 4 frames only (exact); --cache_wav keeps the read audio in memory.

    python decoder_ft.py --gta .../mvc/gta/p6_V --shards .../mvc/mimi/train-clean-100 --raw .../LibriTTS_R \
        --out .../mvc/decoder/p6_V --steps 30000
The checkpoint (<out>/decoder_last.pt, decoder_<step>.pt) holds the three decoder modules' state dicts; mimi_io.MimiIO
(decoder_ckpt=...) loads it for decode_latent only.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HOP = 1920
SR = 24000


# ── losses and discriminator ─────────────────────────────

def mel_bank(sr, n_fft, n_mels):
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2)


class MultiMel:
    """Multi-scale log-mel L1 (+ L2 on the linear mel), the EnCodec reconstruction term."""

    def __init__(self, torch, device, scales=((512, 64), (1024, 80), (2048, 128))):
        self.torch, self.scales = torch, []
        for n_fft, n_mels in scales:
            fb = torch.as_tensor(mel_bank(SR, n_fft, n_mels), dtype=torch.float32, device=device)
            self.scales.append((n_fft, fb, torch.hann_window(n_fft, device=device)))

    def __call__(self, y_hat, y):
        torch = self.torch
        loss = 0.0
        for n_fft, fb, win in self.scales:
            def mel(x):
                s = torch.stft(x.float(), n_fft, n_fft // 4, window=win, return_complex=True).abs()
                return fb @ s
            a, b = mel(y_hat), mel(y)
            loss = loss + (torch.log(a + 1e-5) - torch.log(b + 1e-5)).abs().mean() + ((a - b) ** 2).mean()
        return loss / len(self.scales)


def make_discriminator(torch):
    nn = torch.nn

    class STFTDisc(nn.Module):
        def __init__(self, n_fft, ch=32):
            super().__init__()
            self.n_fft = n_fft
            self.register_buffer('win', torch.hann_window(n_fft), persistent=False)
            k, d = (3, 9), (1, 2, 4)
            layers = [nn.Conv2d(2, ch, (3, 9), padding=(1, 4))]
            for dil in d:
                layers.append(nn.Conv2d(ch, ch, k, stride=(1, 2), dilation=(dil, 1), padding=(dil, 4)))
            layers.append(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)))
            self.convs = nn.ModuleList([nn.utils.parametrizations.weight_norm(c) for c in layers])
            self.out = nn.utils.parametrizations.weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

        def forward(self, x):
            s = torch.stft(x.float(), self.n_fft, self.n_fft // 4, window=self.win, return_complex=True)
            h = torch.stack([s.real, s.imag], 1).transpose(2, 3)       # [B, 2, frames, bins]
            feats = []
            for c in self.convs:
                h = torch.nn.functional.leaky_relu(c(h), 0.2)
                feats.append(h)
            return self.out(h), feats

    class MSSTFTDisc(nn.Module):
        def __init__(self, ffts=(2048, 1024, 512, 256, 128)):
            super().__init__()
            self.ds = nn.ModuleList([STFTDisc(n) for n in ffts])

        def forward(self, x):
            return [d(x) for d in self.ds]

    return MSSTFTDisc()


def d_loss_fn(disc, y_hat, y):
    """LSGAN discriminator loss on a detached generator output."""
    real, fake = disc(y), disc(y_hat.detach())
    return sum(((1 - r) ** 2).mean() + (f ** 2).mean() for (r, _), (f, _) in zip(real, fake)) / len(real)


def g_loss_fn(torch, disc, y_hat, y, fm_rows=None):
    """(LSGAN generator loss, relative feature-matching L1) against the current discriminator. fm_rows: a boolean mask
    of the batch rows that feature matching applies to (None = all rows; no row = 0)."""
    with torch.no_grad():
        real = disc(y)
    fake = disc(y_hat)
    adv = sum(((1 - f) ** 2).mean() for f, _ in fake) / len(fake)
    if fm_rows is not None and not bool(fm_rows.any()):
        return adv, y_hat.sum() * 0.0
    sel = (lambda t: t) if fm_rows is None else (lambda t: t[fm_rows])
    fm = sum(sum((sel(a) - sel(b)).abs().mean() / (sel(b).abs().mean() + 1e-6) for a, b in zip(ff, rf)) / len(ff)
             for (_, ff), (_, rf) in zip(fake, real)) / len(fake)
    return adv, fm


class Anchor:
    """24-band log-mel L1 at n_fft 1024 between two signals (the R5b envelope anchor to the stock decoder)."""

    def __init__(self, torch, device, n_fft=1024, n_mels=24):
        self.torch, self.n_fft = torch, n_fft
        self.fb = torch.as_tensor(mel_bank(SR, n_fft, n_mels), dtype=torch.float32, device=device)
        self.win = torch.hann_window(n_fft, device=device)

    def __call__(self, a, b):
        torch = self.torch
        mel = lambda x: self.fb @ torch.stft(x.float(), self.n_fft, self.n_fft // 4, window=self.win,  # noqa: E731
                                             return_complex=True).abs()
        return (torch.log(mel(a) + 1e-5) - torch.log(mel(b) + 1e-5)).abs().mean()


# ── data ─────────────────────────────────────────────────

class Pairs:
    """Aligned (generated latent, real latent, wav) examples: the GTA npz files of gta_latents.py and the shards they
    came from. sample() returns a window of `frames` frames (+ margins where the utterance allows)."""

    def __init__(self, gta_dir, shard_dirs, raw, reader=None, cache_wav=False):
        import encode_mimi as em
        import encode_perturbed as ep
        self.reader = reader or em.read_wav
        self.cache = {} if cache_wav else None
        rows = {}
        for d in shard_dirs:
            for js in sorted(Path(d).glob('shard_*.json')):
                base = str(js)[:-5]
                for r in json.load(open(js)):
                    rows[r['id']] = (base, r)
        self.items, self.lat = [], {}
        for f in sorted(Path(gta_dir).glob('gta_*.npz')):
            z = np.load(f)
            for uid, off, n in zip(z['ids'], z['offsets'], z['frames']):
                uid = str(uid)
                if uid in rows:
                    base, r = rows[uid]
                    self.items.append(dict(id=uid, base=base, row=r, gta=(str(f), int(off), int(n)),
                                           wav=ep.wav_path(raw, r)))
        if not self.items:
            raise ValueError(f'no GTA utterances under {gta_dir} match the shards')
        self._gta, self._real = {}, {}

    def _gta_arr(self, f):
        if f not in self._gta:
            self._gta[f] = np.load(f)['lat']
        return self._gta[f]

    def _real_arr(self, base):
        if base not in self._real:
            self._real[base] = np.load(base + '.lat.npy', mmap_mode='r')
        return self._real[base]

    def wav(self, path):
        if self.cache is None:
            return self.reader(path)
        if path not in self.cache:
            self.cache[path] = np.round(np.clip(self.reader(path), -1, 1) * 32767).astype(np.int16)
        return self.cache[path].astype(np.float32) / 32767

    def sample(self, rng, frames, margin, p_real=0.5, prefix=False, with_real=False):
        """(latent window, audio of the scored frames, index of the first scored frame in the window[, is_real]).
        prefix: the latent window starts at the utterance start and ends at the scored window (no right margin)."""
        while True:
            it = self.items[rng.integers(len(self.items))]
            f, off, n = it['gta']
            if n >= frames:
                break
        a = int(rng.integers(0, n - frames + 1))
        lo, hi = (0, a + frames) if prefix else (max(0, a - margin), min(n, a + frames + margin))
        is_real = bool(rng.random() < p_real)
        if is_real:
            r = it['row']
            lat = np.asarray(self._real_arr(it['base'])[r['offset'] + lo:r['offset'] + hi], dtype=np.float32)
        else:
            lat = np.asarray(self._gta_arr(f)[off + lo:off + hi], dtype=np.float32)
        wav = self.wav(it['wav'])
        y = np.zeros(frames * HOP, np.float32)
        seg = wav[a * HOP:(a + frames) * HOP]
        y[:len(seg)] = seg
        return (lat, y, a - lo, is_real) if with_real else (lat, y, a - lo)


# ── model ────────────────────────────────────────────────

def decode_windows(torch, io, mods, xs, starts, frames, prefix=False):
    """Decode each latent window x with the decoder modules mods and return the audio of its scored frames
    [s0, s0 + frames) as [B, frames * HOP]. prefix: x starts at the utterance start and ends at the scored window; the
    decoder is causal, so upsample + transformer run on the whole prefix and SEANet on the window plus 4 frames."""
    outs = []
    for x, s0 in zip(xs, starts):
        emb = io.decoder_input(x)
        e = mods['upsample'](emb)
        h = mods['decoder_transformer'](e.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
        if prefix:
            h0 = 2 * max(0, s0 - 4)
            w = mods['decoder'](h[..., h0:2 * (s0 + frames)])[0, 0]
            outs.append(w[len(w) - frames * HOP:])
        else:
            w = mods['decoder'](h)[0, 0]
            outs.append(w[s0 * HOP:(s0 + frames) * HOP])
    return torch.stack([torch.nn.functional.pad(o, (0, frames * HOP - len(o))) for o in outs])


def train(a, io=None, log=print):
    import torch
    import mimi_io
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device
    io = io or mimi_io.MimiIO(dev)
    m = io.m
    gen = torch.nn.ModuleDict(dict(upsample=m.upsample, decoder_transformer=m.decoder_transformer, decoder=m.decoder))
    for p in m.parameters():
        p.requires_grad_(False)
    for p in gen.parameters():
        p.requires_grad_(True)
    disc = make_discriminator(torch).to(dev)
    mel = MultiMel(torch, dev)
    anchor = Anchor(torch, dev) if a.w_anchor else None
    stock = None
    if anchor is not None:                                   # a frozen copy of the stock decoder for the anchor targets
        import copy
        stock = copy.deepcopy(gen).eval()
        for p in stock.parameters():
            p.requires_grad_(False)
    og = torch.optim.AdamW(gen.parameters(), lr=a.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(disc.parameters(), lr=a.lr_d, betas=(0.8, 0.99))
    data = Pairs(a.gta, a.shards.split(','), a.raw, reader=a.reader if hasattr(a, 'reader') else None,
                 cache_wav=a.cache_wav)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'args.json').write_text(json.dumps({k: v for k, v in vars(a).items() if k != 'reader'}, indent=1))
    log(f'{len(data.items)} GTA utterances; decoder {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params, '
        f'discriminator {sum(p.numel() for p in disc.parameters()) / 1e6:.1f}M')

    def batch():
        xs, ys, starts, real = [], [], [], []
        for _ in range(a.batch):
            lat, y, s0, r = data.sample(rng, a.frames, a.margin, a.p_real, prefix=a.prefix, with_real=True)
            xs.append(lat)
            ys.append(y)
            starts.append(s0)
            real.append(r)
        return xs, torch.as_tensor(np.stack(ys), device=dev), starts, torch.as_tensor(real, device=dev)

    def decode(xs, starts, mods=None):
        return decode_windows(torch, io, mods or gen, xs, starts, a.frames, a.prefix)

    t0, rec = time.time(), {}
    for step in range(1, a.steps + 1):
        xs, y, starts, real = batch()
        warm = step <= a.d_warmup
        with torch.set_grad_enabled(not warm):
            y_hat = decode(xs, starts)
        d_loss = d_loss_fn(disc, y_hat, y)
        od.zero_grad()
        d_loss.backward()
        od.step()
        if not warm:
            pair = real if a.pair_losses == 'real' else torch.ones_like(real)
            l_mel = mel(y_hat[pair], y[pair]) if bool(pair.any()) else y_hat.sum() * 0.0
            g_adv, fm = g_loss_fn(torch, disc, y_hat, y, fm_rows=None if a.pair_losses == 'all' else pair)
            loss = a.w_mel * l_mel + a.w_adv * g_adv + a.w_fm * fm
            rec = dict(mel=float(l_mel), adv=float(g_adv), fm=float(fm))
            if anchor is not None and bool((~real).any()):
                g = ~real
                with torch.no_grad():
                    ref = decode([x for x, k in zip(xs, g.tolist()) if k], [s for s, k in zip(starts, g.tolist()) if k],
                                 stock)
                l_anc = anchor(y_hat[g], ref)
                loss = loss + a.w_anchor * l_anc
                rec['anchor'] = float(l_anc)
            og.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
            og.step()
        if step % a.log_every == 0:
            r = dict(step=step, d=float(d_loss), warm=warm, steps_per_s=round(a.log_every / (time.time() - t0), 3), **rec)
            log(json.dumps(r))
            with open(out / 'train.log', 'a') as f:
                f.write(json.dumps(r) + '\n')
            t0 = time.time()
        if step % a.ckpt_every == 0 or step == a.steps:
            sd = {k: v.state_dict() for k, v in gen.items()}
            torch.save(dict(step=step, decoder=sd), out / f'decoder_{step:06d}.pt')
            torch.save(dict(step=step, decoder=sd), out / 'decoder_last.pt')
    return a.steps


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gta', required=True, help='directory of gta_*.npz written by gta_latents.py')
    ap.add_argument('--shards', required=True, help='comma list of the shard directories the GTA utterances come from')
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--frames', type=int, default=32, help='scored window, frames (32 = 2.56 s)')
    ap.add_argument('--margin', type=int, default=4, help='context frames decoded on each side and not scored')
    ap.add_argument('--p_real', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--lr_d', type=float, default=1e-4)
    ap.add_argument('--w_mel', type=float, default=1.0)
    ap.add_argument('--w_adv', type=float, default=3.0)
    ap.add_argument('--w_fm', type=float, default=3.0)
    ap.add_argument('--d_warmup', type=int, default=1000)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--pair_losses', choices=('all', 'real'), default='all',
                    help='rows that get the paired mel and feature-matching losses (R5b: real)')
    ap.add_argument('--w_anchor', type=float, default=0.0, help='GTA-row log-mel anchor to the frozen stock decoder')
    ap.add_argument('--prefix', action='store_true', help='decode from the utterance start (causal, no right margin)')
    ap.add_argument('--cache_wav', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    return ap.parse_args(argv)


if __name__ == '__main__':
    train(parse())
