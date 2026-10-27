"""Mimi-VC behind the baselines adapter contract (benchmark/baselines/__init__.py): SR, load, convert, describe.

convert(src, ref, sr): the source's Mimi codes give the content tokens (CB0; optionally of a perturbed copy), the
reference's Mimi latents give the prompt (its first prompt_max frames, the reference encoded in 30 s chunks as the
published pipeline does), the flow model generates the target latent, and Mimi decodes it (unquantised path, or the
re-quantised codes with decode='codes'). Reference features are cached per reference array (the harness passes the
same object for every pair of a target).
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SR = 24000
CHUNK_S = 30
DEFAULTS = dict(run_dir=None, ckpt='ema_last.pt', steps=16, cfg=0.7, sway=-1.0, prompt_max=None, prompt_chunk=750,
                perturb_src=False, decode='latent', seed_from_pair=True, mode='convert', whisper_src='clean',
                solver='euler', temp=1.0, cfg_rescale=0.0, sharpen=0.0, gv=0.0, stats_path=None, decoder_ckpt=None,
                init_t=0.0, match_rms=False, relax=0.0, relax_k=2, cfg_branch='std', neg_src=0.0, neg_src_until=1.0)
# Naturalness knobs (25 Oct, after P6): solver/temp/cfg_rescale act in the sampler (model.sample_ensemble), sharpen/gv
# post-process the standardised output (model.postprocess). stats_path: append one JSON line per conversion with the
# smoothness (model.smoothness) and per-dimension spread of the generated, source and prompt latents. decoder_ckpt: a
# decoder_ft.py checkpoint (R5) used for decode='latent' and the copy_latent / knn / ot controls. init_t > 0 (SDEdit
# probe, meant for same-speaker reconstruction): start the ODE from the source's own latent noised to init_t.
# match_rms: scale the converted waveform to the source's RMS level (loudness control for UTMOS comparisons).
# relax (0..1) and relax_k: the damped Mimi round trip MimiIO.relax on the converted latent (convert mode only, after
# the stats record), so the latent route, the codes route and gta_latents.py all see it.
# neg_src (>= 0): source-negative guidance. Each ODE step also runs the model with the source's own clean latent as the
# prompt and the source's speaker vector (windowed as the reference's), and adds neg_src (v_cond - v_source) to the
# velocity, steering the output away from the voice the model would give the source (model.sample_ensemble);
# neg_src_until < 1 applies it only at the steps with t < neg_src_until (guidance interval).
# Prompt use follows the run: a pair-sampler run (scaffold, prompts of at most 20 s in training) gets the first
# prompt_max (default 250) frames of the reference; a v1 run (multi-utterance prompts up to 60 s) gets the whole
# reference, cut into chunks of at most prompt_chunk frames whose velocities are averaged (180 s = three 60 s chunks).
# The speaker vector of v1 runs is the renormalised mean of the L2-normalised CAM++ embeddings of 10 s windows.
# mode: 'convert' (the VC model), or modes that need no model: the decode controls 'copy_latent' (the source's own
# latent through the unquantised path) and 'copy_codes' (its 32 codes through MimiModel.decode), and the training-free
# Mimi floors 'knn_latent' (each source frame -> mean of its 4 cosine-nearest reference latent frames) and 'ot_latent'
# (Gaussian optimal transport of the source latent distribution onto the reference's).


class MimiVC:
    SR = SR

    def __init__(self):
        self.cfg, self.ready = dict(DEFAULTS), False
        self.ref_cache = {}

    def load(self, device, **cfg):
        bad = set(cfg) - set(DEFAULTS)
        if bad:
            raise ValueError(f'unknown settings {sorted(bad)}')
        if not 0.0 <= float(cfg.get('relax', 0.0)) <= 1.0:
            raise ValueError('relax must be in [0, 1]')
        if float(cfg.get('neg_src', 0.0)) < 0:
            raise ValueError('neg_src must be >= 0')
        self.cfg.update(cfg)
        import torch
        import mimi_io
        import train
        self.torch, self.device = torch, device
        if self.cfg['mode'] != 'convert':
            self.io, self.ckpt_path, self.step, self.train_args, self.spk = \
                mimi_io.MimiIO(device, decoder_ckpt=self.cfg['decoder_ckpt']), None, None, {}, None
            self.ready = True
            return
        if not self.cfg['run_dir']:
            raise ValueError('run_dir is required')
        run = Path(self.cfg['run_dir'])
        self.ckpt_path = str(run / self.cfg['ckpt'])
        ck = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        a = train.parse(['--train_dirs', 'x', '--val_dirs', 'x', '--out', str(run)])
        for k, v in ck['args'].items():
            setattr(a, k, v)
        self.model = train.build(a).to(device).eval()
        sd = ck['ema'] if 'ema' in ck else ck['model']
        self.model.load_state_dict({k: v.to(self.model.state_dict()[k].dtype) for k, v in sd.items()})
        st = np.load(run / 'stats.npz')
        self.mean, self.std = st['mean'], st['std']
        self.step, self.train_args = ck.get('step'), ck['args']
        self.io = mimi_io.MimiIO(device, decoder_ckpt=self.cfg['decoder_ckpt'])
        self.train = train
        self.spk = None
        if getattr(a, 'spk', 0):
            import spk_embed
            self.spk = spk_embed.CamPlus(device)
        self.wf = None
        if getattr(a, 'whisper', 0):
            import whisper_feats
            self.wf = whisper_feats.WhisperFeat(device)
        self.ready = True

    def ref_features(self, ref):
        key = id(ref)
        if key not in self.ref_cache:
            n = CHUNK_S * SR
            lats, codes = [], []
            for i in range(0, len(ref), n):
                chunk = np.asarray(ref[i:i + n], dtype=np.float32)
                if len(chunk) <= SR:                     # chunks of 1 s or less are dropped, as the published pipeline
                    continue
                lat, cod = self.io.encode(chunk)
                lats.append(lat)
                codes.append(cod)
            lat, cod = np.concatenate(lats), np.concatenate(codes)
            v1 = self.train_args.get('sampler') == 'prompt'
            P = self.cfg['prompt_max'] or (None if v1 else 250)
            lat, cod = lat[:P], cod[:P]
            n = self.cfg['prompt_chunk'] if v1 else len(lat)
            k = -(-len(lat) // n)
            chunks = [(lat[i * len(lat) // k:(i + 1) * len(lat) // k], cod[i * len(lat) // k:(i + 1) * len(lat) // k])
                      for i in range(k)]
            g = self.spk_vector(ref, v1)
            self.ref_cache = {key: (chunks, g, ref)}     # keep the array alive so its id stays unique
        return self.ref_cache[key][:2]

    def spk_vector(self, r, v1=True):
        """The global speaker vector of audio r: v1 runs the renormalised mean of L2-normalised CAM++ embeddings of 10 s
        windows, older runs one embedding; None without a speaker model."""
        if self.spk is None:
            return None
        r = np.asarray(r, dtype=np.float32)
        if not v1:
            return self.spk(r, SR)
        w = 10 * SR
        vs = [self.spk(r[i:i + w], SR) for i in range(0, max(1, len(r) - w // 2), w)] or [self.spk(r, SR)]
        vs = [v / max(np.linalg.norm(v), 1e-8) for v in vs]
        m = np.mean(vs, 0)
        return m / max(np.linalg.norm(m), 1e-8)

    def ref_latent(self, ref):
        key = ('lat', id(ref))
        if key not in self.ref_cache:
            n = CHUNK_S * SR
            lats = [self.io.encode(np.asarray(ref[i:i + n], dtype=np.float32))[0] for i in range(0, len(ref), n)
                    if len(ref[i:i + n]) > SR]
            self.ref_cache = {key: (np.concatenate(lats), ref)}
        return self.ref_cache[key][0]

    def convert(self, src, ref, sr):
        if sr != SR:
            raise ValueError(f'expects {SR} Hz input')
        c = self.cfg
        s = np.asarray(src, dtype=np.float32)
        if c['mode'] != 'convert':
            lat, codes = self.io.encode(s)
            if c['mode'] == 'copy_latent':
                wav = self.io.decode_latent(lat)
            elif c['mode'] == 'copy_codes':
                wav = self.io.decode_codes(codes)
            else:
                ref_lat = self.ref_latent(ref)
                x = knn_latent(lat, ref_lat) if c['mode'] == 'knn_latent' else ot_latent(lat, ref_lat)
                wav = self.io.decode_latent(x)
            return np.asarray(wav[:len(src)], dtype=np.float32), SR
        x = self.convert_latent(s, ref)
        wav = self.io.decode_latent(x) if c['decode'] == 'latent' else self.io.decode_codes(self.io.quantize(x))
        wav = np.asarray(wav[:len(src)], dtype=np.float32)
        if c['match_rms']:
            r = float(np.sqrt(np.mean(wav ** 2)))
            if r > 0:
                wav = wav * (float(np.sqrt(np.mean(s ** 2))) / r)
        return wav, SR

    def convert_latent(self, s, ref):
        """The generated (un-standardised) latent [T, 512] of convert() for a 24 kHz source s, before decoding."""
        c = self.cfg
        s = np.asarray(s, dtype=np.float32)
        neg = {}
        if c['neg_src'] > 0:                             # the clean source's own voice, before any perturbation
            n_lat, n_codes = self.io.encode(s)
            neg = dict(neg_lat=n_lat, neg_codes=n_codes, w_neg=c['neg_src'], neg_until=c['neg_src_until'],
                       neg_spk=self.spk_vector(s, self.train_args.get('sampler') == 'prompt'))
        if c['perturb_src']:
            import perturb
            s = perturb.perturb(s, SR, np.random.default_rng(int(np.random.randint(2 ** 31))))
        src_lat, codes = self.io.encode(s)
        cont = None
        if self.wf is not None:
            w = s
            if c['whisper_src'] == 'perturbed' and not c['perturb_src']:
                import perturb
                w = perturb.perturb(s, SR, np.random.default_rng(int(np.random.randint(2 ** 31))))
            cont = self.wf(w)
        chunks, g = self.ref_features(ref)
        seed = int(np.random.randint(2 ** 31)) if c['seed_from_pair'] else 0
        x, z = self.train.convert_latents(self.model, codes, chunks[0][0], chunks[0][1], self.mean, self.std, self.device,
                                          steps=c['steps'], cfg=c['cfg'], sway=c['sway'], seed=seed, spk=g,
                                          pos_mode=self.train_args.get('pos', 'continuous'), src_cont=cont,
                                          seg_len=self.train_args.get('seg_len', 125), chunks=chunks, solver=c['solver'],
                                          temp=c['temp'], cfg_rescale=c['cfg_rescale'], sharpen=c['sharpen'], gv=c['gv'],
                                          return_std=True, init_lat=src_lat if c['init_t'] > 0 else None,
                                          t_start=c['init_t'], cfg_branch=c['cfg_branch'], **neg)
        if c['stats_path']:
            from model import smoothness
            zs = (np.asarray(src_lat) - self.mean) / self.std
            zr = (np.concatenate([l for l, _ in chunks]) - self.mean) / self.std
            rec = dict(frames=len(z), smooth_gen=smoothness(z), smooth_src=smoothness(zs), smooth_ref=smoothness(zr),
                       std_gen=float(z.std(0).mean()), std_src=float(zs.std(0).mean()), std_ref=float(zr.std(0).mean()))
            with open(c['stats_path'], 'a') as f:
                f.write(json.dumps(rec) + '\n')
        if c['relax'] > 0:
            x = self.io.relax(x, c['relax'], c['relax_k'])
        return x

    def describe(self):
        here = Path(__file__).resolve().parent
        h = hashlib.sha256()
        for f in sorted(here.glob('*.py')):
            h.update(f.read_bytes())
        return dict(repo='duplex/mvc (MimiFlow-VC)', commit=h.hexdigest()[:16], checkpoints=[p for p in [self.ckpt_path] if p],
                    sr_out=SR, settings=dict(self.cfg, train_step=self.step, train_args=self.train_args))


def knn_latent(src, ref, k=4):
    """kNN-VC in Mimi's latent: each source frame becomes the mean of its k cosine-nearest reference frames."""
    a = src / np.maximum(np.linalg.norm(src, axis=1, keepdims=True), 1e-8)
    b = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-8)
    idx = np.argsort(-(a @ b.T), axis=1)[:, :k]
    return ref[idx].mean(1).astype(np.float32)


def _sqrtm(S):
    w, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def ot_latent(src, ref, shrink=1e-3):
    """Gaussian optimal transport: x -> mu_r + A (x - mu_s), A the Monge map between N(mu_s, S_s) and N(mu_r, S_r)
    (covariances shrunk by shrink * mean variance towards the identity)."""
    ms, mr = src.mean(0), ref.mean(0)
    def cov(x, m):
        c = np.cov((x - m).T) if len(x) > 1 else np.zeros((x.shape[1], x.shape[1]))
        return c + shrink * max(np.trace(c) / len(c), 1e-6) * np.eye(len(c))
    Ss, Sr = cov(src, ms), cov(ref, mr)
    Rs = _sqrtm(Ss)
    w, V = np.linalg.eigh(Ss)
    Rs_inv = (V / np.sqrt(np.clip(w, 1e-12, None))) @ V.T
    A = Rs_inv @ _sqrtm(Rs @ Sr @ Rs) @ Rs_inv
    return (mr + (src - ms) @ A.T).astype(np.float32)
