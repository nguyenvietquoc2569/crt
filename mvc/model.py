"""MimiFlow-VC: in-context conditional flow matching on Mimi's continuous latent (12.5 Hz, 512-d).

Sequence layout per example: [prompt frames | target frames], packed and right-padded. Per frame the network sees
  x_tau     the noisy latent (rectified flow: x_tau = (1 - tau) noise + tau x1), over the whole sequence,
  cond      the clean latent on prompt frames, zero on target frames (the voice to copy, in context),
  content   an embedding of the frame's content token (CB0, ideally of information-perturbed audio),
  region    a learned prompt / target embedding,
and predicts the velocity x1 - noise; the loss counts target frames only. A DiT (adaLN-zero on tau, rotary positions,
bidirectional attention with key padding) does the work. Classifier-free guidance drops cond (and optionally content)
during training and mixes conditional and unconditional velocities at sampling.
"""
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

LAT = 512
VOCAB = 2048


def rope(x, pos, base=10000.0):
    """Rotary embedding on the last dim of x [B, H, S, D] for integer positions pos [B, S]."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos[:, None, :, None].float() * inv[None, None, None, :]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)
    return out.to(x.dtype)


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t[:, None].float() * 1000.0 * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class Block(nn.Module):
    def __init__(self, d, heads, mlp=4.0, dropout=0.0, single=False):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(approximate='tanh'), nn.Linear(int(d * mlp), d))
        if single:                               # adaLN-single (PixArt): a shared projection plus a per-block offset
            self.ada_off = nn.Parameter(torch.zeros(6 * d))
            self.ada = None
        else:
            self.ada = nn.Linear(d, 6 * d)
            nn.init.zeros_(self.ada.weight)
            nn.init.zeros_(self.ada.bias)
        self.drop = dropout

    def forward(self, x, c, pos, kmask):
        mod = self.ada(c) if self.ada is not None else c + self.ada_off   # c is the shared 6d vector when single
        s1, b1, g1, s2, b2, g2 = mod[:, None].chunk(6, -1)
        h = self.n1(x) * (1 + s1) + b1
        B, S, D = h.shape
        q, k, v = self.qkv(h).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=kmask[:, None, None, :],
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(B, S, D))
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class MimiFlow(nn.Module):
    def __init__(self, d=768, layers=12, heads=12, mlp=4.0, content_vocab=VOCAB, content_dim=256, dropout=0.0, spk_dim=0,
                 cont_dim=0, ada_single=False):
        super().__init__()
        self.d = d
        self.ada_single = ada_single
        self.spk_dim = spk_dim
        self.cont_dim = cont_dim                  # optional continuous content stream (e.g. Whisper, 768-d), target frames
        self.cont = nn.Sequential(nn.LayerNorm(cont_dim), nn.Linear(cont_dim, content_dim)) if cont_dim else None
        self.spk = nn.Sequential(nn.Linear(spk_dim, d), nn.SiLU(), nn.Linear(d, d)) if spk_dim else None
        self.content = nn.Embedding(content_vocab + 1, content_dim)        # last index = "no content" (CFG drop)
        self.null_content = content_vocab
        self.region = nn.Embedding(2, d)
        self.inp = nn.Linear(2 * LAT + content_dim, d)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp, dropout, ada_single) for _ in range(layers)])
        if ada_single:
            self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
            nn.init.zeros_(self.ada_shared[1].weight)
            nn.init.zeros_(self.ada_shared[1].bias)
        self.n_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, LAT)
        nn.init.zeros_(self.ada_out.weight)
        nn.init.zeros_(self.ada_out.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x_tau, cond, content, region, mask, tau, g=None, pos=None, cc=None):
        """x_tau, cond [B, S, 512]; content [B, S] long; region [B, S] (0 prompt, 1 target); mask [B, S] bool; tau [B];
        g [B, spk_dim] global speaker embedding (zeros = none), used when the model has spk_dim."""
        ce = self.content(content)
        if self.cont is not None and cc is not None:
            ce = ce + self.cont(cc) * (cc.abs().sum(-1, keepdim=True) > 0)        # all-zero rows = no stream there
        h = self.inp(torch.cat([x_tau, cond, ce], -1)) + self.region(region)
        c = self.t_mlp(timestep_embedding(tau, 256))
        if self.spk is not None:
            c = c + self.spk(g if g is not None else torch.zeros(len(tau), self.spk_dim, device=tau.device))
        if pos is None:
            pos = torch.cumsum(mask.long(), 1) - 1
        cb = self.ada_shared(c) if self.ada_single else c
        for blk in self.blocks:
            h = blk(h, cb, pos, mask)
        s, b = self.ada_out(c)[:, None].chunk(2, -1)
        return self.out(self.n_out(h) * (1 + s) + b)


def positions(region, mask, mode='restart', seg=None, generator=None, max_offset=4096, stride=512):
    """Rotary positions.
    'restart'     the prompt 0..P-1 and the target 0..T-1 again, so a longer reference never pushes the target past
                  the trained range;
    'continuous'  the packed sequence 0..P+T-1;
    'segment_random' / 'segment' (needs seg [B, S]: 0 target, 1..n prompt segments, -1 padding): every segment counts
                  0..n-1 from its own offset, drawn per segment from U{0..max_offset-1} in training ('segment_random'),
                  so the relative phase between target and prompt carries no information, and fixed at inference
                  ('segment': target 0, prompt segment k at stride*k)."""
    if mode == 'continuous':
        return torch.cumsum(mask.long(), 1) - 1
    if mode in ('segment', 'segment_random'):
        B, S = seg.shape
        ar = torch.arange(S, device=seg.device).expand(B, S)
        start = torch.ones_like(seg, dtype=torch.bool)
        start[:, 1:] = seg[:, 1:] != seg[:, :-1]
        run0 = torch.cummax(torch.where(start, ar, torch.zeros_like(ar)), 1).values
        within = ar - run0
        sid = seg.clamp(min=0)
        if mode == 'segment_random':
            off = torch.randint(0, max_offset, (B, int(sid.max()) + 1), device=seg.device, generator=generator)
            base = off.gather(1, sid)
        else:
            base = sid * stride
        return torch.where(seg >= 0, within + base, torch.zeros_like(within))
    p = torch.cumsum((mask & (region == 0)).long(), 1) - 1
    t = torch.cumsum((mask & (region == 1)).long(), 1) - 1
    return torch.where(region == 1, t, p).clamp(min=0)


def pack(prompt_lat, prompt_mask, target_lat, target_mask, prompt_content, target_content):
    """[prompt | target] per example, right-padded. Returns lat, content, region, mask, target index map (B, S) bool."""
    B = prompt_lat.shape[0]
    lp, lt = prompt_mask.sum(1), target_mask.sum(1)
    S = int((lp + lt).max())
    dev = prompt_lat.device
    lat = torch.zeros(B, S, LAT, device=dev, dtype=prompt_lat.dtype)
    content = torch.zeros(B, S, dtype=torch.long, device=dev)
    region = torch.zeros(B, S, dtype=torch.long, device=dev)
    mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
    is_tgt = torch.zeros(B, S, dtype=torch.bool, device=dev)
    for i in range(B):
        p, t = int(lp[i]), int(lt[i])
        lat[i, :p] = prompt_lat[i, :p]
        lat[i, p:p + t] = target_lat[i, :t]
        content[i, :p] = prompt_content[i, :p]
        content[i, p:p + t] = target_content[i, :t]
        region[i, p:p + t] = 1
        mask[i, :p + t] = True
        is_tgt[i, p:p + t] = True
    return lat, content, region, mask, is_tgt


def flow_loss(model, lat, content, region, mask, is_tgt, p_drop_cond=0.1, p_drop_content=0.05, generator=None, g=None,
              pos=None, tau_dist='uniform', cc=None, p_drop_spk=0.0, null_noise_prompt=0.0):
    """Rectified-flow loss on target frames; condition dropout for classifier-free guidance (the speaker embedding g
    is dropped together with the prompt). null_noise_prompt: on that share of the condition-dropped rows the prompt
    frames of x_tau become fresh noise, so the unconditional branch can be trained to see no voice at all (N1)."""
    B = lat.shape[0]
    dev = lat.device
    if tau_dist == 'logit_normal':            # SD3: more weight on mid-range tau, where the velocity is hardest
        tau = torch.sigmoid(torch.randn(B, device=dev, generator=generator))
    else:
        tau = torch.rand(B, device=dev, generator=generator)
    noise = torch.randn(lat.shape, device=dev, generator=generator, dtype=lat.dtype)
    x_tau = (1 - tau[:, None, None]) * noise + tau[:, None, None] * lat
    cond = lat * (~is_tgt & mask)[..., None]
    drop = torch.rand(B, device=dev, generator=generator) < p_drop_cond
    cond = cond * (~drop)[:, None, None]
    if g is not None:
        dropg = drop | (torch.rand(B, device=dev, generator=generator) < p_drop_spk)   # speaker vector alone, too
        g = g * (~dropg)[:, None]
    if null_noise_prompt > 0:
        rows = drop & (torch.rand(B, device=dev, generator=generator) < null_noise_prompt)
        fresh = torch.randn(lat.shape, device=dev, generator=generator, dtype=lat.dtype)
        x_tau = torch.where(((~is_tgt & mask) & rows[:, None])[..., None], fresh, x_tau)
    dropc = torch.rand(B, device=dev, generator=generator) < p_drop_content
    content = torch.where(dropc[:, None], torch.full_like(content, model.null_content), content)
    if cc is not None:
        cc = cc * (~dropc)[:, None, None]
    v = model(x_tau, cond, content, region, mask, tau, g, pos, cc)
    w = is_tgt.float()
    err = ((v.float() - (lat - noise).float()) ** 2).mean(-1)
    return (err * w).sum() / w.sum().clamp(min=1)


@torch.no_grad()
def sample(model, cond_lat, content, region, mask, is_tgt, steps=16, cfg=0.7, sway=-1.0, generator=None, cfg_content=False,
           g=None, pos=None, cc=None, solver='euler', temp=1.0, cfg_rescale=0.0, neg=None, w_neg=0.0, neg_until=1.0):
    """ODE integration from noise to x1 on target frames. Prompt frames are given as cond and, at every step, set
    to the exact interpolation (1 - t) eps + t x1 they had in training. Returns [B, S, 512]."""
    return sample_ensemble(model, cond_lat, content, region, mask, is_tgt, steps, cfg, sway, generator, cfg_content,
                           g, pos, cc, groups=None, solver=solver, temp=temp, cfg_rescale=cfg_rescale, neg=neg,
                           w_neg=w_neg, neg_until=neg_until)


@torch.no_grad()
def sample_ensemble(model, cond_lat, content, region, mask, is_tgt, steps=16, cfg=0.7, sway=-1.0, generator=None,
                    cfg_content=False, g=None, pos=None, cc=None, groups=None, solver='euler', temp=1.0, cfg_rescale=0.0,
                    x_init=None, t_start=0.0, cfg_branch='std', neg=None, w_neg=0.0, neg_until=1.0):
    """As sample, but the batch rows of each group (a list of row-index lists, all rows of a group sharing the same
    target length) share one target state, and their target velocities are averaged at every step: one conversion
    conditioned on several prompt chunks (the 180 s chunk ensemble). groups=None: every row on its own.
    solver: 'euler' (1 velocity per step), 'midpoint' or 'heun' (2 per step, second order). temp scales the starting
    noise of target frames (prompt frames keep unit noise, as in training). cfg_rescale (0..1) blends the guided
    velocity with its version rescaled to the conditional velocity's per-row standard deviation (Lin et al. 2024),
    which keeps a large guidance weight from inflating the velocity. x_init with t_start in (0, 1) (SDEdit probe): target
    frames start at (1 - t_start) eps + t_start x_init and the time grid is mapped onto [t_start, 1].
    cfg_branch 'std': the unconditional pass sees the same x (noisy prompt frames included); 'noiseprompt': its prompt
    frames are the pure starting noise (for models trained with null_noise_prompt).
    neg, w_neg (source-negative guidance): neg is a packed one-row example dict(cond_lat, content, region, mask, is_tgt,
    g, pos, cc) holding a voice to steer away from (the source's own latents as prompt and its speaker vector) with the
    same target frames as every row of a single group; its prompt frames get their own noise (drawn after eps) and the
    training interpolation, its target frames the shared state, and every target velocity becomes
    v + w_neg (mean_group v_cond - v_neg), v_cond the conditional (unguided) velocity, at the steps with t < neg_until
    (a guidance interval). w_neg = 0 leaves the sampler unchanged."""
    B, S, D = cond_lat.shape
    dev = cond_lat.device
    groups = groups or [[i] for i in range(B)]
    prm = (~is_tgt & mask)[..., None]
    eps = torch.randn(cond_lat.shape, device=dev, generator=generator, dtype=cond_lat.dtype)
    x = torch.where(is_tgt[..., None], eps * temp, eps)
    if x_init is not None and t_start > 0:
        x = torch.where(is_tgt[..., None], (1 - t_start) * x + t_start * x_init.to(x.dtype), x)
    for grp in groups:                                   # one shared starting noise per group on target frames
        lead = grp[0]
        for i in grp[1:]:
            x[i][is_tgt[i]] = x[lead][is_tgt[lead]]
    cond = cond_lat * prm
    if w_neg:
        if neg is None or len(groups) != 1:
            raise ValueError('source-negative guidance needs neg and a single group')
        n_prm = (~neg['is_tgt'] & neg['mask'])[..., None]
        n_eps = torch.randn(neg['cond_lat'].shape, device=dev, generator=generator, dtype=cond_lat.dtype)
        n_cond = neg['cond_lat'] * n_prm
    ts = torch.linspace(0, 1, steps + 1, device=dev)
    if sway is not None:
        ts = ts + sway * (torch.cos(math.pi / 2 * ts) - 1 + ts)
    if x_init is not None and t_start > 0:
        ts = t_start + (1 - t_start) * ts
    null_c = torch.full_like(content, model.null_content) if cfg_content else content
    w_tgt = is_tgt[..., None].to(cond_lat.dtype)

    def velocity(x, t):
        x = torch.where(prm, (1 - t) * eps + t * cond_lat, x)
        tt = t.expand(B)
        v = model(x, cond, content, region, mask, tt, g, pos, cc)
        v_cond = v
        if cfg:
            xu = torch.where(prm, eps, x) if cfg_branch == 'noiseprompt' else x
            vu = model(xu, torch.zeros_like(cond), null_c, region, mask, tt, None if g is None else torch.zeros_like(g), pos,
                       None if (cc is None or cfg_content) else cc)
            vg = v + cfg * (v - vu)
            if cfg_rescale:
                sd = lambda y: (((y.float() ** 2) * w_tgt).sum((1, 2)) / (w_tgt.sum((1, 2)) * y.shape[-1]).clamp(min=1)).sqrt()  # noqa: E731
                vr = vg * (sd(v) / sd(vg).clamp(min=1e-6)).to(vg.dtype)[:, None, None]
                vg = cfg_rescale * vr + (1 - cfg_rescale) * vg
            v = vg
        v = v.to(x.dtype)
        for grp in groups:
            if len(grp) > 1:
                vm = torch.stack([v[j][is_tgt[j]] for j in grp]).mean(0)
                for j in grp:
                    v[j][is_tgt[j]] = vm
        if w_neg and float(t) < neg_until:
            grp = groups[0]
            xn = torch.where(n_prm, (1 - t) * n_eps + t * neg['cond_lat'], torch.zeros_like(n_eps))
            xn[0][neg['is_tgt'][0]] = x[grp[0]][is_tgt[grp[0]]]
            vn = model(xn, n_cond, neg['content'], neg['region'], neg['mask'], t.expand(1), neg.get('g'), neg.get('pos'),
                       neg.get('cc'))[0][neg['is_tgt'][0]]
            vc = torch.stack([v_cond[j][is_tgt[j]] for j in grp]).mean(0)
            d = (w_neg * (vc.float() - vn.float())).to(v.dtype)
            for j in grp:
                v[j][is_tgt[j]] = v[j][is_tgt[j]] + d
        return v

    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        h = t1 - t0
        v0 = velocity(x, t0)
        if solver == 'midpoint':
            x = x + h * velocity(x + 0.5 * h * v0, t0 + 0.5 * h)
        elif solver == 'heun':
            v1 = velocity(x + h * v0, t1)
            x = x + 0.5 * h * (v0 + v1)
        elif solver == 'euler':
            x = x + h * v0
        else:
            raise ValueError(f'unknown solver {solver}')
    return torch.where(is_tgt[..., None], x, cond_lat)


def smoothness(z):
    """Normalised frame-to-frame energy of a latent [T, D]: mean over dimensions of E[(z_t - z_t-1)^2] / (2 Var_t(z)).
    About 1 for white noise and near 0 for slowly varying signals: over-smoothed outputs score lower than real latents."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) < 3:
        return float('nan')
    var = z.var(0) + 1e-9
    return float(np.mean(((z[1:] - z[:-1]) ** 2).mean(0) / (2 * var)))


def postprocess(z, ref=None, sharpen=0.0, gv=0.0):
    """Post-processing of a generated standardised latent z [T, D]:
    sharpen  temporal unsharp mask z + a (z - MA3(z)), the inverse of the 3-frame smoothing the Mimi decoder punishes;
    gv       global-variance restoration toward the reference ref [P, D] (standardised): each dimension's deviation from
             the utterance mean is scaled by clip((std_ref / std_z) ** gv, 1, 3), as in the journal's GV decoder."""
    z = np.asarray(z, dtype=np.float32).copy()
    if sharpen and len(z) >= 3:
        pad = np.concatenate([z[:1], z, z[-1:]])
        ma = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
        z = z + sharpen * (z - ma)
    if gv and ref is not None and len(z) > 1 and len(ref) > 1:
        m = z.mean(0)
        gain = np.clip((np.asarray(ref).std(0) / np.maximum(z.std(0), 1e-6)) ** gv, 1.0, 3.0)
        z = m + (z - m) * gain
    return z


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in self.shadow.items()})


def pack_target_stream(arr, is_tgt):
    """Scatter a per-target-frame array arr [B, T, D] (right-padded) into the packed layout [B, S, D]; zeros elsewhere."""
    B, S = is_tgt.shape
    out = torch.zeros(B, S, arr.shape[-1], device=arr.device, dtype=arr.dtype)
    for i in range(B):
        idx = torch.nonzero(is_tgt[i]).squeeze(1)
        out[i, idx] = arr[i, :len(idx)]
    return out


def pack_seg(prompt_seg, prompt_mask, target_mask):
    """Segment ids in the packed layout: prompt frames keep their ids (1..n), target frames 0, padding -1."""
    B = prompt_seg.shape[0]
    lp, lt = prompt_mask.sum(1), target_mask.sum(1)
    S = int((lp + lt).max())
    seg = torch.full((B, S), -1, dtype=torch.long, device=prompt_seg.device)
    for i in range(B):
        p, t = int(lp[i]), int(lt[i])
        seg[i, :p] = prompt_seg[i, :p]
        seg[i, p:p + t] = 0
    return seg
