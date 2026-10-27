"""Tests of mvc/model.py: packing, padding independence, zero-initialised output, a loss that counts target frames
only, the Euler sampler on a field with a known solution, guidance, EMA and a tiny overfit run."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as mdl  # noqa: E402


def tiny(**kw):
    torch.manual_seed(0)
    return mdl.MimiFlow(d=64, layers=2, heads=4, content_dim=16, **kw)


def batch(B=3, P=(5, 7, 4), T=(6, 3, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    Pm, Tm = max(P), max(T)
    pl, tl = torch.randn(B, Pm, 512, generator=g), torch.randn(B, Tm, 512, generator=g)
    pm = torch.arange(Pm)[None] < torch.tensor(P)[:, None]
    tm = torch.arange(Tm)[None] < torch.tensor(T)[:, None]
    pc, tc = torch.randint(0, 2048, (B, Pm), generator=g), torch.randint(0, 2048, (B, Tm), generator=g)
    return pl * pm[..., None], pm, tl * tm[..., None], tm, pc, tc


def test_pack_layout():
    pl, pm, tl, tm, pc, tc = batch()
    lat, content, region, mask, is_tgt = mdl.pack(pl, pm, tl, tm, pc, tc)
    assert lat.shape == (3, 12, 512) and mask.sum(1).tolist() == [11, 10, 12]
    assert torch.equal(lat[0, :5], pl[0, :5]) and torch.equal(lat[0, 5:11], tl[0, :6]) and lat[0, 11:].abs().sum() == 0
    assert region[1].tolist() == [0] * 7 + [1] * 3 + [0, 0] and is_tgt[1].tolist() == [False] * 7 + [True] * 3 + [False] * 2
    assert torch.equal(content[2, 4:12], tc[2, :8])


def test_zero_init_and_padding_independence():
    m = tiny().eval()
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    tau = torch.full((3,), 0.3)
    assert m(lat, lat, content, region, mask, tau).abs().max() == 0          # zero-initialised output head
    for p in m.parameters():                                                 # make the network non-trivial
        torch.nn.init.normal_(p, std=0.05)
    out = m(lat, lat, content, region, mask, tau)
    lat2 = lat.clone()
    lat2[~mask] = 123.0                                                      # garbage in the padding
    content2 = content.clone()
    content2[~mask] = 5
    out2 = m(lat2, lat2, content2, region, mask, tau)
    assert torch.allclose(out[mask], out2[mask], atol=1e-5)


def test_loss_ignores_prompt_and_padding_targets():
    m = tiny()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    g = torch.Generator().manual_seed(1)
    l1 = mdl.flow_loss(m, lat, content, region, mask, is_tgt, p_drop_cond=0, p_drop_content=0, generator=g)
    assert torch.isfinite(l1) and l1 > 0
    lat2 = lat.clone()
    lat2[~mask] = 50.0
    g = torch.Generator().manual_seed(1)
    l2 = mdl.flow_loss(m, lat2, content, region, mask, is_tgt, p_drop_cond=0, p_drop_content=0, generator=g)
    assert torch.allclose(l1, l2, atol=1e-5)


class PointField(torch.nn.Module):
    """The exact rectified-flow velocity towards a fixed point x1: v = (x1 - x) / (1 - t)."""
    null_content = 2048

    def __init__(self, x1):
        super().__init__()
        self.x1 = x1

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        return (self.x1 - x) / (1 - t[:, None, None]).clamp(min=1e-4)


def test_sampler_reaches_the_point_and_keeps_the_prompt():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    x1 = torch.randn_like(lat)
    out = mdl.sample(PointField(x1), lat, content, region, mask, is_tgt, steps=8, cfg=0, sway=None)
    assert torch.allclose(out[is_tgt], x1[is_tgt], atol=1e-3)
    assert torch.equal(out[~is_tgt], lat[~is_tgt])
    out_s = mdl.sample(PointField(x1), lat, content, region, mask, is_tgt, steps=8, cfg=0.7, sway=-1.0)
    assert torch.allclose(out_s[is_tgt], x1[is_tgt], atol=1e-2)             # guidance of identical branches is a no-op


def test_ema_tracks_parameters():
    m = tiny()
    e = mdl.EMA(m, decay=0.5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    e.update(m)
    m2 = tiny()
    e.copy_to(m2)
    k = 'inp.weight'
    assert torch.allclose(m2.state_dict()[k], m.state_dict()[k] - 0.5, atol=1e-6)


def test_tiny_overfit():
    torch.manual_seed(0)
    m = tiny()
    lat, content, region, mask, is_tgt = mdl.pack(*batch(seed=3))
    lat = lat * 0.3
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    first = None
    for step in range(150):                          # fixed noise and tau: a pure memorisation check
        g = torch.Generator().manual_seed(0)
        loss = mdl.flow_loss(m, lat, content, region, mask, is_tgt, p_drop_cond=0, p_drop_content=0, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss.detach())
    assert float(loss.detach()) < 0.6 * first


def test_speaker_conditioning_changes_output_and_drops_with_the_prompt():
    torch.manual_seed(0)
    m = mdl.MimiFlow(d=64, layers=2, heads=4, content_dim=16, spk_dim=8)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    tau = torch.full((3,), 0.4)
    g1, g2 = torch.randn(3, 8), torch.randn(3, 8)
    o1, o2 = m(lat, lat, content, region, mask, tau, g1), m(lat, lat, content, region, mask, tau, g2)
    assert not torch.allclose(o1, o2)
    assert torch.allclose(m(lat, lat, content, region, mask, tau, None), m(lat, lat, content, region, mask, tau, torch.zeros(3, 8)))
    loss = mdl.flow_loss(m, lat, content, region, mask, is_tgt, p_drop_cond=1.0, p_drop_content=0, g=g1)
    assert torch.isfinite(loss)
    out = mdl.sample(m, lat, content, region, mask, is_tgt, steps=2, cfg=0.7, g=g1)
    assert out.shape == lat.shape


def test_positions_restart_and_continuous():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    r = mdl.positions(region, mask, 'restart')
    assert r[0, :5].tolist() == list(range(5)) and r[0, 5:11].tolist() == list(range(6))
    c = mdl.positions(region, mask, 'continuous')
    assert c[0, :11].tolist() == list(range(11))
    m = tiny().eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    tau = torch.full((3,), 0.5)
    assert not torch.allclose(m(lat, lat, content, region, mask, tau, None, r), m(lat, lat, content, region, mask, tau, None, c))


def test_continuous_content_stream_and_target_scatter():
    torch.manual_seed(0)
    m = mdl.MimiFlow(d=64, layers=2, heads=4, content_dim=16, cont_dim=12)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    arr = torch.randn(3, 8, 12)
    cc = mdl.pack_target_stream(arr, is_tgt)
    assert cc.shape == (3, 12, 12) and torch.equal(cc[1, 7:10], arr[1, :3]) and cc[1, :7].abs().sum() == 0
    tau = torch.full((3,), 0.3)
    assert not torch.allclose(m(lat, lat, content, region, mask, tau, cc=cc), m(lat, lat, content, region, mask, tau))
    loss = mdl.flow_loss(m, lat, content, region, mask, is_tgt, cc=cc, p_drop_content=1.0)
    assert torch.isfinite(loss)
    assert mdl.sample(m, lat, content, region, mask, is_tgt, steps=2, cc=cc).shape == lat.shape


def test_segment_positions():
    seg = torch.tensor([[1, 1, 1, 2, 2, 0, 0, 0, -1], [1, 1, 0, 0, 0, 0, -1, -1, -1]])
    region = (seg == 0).long()
    mask = seg >= 0
    p = mdl.positions(region, mask, 'segment', seg=seg, stride=100)
    assert p[0].tolist() == [100, 101, 102, 200, 201, 0, 1, 2, 0] and p[1].tolist() == [100, 101, 0, 1, 2, 3, 0, 0, 0]
    g = torch.Generator().manual_seed(0)
    r = mdl.positions(region, mask, 'segment_random', seg=seg, generator=g)
    for b in range(2):                                   # contiguous runs inside every segment
        for s in seg[b].unique().tolist():
            if s < 0:
                continue
            idx = (seg[b] == s).nonzero().squeeze(1)
            v = r[b, idx]
            assert torch.equal(v - v[0], torch.arange(len(idx)))
    pl, pm, tl, tm, pc, tc = batch()
    pseg = torch.where(pm, torch.tensor([1, 1, 2, 2, 2, 3, 3])[None].expand(3, -1), torch.full_like(pc, -1))
    packed = mdl.pack_seg(pseg, pm, tm)
    lat, content, region, mask, is_tgt = mdl.pack(pl, pm, tl, tm, pc, tc)
    assert packed.shape == mask.shape and torch.equal(packed == 0, is_tgt) and torch.equal(packed >= 0, mask)


def test_ada_single_and_speaker_dropout():
    torch.manual_seed(0)
    m = mdl.MimiFlow(d=64, layers=3, heads=4, content_dim=16, spk_dim=8, ada_single=True)
    n_single = sum(p.numel() for p in m.parameters())
    n_block = sum(p.numel() for p in mdl.MimiFlow(d=64, layers=3, heads=4, content_dim=16, spk_dim=8).parameters())
    assert n_single < n_block
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    tau = torch.full((3,), 0.3)
    assert m(lat, lat, content, region, mask, tau).abs().max() == 0      # zero-initialised paths still give 0
    loss = mdl.flow_loss(m, lat, content, region, mask, is_tgt, g=torch.randn(3, 8), p_drop_spk=1.0)
    assert torch.isfinite(loss)


def test_ensemble_shares_one_target_and_averages_velocities():
    pl, pm, tl, tm, pc, tc = batch(B=3, P=(5, 7, 4), T=(6, 6, 6))
    lat, content, region, mask, is_tgt = mdl.pack(pl, pm, tl, tm, pc, tc)
    x1 = torch.randn_like(lat)
    for i in range(3):                                   # different targets per row: the group must average them
        x1[i][is_tgt[i]] = float(i)
    out = mdl.sample_ensemble(PointField(x1), lat, content, region, mask, is_tgt, steps=8, cfg=0, sway=None,
                              groups=[[0, 1, 2]])
    t0, t1, t2 = (out[i][is_tgt[i]] for i in range(3))
    assert torch.allclose(t0, t1) and torch.allclose(t1, t2)
    assert torch.allclose(t0, torch.ones_like(t0), atol=1e-3)            # the mean of targets 0, 1, 2


class PromptProbe(torch.nn.Module):
    """Records the prompt-frame input at every step."""
    null_content = 2048

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        self.seen.append((t[0].item(), x.clone()))
        return torch.zeros_like(x)


def test_prompt_frames_follow_the_training_interpolation():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    m = PromptProbe()
    torch.manual_seed(0)
    mdl.sample(m, lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=None, generator=torch.Generator().manual_seed(3))
    prm = ~is_tgt & mask
    eps = torch.randn(lat.shape, generator=torch.Generator().manual_seed(3))
    for t, x in m.seen:
        assert torch.allclose(x[prm], ((1 - t) * eps + t * lat)[prm], atol=1e-5)


class LinearField(torch.nn.Module):
    """dx/dt = a x: the exact solution x(1) = x(0) e^a makes solver error measurable."""
    null_content = 2048

    def __init__(self, a=1.0):
        super().__init__()
        self.a = a

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        return self.a * x


def test_second_order_solvers_beat_euler_and_temp_scales_the_start():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    x0 = torch.randn(lat.shape, generator=torch.Generator().manual_seed(0))      # the sampler's own first draw
    exact = x0 * math.e
    err = {}
    for s in ('euler', 'midpoint', 'heun'):
        out = mdl.sample(LinearField(), lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=None,
                         generator=torch.Generator().manual_seed(0), solver=s)
        err[s] = float((out[is_tgt] - exact[is_tgt]).abs().max())
        assert torch.equal(out[~is_tgt], lat[~is_tgt])
    assert err['midpoint'] < 0.15 * err['euler'] and err['heun'] < 0.15 * err['euler']
    hot = mdl.sample(LinearField(), lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=None,
                     generator=torch.Generator().manual_seed(0), temp=0.5)
    cold = mdl.sample(LinearField(), lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=None,
                      generator=torch.Generator().manual_seed(0))
    assert torch.allclose(hot[is_tgt], 0.5 * cold[is_tgt], atol=1e-5)
    with pytest.raises(ValueError, match='unknown solver'):
        mdl.sample(LinearField(), lat, content, region, mask, is_tgt, steps=2, cfg=0, solver='rk9')


class GuidedField(torch.nn.Module):
    """Velocity 1 when conditioned (non-zero cond), 0 when the prompt is dropped: guidance multiplies it by 1 + w."""
    null_content = 2048

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        on = (cond.abs().sum((1, 2)) > 0).float()[:, None, None]
        return torch.ones_like(x) * on


def test_cfg_rescale_restores_the_conditional_scale():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    x0 = torch.randn(lat.shape, generator=torch.Generator().manual_seed(0))
    plain = mdl.sample(GuidedField(), lat, content, region, mask, is_tgt, steps=4, cfg=2.0, sway=None,
                       generator=torch.Generator().manual_seed(0))
    resc = mdl.sample(GuidedField(), lat, content, region, mask, is_tgt, steps=4, cfg=2.0, sway=None,
                      generator=torch.Generator().manual_seed(0), cfg_rescale=1.0)
    assert torch.allclose(plain[is_tgt], x0[is_tgt] + 3.0, atol=1e-4)
    assert torch.allclose(resc[is_tgt], x0[is_tgt] + 1.0, atol=1e-4)


def test_smoothness_and_postprocess():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((400, 16)).astype(np.float32)
    pad = np.concatenate([noise[:1], noise, noise[-1:]])
    ma = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
    slow = np.sin(np.linspace(0, 6, 400))[:, None] * np.ones((1, 16))
    assert 0.85 < mdl.smoothness(noise) < 1.15 and mdl.smoothness(slow) < 0.01 and mdl.smoothness(ma) < 0.5
    assert mdl.smoothness(mdl.postprocess(ma, sharpen=1.0)) > mdl.smoothness(ma) + 0.2
    z = 0.5 * noise
    assert np.allclose(mdl.postprocess(z, noise, gv=1.0).std(0), noise.std(0), rtol=0.05)
    assert np.allclose(mdl.postprocess(z, noise, gv=0.5).std(0), 0.5 * noise.std(0) * 2 ** 0.5, rtol=0.05)
    assert np.allclose(mdl.postprocess(noise, z, gv=1.0), noise)                    # never shrinks (gain >= 1)
    assert np.allclose(mdl.postprocess(0.1 * noise, noise, gv=1.0).std(0), 0.3 * noise.std(0), rtol=0.05)  # gain <= 3
    assert np.array_equal(mdl.postprocess(noise), noise)


class RecordingZeroField(torch.nn.Module):
    """Zero velocity; records the times it is queried at."""
    null_content = 2048

    def __init__(self):
        super().__init__()
        self.ts = []

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        self.ts.append(float(t[0]))
        return torch.zeros_like(x)


def test_sdedit_start_mixes_the_init_and_maps_the_time_grid():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    x0 = torch.randn(lat.shape, generator=torch.Generator().manual_seed(0))      # the sampler's own first draw
    init = torch.randn(lat.shape, generator=torch.Generator().manual_seed(5))
    f = RecordingZeroField()
    out = mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=-1.0,
                              generator=torch.Generator().manual_seed(0), x_init=init, t_start=0.6)
    assert torch.allclose(out[is_tgt], (0.4 * x0 + 0.6 * init)[is_tgt], atol=1e-6)
    assert torch.equal(out[~is_tgt], lat[~is_tgt])
    assert len(f.ts) == 4 and abs(f.ts[0] - 0.6) < 1e-6 and all(0.6 <= t < 1 for t in f.ts) and f.ts == sorted(f.ts)
    f0 = RecordingZeroField()
    plain = mdl.sample_ensemble(f0, lat, content, region, mask, is_tgt, steps=4, cfg=0, sway=-1.0,
                                generator=torch.Generator().manual_seed(0), x_init=init, t_start=0.0)
    assert torch.allclose(plain[is_tgt], x0[is_tgt]) and f0.ts[0] == 0.0          # t_start 0 ignores the init


def test_null_noise_prompt_hides_the_prompt_from_dropped_rows():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    m = tiny()
    torch.manual_seed(1)
    with torch.no_grad():                                   # the output layer is zero-initialised: give it weights
        for prm_ in m.parameters():
            prm_.add_(0.05 * torch.randn_like(prm_))
    base = mdl.flow_loss(m, lat, content, region, mask, is_tgt, generator=torch.Generator().manual_seed(3))
    same = mdl.flow_loss(m, lat, content, region, mask, is_tgt, generator=torch.Generator().manual_seed(3),
                         null_noise_prompt=0.0)
    assert torch.equal(base, same)                                          # 0 keeps the loss bit for bit
    other = torch.where((~is_tgt & mask)[..., None], lat * 3 + 1, lat)      # different prompt latents only
    l1 = mdl.flow_loss(m, lat, content, region, mask, is_tgt, p_drop_cond=1.0, generator=torch.Generator().manual_seed(4),
                       null_noise_prompt=1.0)
    l2 = mdl.flow_loss(m, other, content, region, mask, is_tgt, p_drop_cond=1.0,
                       generator=torch.Generator().manual_seed(4), null_noise_prompt=1.0)
    assert torch.allclose(l1, l2)
    l3 = mdl.flow_loss(m, other, content, region, mask, is_tgt, p_drop_cond=1.0, generator=torch.Generator().manual_seed(4))
    l4 = mdl.flow_loss(m, lat, content, region, mask, is_tgt, p_drop_cond=1.0, generator=torch.Generator().manual_seed(4))
    assert not torch.allclose(l3, l4)                                       # the standard null still sees the prompt


class UncondRecorder(torch.nn.Module):
    null_content = 2048

    def __init__(self):
        super().__init__()
        self.unc = []

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        if cond.abs().sum() == 0:
            self.unc.append(x.clone())
        return torch.zeros_like(x)


def test_noiseprompt_branch_shows_pure_noise_in_the_prompt():
    lat, content, region, mask, is_tgt = mdl.pack(*batch())
    a = mdl.sample(tiny(), lat, content, region, mask, is_tgt, steps=3, generator=torch.Generator().manual_seed(0))
    b = mdl.sample_ensemble(tiny(), lat, content, region, mask, is_tgt, steps=3, generator=torch.Generator().manual_seed(0),
                            cfg_branch='std')
    assert torch.equal(a, b)
    prm = (~is_tgt & mask)
    seen = []
    for scale in (1.0, 5.0):
        r = UncondRecorder()
        mdl.sample_ensemble(r, lat * scale, content, region, mask, is_tgt, steps=3, cfg=1.0,
                            generator=torch.Generator().manual_seed(0), cfg_branch='noiseprompt')
        seen.append(torch.stack([u[prm] for u in r.unc]))
    assert torch.equal(seen[0], seen[1])                                    # the unconditional pass never sees the prompt


class PromptMeanField(torch.nn.Module):
    """Target velocity = mean of the cond prompt frames - x (pulls the target toward the prompt's mean voice)."""
    null_content = 2048

    def forward(self, x, cond, content, region, mask, t, g=None, pos=None, cc=None):
        prm = (cond.abs().sum(-1, keepdim=True) > 0).float()
        m = (cond * prm).sum(1, keepdim=True) / prm.sum(1, keepdim=True).clamp(min=1)
        return m - x


def one_example(value, P=5, T=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pl = torch.full((1, P, 512), float(value))
    tl = torch.zeros(1, T, 512)
    pc, tc = torch.randint(0, 2048, (1, P), generator=g), torch.randint(0, 2048, (1, T), generator=g)
    return mdl.pack(pl, torch.ones(1, P, dtype=torch.bool), tl, torch.ones(1, T, dtype=torch.bool), pc, tc)


def test_source_negative_guidance_zero_weight_and_identical_branch_change_nothing():
    lat, content, region, mask, is_tgt = mdl.pack(*batch(B=1, P=(5,), T=(6,)))
    m = tiny().eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.05)
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    base = mdl.sample_ensemble(m, lat, content, region, mask, is_tgt, steps=4, cfg=0.7, generator=gen())
    neg = dict(cond_lat=lat, content=content, region=region, mask=mask, is_tgt=is_tgt)
    same = mdl.sample_ensemble(m, lat, content, region, mask, is_tgt, steps=4, cfg=0.7, generator=gen(), neg=neg, w_neg=0.0)
    assert torch.equal(base, same)
    f = PromptMeanField()                                  # depends on cond only: an identical negative branch cancels
    a = mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=4, cfg=0, generator=gen())
    b = mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=4, cfg=0, generator=gen(), neg=neg, w_neg=1.0)
    assert torch.allclose(a[is_tgt], b[is_tgt], atol=1e-5)
    with pytest.raises(ValueError):
        mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=2, cfg=0, w_neg=1.0)


def test_source_negative_guidance_steers_away_from_the_negative_voice():
    lat, content, region, mask, is_tgt = one_example(+1.0)
    nlat, ncontent, nregion, nmask, nis_tgt = one_example(-1.0, P=3)
    neg = dict(cond_lat=nlat, content=ncontent, region=nregion, mask=nmask, is_tgt=nis_tgt)
    f = PromptMeanField()
    out = {}
    for w in (0.0, 1.0):
        x = mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=64, cfg=0, sway=None, temp=0.0,
                                generator=torch.Generator().manual_seed(0), neg=neg, w_neg=w)
        out[w] = float(x[is_tgt].mean())
    # dx/dt = (1 + w) a - w b - x from 0 with a = +1, b = -1: x(1) = (1 + 2w)(1 - 1/e) (Euler, 64 steps: close)
    assert abs(out[0.0] - (1 - math.exp(-1))) < 0.01 and abs(out[1.0] - 3 * (1 - math.exp(-1))) < 0.02
    assert out[1.0] > out[0.0]
    x = mdl.sample_ensemble(f, lat, content, region, mask, is_tgt, steps=64, cfg=0, sway=None, temp=0.0,
                            generator=torch.Generator().manual_seed(0), neg=neg, w_neg=1.0, neg_until=0.5)
    assert out[0.0] < float(x[is_tgt].mean()) < out[1.0]               # a guidance interval: in between
