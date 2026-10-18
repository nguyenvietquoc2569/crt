"""Unit tests of the round-7 Gaussian optimal transport of crt_decode_lab.py (gaussian_ot, parse_ot)."""
import numpy as np
import pytest

import crt_decode_lab as cdl


def frames(rng, n, mean, cov):
    return rng.multivariate_normal(mean, cov, size=n)


def rand_cov(rng, d, scale=1.0):
    a = rng.normal(size=(d, d))
    return scale * (a @ a.T / d + 0.1 * np.eye(d))


def pc_coords(x, yr, k):
    """Coordinates in the reference's top-k PCs, centred on the speech-frame mean of x (all frames here)."""
    mr = yr.mean(0)
    ev, U = np.linalg.eigh((yr - mr).T @ (yr - mr) / len(yr))
    return (x - x.mean(0)) @ U[:, -k:], ev[-k:], U[:, -k:]


@pytest.fixture
def rng():
    return np.random.default_rng(20261018)


def test_moments_matched_exactly(rng):
    """beta = 0: the output's speech-frame mean is the reference mean and its top-k covariance the reference's."""
    d, k = 8, 4
    x = frames(rng, 300, rng.normal(size=d), rand_cov(rng, d))
    yr = frames(rng, 400, rng.normal(size=d) + 3, rand_cov(rng, d, 4.0))
    sp = np.ones(len(x), bool)
    dx, A = cdl.gaussian_ot(x, sp, yr, k, beta=0.0)
    xo = x + dx
    np.testing.assert_allclose(xo.mean(0), yr.mean(0), atol=1e-9)
    P, vr, _ = pc_coords(xo, yr, k)
    np.testing.assert_allclose(P.T @ P / len(P), np.diag(vr), atol=1e-6)   # exact up to the 1e-9 trace ridge


def test_map_is_symmetric_and_transports_covariance(rng):
    d, k = 6, 6
    x = frames(rng, 200, np.zeros(d), rand_cov(rng, d))
    yr = frames(rng, 300, np.zeros(d), rand_cov(rng, d, 2.0))
    sp = np.ones(len(x), bool)
    _, A = cdl.gaussian_ot(x, sp, yr, k, beta=0.0)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert np.linalg.eigvalsh(A).min() > 0
    P, vr, _ = pc_coords(x, yr, k)
    Sx = P.T @ P / len(P)
    np.testing.assert_allclose(A @ Sx @ A, np.diag(vr), atol=1e-6)                # exact up to the 1e-9 trace ridge


def test_identity_when_output_is_the_reference(rng):
    d = 5
    yr = frames(rng, 250, rng.normal(size=d), rand_cov(rng, d))
    dx, A = cdl.gaussian_ot(yr.copy(), np.ones(len(yr), bool), yr, 3, beta=0.0)
    np.testing.assert_allclose(A, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(dx, 0.0, atol=1e-8)


def test_complement_only_shifted(rng):
    """Outside the top-k subspace a frame moves by the mean shift alone (and not at all with keep_mean)."""
    d, k = 7, 2
    x = frames(rng, 120, rng.normal(size=d), rand_cov(rng, d))
    yr = frames(rng, 200, rng.normal(size=d), rand_cov(rng, d, 3.0))
    sp = np.ones(len(x), bool)
    _, _, U = pc_coords(x, yr, k)
    Q = np.eye(d) - U @ U.T                                     # projector on the complement
    dx, _ = cdl.gaussian_ot(x, sp, yr, k)
    np.testing.assert_allclose(dx @ Q, np.broadcast_to((yr.mean(0) - x.mean(0)) @ Q, x.shape), atol=1e-9)
    dx_nm, _ = cdl.gaussian_ot(x, sp, yr, k, keep_mean=True)
    np.testing.assert_allclose(dx_nm @ Q, 0.0, atol=1e-9)


def test_keep_mean_keeps_speech_mean(rng):
    d = 6
    x = frames(rng, 150, rng.normal(size=d), rand_cov(rng, d))
    yr = frames(rng, 150, rng.normal(size=d) - 2, rand_cov(rng, d))
    sp = rng.random(len(x)) > 0.3
    dx, _ = cdl.gaussian_ot(x, sp, yr, 3, keep_mean=True)
    np.testing.assert_allclose((x + dx)[sp].mean(0), x[sp].mean(0), atol=1e-9)


def test_k0_is_the_mean_shift(rng):
    x = frames(rng, 50, np.zeros(4), np.eye(4))
    yr = frames(rng, 80, np.ones(4), np.eye(4))
    sp = np.ones(50, bool)
    dx, A = cdl.gaussian_ot(x, sp, yr, 0)
    assert A.shape == (0, 0)
    np.testing.assert_allclose(dx, np.broadcast_to(yr.mean(0) - x.mean(0), x.shape))
    dx0, _ = cdl.gaussian_ot(x, sp, yr, 0, keep_mean=True)
    assert not dx0.any()


def test_statistics_from_speech_frames_only(rng):
    """Silence frames (sp False) do not enter the statistics, but the map is applied to them."""
    d = 4
    x = frames(rng, 100, np.zeros(d), rand_cov(rng, d))
    yr = frames(rng, 100, np.ones(d), rand_cov(rng, d))
    sp = np.ones(100, bool)
    sp[:20] = False
    x2 = x.copy()
    x2[:20] = 1e3                                                # outliers in the silence frames
    dx1, A1 = cdl.gaussian_ot(x, sp, yr, 2)
    dx2, A2 = cdl.gaussian_ot(x2, sp, yr, 2)
    np.testing.assert_allclose(A1, A2)
    np.testing.assert_allclose(dx1[20:], dx2[20:])
    assert np.abs(dx2[:20] - dx1[:20]).max() > 0                # silence frames are mapped too


def test_diagonal_covariance_gives_gv1_gains(rng):
    """When Sx is diagonal in the reference's PC basis, A = diag(sqrt(vr / vx)): the gv1 gains without the clamp."""
    d, k = 5, 5
    U, _ = np.linalg.qr(rng.normal(size=(d, d)))
    vr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    yr = rng.normal(size=(5000, d)) * np.sqrt(vr[::-1]) @ U.T       # reference with known PCs
    ev, Ur = np.linalg.eigh((yr - yr.mean(0)).T @ (yr - yr.mean(0)) / len(yr))
    vx = np.array([0.5, 0.7, 1.0, 1.5, 3.0])
    Z = rng.normal(size=(4000, d))
    Z -= Z.mean(0)
    Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z / len(Z))).T   # exactly white
    x = 2.0 + (Z * np.sqrt(vx)) @ Ur.T                              # Sx = diag(vx) exactly in the reference's PCs
    _, A = cdl.gaussian_ot(x, np.ones(len(x), bool), yr, k, beta=0.0)
    np.testing.assert_allclose(A, np.diag(np.sqrt(ev / vx)), atol=1e-6)


def test_rank_deficient_output_is_finite(rng):
    """Fewer speech frames than dimensions: the shrinkage keeps Sx positive definite and the output finite."""
    d, k = 64, 64
    x = rng.normal(size=(48, d))
    yr = rng.normal(size=(375, d)) * 2
    dx, A = cdl.gaussian_ot(x, np.ones(48, bool), yr, k)
    assert np.isfinite(dx).all() and np.isfinite(A).all()
    assert np.linalg.eigvalsh(A).min() > 0


def test_correction_lies_in_the_span_of_reference_and_output(rng):
    """A rank-r decoder space (as range(W) in 512-d): the correction stays in the span of the data."""
    r, d = 3, 8
    B = np.linalg.qr(rng.normal(size=(d, r)))[0]                     # orthonormal basis of the subspace
    x = rng.normal(size=(100, r)) @ B.T
    yr = (rng.normal(size=(200, r)) * 2 + 1) @ B.T
    dx, _ = cdl.gaussian_ot(x, np.ones(100, bool), yr, r)
    np.testing.assert_allclose(dx - dx @ B @ B.T, 0.0, atol=1e-8)


@pytest.mark.parametrize('name, want', [
    ('ot16_src_cont', (16, False, 'src_cont', False)),
    ('ot0_src_cont', (0, False, 'src_cont', False)),
    ('ot16nm_src_cont', (16, True, 'src_cont', False)),
    ('ot64_pm_cont_T1', (64, False, 'pm_cont_T1', False)),
    ('ot16_shrink0.85_cont', (16, False, 'shrink0.85_cont', False)),
    ('ot8nm_knn4_cont', (8, True, 'knn4_cont', False)),
    ('otf16_src_cont', (16, False, 'src_cont', True)),
    ('otf128nm_pm_cont_T1', (128, True, 'pm_cont_T1', True)),
])
def test_parse_ot(name, want):
    assert cdl.parse_ot(name) == want


@pytest.mark.parametrize('name', ['ot_src_cont', 'ot16_faiss', 'ot16x_src_cont', 'otnm_src_cont', 'ot16_pm_rq_T1',
                                  'otff16_src_cont', 'fot16_src_cont'])
def test_parse_ot_rejects(name):
    with pytest.raises(SystemExit):
        cdl.parse_ot(name)


# ---- run_pair wiring, with a small random-codebook stand-in for the Mimi lab (real torch, CPU) ----

torch = pytest.importorskip('torch')

V, DAC, DDEC, NCB, T, TREF = 8, 6, 10, 18, 24, 60


class FakeLab:
    """The parts of crt_decode_lab.Lab that run_pair's round-7 branches touch; decode_latent records its inputs."""
    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.torch, self.device = torch, 'cpu'
        self.E = torch.randn(31, V, DAC, generator=g)
        W = torch.randn(DDEC, DAC, generator=g)
        # as in Mimi, the projection weight is a module parameter (requires grad) outside any no_grad block
        self.ac = type('AC', (), {'output_proj': type('P', (), {'weight': torch.nn.Parameter(W[:, :, None])})()})()
        self.Edec = self.E @ W.T
        self.E0dec = torch.randn(V, DDEC, generator=g)
        self.chunk = 1000
        self.calls = []

    def latent(self, codes, start=0):
        return sum(self.E[start + k][codes[k]] for k in range(codes.shape[0]))

    def decode_latent(self, cb0, ac):
        self.calls.append((cb0.clone(), ac.clone()))
        return np.zeros(10, np.float32)

    def full_input(self, cb0, ac):
        return self.E0dec[cb0] + ac @ self.ac.output_proj.weight[:, :, 0].detach().T

    def decode_input(self, x):
        self.calls.append(('input', x.clone()))
        return np.zeros(10, np.float32)


def fake_dump(seed=1):
    r = np.random.default_rng(seed)
    return {'crt_first_cb': 1,
            'retrieved_codes': r.integers(0, V, (32, T)).astype(np.int16),
            'src_codes': r.integers(0, V, (32, T)).astype(np.int16),
            'crt_logits': r.normal(size=(NCB, T, V)).astype(np.float16),
            'ref_codes': r.integers(0, V, (TREF, 32)).astype(np.int16),
            'cand_idx': r.integers(0, TREF, (T, 8)), 'cand_sim': r.random((T, 8)).astype(np.float32),
            'cand_logp': r.normal(size=(NCB, 8, T)).astype(np.float32), 'path': r.integers(0, TREF, T)}


def speech_stats(lab, a, ref):
    """(speech-frame mean of W a_ref, its top-k basis and variances) as run_pair computes them."""
    W = lab.ac.output_proj.weight[:, :, 0].detach()
    yr = lab.Edec[torch.arange(31)[:, None], torch.as_tensor(ref[:, 1:].T.astype(np.int64))].sum(0)
    nr = yr.norm(dim=1)
    yr = yr[nr > nr.quantile(0.2)].double().numpy()
    return yr, W


def test_src_cont_is_the_source_latent_with_its_cb0():
    lab, z = FakeLab(), fake_dump()
    cdl.run_pair(lab, z, ['src_cont'])
    cb0, a = lab.calls[0]
    np.testing.assert_array_equal(cb0.numpy(), z['src_codes'][0])
    np.testing.assert_allclose(a.numpy(), lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64))).numpy())


@pytest.mark.parametrize('k', [0, 3, 6])
def test_ot_src_moments_in_decoder_space(monkeypatch, k):
    """ot<k>_src_cont with beta = 0: the output's W a has the reference's speech-frame mean and top-k covariance,
    the source CB0 is kept, and the correction survives the back-projection through pinv(W) exactly."""
    monkeypatch.setattr(cdl, 'OT_BETA', 0.0)
    monkeypatch.setattr(cdl.gaussian_ot, '__defaults__', (False, 0.0))
    lab, z = FakeLab(), fake_dump()
    stats = cdl.run_pair(lab, z, [f'ot{k}_src_cont'])[f'ot{k}_src_cont'][1]
    cb0, a = lab.calls[0]
    np.testing.assert_array_equal(cb0.numpy(), z['src_codes'][0])
    yr, W = speech_stats(lab, a, z['ref_codes'])
    y = (a @ W.T).double().numpy()
    a_src = lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64)))
    y0 = (a_src @ W.T).double().numpy()
    n0 = np.linalg.norm(y0, axis=1)
    sp = n0 >= np.quantile(n0, 0.2)                     # the speech frames of the source latent
    np.testing.assert_allclose(y[sp].mean(0), yr.mean(0), atol=1e-4)
    if k:
        ev, U = np.linalg.eigh((yr - yr.mean(0)).T @ (yr - yr.mean(0)) / len(yr))
        P = (y[sp] - y[sp].mean(0)) @ U[:, -k:]
        np.testing.assert_allclose(P.T @ P / sp.sum(), np.diag(ev[-k:]), atol=1e-3)
    assert stats['ot_k'] == k


def test_ot_nm_keeps_the_speech_mean():
    lab, z = FakeLab(), fake_dump()
    cdl.run_pair(lab, z, ['ot3nm_src_cont'])
    _, a = lab.calls[0]
    W = lab.ac.output_proj.weight[:, :, 0].detach()
    a_src = lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64)))
    y, y0 = (a @ W.T).double().numpy(), (a_src @ W.T).double().numpy()
    n0 = np.linalg.norm(y0, axis=1)
    sp = n0 >= np.quantile(n0, 0.2)
    np.testing.assert_allclose(y[sp].mean(0), y0[sp].mean(0), atol=1e-4)


def test_ot_on_a_crt_base_uses_the_retrieved_cb0_slot():
    """Non-source bases decode with retr[0], which --cb0 src has already replaced by the source CB0 (with_anchor)."""
    lab, z = FakeLab(), fake_dump()
    cdl.run_pair(lab, z, ['ot3_pm_cont_T1'])
    cb0, a = lab.calls[0]
    np.testing.assert_array_equal(cb0.numpy(), z['retrieved_codes'][0])
    assert a.shape == (T, DAC) and torch.isfinite(a).all()


@pytest.mark.parametrize('k', [0, 4, 10])
def test_otf_maps_the_whole_decoder_input(monkeypatch, k):
    """otf<k>_src_cont with beta = 0: the decoded input has the reference's speech-frame mean and top-k covariance of
    the reference's full decoder inputs (semantic + acoustic), on the speech frames of the source's acoustic part."""
    monkeypatch.setattr(cdl.gaussian_ot, '__defaults__', (False, 0.0))
    lab, z = FakeLab(), fake_dump()
    st = cdl.run_pair(lab, z, [f'otf{k}_src_cont'])[f'otf{k}_src_cont'][1]
    tag, x = lab.calls[0]
    assert tag == 'input' and st['ot_full'] and st['ot_k'] == k
    W = lab.ac.output_proj.weight[:, :, 0].detach()
    ref = torch.as_tensor(z['ref_codes'].T.astype(np.int64))
    a_ref = lab.latent(ref[1:])
    nr = (a_ref @ W.T).norm(dim=1)
    xr = lab.full_input(ref[0], a_ref)[nr > nr.quantile(0.2)].double().numpy()
    a_src = lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64)))
    n0 = (a_src @ W.T).norm(dim=1).numpy()
    sp = n0 >= np.quantile(n0, 0.2)
    xo = x.double().numpy()
    np.testing.assert_allclose(xo[sp].mean(0), xr.mean(0), atol=1e-4)
    if k:
        ev, U = np.linalg.eigh((xr - xr.mean(0)).T @ (xr - xr.mean(0)) / len(xr))
        P = (xo[sp] - xo[sp].mean(0)) @ U[:, -k:]
        np.testing.assert_allclose(P.T @ P / sp.sum(), np.diag(ev[-k:]), atol=1e-3)


def test_otf_without_transport_is_the_source_input():
    """otf0nm (no mean shift, no subspace) decodes exactly the source's own decoder input."""
    lab, z = FakeLab(), fake_dump()
    cdl.run_pair(lab, z, ['otf0nm_src_cont'])
    _, x = lab.calls[0]
    s = torch.as_tensor(z['src_codes'].astype(np.int64))
    np.testing.assert_allclose(x.numpy(), lab.full_input(s[0], lab.latent(s[1:])).numpy(), atol=1e-6)
