"""Tests of mvc/decoder_ft.py (R5): the mel loss, the multi-scale STFT discriminator and its losses, frame-aligned
GTA pairs with margins and real-latent mixing, and a tiny end-to-end fine-tune on fake decoder modules."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decoder_ft as df  # noqa: E402
import encode_mimi as em  # noqa: E402

HOP = df.HOP


def test_mel_loss_and_discriminator():
    mel = df.MultiMel(torch, 'cpu')
    y = torch.randn(2, 24000)
    assert float(mel(y, y)) == pytest.approx(0.0, abs=1e-6) and float(mel(0.5 * y, y)) > 0.05
    disc = df.make_discriminator(torch)
    out = disc(y)
    assert len(out) == 5 and all(o.shape[0] == 2 and len(f) == 5 for o, f in out)
    y_hat = (0.3 * torch.randn(2, 24000)).requires_grad_()
    d = df.d_loss_fn(disc, y_hat, y)
    adv, fm = df.g_loss_fn(torch, disc, y_hat, y)
    (adv + fm).backward()
    assert torch.isfinite(d) and torch.isfinite(adv) and torch.isfinite(fm) and y_hat.grad.abs().sum() > 0


def write_data(tmp_path, n_utt=6, frames=40):
    """One shard of real latents (value = 1) + a GTA npz (value = 2) + wavs whose samples encode the frame index."""
    shard = tmp_path / 'train-clean-100'
    shard.mkdir()
    rows, lats, cods, off, wavs = [], [], [], 0, {}
    for u in range(n_utt):
        rows.append(dict(id=f's_c_{u}', speaker='s', chapter='c', subset='train-clean-100', text='', offset=off,
                         frames=frames, samples=(frames + 2) * HOP))
        lats.append(np.ones((frames, 512), np.float32))
        cods.append(np.zeros((frames, 32), np.int64))
        wavs[f's_c_{u}'] = (np.arange((frames + 2) * HOP) // HOP).astype(np.float32)
        off += frames
    em.write_shard(shard / 'shard_0000', rows, lats, cods)
    gta = tmp_path / 'gta'
    gta.mkdir()
    np.savez(gta / 'gta_0000.npz', ids=np.array([r['id'] for r in rows]), offsets=np.arange(n_utt) * frames,
             frames=np.full(n_utt, frames), lat=np.full((n_utt * frames, 512), 2.0, np.float16))
    reader = lambda p: wavs[Path(p).stem]  # noqa: E731
    return shard, gta, reader


def test_pairs_are_frame_aligned_with_margins(tmp_path):
    shard, gta, reader = write_data(tmp_path)
    data = df.Pairs(gta, [str(shard)], '/raw', reader=reader)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        lat, y, s0 = data.sample(rng, frames=8, margin=3, p_real=0.5)
        a = int(y[0])                                               # the wav's first sample is its frame index
        assert np.array_equal(y, np.repeat(np.arange(a, a + 8), HOP).astype(np.float32))
        assert 0 <= s0 <= 3 and len(lat) == min(40, a + 8 + 3) - max(0, a - 3) and s0 == a - max(0, a - 3)
        seen.add(float(lat[0, 0]))
    assert seen == {1.0, 2.0}                                      # real and generated latents both appear


class FakeIO:
    """Mimi's decode path in miniature: 512 -> 8 channels, x2 upsample, a 'transformer', x960 to samples."""

    def __init__(self):
        nn = torch.nn
        self.proj = nn.Linear(512, 8)
        for p in self.proj.parameters():
            p.requires_grad_(False)

        class Tr(nn.Module):
            def __init__(self):
                super().__init__()
                self.l = nn.Linear(8, 8)

            def forward(self, x, return_dict=False):
                return (self.l(x),)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = nn.ConvTranspose1d(8, 8, 2, 2)
                self.decoder_transformer = Tr()
                self.decoder = nn.ConvTranspose1d(8, 1, 960, 960)

        self.m = M()

    def decoder_input(self, x):
        return self.proj(torch.as_tensor(np.asarray(x), dtype=torch.float32)).T[None]


def test_tiny_fine_tune_writes_decoder_checkpoints(tmp_path):
    shard, gta, reader = write_data(tmp_path)
    a = df.parse(['--gta', str(gta), '--shards', str(shard), '--raw', '/raw', '--out', str(tmp_path / 'out'),
                  '--device', 'cpu', '--steps', '4', '--batch', '2', '--frames', '4', '--margin', '1', '--d_warmup', '2',
                  '--log_every', '2', '--ckpt_every', '2'])
    a.reader = reader
    io = FakeIO()
    before = io.m.decoder.weight.detach().clone()
    assert df.train(a, io=io, log=lambda m: None) == 4
    ck = torch.load(tmp_path / 'out' / 'decoder_last.pt', weights_only=False)
    assert ck['step'] == 4 and set(ck['decoder']) == {'upsample', 'decoder_transformer', 'decoder'}
    assert not torch.equal(io.m.decoder.weight, before)                       # the generator trained after warm-up
    logs = [json.loads(l) for l in open(tmp_path / 'out' / 'train.log')]
    assert logs[0]['warm'] is True and logs[-1]['warm'] is False and 'mel' in logs[-1]
    assert (tmp_path / 'out' / 'decoder_000002.pt').exists()


def test_prefix_pairs_and_window_decode_match_the_full_decode(tmp_path):
    shard, gta, reader = write_data(tmp_path)
    data = df.Pairs(gta, [str(shard)], '/raw', reader=reader)                    # frame-index audio: no int16 cache
    rng = np.random.default_rng(1)
    io = FakeIO()
    mods = dict(upsample=io.m.upsample, decoder_transformer=io.m.decoder_transformer, decoder=io.m.decoder)
    for _ in range(10):
        lat, y, s0, is_real = data.sample(rng, frames=8, margin=3, prefix=True, with_real=True)
        a = int(y[0])
        assert s0 == a and len(lat) == a + 8 and isinstance(is_real, bool)       # utterance start .. window end
        with torch.no_grad():
            w = df.decode_windows(torch, io, mods, [lat], [s0], 8, prefix=True)[0]
            full = io.m.decoder(io.m.decoder_transformer(io.m.upsample(io.decoder_input(lat)).transpose(1, 2))[0]
                                .transpose(1, 2))[0, 0][s0 * HOP:(s0 + 8) * HOP]
        assert torch.allclose(w, full, atol=1e-5)
    quiet = lambda p: reader(p) / 100.0  # noqa: E731
    cached = df.Pairs(gta, [str(shard)], '/raw', reader=quiet, cache_wav=True)
    p = cached.items[0]['wav']
    assert np.allclose(cached.wav(p), quiet(p), atol=1 / 32767) and p in cached.cache   # the int16 cache reads back


def test_r5b_losses_pair_only_real_rows_and_anchor_starts_at_zero(tmp_path):
    disc = df.make_discriminator(torch)
    y = torch.randn(3, 24000)
    y_hat = (0.3 * torch.randn(3, 24000)).requires_grad_()
    adv, fm = df.g_loss_fn(torch, disc, y_hat, y, fm_rows=torch.tensor([False, False, False]))
    assert float(fm) == 0.0
    fm.backward()
    assert y_hat.grad is None or float(y_hat.grad.abs().sum()) == 0.0        # no paired gradient on an all-GTA batch
    anc = df.Anchor(torch, 'cpu')
    assert float(anc(y, y)) == 0.0 and float(anc(0.5 * y, y)) > 0.1
    shard, gta, reader = write_data(tmp_path)
    a = df.parse(['--gta', str(gta), '--shards', str(shard), '--raw', '/raw', '--out', str(tmp_path / 'out'),
                  '--device', 'cpu', '--steps', '3', '--batch', '2', '--frames', '4', '--d_warmup', '1', '--log_every', '1',
                  '--ckpt_every', '3', '--pair_losses', 'real', '--w_anchor', '1', '--prefix', '--cache_wav',
                  '--p_real', '0'])
    a.reader = reader
    assert df.train(a, io=FakeIO(), log=lambda m: None) == 3
    logs = [json.loads(l) for l in open(tmp_path / 'out' / 'train.log')]
    first = [r for r in logs if not r['warm']][0]
    assert first['mel'] == 0.0 and first['fm'] == 0.0 and first['anchor'] == pytest.approx(0.0, abs=1e-6)
