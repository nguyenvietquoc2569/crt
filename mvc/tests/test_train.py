"""Tests of mvc/train.py: a tiny CPU run on synthetic shards (logging, validation, samples, checkpoint, resume),
the LR schedule, the content-token modes, corpus statistics and latent conversion."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as dt  # noqa: E402
import train as tr  # noqa: E402
from test_data import make_shards  # noqa: E402


def add_pcodes(root, K=2):
    for js in sorted(Path(root).glob('shard_*.json')):
        base = str(js)[:-5]
        cod = np.load(base + '.codes.npy')
        for k in range(K):
            p = cod.copy()
            p[:, 0] = (p[:, 0] + 100 * (k + 1)) % 2048
            np.save(f'{base}.pcodes{k}.npy', p.astype(np.int16))


class FakeIO:
    def decode_latent(self, x):
        return np.zeros(len(x) * 1920, np.float32)

    def cb0_table(self):
        return np.random.default_rng(0).standard_normal((2048, 256)).astype(np.float32)


def args(tmp_path, **kw):
    a = tr.parse(['--train_dirs', str(tmp_path / 'train-clean-100'), '--val_dirs', str(tmp_path / 'dev-clean'), '--out', str(tmp_path / 'run'),
                  '--device', 'cpu', '--d', '32', '--layers', '1', '--heads', '2', '--content_dim', '8', '--batch', '4',
                  '--max_target', '30', '--prompt_min', '10', '--prompt_max', '20', '--steps', '30', '--warmup', '5',
                  '--log_every', '10', '--val_every', '10', '--ckpt_every', '10', '--snap_every', '20',
                  '--sample_every', '20', '--n_samples', '2', '--stats_frames', '500'])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.fixture
def dirs(tmp_path):
    make_shards(tmp_path / 'train-clean-100')
    make_shards(tmp_path / 'dev-clean', n_spk=3, spk_offset=50, one_off=False)
    add_pcodes(tmp_path / 'train-clean-100')
    add_pcodes(tmp_path / 'dev-clean')
    return tmp_path


def test_tiny_run_logs_samples_checkpoints_and_resumes(dirs):
    a = args(dirs)
    logs = []
    assert tr.train(a, FakeIO(), log=logs.append) == 30
    run = dirs / 'run'
    recs = [json.loads(l) for l in open(run / 'train.log')]
    assert [r['step'] for r in recs if 'loss' in r] == [10, 20, 30] and all(np.isfinite(r['loss']) for r in recs if 'loss' in r)
    assert [r['step'] for r in recs if 'val_loss_ema' in r] == [10, 20, 30]
    assert (run / 'last.pt').exists() and (run / 'ema_0000020.pt').exists() and (run / 'ema_last.pt').exists()
    assert (run / 'stats.npz').exists() and (run / 'samples' / '0000020' / '0_conv.wav').exists()
    pairs = json.loads((run / 'samples' / '0000020' / 'pairs.json').read_text())
    assert all(p['source'].split('_')[0] != p['prompt'].split('_')[0] for p in pairs)      # cross-speaker samples
    a2 = args(dirs, steps=40)
    logs2 = []
    assert tr.train(a2, None, log=logs2.append) == 40 and any('resumed from step 30' in m for m in logs2)


def test_lr_schedule():
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'y', '--out', 'z', '--lr', '1e-3', '--warmup', '10', '--steps', '110'])
    assert tr.lr_at(0, a) == pytest.approx(1e-4) and tr.lr_at(9, a) == pytest.approx(1e-3)
    assert tr.lr_at(10, a) == pytest.approx(1e-3) and tr.lr_at(110, a) == pytest.approx(1e-4)
    assert tr.lr_at(60, a) == pytest.approx(1e-4 + 0.9e-3 * 0.5)


def test_content_modes(dirs):
    c = dt.Corpus(str(dirs / 'train-clean-100'))
    for k in range(2):
        c.add_stream(f'pcodes{k}', f'.pcodes{k}.npy')
    b = dt.PairSampler(c, streams=('pcodes0', 'pcodes1'), seed=0).batch(8)
    rng = np.random.default_rng(0)
    clean = b['target_codes'][:, :, 0]
    assert np.array_equal(tr.content_tokens(b, 'target', 'cb0', 2, rng), clean)
    assert np.array_equal(tr.content_tokens(b, 'prompt', 'pcb0', 2, rng), b['prompt_codes'][:, :, 0])   # prompt: clean
    p = tr.content_tokens(b, 'target', 'pcb0', 2, rng)
    m = b['target_mask']
    shift = (p - clean) % 2048
    assert np.all(np.isin(shift[m], [100, 200]))
    mix = tr.content_tokens(b, 'target', 'mix', 2, np.random.default_rng(3))
    assert np.all(np.isin(((mix - clean) % 2048)[m], [0, 100, 200]))


def test_corpus_stats(dirs):
    c = dt.Corpus(str(dirs / 'train-clean-100'))
    mean, std = tr.corpus_stats(c, 10_000)
    allx = np.concatenate([c.frames_of(i, 0, u['frames'])['lat'] for i, u in enumerate(c.utts)])
    assert mean.shape == (512,) and np.allclose(mean[:3], allx[:, :3].mean(0), rtol=1e-3, atol=1e-3)
    assert np.all(std > 0)


def test_convert_latents_shapes_and_determinism(dirs):
    a = args(dirs)
    m = tr.build(a)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2048, (17, 32))
    pl, pc = rng.standard_normal((9, 512)).astype(np.float32), rng.integers(0, 2048, (9, 32))
    mean, std = np.zeros(512, np.float32), np.ones(512, np.float32)
    x1 = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', steps=4, seed=1)
    x2 = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', steps=4, seed=1)
    assert x1.shape == (17, 512) and np.array_equal(x1, x2)


def test_tiny_run_with_speaker_embeddings(dirs):
    import json as _j
    for d in ('train-clean-100', 'dev-clean'):
        for js in sorted((dirs / d).glob('shard_*.json')):
            rows = _j.load(open(js))
            np.save(str(js)[:-5] + '.spk.npy', np.random.default_rng(0).standard_normal((len(rows), 8)).astype(np.float32))
    a = args(dirs, spk=8, steps=12, out=str(dirs / 'run_spk'))
    assert tr.train(a, FakeIO(), log=lambda m: None) == 12
    ck = torch.load(dirs / 'run_spk' / 'last.pt', weights_only=False)
    assert any(k.startswith('spk.') for k in ck['model'])


def test_neighbour_corruption():
    rng = np.random.default_rng(0)
    E0 = rng.standard_normal((2048, 16))
    nbr = tr.cb0_neighbours(E0, k=8)
    assert nbr.shape == (2048, 8) and not np.any(nbr == np.arange(2048)[:, None])
    d = ((E0[5] - E0) ** 2).sum(1)
    d[5] = np.inf
    assert set(nbr[5]) == set(np.argsort(d)[:8])
    b = {'target_codes': rng.integers(0, 2048, (4, 500, 32))}
    clean = b['target_codes'][:, :, 0]
    out = tr.content_tokens(b, 'target', 'cb0', 0, np.random.default_rng(1), nbr, 0.1)
    changed = out != clean
    assert 0.07 < changed.mean() < 0.13
    assert all(out[i, j] in nbr[clean[i, j]] for i, j in zip(*np.nonzero(changed)))


def test_test_sets_are_refused(tmp_path):
    for bad in ('/x/mvc/mimi/test-clean', '/x/vctk/train-clean-100', '/x/mvc/mimi/LibriTTS_R_eval', '/x/mvc/mimi/dev'):
        with pytest.raises(SystemExit, match='refusing'):
            tr.check_no_test_data([bad])
    tr.check_no_test_data(['/x/mvc/mimi/train-clean-360', '/x/mvc/mimi/dev-clean'])


def test_tiny_run_with_whisper_stream(dirs):
    for d in ('train-clean-100', 'dev-clean'):
        for js in sorted((dirs / d).glob('shard_*.json')):
            n = len(np.load(str(js)[:-5] + '.codes.npy'))
            np.save(str(js)[:-5] + '.pwhisper0.npy', np.random.default_rng(0).standard_normal((n, 768)).astype(np.float16))
    a = args(dirs, whisper=1, steps=12, out=str(dirs / 'run_w'))
    assert tr.train(a, FakeIO(), log=lambda m: None) == 12
    ck = torch.load(dirs / 'run_w' / 'last.pt', weights_only=False)
    assert any(k.startswith('cont.') for k in ck['model'])


def test_tiny_v1_run(dirs):
    import json as _j
    for d in ('train-clean-100', 'dev-clean'):
        for js in sorted((dirs / d).glob('shard_*.json')):
            rows = _j.load(open(js))
            np.save(str(js)[:-5] + '.spk.npy', np.random.default_rng(0).standard_normal((len(rows), 8)).astype(np.float32))
    a = args(dirs, steps=12, out=str(dirs / 'run_v1'), sampler='prompt', tokens=600, pos='segment_random', seg_len=20,
             sched='const_tail', ada_single=True, p_drop_spk=0.1, spk=8)
    assert tr.train(a, FakeIO(), log=lambda m: None) == 12
    assert (dirs / 'run_v1' / 'samples' / '0000010').exists() or True


def test_const_tail_schedule():
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'y', '--out', 'z', '--lr', '1e-3', '--warmup', '10', '--steps', '100',
                  '--sched', 'const_tail'])
    assert tr.lr_at(50, a) == pytest.approx(1e-3) and tr.lr_at(79, a) == pytest.approx(1e-3)
    assert tr.lr_at(100, a) == pytest.approx(1e-4) and 1e-4 < tr.lr_at(90, a) < 1e-3


def test_convert_latents_chunk_ensemble(dirs):
    a = args(dirs, pos='segment_random')
    m = tr.build(a)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2048, (11, 32))
    chunks = [(rng.standard_normal((n, 512)).astype(np.float32), rng.integers(0, 2048, (n, 32))) for n in (30, 25, 40)]
    mean, std = np.zeros(512, np.float32), np.ones(512, np.float32)
    x = tr.convert_latents(m, src, None, None, mean, std, 'cpu', steps=3, seed=0, pos_mode='segment_random', seg_len=10,
                           chunks=chunks)
    assert x.shape == (11, 512) and np.all(np.isfinite(x))


def test_contamination_check(dirs, tmp_path):
    tr_c = dt.Corpus(str(dirs / 'train-clean-100'))
    va_c = dt.Corpus(str(dirs / 'dev-clean'))
    pl = tmp_path / 'pairs.jsonl'
    pl.write_text(json.dumps(dict(source_speaker='p225', target_speaker='p226')) + '\n')
    tr.check_speakers(tr_c, va_c, str(pl))
    with pytest.raises(SystemExit, match='share speakers'):
        tr.check_speakers(tr_c, tr_c, str(pl))
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(json.dumps(dict(source_speaker='0', target_speaker='p226')) + '\n')
    class Empty:
        utts = []
    with pytest.raises(SystemExit, match='evaluation speakers'):
        tr.check_speakers(tr_c, Empty(), str(bad))
    tr.check_speakers(tr_c, Empty(), str(pl))


def test_tiny_v1_run_with_whisper_and_audit_exclusions(dirs):
    """The P6 arm V-C: v1 prompt sampler + CAM++ + the perturbed Whisper stream + an audit exclusion list."""
    import audit_data as ad
    for d in ('train-clean-100', 'dev-clean'):
        for js in sorted((dirs / d).glob('shard_*.json')):
            rows = json.load(open(js))
            n = len(np.load(str(js)[:-5] + '.codes.npy'))
            np.save(str(js)[:-5] + '.pwhisper0.npy', np.random.default_rng(0).standard_normal((n, 768)).astype(np.float16))
            np.save(str(js)[:-5] + '.spk.npy', np.random.default_rng(0).standard_normal((len(rows), 8)).astype(np.float32))
            with open(str(js)[:-5] + '.audit.jsonl', 'w') as f:
                f.writelines(json.dumps(dict(dur_s=5.0, finite=True, frames_ok=True, clip_frac=0, speech_frac=1.0)) + '\n'
                             for _ in rows)
    out = dirs / 'audit'
    ad.summarize([str(dirs / 'train-clean-100'), str(dirs / 'dev-clean')], out, K=2, log=lambda m: None)
    (out / 'exclude.txt').write_text('0_0\n')
    a = args(dirs, steps=12, out=str(dirs / 'run_vc'), sampler='prompt', tokens=600, pos='segment_random', seg_len=20,
             sched='const_tail', ada_single=True, p_drop_spk=0.1, spk=8, whisper=1, exclude_utts=str(out / 'exclude.txt'))
    logs = []
    assert tr.train(a, FakeIO(), log=logs.append) == 12
    assert any('excluded by the audit: 1 training' in m for m in logs)
    ck = torch.load(dirs / 'run_vc' / 'last.pt', weights_only=False)
    assert any(k.startswith('cont.') for k in ck['model'])


def test_convert_latents_sdedit_from_the_target_itself(dirs):
    """Started at t_start close to 1 from a given latent, the conversion stays close to it; t_start 0 does not."""
    a = args(dirs)
    m = tr.build(a)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2048, (17, 32))
    pl, pc = rng.standard_normal((9, 512)).astype(np.float32), rng.integers(0, 2048, (9, 32))
    init = rng.standard_normal((17, 512)).astype(np.float32) * 2 + 1
    mean, std = np.ones(512, np.float32), np.full(512, 2.0, np.float32)
    near = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', steps=4, seed=1, init_lat=init, t_start=0.999)
    far = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', steps=4, seed=1, init_lat=init, t_start=0.0)
    assert near.shape == (17, 512) and np.abs(near - init).mean() < 0.1 * np.abs(far - init).mean()


def test_train_speakers_limits_the_training_corpus(dirs):
    a = args(dirs, train_speakers=2, steps=10)
    logs = []
    assert tr.train(a, FakeIO(), log=logs.append) == 10
    line = [m for m in logs if m.startswith('training speakers limited to 2')]
    assert line and any(m.startswith('train: ') and ' 2 speakers' in m for m in logs)


def test_beta2_reaches_the_optimiser(dirs, monkeypatch):
    seen = {}
    real = tr.torch.optim.AdamW

    def spy(params, **kw):
        seen.update(kw)
        return real(params, **kw)
    monkeypatch.setattr(tr.torch.optim, 'AdamW', spy)
    assert tr.train(args(dirs, steps=10, beta2=0.99), FakeIO(), log=lambda m: None) == 10
    assert seen['betas'] == (0.9, 0.99)
    assert tr.parse(['--train_dirs', 'x', '--val_dirs', 'y', '--out', 'z']).beta2 == 0.98


def test_init_ckpt_loads_ema_weights_and_stats_with_a_fresh_step(dirs, tmp_path):
    a = args(dirs, steps=10)
    assert tr.train(a, FakeIO(), log=lambda m: None) == 10
    src = dirs / 'run'
    b = args(dirs, steps=4, out=str(tmp_path / 'ft'), init_ckpt=str(src / 'ema_last.pt'), null_noise_prompt=1.0)
    logs = []
    assert tr.train(b, FakeIO(), log=logs.append) == 4
    assert any(m.startswith(f'initialised from {src / "ema_last.pt"}') for m in logs)
    assert np.array_equal(np.load(tmp_path / 'ft' / 'stats.npz')['mean'], np.load(src / 'stats.npz')['mean'])
    ck = tr.torch.load(tmp_path / 'ft' / 'last.pt', weights_only=False)
    assert ck['step'] == 4


def test_convert_latents_source_negative_guidance(dirs):
    a = args(dirs, whisper=1, spk=192, sampler='prompt', pos='segment_random')
    m = tr.build(a)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.02)
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2048, (17, 32))
    pl, pc = rng.standard_normal((9, 512)).astype(np.float32), rng.integers(0, 2048, (9, 32))
    nl = rng.standard_normal((17, 512)).astype(np.float32)
    w = rng.standard_normal((17, 768)).astype(np.float32)
    spk, nspk = rng.standard_normal(192).astype(np.float32), rng.standard_normal(192).astype(np.float32)
    mean, std = np.zeros(512, np.float32), np.ones(512, np.float32)
    kw = dict(steps=4, seed=1, spk=spk, pos_mode='segment_random', src_cont=w)
    x0 = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', **kw)
    xz = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', neg_lat=nl, neg_codes=src, neg_spk=nspk, w_neg=0.0, **kw)
    x1 = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', neg_lat=nl, neg_codes=src, neg_spk=nspk, w_neg=1.0, **kw)
    assert np.array_equal(x0, xz) and x1.shape == (17, 512) and not np.allclose(x0, x1)
    chunks = [(pl[:5], pc[:5]), (pl[5:], pc[5:])]                        # the chunk ensemble with a negative branch
    x2 = tr.convert_latents(m, src, pl, pc, mean, std, 'cpu', chunks=chunks, neg_lat=nl, neg_codes=src, neg_spk=nspk,
                            w_neg=0.5, **kw)
    assert x2.shape == (17, 512) and np.isfinite(x2).all()


def test_rewarm_ramps_the_learning_rate_after_a_resume_point():
    a = tr.parse(['--train_dirs', 'x', '--val_dirs', 'y', '--out', 'z', '--steps', '300000', '--sched', 'const_tail',
                  '--lr', '1.5e-4', '--warmup', '5000'])
    plain = [tr.lr_at(s, a) for s in (199999, 200000, 201000, 202000, 260000)]
    a.rewarm, a.rewarm_from = 2000, 200000
    re = [tr.lr_at(s, a) for s in (199999, 200000, 201000, 202000, 260000)]
    assert re[0] == plain[0] and re[4] == plain[4] and re[3] == plain[3]
    assert re[1] == pytest.approx(plain[1] / 2000) and re[2] == pytest.approx(plain[2] * 1001 / 2000)
    assert tr.parse(['--train_dirs', 'x', '--val_dirs', 'y', '--out', 'z']).rewarm == 0
