"""Tests of mvc/probe_decoder.py with a toy stand-in for the whole of Mimi (encoder, split RVQ, decoder; no download):
every condition's wav against its route, the 16-bit copy, the decoder-input noise scale, utterance and reference
selection (E-only Dev references), resume and the refused rerun (sources, device, code), rows checked before any
encode and passing the journal's metrics.check_row, the cgroup and mount readers, and the summary's capped WER,
deltas, P1 decisions, the K6 ratio at a common drop and its own-metric reading."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import encode_perturbed as ep  # noqa: E402
import probe_decoder as pdec  # noqa: E402

torch = pytest.importorskip('torch')
import mimi_io  # noqa: E402

BENCH = Path(__file__).resolve().parents[2] / 'benchmark'
OLD = '/datastore/nvquoc/crt/vctk'
EPS = [0.05, 0.1, 0.2, 0.4]


class Codebook(torch.nn.Module):
    def __init__(self, g, size=64, dim=16):
        super().__init__()
        self.register_buffer('embed_sum', 0.15 * torch.randn(size, dim, generator=g))
        self.register_buffer('cluster_usage', torch.ones(size))
        self.epsilon = 1e-5

    @property
    def embed(self):
        return self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]


class Level(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.codebook = Codebook(g)


class RVQ(torch.nn.Module):
    """As HF's MimiResidualVectorQuantizer: bias-free 1x1 projections around n residual codebooks; decode sums only the
    levels it is given."""

    def __init__(self, n, g, dim=16):
        super().__init__()
        self.input_proj = torch.nn.Conv1d(512, dim, 1, bias=False)
        self.output_proj = torch.nn.Conv1d(dim, 512, 1, bias=False)
        self.layers = torch.nn.ModuleList([Level(g) for _ in range(n)])

    def encode(self, x):
        r, out = self.input_proj(x).transpose(1, 2), []
        for lv in self.layers:
            E = lv.codebook.embed
            i = ((r[..., None, :] - E) ** 2).sum(-1).argmin(-1)
            r = r - E[i]
            out.append(i)
        return torch.stack(out)

    def decode(self, codes):
        q = sum(torch.nn.functional.embedding(c, lv.codebook.embed) for c, lv in zip(codes.unbind(1), self.layers))
        return self.output_proj(q.transpose(1, 2))


class Split(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.semantic_residual_vector_quantizer = RVQ(1, g)
        self.acoustic_residual_vector_quantizer = RVQ(31, g)

    def encode(self, x):
        return torch.cat([self.semantic_residual_vector_quantizer.encode(x),
                          self.acoustic_residual_vector_quantizer.encode(x)])

    def decode(self, codes):
        q = self.semantic_residual_vector_quantizer.decode(codes[:, :1])
        if codes.shape[1] > 1:
            q = q + self.acoustic_residual_vector_quantizer.decode(codes[:, 1:])
        return q


class FullMimi(torch.nn.Module):
    """Mimi's module layout at toy size: one 512-d frame per 1920 samples, a split RVQ (1 semantic + 31 acoustic
    levels), 2x upsampling, a decoder transformer and a frame decoder; decode(codes) as HF's MimiModel._decode_frame."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        self.w = torch.nn.Parameter(torch.randn(512, em.HOP, generator=g) * 0.02)
        self.quantizer = Split(g)
        self.tf = torch.nn.Linear(512, 512)
        self.wd = torch.nn.Parameter(torch.randn(512, em.HOP // 2, generator=g) * 0.08)

    def encoder(self, x):
        B, _, L = x.shape
        T = -(-L // em.HOP)
        x = torch.nn.functional.pad(x, (0, T * em.HOP - L))
        return torch.einsum('bth,dh->bdt', x.reshape(B, T, em.HOP), self.w)

    def encoder_transformer(self, h, return_dict=False):
        return (torch.tanh(h),)

    def downsample(self, h):
        return h

    def upsample(self, x):
        return x.repeat_interleave(2, dim=2)

    def decoder_transformer(self, h, return_dict=False):
        return (h + 0.1 * self.tf(h),)

    def decoder(self, h):
        B, _, T = h.shape
        return torch.einsum('bdt,dk->btk', h, self.wd).reshape(B, 1, T * (em.HOP // 2))

    def decode(self, codes):
        emb = self.upsample(self.quantizer.decode(codes))
        h = self.decoder_transformer(emb.transpose(1, 2))[0].transpose(1, 2)
        return type('Out', (), {'audio_values': self.decoder(h)})


def models():
    m = FullMimi().eval()
    return mimi_io.MimiIO('cpu', model=m), pdec.Half(copy.deepcopy(m).half(), 'cpu'), m


def vctk(spk, i):
    return f'{OLD}/{spk}/{spk}_{i:03d}.flac'


def corpus(tmp_path, n_spk=3, n_utt=5):
    """A dev-clean-like shard (only its rows are used) with raw audio, and a 40-row Dev pair list of two speakers."""
    rng = np.random.default_rng(0)
    raw, audio, rows, off = tmp_path / 'raw', {}, [], 0
    for s in range(n_spk):
        for u in range(n_utt):
            n = int(em.SR * (0.6 + 0.3 * u))                       # u = 0 is under min_sec; u = 1 has no transcript
            r = dict(id=f'{10 + s}_7_{u:06d}_000000', speaker=str(10 + s), chapter=str(7 + u % 2), subset='dev-clean',
                     text='' if u == 1 else f'words of {s} {u}', offset=off, frames=em.stored_frames(n), samples=n)
            audio[ep.wav_path(raw, r)] = (0.3 * rng.standard_normal(n)).astype(np.float32)
            rows.append(r)
            off += r['frames']
    d = tmp_path / 'mimi' / 'dev-clean'
    d.mkdir(parents=True)
    em.write_shard(d / 'shard_0000', rows, [np.zeros((r['frames'], 512)) for r in rows],
                   [np.zeros((r['frames'], 32)) for r in rows])
    pairs = []
    for k in range(40):
        s, t = ('pA', 'pB') if k % 2 == 0 else ('pB', 'pA')
        pairs.append(dict(pair_id=f'dev__{s}_{k:03d}__{t}', set='dev', source_speaker=s, source_utt=f'{s}_{k:03d}',
                          source_path=vctk(s, k), source_text=f'sentence {k}', target_speaker=t, wer_primary=k % 3 == 0,
                          E={'paths': [vctk(t, 100 + j) for j in range(12)]}))
    new = str(tmp_path / 'vctk')
    for r in pairs[::20]:
        audio[new + r['source_path'][len(OLD):]] = (0.3 * rng.standard_normal(int(em.SR * 1.7))).astype(np.float32)
    pp = tmp_path / 'dev.jsonl'
    pp.write_text(''.join(json.dumps(r) + '\n' for r in pairs))
    return audio, d, pp, new


def setup(tmp_path, *extra):
    audio, shards, pairs, new = corpus(tmp_path)
    a = pdec.parse(['run', '--shards_dev', str(shards), '--raw', str(tmp_path / 'raw'), '--n_libri', '4', '--pairs',
                    str(pairs), '--path_map', f'{OLD}={new}', '--journal', str(BENCH), '--out', str(tmp_path / 'p1'),
                    '--local_disk', str(tmp_path), *extra])
    return a, audio


def read(path):
    import soundfile as sf
    w, sr = sf.read(path, dtype='float32')
    return w, sr, sf.info(path).subtype


def journal_metrics():
    pytest.importorskip('librosa')
    spec = importlib.util.spec_from_file_location('journal_metrics_test', BENCH / 'metrics.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_condition_is_its_route(tmp_path):
    a, audio = setup(tmp_path)
    io, half, _ = models()
    logs = []
    res = pdec.run(a, io, half, reader=audio.__getitem__, log=logs.append)
    conds = pdec.conditions(EPS)
    out = tmp_path / 'p1'
    assert len(conds) == 15 and res['n_utts'] == 6 and res['parts'] == {'dev-clean': 4, 'vctk-dev': 2}
    assert res['new_wavs'] == 6 * 15 and res['n_rows'] == 90 and 'P1: 6 utterances' in logs[-1]
    st = np.load(out / 'stats.npz')
    items = [json.loads(line) for line in open(out / 'rows_codes32_fp16.jsonl')]
    lats = []
    for it in items:
        src = audio[it['source_wav']]
        lat, codes = io.encode(src)
        lats.append(lat)
        xi = np.random.default_rng(ep.utt_seed(a.seed, it['pair_id'], 0)).standard_normal(lat.shape).astype(np.float32)
        with torch.no_grad():
            dec = io._decode_input(io.decoder_input(lat) + res['dec_noise_scale'] * 0.2 * torch.as_tensor(xi.T[None]))
        want = {'codes32_fp16': half.decode_codes(codes), 'codes32_fp32': io.decode_codes(codes),
                'cont_fp32': io.decode_latent(lat), 'cont_fp16': half.decode_latent(lat),
                'q8': half.decode_codes(codes[:, :8]), 'q16': half.decode_codes(codes[:, :16]),
                'smooth3': io.decode_latent(pdec.smooth3(lat)),
                'noise_eps0.1': io.decode_latent(lat + st['std'] * 0.1 * xi), 'dec_noise_eps0.2': dec}
        got = {}
        for c in conds:
            w, sr, sub = read(out / c / f"{it['pair_id']}.wav")
            assert sr == 24000 and sub == 'PCM_16' and len(w) == len(src)
            got[c] = w
            if c in want:
                assert np.abs(w - np.clip(want[c][:len(src)], -1, 1)).max() < 1e-4, c
        for c in ('q8', 'q16', 'codes32_fp32', 'cont_fp32', 'smooth3'):
            assert np.abs(got[c] - got['codes32_fp16']).max() > 1e-3, c
        for c in ('noise_eps0.05', 'dec_noise_eps0.05', 'cont_fp16'):
            assert np.abs(got[c] - got['cont_fp32']).max() > 1e-4, c
    x = np.concatenate(lats).astype(np.float64)
    assert np.allclose(st['mean'], x.mean(0), atol=1e-5) and np.allclose(st['std'], x.std(0), atol=1e-5)
    assert not list(out.rglob('*.tmp*'))


def test_half_route_is_mimi_io_in_another_dtype():
    io, half, m = models()
    same = pdec.Half(copy.deepcopy(m), 'cpu')
    rng = np.random.default_rng(1)
    lat, codes = io.encode((0.3 * rng.standard_normal(20000)).astype(np.float32))
    assert half.dtype == torch.float16 and same.dtype == torch.float32
    assert np.allclose(same.decode_latent(lat), io.decode_latent(lat), atol=1e-6)
    assert np.allclose(same.decode_codes(codes), io.decode_codes(codes), atol=1e-6)
    d16 = half.decode_latent(lat) - io.decode_latent(lat)
    assert 0 < np.abs(d16).max() < 1e-2


def test_decoder_noise_has_the_decoder_input_energy_of_the_latent_noise():
    io, _, m = models()
    std = np.random.default_rng(2).uniform(0.2, 2.0, 512).astype(np.float32)
    s = pdec.dec_noise_scale(io, std)
    q = m.quantizer
    P = sum(r.output_proj.weight[:, :, 0] @ r.input_proj.weight[:, :, 0]
            for r in (q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer)).detach().double()
    assert s ** 2 == pytest.approx(float(((P * torch.as_tensor(std, dtype=torch.float64)) ** 2).sum()) / 512, rel=1e-5)
    xi = np.random.default_rng(3).standard_normal((4000, 512)).astype(np.float32)
    with torch.no_grad():
        e_lat = float((io.decoder_input(std * xi).double() ** 2).sum()) / 4000
    assert e_lat == pytest.approx(512 * s ** 2, rel=0.05)


def test_resume_new_conditions_and_refused_rerun(tmp_path):
    a, audio = setup(tmp_path)
    io, half, _ = models()
    pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)
    f = next((tmp_path / 'p1' / 'cont_fp32').glob('*.wav'))
    stamp = f.stat().st_mtime_ns
    assert pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)['new_wavs'] == 0
    assert f.stat().st_mtime_ns == stamp
    a.eps = '0.05,0.1,0.2,0.3,0.4'
    res = pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)
    assert res['new_wavs'] == 6 * 2 and len(open(tmp_path / 'p1' / 'rows_all.jsonl').readlines()) == 6 * 17
    a.seed = 1
    with pytest.raises(SystemExit, match='other items, seed; use another --out'):
        pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)


def test_rerun_with_other_sources_device_or_code_is_refused(tmp_path, monkeypatch):
    a, audio = setup(tmp_path)
    io, half, _ = models()
    pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)
    cfg = json.load(open(tmp_path / 'p1' / 'probe_cfg.json'))
    assert cfg['device'] == 'cuda' and len(cfg['code']) == 16
    assert [k for k, _ in cfg['items']][-2:] == ['vctk__pA_000', 'vctk__pA_020']
    assert cfg['items'][-1][1] == f'{tmp_path}/vctk/pA/pA_020.flac'
    calls = []

    def reader(p):
        calls.append(p)
        return audio[p]
    for k, v, what in (('device', 'cpu', 'device'), ('path_map', f'{OLD}={tmp_path}/other', 'items'),
                       ('raw', str(tmp_path / 'raw2'), 'items')):
        b = copy.copy(a)
        setattr(b, k, v)
        with pytest.raises(SystemExit, match=f'other {what}; use another --out'):
            pdec.run(b, io, half, reader=reader, log=lambda m: None)

    def smooth3(lat):
        return lat
    monkeypatch.setattr(pdec, 'smooth3', smooth3)
    with pytest.raises(SystemExit, match='other code; use another --out'):
        pdec.run(a, io, half, reader=reader, log=lambda m: None)
    assert calls == []


def test_rows_are_checked_before_anything_is_encoded(tmp_path):
    a, audio = setup(tmp_path, '--min_sec', '1.7')         # one long utterance per dev-clean speaker: no references
    io, half, _ = models()
    calls = []

    def reader(p):
        calls.append(p)
        return audio[p]
    with pytest.raises(ValueError, match='invalid sim_ref_wavs'):
        pdec.run(a, io, half, reader=reader, log=lambda m: None)
    j = tmp_path / 'broken'
    j.mkdir()
    (j / 'metrics.py').write_text('raise RuntimeError("broken metrics")\n')
    b, _ = setup(tmp_path / 'b', '--journal', str(j))
    with pytest.raises(RuntimeError, match='broken metrics'):
        pdec.run(b, io, half, reader=reader, log=lambda m: None)
    assert calls == [] and not (tmp_path / 'p1').exists() and not (tmp_path / 'b' / 'p1').exists()


def test_given_stats_are_used(tmp_path):
    a, audio = setup(tmp_path)
    sp = tmp_path / 'st.npz'
    np.savez(sp, mean=np.zeros(512, np.float32), std=np.full(512, 0.5, np.float32))
    a.stats = str(sp)
    io, half, _ = models()
    res = pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)
    assert res['stats'] == str(sp) and np.all(np.load(tmp_path / 'p1' / 'stats.npz')['std'] == 0.5)


def test_rows_pass_the_journal_check_and_the_reference_rule(tmp_path):
    a, audio = setup(tmp_path)
    io, half, _ = models()
    res = pdec.run(a, io, half, reader=audio.__getitem__, log=lambda m: None)
    rows = [json.loads(line) for line in open(tmp_path / 'p1' / 'rows_all.jsonl')]
    assert len({(r['pair_id'], r['system']) for r in rows}) == len(rows) == 90
    for c in pdec.conditions(EPS):
        assert [r['system'] for r in map(json.loads, open(tmp_path / 'p1' / f'rows_{c}.jsonl'))] == [c] * 6
    for r in rows:
        assert r['parallel_wav'] is None and r['wer_primary'] is True and 'R' not in r and r['ref_text']
        assert r['wav'] == str(tmp_path / 'p1' / r['system'] / f"{r['pair_id']}.wav")
        assert 1 <= len(r['sim_ref_wavs']) <= 10 and r['source_wav'] not in r['sim_ref_wavs']
        assert all(f"/{r['source_speaker']}/" in p for p in r['sim_ref_wavs'] + [r['source_wav']])
    m = journal_metrics()
    assert res['checked_with'].startswith('metrics.check_row')
    for r in rows:
        m.check_row(m.as_metric_row(r), r.get('R'))
    bad = dict(rows[0], sim_ref_wavs=rows[0]['sim_ref_wavs'] + [rows[0]['source_wav']])
    with pytest.raises(ValueError, match='source_wav is in sim_ref_wavs'):
        m.check_row(bad)
    with pytest.raises(ValueError, match='invalid sim_ref_wavs'):
        pdec.check_row(bad)
    with pytest.raises(ValueError):
        pdec.check_row(dict(rows[0], sim_ref_wavs=[]))


def test_libri_selection_and_reference_pool():
    rows = [dict(id=f'{s}_{u}', speaker=s, text='' if u == 3 else 'a b', samples=int(em.SR * (0.5 + 0.4 * u)))
            for s in ('x', 'y', 'z') for u in range(6)]                 # u < 2 are under 1 s, u = 3 has no transcript
    pick, pool = pdec.select_libri(rows, 7, seed=0)
    assert len(pick) == 7 and len({r['id'] for r in pick}) == 7
    assert all(r['text'] and r['samples'] >= em.SR for r in pick)
    assert [r['speaker'] for r in pick[:6]] == ['x', 'y', 'z'] * 2 and pick[6]['speaker'] == 'x'
    assert pdec.select_libri(rows, 7, seed=0)[0] == pick and pdec.select_libri(rows, 7, seed=1)[0] != pick
    for s, v in pool.items():
        probed = [r['id'] for r in v if r in pick]
        assert sorted(r['id'] for r in v) == [f'{s}_{u}' for u in range(2, 6)]  # long ones, with or without text
        assert [r['id'] for r in v[-len(probed):]] == probed                    # the probed utterances last
    assert len(pdec.select_libri(rows, 100, seed=0)[0]) == 9
    assert pdec.refs_of(['/a/1.wav', '/a/2.wav', '/a//3.wav'], '/a/2.wav', k=5) == ['/a/1.wav', '/a//3.wav']
    assert len(pdec.refs_of([f'/a/{i}' for i in range(20)], '/a/0')) == 10


def test_dev_items_are_the_screen_sources(tmp_path):
    _, _, pairs, new = corpus(tmp_path)
    rows = [json.loads(line) for line in open(pairs)]
    items = pdec.dev_items(rows, 50, pdec.path_mapper(f'{OLD}={new}'))
    assert [i['key'] for i in items] == ['vctk__pA_000', 'vctk__pA_020']
    assert items[0]['path'] == f'{new}/pA/pA_000.flac' and items[0]['text'] == 'sentence 0'
    assert items[0]['refs'] == [f'{new}/pA/pA_{100 + j:03d}.flac' for j in range(10)]
    assert len(pdec.dev_items(rows, 1, pdec.path_mapper(''))) == 1
    assert pdec.path_mapper('')('/x/y') == '/x/y' and pdec.path_mapper('/x=/z')('/x/y') == '/z/y'


def test_dev_references_are_the_speaker_E_only():
    def row(s, k, t, E):
        return dict(source_speaker=s, source_utt=f'{s}_{k:03d}', source_path=f'/v/{s}/{s}_{k:03d}.wav',
                    source_text=f'text {k}', target_speaker=t, E={'paths': E})
    eb, ec = [f'/v/pB/pB_{100 + j}.wav' for j in range(8)], [f'/v/pC/pC_{100 + j}.wav' for j in range(9)]
    rows = [row('pB', k, 'pC', ec) if k < 20 else row('pA', k, 'pB', eb) for k in range(40)]
    items = pdec.dev_items(rows, 50, pdec.path_mapper(''))
    assert [i['key'] for i in items] == ['vctk__pB_000', 'vctk__pA_020']
    assert items[0]['refs'] == eb                                                  # 8 E, not padded with sources
    assert items[1]['refs'] == [f'/v/pA/pA_{k:03d}.wav' for k in range(21, 31)]   # pA has no E: its other sources


def test_smooth3_and_eps_interpolation():
    x = np.arange(12, dtype=np.float32).reshape(4, 3) ** 2
    y = pdec.smooth3(x)
    assert np.allclose(y[1], x[:3].mean(0)) and np.allclose(y[0], x[:2].mean(0)) and np.allclose(y[-1], x[-2:].mean(0))
    assert np.allclose(pdec.smooth3(np.ones((5, 2), np.float32)), 1) and np.allclose(pdec.smooth3(x[:1]), x[:1])
    assert pdec.eps_at_drop([(0.1, 0.15), (0.05, 0.075), (0.2, 0.3), (0.4, 0.6)]) == pytest.approx(0.1 + 0.1 / 3)
    assert pdec.eps_at_drop([(0.05, 0.4)]) == pytest.approx(0.025)
    assert pdec.eps_at_drop([(0.05, 0.0), (0.4, 0.19)]) is None


def test_cgroup_memory_v2_v1_and_host_info(tmp_path):
    d = tmp_path / 'sys' / 'cgroup' / 'job_1'
    (d / 'step_0').mkdir(parents=True)
    for p, lim, cur in ((d, '1000', '400'), (d / 'step_0', 'max', '100')):
        (p / 'memory.max').write_text(lim + '\n')
        (p / 'memory.current').write_text(cur + '\n')
    proc = tmp_path / 'cgroup'
    proc.write_text('0::/job_1/step_0\n')
    assert pdec.cgroup_memory(str(proc), str(tmp_path / 'sys')) == (1000, 400)
    v1 = tmp_path / 'sys' / 'cgroup' / 'memory' / 'slurm' / 'job_2'
    v1.mkdir(parents=True)
    (v1 / 'memory.limit_in_bytes').write_text('2000')
    (v1 / 'memory.usage_in_bytes').write_text('500')
    proc.write_text('7:cpu,cpuacct:/slurm/job_2\n5:memory:/slurm/job_2\n')
    assert pdec.cgroup_memory(str(proc), str(tmp_path / 'sys')) == (2000, 500)
    assert pdec.cgroup_memory(str(tmp_path / 'none'), str(tmp_path / 'sys')) is None
    h = pdec.host_info([str(tmp_path), str(tmp_path / 'missing')], mounts=str(tmp_path / 'none'))
    assert list(h['disk_free_gb']) == [str(tmp_path)] and h['disk_free_gb'][str(tmp_path)] >= 0
    assert h['local_mounts'] == {}


def test_local_mounts_skip_network_and_pseudo_filesystems(tmp_path):
    for d in ('t', 'n', 'l', 'f', 'o'):
        (tmp_path / d).mkdir()
    lines = [f'/dev/sda1 {tmp_path} ext4 rw 0 0', f'tmpfs {tmp_path}/t tmpfs rw 0 0',
             f'srv:/home {tmp_path}/n nfs4 rw 0 0', f'10.0.0.1@o2ib:/fs {tmp_path}/l lustre rw 0 0',
             f'sshfs {tmp_path}/f fuse.sshfs rw 0 0',
             f'overlay {tmp_path}/o overlay rw 0 0', 'proc /proc proc rw 0 0', 'cgroup2 /sys/fs/cgroup cgroup2 rw 0 0',
             f'/dev/sdb1 {tmp_path}/gone xfs rw 0 0']
    mounts = tmp_path / 'mounts'
    mounts.write_text('\n'.join(lines) + '\n')
    m = pdec.local_mounts(str(mounts))
    assert sorted(m) == [str(tmp_path), f'{tmp_path}/t']
    assert m[str(tmp_path)][0] == 'ext4' and m[f'{tmp_path}/t'][0] == 'tmpfs'
    assert m[str(tmp_path)][1] >= 0 and pdec.host_info([], mounts=str(mounts))['local_mounts'] == m


def mrow(pid, system, utmos, sim, errs=0, part='dev-clean'):
    return dict(pair_id=pid, system=system, part=part, utmos=utmos, sim_wavlm=sim,
                wer_large_v3=dict(S=errs, D=0, I=0, N=10, cS=0, cD=0, cI=0, cN=40))


def metric_rows(cont=3.99, a_lat=1.5, a_dec=3.0):
    rows = []
    for k in range(20):
        base = {'codes32_fp16': (4.0, 0.95, 0), 'codes32_fp32': (4.0, 0.95, 0), 'cont_fp32': (cont, 0.945, 0),
                'cont_fp16': (cont - 0.01, 0.945, 0), 'q8': (3.0, 0.80, 15 if k == 0 else 1), 'q16': (3.5, 0.9, 0),
                'smooth3': (3.8, 0.94, 0)}
        for e in EPS:
            base[f'noise_eps{e:g}'] = (cont - a_lat * e, 0.94, 0)
            base[f'dec_noise_eps{e:g}'] = (cont - a_dec * e, 0.93, 0)
        rows += [mrow(f'u{k}', c, u, s, w, 'dev-clean' if k < 15 else 'vctk-dev') for c, (u, s, w) in base.items()]
    return rows + [dict(pair_id='u19', system='smooth3', error='boom')]


def test_summary_capped_wer_deltas_and_decisions(tmp_path):
    d = tmp_path / 'm'
    d.mkdir()
    rows = metric_rows()
    half = len(rows) // 2
    for name, part in (('a.jsonl', rows[:half]), ('b.jsonl', rows[half:])):
        (d / name).write_text(''.join(json.dumps(r) + '\n' for r in part))
    out = tmp_path / 'sum.json'
    pdec.main(['summarize', '--metrics', str(d), '--json', str(out)])
    s = json.load(open(out))
    assert s['n_common'] == 19 and s['n_dropped'] == 1 and list(s['table'])[:2] == ['codes32_fp16', 'codes32_fp32']
    q8 = s['table']['q8']
    assert q8['wer_capped'] == pytest.approx(100 * (10 + 18) / 190, abs=1e-4)          # u0's 15 errors capped at N
    assert q8['d_wer'] == pytest.approx(q8['wer_capped'], abs=1e-4) and q8['d_utmos'] == pytest.approx(-1.0)
    assert q8['d_sim'] == pytest.approx(-0.15) and q8['se_d_utmos'] == pytest.approx(0.0, abs=1e-6)
    assert s['parts']['dev-clean']['q8']['n'] == 15 and s['parts']['vctk-dev']['q8']['n'] == 4
    dec = s['decisions']
    assert dec['continuous_allowed'] is True and dec['cont_fp16_differs'] is False
    assert dec['eps_0_2']['noise'] == pytest.approx(0.1 + 0.1 / 3, abs=1e-4)
    assert dec['eps_0_2']['dec_noise'] == pytest.approx(0.05 + 0.05 / 3, abs=1e-4)
    assert dec['k6_harm_ratio'] == pytest.approx(4.0, abs=1e-3) and dec['k6_arm'] is True
    assert dec['k6_drop_level'] == 0.2 and dec['k6_harm_ratio_method'] == 'MSE ratio at the 0.2 drop'
    assert s['probe_json'] is None and 'k6_harm_ratio_own_metric' not in dec
    assert dec['r5_early_nfe32'] is False and dec['utmos_drop']['noise']['0.4'] == pytest.approx(0.6)


def test_summary_failing_route_and_ratio_at_the_common_drop():
    rows = {(r['pair_id'], r['system']): r for r in metric_rows(cont=3.9, a_lat=0.25, a_dec=0.6)}
    s = pdec.summarize(rows)
    d = s['decisions']
    assert d['continuous_allowed'] is False and d['eps_0_2']['noise'] is None
    assert d['eps_0_2']['dec_noise'] == pytest.approx(0.2 / 0.6, abs=1e-6)
    assert d['k6_harm_ratio_method'] == 'MSE ratio at the common drop k6_drop_level'
    assert d['k6_drop_level'] == pytest.approx(0.1) and d['k6_harm_ratio'] == pytest.approx((0.4 / (0.1 / 0.6)) ** 2)
    assert d['k6_arm'] is True and d['r5_early_nfe32'] is False
    rows[('u0', 'cont_fp16')]['utmos'] = 3.9 - 0.6
    assert pdec.summarize(rows)['decisions']['cont_fp16_differs'] is True
    with pytest.raises(SystemExit, match='no complete rows'):
        pdec.summarize({})


def test_k6_ratio_is_an_mse_ratio_when_the_latent_noise_never_drops_0_2():
    """Drops linear in e (0.45 e latent, 0.75 e decoder-input): the latent space never reaches 0.2 by e = 0.4, the MSE
    ratio at equal harm is (0.75 / 0.45)^2 = 2.78, an arm; the ratio of the summed drops would say 1.67, no arm."""
    rows = {(r['pair_id'], r['system']): r for r in metric_rows(cont=3.9, a_lat=0.45, a_dec=0.75)}
    d = pdec.summarize(rows)['decisions']
    assert d['eps_0_2']['noise'] is None and d['k6_drop_level'] == pytest.approx(0.18)
    assert d['k6_harm_ratio'] == pytest.approx((0.75 / 0.45) ** 2) and d['k6_arm'] is True
    rows = {(r['pair_id'], r['system']): r for r in metric_rows(cont=3.9, a_lat=0.0, a_dec=0.75)}
    d = pdec.summarize(rows)['decisions']
    assert d['k6_harm_ratio'] is None and d['k6_drop_level'] is None and d['k6_arm'] is False


def test_summary_reads_the_noise_scale_for_the_own_metric_reading(tmp_path, capsys):
    d = tmp_path / 'm'
    d.mkdir()
    rows = metric_rows()
    (d / 'a.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    pj = tmp_path / 'probe_decoder.json'
    pj.write_text(json.dumps({'dec_noise_scale': 2.0}))
    out = tmp_path / 'sum.json'
    for spec in (str(d), str(d / 'a.jsonl'), f'{d}/'):
        pdec.main(['summarize', '--metrics', spec, '--json', str(out)])
        s = json.load(open(out))
        dec = s['decisions']
        assert s['probe_json'] == str(pj) and dec['dec_noise_scale'] == 2.0
        assert dec['k6_harm_ratio'] == pytest.approx(4.0, abs=1e-3) and dec['k6_arm'] is True
        assert dec['k6_harm_ratio_own_metric'] == pytest.approx(1.0, abs=1e-3) and dec['k6_arm_own_metric'] is False
        assert dec['eps_0_2_dec_input_units'] == pytest.approx(2 * (0.05 + 0.05 / 3), abs=1e-4)
    assert 'per unit of each space' in capsys.readouterr().out
    pdec.main(['summarize', '--metrics', str(d), '--json', str(out), '--probe_json', ''])
    s = json.load(open(out))
    assert s['probe_json'] is None and 'k6_harm_ratio_own_metric' not in s['decisions']
    other = tmp_path / 'other.json'
    other.write_text(json.dumps({'dec_noise_scale': 0.5}))
    pdec.main(['summarize', '--metrics', str(d), '--json', str(out), '--probe_json', str(other)])
    assert json.load(open(out))['decisions']['k6_harm_ratio_own_metric'] == pytest.approx(16.0, abs=1e-2)
