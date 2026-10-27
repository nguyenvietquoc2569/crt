"""Tests of mvc/probe_content_ctc.py on synthetic shards whose CB0, perturbed CB0, soft profile and latent spell the
transcript one character per frame (no Mimi download, no GPU): text normalisation, the CTC length guard and the
selection, greedy decoding and CER, the soft stream and soft bag, the probe's shapes and padding independence, a tiny
probe that learns to read, the gates, resume and idempotence of the JSON, streams split across runs, the pcb0 idf of
the soft bags, the early check of the arrays, and the VCTK inputs with the causal stand-in for Mimi."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import encode_perturbed as ep  # noqa: E402
import probe_content_ctc as pc  # noqa: E402
import probe_speaker_info as ps  # noqa: E402
from test_encode_mimi import FakeMimi  # noqa: E402

LETTERS = list("abcdefghijklmnopqrstuvwxyz'")
SIL = 5


def make_shards(root, prefix, n_spk=4, n_utt=16, seed=0, odd_rows=False):
    """Two shards. Frame t of an utterance holds character t of its normalised text (plus a silence frame at each
    end): CB0 = 100 + symbol, perturbed CB0 the same with 10 % random tokens, soft top-1 = perturbed CB0 at d2 near 1,
    latent a fixed random (rank-16) vector per symbol plus noise. odd_rows adds one utterance over 4 s and one CTC
    cannot align."""
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(99)
    P = (g.standard_normal((len(pc.CHARS) + 1, 16)) @ g.standard_normal((16, 512)) / 4).astype(np.float32)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    utts = [(s, u, ' '.join(''.join(rng.choice(LETTERS, rng.integers(1, 6))) for _ in range(rng.integers(2, 5)))
             .capitalize() + '.') for s in range(n_spk) for u in range(n_utt)]
    if odd_rows:
        utts += [(0, 90, 'The cat sat on the big red mat, look! ' * 2), (0, 91, 'Look look look.')]
    half = len(utts) // 2
    for si, part in enumerate((utts[:half], utts[half:])):
        rows, lats, cods, pcods, sidx, sd2, off = [], [], [], [], [], [], 0
        for s, u, text in part:
            sym = np.r_[0, pc.encode_text(pc.normalise(text)), 0]
            if u == 91:
                sym = sym[:3]                                      # 3 frames for 13 characters
            T = len(sym)
            tok = np.where(sym > 0, 100 + sym, SIL)
            cod = rng.integers(0, 2048, (T, 32))
            cod[:, 0] = tok
            pcod = cod.copy()
            noise = rng.random(T) < 0.1
            pcod[noise, 0] = rng.integers(0, 2048, noise.sum())
            idx = rng.integers(0, 2048, (T, 8))
            idx[:, 0] = pcod[:, 0]
            d2 = np.sort(2 + 3 * rng.random((T, 8)), axis=1)
            d2[:, 0] = 1 + 0.1 * rng.random(T)
            rows.append(dict(id=f'{prefix}{s}_{u}', speaker=f'{prefix}{s}', chapter=str(u % 2), subset='x', text=text,
                             offset=off, frames=T, samples=T * em.HOP))
            lats.append(P[sym] + 0.1 * rng.standard_normal((T, 512)))
            cods.append(cod)
            pcods.append(pcod)
            sidx.append(idx)
            sd2.append(d2)
            off += T
        base = str(root / f'shard_{si:04d}')
        em.write_shard(base, rows, lats, cods)
        np.save(base + '.pcodes0.npy', np.concatenate(pcods).astype(np.int16))
        np.save(base + '.psoft0.idx.npy', np.concatenate(sidx).astype(np.int16))
        np.save(base + '.psoft0.d2.npy', np.concatenate(sd2).astype(np.float16))
    return root


@pytest.fixture
def dirs(tmp_path):
    make_shards(tmp_path / 'train', 'tr', n_utt=30, odd_rows=True)
    make_shards(tmp_path / 'dev', 'dv', n_utt=6, seed=1)
    return tmp_path


def argv(d, out='p2', **kw):
    opts = dict(train=d / 'train', dev=d / 'dev', out=d / out, device='cpu', d=32, layers=1, heads=2, ff=64,
                dropout=0.0, batch=16, steps=200, warmup=20, lr=3e-3, max_sec=4, n_eval=20, log_every=100,
                ckpt_every=100, hours=10)
    opts.update(kw)
    return [x for k, v in opts.items() for x in (f'--{k}', str(v))]


def test_normalise_labels_and_greedy():
    assert pc.normalise("Hello, World!  It's 3 O'Clock--now.\n") == "hello world it's o'clock now"
    assert pc.normalise('Café, naïve') == 'caf na ve'
    assert len(pc.CHARS) == 28 and pc.encode_text("ab 'z").tolist() == [3, 4, 1, 2, 28]
    assert pc.greedy([0, 3, 3, 0, 3, 1, 1, 0, 4]) == 'aa b' and pc.greedy([]) == '' and pc.greedy([0, 0]) == ''


def test_ctc_guard_and_selection():
    assert pc.ctc_frames_needed(pc.encode_text('hello')) == 6 and pc.ctc_frames_needed(pc.encode_text('ab')) == 2
    assert pc.why_skip(dict(text='hello', frames=1), 50) == 'ctc'          # 4 output frames < 6
    assert pc.why_skip(dict(text='hello', frames=2), 50) is None
    assert pc.why_skip(dict(text='hello', frames=60), 50) == 'too_long'
    assert pc.why_skip(dict(text='!!! 42', frames=10), 50) == 'empty'
    rows = [dict(text='a b', frames=5, samples=5 * em.HOP) for _ in range(20)]
    rows.append(dict(text='hello', frames=1, samples=em.HOP))
    keep, info = pc.select(rows, 50, seed=0, n=8)
    assert len(keep) == 8 == len(set(keep)) and 20 not in keep
    assert info['skipped'] == {'ctc': 1} and info['n_skipped'] == 1 and info['n_rows'] == 21
    assert pc.select(rows, 50, seed=0, n=8)[0] == keep != pc.select(rows, 50, seed=1, n=8)[0]
    keep, info = pc.select(rows, 50, seed=0, hours=12 * em.HOP / em.SR / 3600)
    assert len(keep) == 3 and info['hours'] == pytest.approx(15 * em.HOP / em.SR / 3600, abs=1e-3)


def naive_edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[-1]


def test_edit_distance_and_cer():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = ''.join(rng.choice(list('ab c'), rng.integers(0, 12)))
        b = ''.join(rng.choice(list('ab c'), rng.integers(0, 12)))
        assert pc.edit_distance(a, b) == naive_edit(a, b)
    assert pc.edit_distance('kitten', 'sitting') == 3 and pc.edit_distance('', 'abc') == 3
    assert pc.cer(['abc', 'de'], ['abd', 'de']) == pytest.approx(1 / 5) and pc.cer(['abc'], ['']) == 1.0


def test_soft_weights_and_bag():
    d2 = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 9.0]], np.float16)
    w = pc.soft_weights(d2, 1.0, 2.0)
    assert np.allclose(w.sum(1), 1) and w[0, 0] > w[0, 1] > w[0, 2] and w[1, 0] == pytest.approx(w[1, 1])
    assert np.allclose(w[0], np.exp([-0.5, -1, -2]) / np.exp([-0.5, -1, -2]).sum(), atol=1e-6)
    assert w[0, 0] > pc.soft_weights(d2, 2.0, 2.0)[0, 0]                     # a higher T is softer
    sharp = pc.soft_weights(d2[:1], 1e-3, 1.0)
    assert np.allclose(sharp, [[1, 0, 0]])
    idx = np.array([[7, 3, 9], [3, 7, 1], [7, 1, 2]])
    bag = pc.soft_bag(idx, pc.soft_weights(np.tile([[1.0, 2.0, 3.0]], (3, 1)), 1e-3, 1.0))
    assert np.allclose(bag, ps.bag(idx[:, 0])) and bag.shape == (2048,)       # T -> 0: the bag of top-1 indices
    stats = dict(sigma0_sq=2.0, mean=np.zeros(512, np.float32), std=np.full(512, 2, np.float32))
    fr = dict(cb0=np.array([1, 2], np.int16), pcb0=np.array([3, 4], np.int16), idx=idx[:2].astype(np.int16), d2=d2,
              lat=np.ones((2, 512), np.float16))
    assert pc.make_input('pcb0', fr.__getitem__, stats)['tok'].tolist() == [3, 4]
    x = pc.make_input('soft_T1', fr.__getitem__, stats)
    assert x['idx'].dtype == np.int64 and np.allclose(x['w'], w)
    assert np.allclose(pc.make_input('lat', fr.__getitem__, stats)['x'], 0.5)
    with pytest.raises(ValueError):
        pc.kind('cb1')


def test_sigma0_and_lat_stats(dirs):
    sh = pc.Shards([str(dirs / 'train')])
    idx = list(range(len(sh.rows)))
    top1 = np.concatenate([np.load(b + '.psoft0.d2.npy')[:, 0].astype(np.float32) for b in sh.bases])
    assert pc.sigma0_sq(sh, idx) == pytest.approx(float(np.median(top1)))
    lat = np.concatenate([np.load(b + '.lat.npy').astype(np.float64) for b in sh.bases])
    mean, std = pc.lat_stats(sh, idx)
    assert np.allclose(mean, lat.mean(0), atol=1e-4) and np.allclose(std, lat.std(0), atol=1e-4)


def test_probe_shapes_size_and_padding_independence():
    big = pc.CTCProbe('tok')
    assert 4.5e6 < sum(p.numel() for p in big.parameters()) < 5.5e6           # PLAN: about 5M
    rng = np.random.default_rng(0)
    for kind, x in (('tok', lambda n: dict(tok=rng.integers(0, 2048, n))),
                    ('soft', lambda n: dict(idx=rng.integers(0, 2048, (n, 8)),
                                            w=pc.soft_weights(rng.random((n, 8)), 1.0, 0.5))),
                    ('lat', lambda n: dict(x=rng.standard_normal((n, 512)).astype(np.float32)))):
        m = pc.CTCProbe(kind, d=32, layers=2, heads=2, ff=64).eval()
        items = [dict(x=x(n), label=np.array([1, 2])) for n in (7, 3)]
        b = pc.collate(items, torch.device('cpu'))
        with torch.no_grad():
            out = m(b)
            alone = m(pc.collate(items[1:], torch.device('cpu')))
        assert out.shape == (2, 4 * 7, 29) and b['lens'].tolist() == [7, 3] and b['targets'].tolist() == [1, 2, 1, 2]
        assert torch.allclose(out[1, :12], alone[0], atol=1e-5)               # padding does not leak


def test_lr_schedule():
    a = pc.parse(['--out', 'x', '--steps', '110', '--warmup', '10', '--lr', '1e-3'])
    assert pc.lr_at(0, a) == pytest.approx(1e-4) and pc.lr_at(9, a) == pytest.approx(1e-3)
    assert pc.lr_at(10, a) == pytest.approx(1e-3) and pc.lr_at(60, a) == pytest.approx(5e-4)
    assert pc.lr_at(110, a) == pytest.approx(0, abs=1e-12)


def test_gates():
    st = dict(pcb0=dict(cer=0.12), lat=dict(cer=0.10), soft_T1=dict(cer=0.10), **{'soft_T0.5': dict(cer=0.09)})
    spk = dict(pcb0=dict(excess=0.05), soft_T1=dict(excess=0.069), **{'soft_T0.5': dict(excess=0.08)})
    g = pc.gates(st, spk)
    assert g['cb0_content_go'] and not g['start_f1']
    assert g['soft']['soft_T1'] == dict(cer_ok=True, speaker_ok=True, p6_arm=True)
    assert g['soft']['soft_T0.5'] == dict(cer_ok=True, speaker_ok=False, p6_arm=False)
    assert g['soft_p6_arm'] == 'soft_T1' and g['soft_T'] == 1.0
    g = pc.gates(dict(pcb0=dict(cer=0.16), lat=dict(cer=0.12)))
    assert not g['cb0_content_go'] and not g['start_f1'] and 'soft' not in g        # above 15 %, below 1.5x
    g = pc.gates(dict(pcb0=dict(cer=0.10), lat=dict(cer=0.06), soft_T1=dict(cer=0.09)))
    assert not g['cb0_content_go'] and g['start_f1'] and g['soft_p6_arm'] is None and g['soft_T'] is None


def test_probe_learns_to_read_and_writes_the_json(dirs):
    logs = []
    assert pc.main(argv(dirs), log=logs.append) == 0
    res = json.loads((dirs / 'p2' / 'p2_content.json').read_text())
    tr, dv = res['data']['train'], res['data']['dev']
    assert tr['skipped'] == {'too_long': 1, 'ctc': 1} and tr['n_used'] == 120 and dv['n_used'] == 20
    assert dv['n_skipped'] == 0 and res['sigma0_sq'] == pytest.approx(1.05, abs=0.06)
    assert list(res['streams']) == ['cb0', 'pcb0', 'soft_T0.5', 'soft_T1', 'lat']
    for s, e in res['streams'].items():
        assert e['n_eval'] == 20 and e['n_skipped'] == 0 and e['steps'] == 200 and e['train_s'] > 0
        assert len(e['examples']) == 3 and e['examples'][0]['ref'] == pc.normalise(e['examples'][0]['ref'])
        assert e['cer'] < 0.2, (s, e['cer'], e['examples'])
    cer = {s: e['cer'] for s, e in res['streams'].items()}
    assert cer['cb0'] < 0.05 and cer['lat'] < 0.05 and cer['pcb0'] > cer['cb0']        # 10 % random tokens cost content
    sid = res['speaker_id']
    assert set(sid) == {'cb0', 'pcb0', 'soft_T0.5', 'soft_T1', 'text'} and sid['text']['excess'] == 0
    assert sid['cb0']['n_speakers'] == 4
    assert set(res['gates']) >= {'cb0_content_go', 'start_f1', 'soft', 'soft_p6_arm', 'soft_T'}
    assert (dirs / 'p2' / 'p2_ckpt_lat.pt').exists() and any('gates' in m for m in logs)


def test_interrupted_stream_resumes_exactly_and_reruns_are_idempotent(dirs, monkeypatch):
    kw = dict(streams='cb0,soft_T1', steps=30, ckpt_every=10, dropout=0.1)
    assert pc.main(argv(dirs, out='ref', **kw), log=lambda m: None) == 0
    orig = pc.train_batch

    def crash(sh, idx, stream, stats, step, a, device):
        if step == 15:
            raise RuntimeError('preempted')
        return orig(sh, idx, stream, stats, step, a, device)

    monkeypatch.setattr(pc, 'train_batch', crash)
    with pytest.raises(RuntimeError, match='preempted'):
        pc.main(argv(dirs, **kw), log=lambda m: None)
    assert torch.load(dirs / 'p2' / 'p2_ckpt_cb0.pt', weights_only=False)['step'] == 10
    monkeypatch.setattr(pc, 'train_batch', orig)
    logs = []
    assert pc.main(argv(dirs, **kw), log=logs.append) == 0 and 'cb0: resumed at step 10' in logs
    for s in ('cb0', 'soft_T1'):
        a = torch.load(dirs / 'ref' / f'p2_ckpt_{s}.pt', weights_only=False)['model']
        b = torch.load(dirs / 'p2' / f'p2_ckpt_{s}.pt', weights_only=False)['model']
        assert all(torch.equal(a[k], b[k]) for k in a)
    ref = json.loads((dirs / 'ref' / 'p2_content.json').read_text())
    res = json.loads((dirs / 'p2' / 'p2_content.json').read_text())
    assert all(ref['streams'][s]['cer'] == res['streams'][s]['cer'] for s in ('cb0', 'soft_T1'))

    monkeypatch.setattr(pc, 'train_stream', no_training)
    logs = []
    assert pc.main(argv(dirs, **kw), log=logs.append) == 0 and 'cb0: finished earlier, kept' in logs
    assert json.loads((dirs / 'p2' / 'p2_content.json').read_text()) == res
    assert pc.load_results(dirs / 'p2' / 'p2_content.json', dict(res['config'], steps=31))['streams'] == {}


def no_training(*args, **kw):
    raise AssertionError('a finished stream was trained again')


def test_streams_split_across_runs_keep_their_speaker_id(dirs):
    path = dirs / 'p2' / 'p2_content.json'
    assert pc.main(argv(dirs, streams='pcb0,soft_T1', steps=10), log=lambda m: None) == 0
    first = json.loads(path.read_text())
    assert pc.main(argv(dirs, streams='soft_T0.5', steps=10), log=lambda m: None) == 0
    res = json.loads(path.read_text())
    assert list(res['streams']) == ['pcb0', 'soft_T1', 'soft_T0.5']
    assert set(res['speaker_id']) == {'cb0', 'pcb0', 'soft_T1', 'soft_T0.5', 'text'}
    assert all(res['speaker_id'][s] == first['speaker_id'][s] for s in ('cb0', 'pcb0', 'soft_T1', 'text'))
    for s in ('soft_T1', 'soft_T0.5'):
        assert set(res['gates']['soft'][s]) == {'cer_ok', 'speaker_ok', 'p6_arm'}


def test_soft_bags_take_the_idf_of_the_pcb0_bag(dirs, monkeypatch):
    sh = pc.Shards([str(dirs / 'dev')])
    monkeypatch.setattr(pc.ps, 'evaluate', lambda X, spk, chapters=None: dict(acc=0.5, F=X))
    res = pc.speaker_id(sh, ['soft_T1000', 'soft_T0.001'], 1.0)
    rows = range(len(sh.rows))
    H = np.stack([ps.bag(sh.frames(i, 'pcb0')) for i in rows])
    W = [pc.soft_weights(sh.frames(i, 'd2'), 1000.0, 1.0) for i in rows]
    assert min(w.min() for w in W) > 0.99 / 8                                  # near uniform over the top 8
    S = np.stack([pc.soft_bag(sh.frames(i, 'idx'), w) for i, w in zip(rows, W)])
    assert (S > 0).sum(0).mean() > 3 * (H > 0).sum(0).mean()                    # every top-8 index counts in its df
    idf = np.log((1 + len(H)) / (1 + (H > 0).sum(0))) + 1
    Y = S * idf
    assert np.allclose(res['soft_T1000']['F'], Y / np.linalg.norm(Y, axis=1, keepdims=True))
    assert not np.allclose(res['soft_T1000']['F'], ps.tfidf(S), atol=1e-3)
    assert np.allclose(res['soft_T0.001']['F'], res['pcb0']['F']) and np.allclose(res['pcb0']['F'], ps.tfidf(H))
    assert list(res) == ['cb0', 'pcb0', 'soft_T1000', 'soft_T0.001', 'text']


@pytest.mark.parametrize('split,suffix', [('dev', '.psoft0.d2.npy'), ('dev', '.pcodes0.npy'),
                                          ('train', '.pcodes0.npy'), ('train', '.lat.npy')])
def test_missing_arrays_stop_before_any_training(dirs, monkeypatch, split, suffix):
    (dirs / split / f'shard_0001{suffix}').unlink()
    if suffix != '.lat.npy':
        assert pc.main(argv(dirs, streams='cb0,lat', steps=5), log=lambda m: None) == 0    # not needed: runs
    monkeypatch.setattr(pc, 'train_stream', no_training)
    with pytest.raises(SystemExit, match=f'^1 arrays missing, e.g. .*{split}/shard_0001{re.escape(suffix)}$'):
        pc.main(argv(dirs), log=lambda m: None)
    assert pc.needed(['cb0', 'soft_T1', 'soft_T0.5', 'lat']) == {'cb0', 'idx', 'd2', 'lat'}


def write_vctk(tmp_path):
    import soundfile as sf
    rng = np.random.default_rng(3)
    rows = []
    for k, (utt, text, sec, primary) in enumerate([('p1_001', 'Please call Stella.', 1.5, True),
                                                   ('p1_002', 'Ask her to bring these things.', 2.0, True),
                                                   ('p2_003', 'Not a primary row.', 1.0, False),
                                                   ('p2_004', 'Six spoons of fresh snow peas.', 5.0, True)]):
        p = tmp_path / 'vctk' / f'{utt}_mic1.wav'
        p.parent.mkdir(exist_ok=True)
        sf.write(p, (0.1 * rng.standard_normal(int(sec * em.SR))).astype(np.float32), em.SR)
        for target in ('p9', 'p8'):                                # each source appears in two pairs
            rows.append(dict(pair_id=f'dev__{utt}__{target}', source_utt=utt, source_path=str(p), source_text=text,
                             wer_primary=primary, target_speaker=target))
    path = tmp_path / 'dev.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def flip(w, sr, rng):
    return (-w + 0.01 * rng.standard_normal(len(w))).astype(np.float32)


def test_vctk_inputs_are_mimi_frames_of_clean_and_perturbed_audio(tmp_path):
    path = write_vctk(tmp_path)
    src = pc.vctk_sources(path, 100)
    assert [r['source_utt'] for r in src] == ['p1_001', 'p1_002', 'p2_004'] and len(pc.vctk_sources(path, 2)) == 2
    enc = em.MimiEncoder('cpu', model=FakeMimi())
    items, info = pc.vctk_inputs(src, enc, 50, 7, 0, flip)
    assert info == dict(n_sources=3, n_eval=2, skipped={'too_long': 1}, n_skipped=1)
    it = items[0]
    w = em.read_wav(src[0]['source_path'])
    n = em.n_frames(len(w))                                         # every frame of the audio, none trimmed
    assert n == 19 > em.stored_frames(len(w))
    wp = np.pad(w, (0, em.TRIM * em.HOP))
    lat, cod = enc([wp])[0]
    plat, pcod = enc([flip(wp, em.SR, np.random.default_rng(ep.utt_seed(7, 'p1_001', 0)))])[0]
    assert it['text'] == 'please call stella' and it['label'].tolist() == pc.encode_text('please call stella').tolist()
    fr = it['frames']
    assert np.array_equal(fr['cb0'], cod[:n, 0]) and np.array_equal(fr['pcb0'], pcod[:n, 0])
    assert fr['cb0'].shape == fr['pcb0'].shape == (n,) and items[1]['frames']['cb0'].shape == (em.n_frames(2 * em.SR),)
    assert fr['idx'].shape == fr['d2'].shape == (n, 8) and fr['lat'].shape == (n, 512) and fr['d2'].dtype == np.float16
    assert np.array_equal(fr['idx'], ep.soft_cb0(enc, plat[:n])[0]) and np.allclose(fr['lat'], lat[:n], atol=1e-2)


def test_main_scores_vctk_sources_and_keeps_them(dirs, monkeypatch):
    path = write_vctk(dirs)
    made, Encoder = [], em.MimiEncoder
    monkeypatch.setattr(pc.em, 'MimiEncoder', lambda device: made.append(device) or Encoder('cpu', model=FakeMimi()))
    monkeypatch.setattr(pc.ep, 'default_perturb', flip)
    kw = dict(streams='pcb0,soft_T1,lat', steps=20, vctk_pairs=path)
    assert pc.main(argv(dirs, **kw), log=lambda m: None) == 0 and made == ['cpu']
    res = json.loads((dirs / 'p2' / 'p2_content.json').read_text())
    assert res['data']['vctk'] == dict(n_sources=3, n_eval=2, skipped={'too_long': 1}, n_skipped=1)
    for s, e in res['streams'].items():
        assert e['vctk']['n_eval'] == 2 and e['vctk']['n_skipped'] == 1 and 0 <= e['vctk']['cer']
    monkeypatch.setattr(pc, 'train_stream', no_training)
    assert pc.main(argv(dirs, **kw), log=lambda m: None) == 0 and made == ['cpu']
    assert json.loads((dirs / 'p2' / 'p2_content.json').read_text()) == res


def test_vctk_inputs_are_built_before_any_training(dirs, monkeypatch):
    path = write_vctk(dirs)

    def offline(device):
        raise OSError('kyutai/mimi is not in the HF cache')

    monkeypatch.setattr(pc.em, 'MimiEncoder', offline)
    monkeypatch.setattr(pc, 'train_stream', no_training)
    with pytest.raises(OSError, match='HF cache'):
        pc.main(argv(dirs, streams='pcb0', vctk_pairs=path), log=lambda m: None)
