"""Tests of mvc/probe_parallel_text.py: the speaker x sentence table from pair rows (shared sources and P, sentence
filter, path map), leave-one-sentence-out against a brute-force reference and without leaking the held-out sentence,
and the whole probe with a fake Mimi whose CB0 carries a speaker offset that a fake perturbation removes."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import encode_mimi as em  # noqa: E402
import encode_perturbed as ep  # noqa: E402
import probe_parallel_text as pt  # noqa: E402
from probe_speaker_info import bag  # noqa: E402

DEV = Path(__file__).resolve().parents[2] / 'paper/journal_extension/results/v1/pairs/dev.jsonl'
OLD, NEW = '/datastore/nvquoc/crt', '/datastore/ceai/crt'


def row(src, tgt, sex, sent, P=True):
    return dict(pair_id=f'dev__{src}_{sent or "x"}__{tgt}', source_speaker=src, source_gender=sex[src],
                source_path=f'{OLD}/vctk/{src}/{src}_{sent or "300"}_mic1.flac', shared_sentence_id=sent,
                target_speaker=tgt, target_gender=sex[tgt],
                P=f'{OLD}/vctk/{tgt}/{tgt}_{sent}_mic1.flac' if sent and P else None)


def test_build_table_from_pair_rows():
    sex = dict(a='F', b='M', c='F', d='M', e='F')
    rows = [row('a', 'c', sex, '001'), row('b', 'a', sex, '001'), row('b', 'c', sex, '001', P=False),
            row('a', 'b', sex, '002'), row('d', 'e', sex, None)]
    table, sx = pt.build_table(rows, min_speakers=3, maps=[(OLD, NEW)])
    assert list(table) == ['001'] and list(table['001']) == ['a', 'b', 'c']          # '002' has 2 speakers: dropped
    assert table['001']['c'] == f'{NEW}/vctk/c/c_001_mic1.flac'                         # from P, path mapped
    assert table['001']['a'] == f'{NEW}/vctk/a/a_001_mic1.flac'
    assert sx == dict(a='F', b='M', c='F')                                              # d, e: not in the table
    assert pt.items_of(table) == [(s, '001', table['001'][s]) for s in 'abc']
    assert pt.map_path('/other/x.flac', [(OLD, NEW)]) == '/other/x.flac'
    t2, _ = pt.build_table(rows, min_speakers=2)
    assert list(t2) == ['001', '002'] and t2['002']['b'].startswith(OLD)


@pytest.mark.skipif(not DEV.exists(), reason='Mac copy of the Dev pair list not present')
def test_dev_pair_list_table():
    table, sx = pt.build_table(pt.load_rows(DEV), 10, [(OLD, NEW)])
    assert len(sx) == 20 and sorted(set(sx.values())) == ['F', 'M']
    assert all(len(v) >= 10 for v in table.values()) and len(table) >= 20
    for sent, v in table.items():
        for spk, p in v.items():
            assert p.startswith(NEW) and Path(p).name.startswith(f'{spk}_{sent}_')


def brute_force(F, spk, sent, sx):
    """Reference implementation, one held-out utterance at a time."""
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    correct, hits, hits_u = 0, defaultdict(list), defaultdict(list)
    for i in range(len(F)):
        tr = [j for j in range(len(F)) if sent[j] != sent[i]]
        cents = {p: unit(np.mean([F[j] for j in tr if spk[j] == p], 0)) for p in sorted(set(spk[j] for j in tr))}
        correct += max(cents, key=lambda p: (cents[p] @ F[i], -sorted(cents).index(p))) == spk[i]
        for h, js in ((hits, tr), (hits_u, [j for j in tr if spk[j] != spk[i]])):
            g = {s: unit(np.mean([F[j] for j in js if sx[j] == s], 0)) for s in sorted(set(sx))}
            h[sx[i]].append(max(g, key=lambda s: (g[s] @ F[i], -sorted(g).index(s))) == sx[i])
    bal = lambda h: np.mean([np.mean(v) for v in h.values()])  # noqa: E731
    return correct / len(F), bal(hits), bal(hits_u)


def test_loso_matches_brute_force():
    rng = np.random.default_rng(3)
    spk, sent, sx = [], [], []
    for p, s in zip('abcdef', 'FFFMMM'):
        for j in range(6):
            if (p, j) not in {('a', 0), ('d', 5), ('f', 2)}:                            # some missing cells
                spk.append(p), sent.append(f'{j:03d}'), sx.append(s)
    F = rng.standard_normal((len(spk), 16)) + np.array([ord(p) - 97 for p in spk])[:, None] * 0.2
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    r = pt.loso(F, spk, sent, sx)
    acc, sex, sex_u = brute_force(F, spk, sent, sx)
    assert r['acc'] == pytest.approx(acc) and r['sex_acc'] == pytest.approx(sex)
    assert r['sex_acc_unseen'] == pytest.approx(sex_u)
    assert r['n_test'] == len(spk) and r['n_speakers'] == 6 and r['chance'] == pytest.approx(1 / 6)
    assert r['n_sentences'] == 6


def test_loso_holds_out_the_test_sentence():
    spk = [p for _ in range(5) for p in 'abcd']
    sent = [f'{j:03d}' for j in range(5) for _ in 'abcd']
    sx = ['F' if p in 'ab' else 'M' for p in spk]
    r = pt.loso(np.eye(len(spk)), spk, sent, sx)       # each utterance orthogonal to all others: no evidence left
    assert r['acc'] == pytest.approx(0.25)              # every score 0 -> first candidate 'a', i.e. chance exactly
    S = np.eye(4)[[ord(p) - 97 for p in spk]]
    r = pt.loso(S, spk, sent, sx)                       # speaker-only features: ID perfect, sex only via own speaker
    assert r['acc'] == 1.0 and r['sex_acc'] == 1.0 and r['sex_acc_unseen'] == 0.5
    X = np.concatenate([S, np.array([[1.0] if s == 'F' else [-1.0] for s in sx])], 1)
    assert pt.loso(X, spk, sent, sx)['sex_acc_unseen'] == 1.0


N_SPK, N_SENT = 20, 12
LT = np.random.default_rng(5).standard_normal((2048, 512))


def utt_wav(i, j):
    """Speaker i, sentence j: per frame, sample 0 holds the sentence's content token, sample 1 the speaker offset."""
    c = np.random.default_rng(100 + j).integers(0, 100, 12 + j)
    w = np.zeros(len(c) * em.HOP - 7, np.float32)
    w[0::em.HOP], w[1::em.HOP] = c / 2048, 100 * i / 2048
    return w


class FakeEnc:
    """Stand-in for MimiEncoder: CB0 = content token + speaker offset, CB1 a permutation of it, lat a table row."""

    def __init__(self):
        self.calls = 0

    def __call__(self, wavs):
        self.calls += len(wavs)
        out = []
        for w in wavs:
            T = em.n_frames(len(w))
            x = np.pad(w, (0, T * em.HOP - len(w)))
            tok = (np.round(x[0::em.HOP] * 2048) + np.round(x[1::em.HOP] * 2048)).astype(np.int64) % 2048
            codes = np.zeros((T, 32), np.int64)
            codes[:, 0], codes[:, 1] = tok, (7 * tok + 3) % 2048
            out.append((LT[tok].astype(np.float32), codes))
        return out


def scramble(w, sr, rng):
    """A 'perturbation' that replaces the speaker offset with a random speaker's (keeps the length)."""
    y = w.copy()
    y[1::em.HOP] = 100 * rng.integers(0, N_SPK) / 2048
    return y


def corpus(tmp_path):
    names = [f'p{i:03d}' for i in range(N_SPK)]
    sex = {p: 'F' if i < N_SPK // 2 else 'M' for i, p in enumerate(names)}
    rows, audio = [], {}
    for j in range(N_SENT):
        sent = f'{j + 1:03d}'
        for i in range(0, N_SPK, 2):                                    # sources i, targets (P) i + 1
            rows.append(row(names[i], names[i + 1], sex, sent))
    rows.append(row(names[0], names[3], sex, '099'))                    # a sentence of 2 speakers: dropped
    rows.append(row(names[1], names[2], sex, None))
    for i, p in enumerate(names):
        for j in range(N_SENT):
            audio[f'{NEW}/vctk/{p}/{p}_{j + 1:03d}_mic1.flac'] = utt_wav(i, j)
    path = tmp_path / 'dev.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path, audio


def test_perturbed_draws_are_seeded_per_utterance():
    enc = FakeEnc()
    items = [('p003', '002', 'x'), ('p011', '007', 'y')]
    audio = dict(x=utt_wav(3, 1), y=utt_wav(11, 6))
    f = pt.encode_items(enc, items, 2, 9, reader=audio.__getitem__, perturb_fn=scramble, log=lambda m: None)
    assert enc.calls == 3 * len(items)
    for i, (spk, sent, path) in enumerate(items):
        n = em.stored_frames(len(audio[path]))
        assert f['frames'][i] == n and f['cb0'][i].sum() == n
        for k in range(2):
            w = scramble(audio[path], em.SR, np.random.default_rng(ep.utt_seed(9, f'{spk}_{sent}', k)))
            assert np.array_equal(f['pcb0'][k, i], bag(enc([w])[0][1][:n, 0]))
    assert not np.array_equal(f['pcb0'][0], f['pcb0'][1])


def test_probe_end_to_end_with_fake_mimi(tmp_path):
    pairs, audio = corpus(tmp_path)
    g = np.random.default_rng(7)                        # codewords of the female offsets (< 1000) share a direction
    sign = np.where(np.arange(2048) < 1000, 1.0, -1.0)[:, None]
    E0 = g.standard_normal((2048, 256)) + 0.3 * sign * g.standard_normal(256)
    encs = []

    def enc_fn():
        encs.append(FakeEnc())
        return encs[-1], E0
    out = tmp_path / 'p0b'
    a = pt.parse(['--pairs', str(pairs), '--out', str(out), '--K', '2', '--path_map', f'{OLD}={NEW}'])
    logs = []
    res = pt.run(a, enc_fn, log=logs.append, reader=audio.__getitem__, perturb_fn=scramble)
    saved = json.loads((out / 'p0b_parallel_text.json').read_text())
    assert saved == json.loads(json.dumps(res)) and (out / 'p0b_features.npz').exists()
    assert not list(out.glob('*.tmp'))
    assert res['n_speakers'] == N_SPK and res['n_sentences'] == N_SENT and res['n_utts'] == N_SPK * N_SENT
    S = res['streams']
    assert set(S) == {'cb0', 'cb0_emb', 'cb1', 'lat', 'pcb0_0', 'pcb0_1', 'pcb0_emb_0', 'pcb0_emb_1'}
    for name in ('cb0', 'cb0_emb', 'cb1', 'lat'):
        assert S[name]['acc'] >= 0.9 and S[name]['sex_acc'] >= 0.9 and S[name]['chance'] == 1 / N_SPK, name
    for name in ('pcb0_0', 'pcb0_1', 'pcb0_emb_0', 'pcb0_emb_1'):
        assert S[name]['acc'] < 0.15 and S[name]['sex_acc'] < 0.7, (name, S[name])
    assert res['gate']['e1_needed'] is False and res['gate']['pcb0_acc'] < 5 / N_SPK
    assert all(x < 0.2 for x in res['cb0_agreement'])
    assert any('E1 not needed' in m for m in logs)

    res2 = pt.run(a, enc_fn, log=logs.append, reader=audio.__getitem__, perturb_fn=scramble)
    assert len(encs) == 1 and res2 == res and any('features reused' in m for m in logs)

    a3 = pt.parse(['--pairs', str(pairs), '--out', str(tmp_path / 'id'), '--K', '1', '--path_map', f'{OLD}={NEW}'])
    res3 = pt.run(a3, enc_fn, log=logs.append, reader=audio.__getitem__, perturb_fn=lambda w, sr, rng: w.copy())
    assert res3['gate']['e1_needed'] is True and res3['streams']['pcb0_0']['acc'] >= 0.9
    assert res3['cb0_agreement'] == [1.0]
    assert set(res3['streams']) == {'cb0', 'cb0_emb', 'cb1', 'lat', 'pcb0_0', 'pcb0_emb_0'}


def test_gate_thresholds():
    r = {f'{t}_{k}': dict(acc=acc, sex_acc=0.6) for t, accs in (('pcb0', (0.25, 0.375)), ('pcb0_emb', (0.1, 0.1)))
         for k, acc in enumerate(accs)}
    g = pt.gate(r, 2, 1 / 16)
    assert g['pcb0_acc'] == 0.3125 and g['e1_needed']                                  # mean of draws = 5x chance
    r['pcb0_1']['acc'] = 0.37
    assert not pt.gate(r, 2, 1 / 16)['e1_needed']
    r['pcb0_emb_0']['sex_acc'] = r['pcb0_emb_1']['sex_acc'] = 0.8
    assert pt.gate(r, 2, 1 / 16)['e1_needed']


def test_main_reads_audio_and_perturbs_in_worker_processes(tmp_path, monkeypatch):
    """main() as on ceai, with the real reader (48 kHz flac -> 24 kHz), the real Praat perturbation in a process pool,
    and only MimiIO replaced."""
    sf = pytest.importorskip('soundfile')
    pytest.importorskip('parselmouth')
    pytest.importorskip('librosa')
    import mimi_io

    class FakeIO:
        def __init__(self, device):
            self.enc = FakeEnc()

        def cb0_table(self):
            return np.random.default_rng(0).standard_normal((2048, 256))
    monkeypatch.setattr(mimi_io, 'MimiIO', FakeIO)
    sex = dict(p001='F', p002='F', p003='M', p004='M')
    rows = []
    for j, sent in enumerate(('001', '002', '003')):
        rows += [row('p001', 'p002', sex, sent), row('p003', 'p004', sex, sent)]
        for i, p in enumerate(sex):
            t = np.arange(int(48000 * (1.0 + 0.2 * j))) / 48000
            f0 = (110, 130, 210, 240)[i]
            w = sum(np.sin(2 * np.pi * h * f0 * t) / h for h in range(1, 6)) * 0.2
            path = tmp_path / f'new/vctk/{p}/{p}_{sent}_mic1.flac'
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, w, 48000)
    pairs = tmp_path / 'dev.jsonl'
    pairs.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'p0b'
    assert pt.main(['--pairs', str(pairs), '--out', str(out), '--K', '1', '--min_speakers', '4', '--workers', '2',
                    '--device', 'cpu', '--path_map', f'{OLD}={tmp_path}/new']) == 0
    res = json.loads((out / 'p0b_parallel_text.json').read_text())
    assert res['n_utts'] == 12 and res['n_speakers'] == 4 and res['speakers'] == sex
    assert set(res['gate']) >= {'e1_needed', 'pcb0_acc', 'pcb0_sex_acc', 'chance', 'rule'}
