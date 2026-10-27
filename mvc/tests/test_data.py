"""Tests of mvc/data.py on synthetic shards: indexing, speaker-consistent pairs from different utterances, crops inside
utterances, extra streams aligned with the target, padding masks and speaker hold-out."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as dt  # noqa: E402
import encode_mimi as em  # noqa: E402


def make_shards(root, n_spk=4, n_utt=3, frames=(40, 90, 200), one_off=True, spk_offset=0):
    """Two shards; frame values encode (speaker, utterance, frame) so crops can be checked exactly."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, lats, cods, off, shard = [], [], [], 0, 0
    utts = [(s, u) for s in range(spk_offset, spk_offset + n_spk) for u in range(n_utt)]
    if one_off:
        utts.append((99, 0))                          # a speaker with one utterance: never sampled
    for k, (s, u) in enumerate(utts):
        T = frames[u % len(frames)]
        lat = np.zeros((T, 512), np.float32)
        lat[:, 0], lat[:, 1], lat[:, 2] = s, u, np.arange(T)
        cod = np.tile(np.arange(T)[:, None] % 2048, (1, 32))
        rows.append(dict(id=f'{s}_{u}', speaker=str(s), chapter='1', subset='x', text='', offset=off, frames=T, samples=T * 1920))
        lats.append(lat)
        cods.append(cod)
        off += T
        if k == len(utts) // 2:
            em.write_shard(root / f'shard_{shard:04d}', rows, lats, cods)
            np.save(str(root / f'shard_{shard:04d}') + '.pcb0.npy', np.concatenate(cods)[:, 0].astype(np.int16) + 7)
            rows, lats, cods, off, shard = [], [], [], 0, shard + 1
    em.write_shard(root / f'shard_{shard:04d}', rows, lats, cods)
    np.save(str(root / f'shard_{shard:04d}') + '.pcb0.npy', np.concatenate(cods)[:, 0].astype(np.int16) + 7)
    return root


def test_corpus_indexes_and_skips_single_utterance_speakers(tmp_path):
    c = dt.Corpus(make_shards(tmp_path / 's'))
    assert len(c.shards) == 2 and len(c.utts) == 13 and c.speakers == ['0', '1', '2', '3']
    assert c.hours() == pytest.approx(sum(u['frames'] for u in c.utts) / 12.5 / 3600)
    assert len(dt.Corpus(make_shards(tmp_path / 't'), min_frames=100).utts) == 4


def test_pairs_are_same_speaker_different_utterance_and_exact_crops(tmp_path):
    c = dt.Corpus(make_shards(tmp_path / 's'))
    c.add_stream('pcb0', '.pcb0.npy')
    smp = dt.PairSampler(c, max_target=60, prompt_range=(20, 50), streams=('pcb0',), seed=1)
    for _ in range(200):
        x = smp.one()
        t, p = x['target'], x['prompt']
        assert x['target_id'] != x['prompt_id'] and x['target_id'].split('_')[0] == x['prompt_id'].split('_')[0] == x['speaker']
        assert set(t['lat'][:, 0]) == {float(x['speaker'])} and set(p['lat'][:, 0]) == {float(x['speaker'])}
        assert len(t['lat']) <= 60 and 1 <= len(p['lat']) <= 50
        fr = t['lat'][:, 2]
        assert np.array_equal(fr, np.arange(fr[0], fr[0] + len(fr)))                # contiguous crop
        assert np.array_equal(t['pcb0'], (t['codes'][:, 0] + 7))                     # stream aligned with target
        assert np.array_equal(p['pcb0'], (p['codes'][:, 0] + 7))                     # and with the prompt


def test_collate_pads_with_masks(tmp_path):
    c = dt.Corpus(make_shards(tmp_path / 's'))
    b = dt.PairSampler(c, max_target=200, seed=2).batch(6)
    B, T = b['target_mask'].shape
    assert b['target_lat'].shape == (B, T, 512) and b['target_codes'].shape == (B, T, 32) and B == 6
    for i in range(B):
        n = b['target_mask'][i].sum()
        assert np.all(b['target_lat'][i, n:] == 0) and b['target_mask'][i, :n].all()
    assert b['prompt_lat'].shape[:2] == b['prompt_mask'].shape


def test_speaker_split_and_restricted_sampler(tmp_path):
    c = dt.Corpus(make_shards(tmp_path / 's'))
    tr, va = dt.split_speakers(c.speakers, 1, seed=0)
    assert len(va) == 1 and not set(tr) & set(va) and sorted(tr + va) == c.speakers
    smp = dt.PairSampler(c, speakers=va, seed=0)
    assert {smp.one()['speaker'] for _ in range(20)} == set(va)
    with pytest.raises(ValueError):
        dt.PairSampler(c, speakers=['99'])
    with pytest.raises(FileNotFoundError):
        c.add_stream('nope', '.missing.npy')


def test_utterance_stream_follows_the_prompt(tmp_path):
    root = make_shards(tmp_path / 's')
    for js in sorted(root.glob('shard_*.json')):
        rows = __import__('json').load(open(js))
        E = np.array([[float(r['speaker']), float(r['id'].split('_')[1])] for r in rows], np.float32)
        np.save(str(js)[:-5] + '.spk.npy', E)
    c = dt.Corpus(root)
    c.add_utt_stream('spk', '.spk.npy')
    smp = dt.PairSampler(c, seed=0)
    for _ in range(50):
        x = smp.one()
        assert x['prompt_utt']['spk'].tolist() == [float(x['speaker']), float(x['prompt_id'].split('_')[1])]
    b = smp.batch(5)
    assert b['prompt_spk'].shape == (5, 2)


def make_multi(root, n_spk=3, n_utt=8, frames=(40, 90, 200, 60), chapters=2):
    """Shards where every frame records (speaker, utterance, frame), utterances spread over `chapters` chapters."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, lats, cods, off = [], [], [], 0
    for s in range(n_spk):
        for u in range(n_utt):
            T = frames[u % len(frames)]
            lat = np.zeros((T, 512), np.float32)
            lat[:, 0], lat[:, 1], lat[:, 2] = s, u, np.arange(T)
            rows.append(dict(id=f'{s}_{u}', speaker=str(s), chapter=str(u % chapters), subset='x', text='', offset=off,
                             frames=T, samples=T * 1920))
            lats.append(lat)
            cods.append(np.zeros((T, 32), np.int64))
            off += T
    em.write_shard(root / 'shard_0000', rows, lats, cods)
    E = np.stack([np.eye(4)[int(r['speaker'])] * (1 + int(r['id'].split('_')[1])) for r in rows]).astype(np.float32)
    np.save(str(root / 'shard_0000') + '.spk.npy', E)
    return root


def test_prompt_sampler_multi_utterance_segments_and_speaker_vector(tmp_path):
    c = dt.Corpus(make_multi(tmp_path / 's'))
    c.add_utt_stream('spk', '.spk.npy')
    smp = dt.PromptSampler(c, max_target=60, seg_len=50, seed=0)
    lens = []
    for _ in range(300):
        x = smp.one()
        p, t = x['prompt'], x['target']
        tu = x['target_id'].split('_')[1]
        assert set(p['lat'][:, 0]) == {float(x['speaker'])} and tu not in {str(int(v)) for v in p['lat'][:, 1]}
        seg = p['seg']
        assert seg[0] == 1 and np.all(np.diff(seg) >= 0) and np.all(np.bincount(seg)[1:] <= 50)
        for sid in np.unique(seg):                       # a segment never mixes utterances, frames are contiguous
            fr = p['lat'][seg == sid]
            assert len(set(fr[:, 1])) == 1 and np.array_equal(fr[:, 2], np.arange(fr[0, 2], fr[0, 2] + len(fr)))
        v = x['prompt_utt']['spk']
        assert np.isclose(np.linalg.norm(v), 1) and np.argmax(v) == int(x['speaker'])
        lens.append(len(p['lat']))
    assert min(lens) >= 38 and max(lens) <= 750 and np.mean([l > 62 for l in lens]) > 0.5


def test_token_budget_batches(tmp_path):
    c = dt.Corpus(make_multi(tmp_path / 's'))
    smp = dt.PromptSampler(c, max_target=60, seed=1)
    for _ in range(5):
        b = smp.batch_tokens(4000, pool=40)
        B, P = b['prompt_mask'].shape
        T = b['target_mask'].shape[1]
        tot = (b['prompt_mask'].sum(1) + b['target_mask'].sum(1)).max()
        assert B >= 1 and B * tot <= 4000 or B == 1
        assert np.all(b['prompt_seg'][~b['prompt_mask']] == -1) and np.all(b['prompt_seg'][b['prompt_mask']] >= 1)
