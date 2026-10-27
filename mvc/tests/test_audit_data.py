"""Tests of mvc/audit_data.py on synthetic audio and shards: the audio statistics, every exclusion rule, the
leave-one-out speaker cosine, the summary and exclusion list, resuming, and the --exclude_utts path of the corpus."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_data as ad  # noqa: E402
import data as dt  # noqa: E402
import encode_mimi as em  # noqa: E402

SR = 24000


def tone(sec=2.0, amp=0.3, sil=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    w = amp * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * rng.random(len(t)))
    return np.concatenate([np.zeros(int(sil[0] * SR)), w, np.zeros(int(sil[1] * SR))]).astype(np.float32)


def test_audio_stats_measure_level_clipping_and_silence():
    s = ad.audio_stats(tone(2.0, sil=(1.5, 0.5)), SR)
    assert s['finite'] and s['dur_s'] == pytest.approx(4.0, abs=0.01)
    assert s['speech_frac'] == pytest.approx(0.5, abs=0.03)
    assert s['lead_sil_s'] == pytest.approx(1.5, abs=0.05) and s['trail_sil_s'] == pytest.approx(0.5, abs=0.05)
    assert s['clip_frac'] == 0 and -20 < s['rms_db'] < -8
    assert s['speech_frac_core'] == pytest.approx(2.0 / 2.5, abs=0.03)          # 1 s of the lead and all 0.5 s of the trail discounted
    short = ad.audio_stats(tone(1.0, sil=(0.6, 0.6)), SR)                       # ordinary edge pauses
    assert short['speech_frac'] < 0.5 and short['speech_frac_core'] > 0.95 and 'low_speech' not in ad.flags(short)
    for empty in (np.concatenate([np.zeros(int(2.4 * SR)), tone(0.2), np.zeros(int(2.4 * SR))]), np.zeros(4 * SR)):
        e = ad.audio_stats(empty, SR)                                           # no relative-only activity test
        assert e['speech_frac'] < 0.1 and 'low_speech' in ad.flags(e)
    clipped = ad.audio_stats(np.clip(tone(amp=3.0), -1, 1), SR)
    assert clipped['clip_frac'] > 0.1
    bad = tone()
    bad[5] = np.nan
    assert ad.audio_stats(bad, SR)['finite'] is False


def test_every_rule_flags_its_case_and_a_clean_utterance_passes():
    ok = dict(finite=True, frames_ok=True, clip_frac=0, speech_frac=0.9, speech_frac_core=0.9, spk_cos=0.8, spk_other=0.3,
              spk_z=0.1, agree0=0.5, agree1=0.55, lat_z=0.3, utmos=4.0, lead_sil_s=0.1, trail_sil_s=0.2)
    assert ad.flags(ok) == []
    cases = [({'error': 'x'}, 'unreadable'), ({'finite': False}, 'not_finite'), ({'frames_ok': False}, 'frames_mismatch'),
             ({'clip_frac': 0.01}, 'clipped'), ({'speech_frac_core': 0.3}, 'low_speech'), ({'spk_other': 0.86}, 'speaker_outlier'),
             ({'spk_z': -5.0}, 'speaker_outlier'), ({'agree0': 0.1, 'agree1': 0.12}, 'perturbation_extreme'),
             ({'agree0': 0.95, 'agree1': 0.92}, 'perturbation_extreme'), ({'lat_z': -7.0}, 'latent_outlier'), ({'utmos': 2.0}, 'low_mos'),
             ({'lead_sil_s': 1.5}, 'long_silence')]
    for change, flag in cases:
        u = {'error': 'x'} if 'error' in change else dict(ok, **change)
        assert ad.flags(u) == [flag], (change, ad.flags(u))
    assert 'long_silence' not in ad.EXCLUDING and set(ad.EXCLUDING) <= {f for _, f in cases}
    assert ad.flags(dict(ok, agree1=0.1)) == [] and ad.flags(dict(ok, spk_other=0.79)) == []   # one bad draw; a near tie


def test_speaker_cosine_is_leave_one_out_and_finds_the_impostor():
    rng = np.random.default_rng(0)
    centre = rng.standard_normal(192)
    utts = [dict(id=f'a{i}', speaker='a', lat_rms=1.0, _emb=centre + 0.3 * rng.standard_normal(192)) for i in range(8)]
    other = rng.standard_normal(192)
    utts += [dict(id=f'b{i}', speaker='b', lat_rms=1.0, _emb=other + 0.3 * rng.standard_normal(192)) for i in range(6)]
    utts.append(dict(id='a_x', speaker='a', lat_rms=1.0, _emb=other + 0.3 * rng.standard_normal(192)))   # b's voice
    utts.append(dict(id='c0', speaker='c', lat_rms=1.0, _emb=rng.standard_normal(192)))          # a lone utterance
    utts.append(dict(id='a_nan', speaker='a', lat_rms=1.0, _emb=np.full(192, np.nan)))           # never poisons a
    out = {u['id']: u for u in ad.add_corpus_stats(utts)}
    assert out['a_x']['spk_cos'] < out['a_x']['spk_other'] and min(out[f'a{i}']['spk_cos'] for i in range(8)) > 0.8
    assert 'speaker_outlier' in ad.flags(out['a_x']) and 'speaker_outlier' not in ad.flags(out['a0'])
    assert out['c0']['spk_cos'] == 1.0 and '_emb' not in out['a0'] and 'spk_cos' not in out['a_nan']


def write_corpus(root, raw, n_spk=3, n_utt=6, sec=1.5):
    """One shard with real wavs under raw/<subset>/<spk>/<chapter>/<id>.wav, CAM++-like embeddings and perturbed codes."""
    rng = np.random.default_rng(1)
    rows, lats, cods, pc, embs, off = [], [], [], [], [], 0
    centres = rng.standard_normal((n_spk, 192))
    for s in range(n_spk):
        for u in range(n_utt):
            uid = f'{s}_c_{u}'
            w = tone(sec + 0.1 * u, seed=s * 10 + u)
            if (s, u) == (0, 1):
                w = np.concatenate([np.zeros(3 * SR, np.float32), w[:SR // 2]])        # mostly silence
            p = Path(raw) / 'train-x' / str(s) / 'c' / f'{uid}.wav'
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, w, SR)
            T = em.stored_frames(len(w))
            lat = rng.standard_normal((T, 512)).astype(np.float32)
            cod = rng.integers(0, 2048, (T, 32))
            p0 = cod.copy()
            p0[: T // 2, 0] = (p0[: T // 2, 0] + 1) % 2048                                # half the CB0 changed
            if (s, u) == (1, 2):
                p0[:, 0] = (p0[:, 0] + 1) % 2048                                           # every CB0 changed
            rows.append(dict(id=uid, speaker=str(s), chapter='c', subset='train-x', text='', offset=off, frames=T, samples=len(w)))
            lats.append(lat)
            cods.append(cod)
            pc.append(p0)
            emb = centres[s] + 0.2 * rng.standard_normal(192)
            if (s, u) == (2, 3):
                emb = rng.standard_normal(192)                                             # wrong speaker label
            embs.append(emb)
            off += T
    d = Path(root) / 'train-x'
    d.mkdir(parents=True, exist_ok=True)
    em.write_shard(d / 'shard_0000', rows, lats, cods)
    np.save(str(d / 'shard_0000') + '.pcodes0.npy', np.concatenate(pc).astype(np.int16))
    np.save(str(d / 'shard_0000') + '.spk.npy', np.stack(embs).astype(np.float32))
    return d


def test_pass_one_and_two_write_the_audit_summary_and_exclusion_list(tmp_path):
    d = write_corpus(tmp_path / 'mimi', tmp_path / 'raw')
    base = str(d / 'shard_0000')
    assert ad.audit_shard(base, str(tmp_path / 'raw'), log=lambda m: None)
    assert not ad.audit_shard(base, str(tmp_path / 'raw'), log=lambda m: None)                 # resumes
    stats = [json.loads(x) for x in open(base + '.audit.jsonl')]
    assert len(stats) == 18 and all(s['frames_ok'] for s in stats) and all(s['lat_rms'] > 0 for s in stats)
    assert sorted(p.name for p in d.glob('shard_*.json')) == ['shard_0000.json']          # invisible to shard globs
    res = ad.summarize([str(d)], tmp_path / 'out', K=1, log=lambda m: None)
    ex = set((tmp_path / 'out' / 'exclude.txt').read_text().split())
    assert ex == {'0_c_1', '1_c_2', '2_c_3'}, ex                                                # silence, perturbation, speaker
    assert res['excluded'] == 3 and res['flag_counts']['low_speech'] == 1
    assert res['flag_counts']['perturbation_extreme'] == 1 and res['flag_counts']['speaker_outlier'] == 1
    assert res['coverage'] == dict(spk_cos=18, agree=18, utmos=0) and len(res['exclude_sha256']) == 64
    fl = [json.loads(l) for l in open(tmp_path / 'out' / 'flags.jsonl')]
    assert {f['id'] for f in fl} >= ex
    assert res['audited_shards'] == ['train-x/shard_0000'] and res['unaudited_shards'] == []
    assert set(res['flag_share_of_hours']) == set(res['flag_counts']) and res['speakers']['n'] == 3
    c = dt.Corpus([str(d)], min_frames=1, exclude=ex)
    assert c.n_excluded == 3 and len(c.utts) == 15 and not ex & {u['id'] for u in c.utts}
    (tmp_path / 'out' / 'exclude.txt').write_text('someone_else\n')                         # never silently replaced
    with pytest.raises(SystemExit, match='different list'):
        ad.summarize([str(d)], tmp_path / 'out', K=1, log=lambda m: None)
    ad.summarize([str(d)], tmp_path / 'out', K=1, force=True, log=lambda m: None)
    assert set((tmp_path / 'out' / 'exclude.txt').read_text().split()) == ex


def test_a_speaker_the_speaker_rule_would_gut_is_reviewed_not_cut():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(192), rng.standard_normal(192)
    utts = [dict(id=f'a{i}', speaker='a', subset='x', dur_s=3.0, lat_rms=1.0,
                 _emb=(a if i < 3 else b) + 0.2 * rng.standard_normal(192)) for i in range(6)]   # half sound like b
    utts += [dict(id=f'b{i}', speaker='b', subset='x', dur_s=3.0, lat_rms=1.0, _emb=b + 0.2 * rng.standard_normal(192))
             for i in range(6)]
    us = ad.add_corpus_stats(utts)
    assert sum('speaker_outlier' in ad.flags(u) for u in us if u['speaker'] == 'a') >= 2


def test_train_refuses_shards_the_audit_did_not_cover(tmp_path):
    import train as tn
    d = write_corpus(tmp_path / 'mimi', tmp_path / 'raw', n_spk=3, n_utt=4, sec=3.0)
    d = d.rename(d.parent / 'train-clean-100')                                   # a subset name train.py accepts
    ad.audit_shard(str(d / 'shard_0000'), str(tmp_path / 'raw'), log=lambda m: None)
    ad.summarize([str(d)], tmp_path / 'out', K=1, log=lambda m: None)
    s = json.load(open(tmp_path / 'out' / 'audit_summary.json'))
    s['audited_shards'] = []
    json.dump(s, open(tmp_path / 'out' / 'audit_summary.json', 'w'))
    a = tn.parse(['--train_dirs', str(d), '--val_dirs', str(d), '--out', str(tmp_path / 'run'), '--device', 'cpu',
                  '--exclude_utts', str(tmp_path / 'out' / 'exclude.txt')])
    with pytest.raises(SystemExit, match='not covered by the audit'):
        tn.train(a, log=lambda m: None)


def test_unreadable_files_are_flagged_not_fatal(tmp_path):
    d = write_corpus(tmp_path / 'mimi', tmp_path / 'raw', n_spk=2, n_utt=3)
    (tmp_path / 'raw' / 'train-x' / '1' / 'c' / '1_c_0.wav').write_bytes(b'not audio')
    ad.audit_shard(str(d / 'shard_0000'), str(tmp_path / 'raw'), log=lambda m: None)
    ad.summarize([str(d)], tmp_path / 'out', K=1, log=lambda m: None)
    assert '1_c_0' in (tmp_path / 'out' / 'exclude.txt').read_text().split()
