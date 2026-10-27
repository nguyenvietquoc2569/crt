"""Tests of mvc/summarize.py: capped WER/CER over wer_primary pairs, means, @src rows, same-pair comparison."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize as sm  # noqa: E402


def row(pid, system, S, D, I, N, sim, utmos, primary=True):
    return dict(pair_id=pid, system=system, wer_primary=primary, sim_wavlm=sim, utmos=utmos,
                wer_large_v3=dict(S=S, D=D, I=I, N=N, cS=S, cD=D, cI=I, cN=4 * N))


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def test_capped_wer_means_and_comparison(tmp_path):
    m = write(tmp_path / 'm.jsonl', [row('a', 'mvc', 1, 0, 0, 10, 0.9, 3.0), row('b', 'mvc', 0, 0, 30, 5, 0.8, 2.0),
                                     row('c', 'mvc', 9, 9, 9, 9, 0.7, 1.0, primary=False)])
    s = write(tmp_path / 's.jsonl', [dict(pair_id=p, system='mvc@src', sim_wavlm=v) for p, v in (('a', 0.5), ('b', 0.6), ('c', 0.7))])
    cm = write(tmp_path / 'cm.jsonl', [row('a', 'cand_pm18_rq', 5, 0, 0, 10, 0.8, 2.5), row('z', 'cand_pm18_rq', 0, 0, 0, 10, 0.1, 1.0),
                                       row('a', 'other', 0, 0, 0, 10, 0.0, 0.0)])
    cs = write(tmp_path / 'cs.jsonl', [dict(pair_id='a', system='cand_pm18_rq@src', sim_wavlm=0.4)])
    out = tmp_path / 'sum.json'
    sm.main(['--metrics', m, '--src', s, '--compare', f'cand_pm18_rq={cm}:{cs}', '--out', str(out)])
    res = json.load(open(out))
    me = res['_system']
    assert me['wer_capped'] == pytest.approx(100 * (1 + 5) / 15, abs=0.01)     # pair b's 30 insertions capped at N=5
    assert me['n'] == 3 and me['n_wer'] == 2 and me['sim'] == pytest.approx(0.8) and me['sim_src'] == pytest.approx(0.6)
    assert me['utmos'] == pytest.approx(2.0) and me['margin'] == pytest.approx(0.2)
    c = res['cand_pm18_rq']
    assert c['n'] == 1 and c['wer_capped'] == 50.0 and c['sim_src'] == pytest.approx(0.4)       # z is not compared


def test_strata_and_leakage_against_target_copy(tmp_path):
    def r(pid, system, g, sim):
        x = row(pid, system, 0, 0, 0, 10, sim, 3.0)
        x['genders'] = g
        return x
    m = write(tmp_path / 'm.jsonl', [r('a', 'mvc', 'F2F', 0.9), r('b', 'mvc', 'F2M', 0.8), r('c', 'mvc', 'M2M', 0.85)])
    s = write(tmp_path / 's.jsonl', [dict(pair_id=p, system='mvc@src', sim_wavlm=v) for p, v in (('a', 0.7), ('b', 0.5), ('c', 0.6))])
    cm = write(tmp_path / 'cm.jsonl', [r(p, 'target_copy', g, 0.95) for p, g in (('a', 'F2F'), ('b', 'F2M'), ('c', 'M2M'))])
    cs = write(tmp_path / 'cs.jsonl', [dict(pair_id=p, system='target_copy@src', sim_wavlm=v) for p, v in (('a', 0.6), ('b', 0.45), ('c', 0.6))])
    out = tmp_path / 'o.json'
    sm.main(['--metrics', m, '--src', s, '--compare', f'target_copy={cm}:{cs}', '--out', str(out)])
    res = json.load(open(out))['_system']['strata']
    assert set(res) >= {'F2F', 'F2M', 'M2M', 'same', 'cross'} and res['same']['n'] == 2 and res['cross']['n'] == 1
    assert res['same']['sim_src'] == pytest.approx(0.65) and res['same']['sim_src_minus_target_copy'] == pytest.approx(0.05)
    assert res['cross']['sim_src_minus_target_copy'] == pytest.approx(0.05)
