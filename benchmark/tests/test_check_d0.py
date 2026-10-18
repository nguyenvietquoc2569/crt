"""Tests of benchmark/check_d0.py, the D0 reproduction checks of ANALYSIS_PLAN_v1.md §8.1 item 9 and §4.4 (D1 and D2
against D0): the S1 / S3 codes, the out_codes of A1-A5 and C1, the C1-C5 WAVs sample for sample or else the metrics
tolerance (identical Whisper large-v3 hypothesis, |ΔSIM| <= 0.001, |ΔUTMOS| <= 0.01), the JSON verdict and the exit
status. The screen (journal/ on the HPC) and the D0 / D1 / D2 output trees are built under tmp_path."""
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

import check_d0
from fixtures_rows import ROOT, conv_row, dev_pairs, pair_row, vctk_pairs, write_jsonl

DEV = ('p225', 'p226', 'p227', 'p228')
# §2.3 / §2.2 names: A1-A5 and C1 (crt_anchor_lab), C1-C5 (crt_decode_lab)
ANC = {'A1': 'R1_s0_pm_rq_T1', 'A2': 'R3_r0_pm_rq_T1', 'A3': 'R3_s0_argmax', 'A4': 'R3_s0_tau0.4', 'A5': 'src0_faiss',
       'C1': 'R3_s0_pm_rq_T1'}
CANDS = {'C1': 'pm_rq_T1', 'C2': 'pm18_rq', 'C3': 'gv1_pm_cont_T1', 'C4': 'gv1_shrink0.85_cont',
         'C5': 'gv0.5_shrink0.75_cont'}
SCREEN_ANC = (['faiss', 'src0_faiss'] + [f'{a}_{c}_{r}' for a in ('R1', 'R3', 'G0.5') for c in ('r0', 's0')
                                         for r in ('tau0.4', 'argmax', 'pm_rq_T1')]
              + [f'R1_r0_tau{t}' for t in (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)])             # job 2634: 28
D0_ANC = SCREEN_ANC[::-1] + [f'R3_r0_tau{t}' for t in (0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)]      # D0, D1: 36
R1R3_ANC = list(ANC.values())                                                                    # anc_r1r3


def metric_row(pair, system, wav, hyp, sim, utmos):
    """A metrics.py output row: the manifest fields plus the scores check_d0 reads."""
    r = conv_row(pair, system, '/x')
    r.update(wav=wav, dur_s=2.0, wer_large_v3={'hyp': hyp, 'ref_norm': 'shared sentence', 'S': 1, 'D': 0, 'I': 0,
                                               'N': 3}, sim_wavlm=sim, utmos=utmos, secs_resemblyzer=0.8)
    return r


@pytest.fixture
def screen(tmp_path):
    """60 Dev rows (screen: rows 0, 20, 40), the arrays, codes, WAVs and scores of D0, and the screen tree J."""
    rng = np.random.default_rng(7)
    pairs = [pair_row('dev', DEV[k % 4], DEV[(k % 4 + 1 + (k // 4) % 3) % 4], k + 1, parallel=k % 2 == 0)
             for k in range(60)]
    ids = [p['pair_id'] for p in pairs[::20]]
    assert len(set(p['pair_id'] for p in pairs)) == 60 and len(ids) == 3
    data = {}
    for p in pairs[::20]:
        t = int(rng.integers(30, 50))
        conf = rng.random((18, t)).astype(np.float32)
        rc = {'src_codes': rng.integers(0, 2048, (32, t)).astype(np.int16),
              'retrieved_codes': rng.integers(0, 2048, (32, t)).astype(np.int16), 'crt_first_cb': np.int8(1),
              'crt_argmax': rng.integers(0, 2048, (18, t)).astype(np.int16), 'crt_conf': conf,
              'crt_logits': rng.standard_normal((18, t, 16)).astype(np.float16),
              **{f'mask_tau{x:g}': conf > np.float32(x) for x in (0.1, 0.4, 0.9)}}
        oc = {s: rng.integers(0, 2048, (32, t)).astype(np.int16) for s in D0_ANC}
        audio = {s: (rng.standard_normal(4800 + 960 * len(data)) * 3000).astype(np.int16) for s in CANDS.values()}
        scores = {s: (f'hypothesis {s} {p["pair_id"]}', round(float(rng.uniform(0.8, 0.95)), 6),
                      round(float(rng.uniform(1.5, 2.5)), 6)) for s in CANDS.values()}
        data[p['pair_id']] = types.SimpleNamespace(pair=p, rc=rc, oc=oc, audio=audio, scores=scores)
    s = types.SimpleNamespace(tmp=tmp_path, path=write_jsonl(tmp_path / 'pairs' / 'dev.jsonl', pairs), pairs=pairs,
                              ids=ids, data=data, J=tmp_path / 'J')
    c = s.J / 'clustering_check'
    write_rc(s, c)
    write_anc(s, s.J / 'anchor_lab', SCREEN_ANC)
    write_cands(s, c / 'lab_r3a', c / 'lab_r3_metrics.jsonl', 'r3a_', list(CANDS.values())[:3], extra='r3a_faiss')
    write_cands(s, c / 'lab_r4p', c / 'lab_r4_metrics.jsonl', 'r4p_', list(CANDS.values())[3:], extra='r4u_faiss')
    return s


def dump_dir(root):
    d = os.path.join(root, 'dumps', 'dev', '30')
    os.makedirs(d, exist_ok=True)
    return d


def write_rc(s, root, mutate=None, mask=True, skip=()):
    """run_conversions dumps (the arrays check_d0 reads, plus logits); mask=False: a faiss-only run, no mask_tau*."""
    for pid, x in s.data.items():
        if pid in skip:
            continue
        a = {k: v.copy() for k, v in x.rc.items() if mask or not k.startswith('mask_tau')}
        if mutate:
            mutate(pid, a)
        np.savez_compressed(os.path.join(dump_dir(root), pid + '.npz'), **a)


def write_anc(s, root, systems, mutate=None):
    """crt_anchor_lab dumps: systems, out_codes and some recipe arrays."""
    for pid, x in s.data.items():
        oc = np.stack([x.oc[n] for n in systems])
        if mutate:
            mutate(pid, systems, oc)
        t = oc.shape[2]
        np.savez_compressed(os.path.join(dump_dir(root), pid + '.npz'), cfg_sha='0123456789abcdef',
                            systems=np.array(systems), out_codes=oc, R1_conf=x.rc['crt_conf'],
                            R3_logits=np.zeros((18, t, 16), np.float16), **{'G0.5_conf': x.rc['crt_conf']},
                            psim_codes=x.rc['src_codes'])


def write_cands(s, wav_dir, metrics_file, prefix, systems, perturb=None, score=None, skip=(), extra=None):
    """crt_decode_lab WAVs (PCM_16) <wav_dir>/<system>/<pair>.wav and their metrics.py rows <prefix><system>; extra:
    the rows of another system in the same file."""
    rows = []
    for pid, x in s.data.items():
        for n in systems:
            if (n, pid) in skip:
                continue
            w = os.path.join(wav_dir, n, pid + '.wav')
            os.makedirs(os.path.dirname(w), exist_ok=True)
            a = x.audio[n].copy()
            sf.write(w, perturb(n, pid, a) if perturb else a, 24000, subtype='PCM_16')
            r = metric_row(x.pair, prefix + n, w, *x.scores[n])
            if score:
                score(n, pid, r)
            rows.append(r)
        if extra:
            rows.append(metric_row(x.pair, extra, '/x/faiss.wav', 'other', 0.5, 1.0))
    write_jsonl(metrics_file, rows)


def write_d0(s, out, rc=None, anc=None, r1r3=None, cand=None):
    """D0's tree (job_d0.slurm): rc/, anc/ (36 systems), anc_r1r3/ (A1-A5, C1), lab/cand and lab/metrics.jsonl; the
    keyword arguments are passed to the writers. Returns the check_d0 arguments of job_d0.slurm step 10."""
    r1r3 = dict(r1r3 or {})
    write_rc(s, out / 'rc', **(rc or {}))
    write_anc(s, out / 'anc', D0_ANC, **(anc or {}))
    write_anc(s, out / 'anc_r1r3', r1r3.pop('systems', R1R3_ANC), **r1r3)
    write_cands(s, out / 'lab' / 'cand', out / 'lab' / 'metrics.jsonl', 'cand_', list(CANDS.values()),
                extra='front_lmbr_k2_g1_T1', **(cand or {}))
    return ['--pairs', s.path, '--ref_screen', s.J, '--rc', out / 'rc', '--anc', out / 'anc', '--anc',
            out / 'anc_r1r3', '--cand', out / 'lab' / 'cand', '--cand_metrics', out / 'lab' / 'metrics.jsonl']


def check(monkeypatch, out, *argv):
    """check_d0.main(): (exit status, verdict)."""
    monkeypatch.setattr(sys, 'argv', ['check_d0.py', *map(str, argv), '--out', str(out)])
    with pytest.raises(SystemExit) as e:
        check_d0.main()
    return e.value.code, json.load(open(out))


def lists_dir(tmp, bad_dev=False):
    """Pair lists for make_pairs.check_lists: dev.jsonl and vctk2k.jsonl with its source-E file."""
    d = tmp / 'lists'
    dev = dev_pairs()
    if bad_dev:              # a source speaker who is not a target
        dev.append(pair_row('dev', 'p229', 'p225', 9))
    write_jsonl(d / 'dev.jsonl', dev)
    write_jsonl(d / 'vctk2k.jsonl', vctk_pairs())
    (d / 'vctk2k_source_E.json').write_text(json.dumps(
        {s: {'paths': [f'{ROOT}/{s}/{s}_{900 + i}_mic1.flac' for i in range(3)], 'durs': [3.0] * 3, 'total_s': 9.0}
         for s in ('p301', 'p302', 'p303')}))
    return d


def test_constants_follow_the_plan():
    assert check_d0.ANC == ANC and check_d0.CANDS == CANDS
    assert (check_d0.SIM_TOL, check_d0.UTMOS_TOL, check_d0.SCREEN_MOD, check_d0.TAU) == (0.001, 0.01, 20, 0.4)


def test_screen_ids_are_the_rows_of_shard_0_of_20(screen):
    assert check_d0.screen_ids(screen.path) == [p['pair_id'] for p in screen.pairs[0::20]] == screen.ids


# ── a reproduced D0 ──────────────────────────────────────────

def test_reproduced_d0_passes(screen, monkeypatch, capsys):
    out = screen.tmp / 'd0'
    argv = write_d0(screen, out)
    stamps = screen.tmp / 'timings.tsv'
    stamps.write_text('1000 1. pair lists\n1100 2. conversions\n\n1600 3. done\n')
    runs = write_jsonl(screen.tmp / 'runs.jsonl', [{'event': 'start'}, {'event': 'end', 'n_errors': 0, 'total_s': 5},
                                                   {'event': 'start'}, {'event': 'end', 'n_errors': 1, 'total_s': 7}])
    code, v = check(monkeypatch, screen.tmp / 'check_d0.json', *argv, '--lists', f'{lists_dir(screen.tmp)}:dev,vctk2k',
                    '--timings', stamps, '--runs', runs, '--sizes', out / 'rc' / 'dumps' / 'dev' / '30', '--sizes',
                    out / 'anc' / 'dumps' / 'dev' / '30', '--label', 'D0 vs the screen')
    assert code == 0 and v['passed'] is True
    assert set(v) == {'written', 'label', 'reference', 'pairs', 'n_screen_pairs', 'tolerance', 'passed', 'summary',
                      'checks', 'info'}
    assert v['n_screen_pairs'] == 3 and v['tolerance'] == {'sim_wavlm': 0.001, 'utmos': 0.01}
    assert v['summary'] == ('D0 vs the screen (3 screen pairs): S1/S3 codes OK (3/3 pairs equal); A1-A5,C1 out_codes '
                            'of anc OK; A1-A5,C1 out_codes of anc_r1r3 OK; C1-C5 WAVs: C1 identical, C2 identical, C3 '
                            'identical, C4 identical, C5 identical; pair lists OK')
    c = v['checks']
    assert set(c) == {'codes', 'out_codes', 'wavs', 'lists'}
    assert (c['codes']['n_equal'], c['codes']['s1_differ'], c['codes']['s3_differ'], c['codes']['missing'],
            c['codes']['mask_differ_tokens']) == (3, [], [], [], 0)
    assert [x['passed'] for x in c['out_codes']] == [True, True]
    assert all(y['n_equal'] == 3 for x in c['out_codes'] for y in x['systems'].values())
    assert list(c['wavs']) == ['C1', 'C2', 'C3', 'C4', 'C5']
    assert all(w['wavs_identical'] and w['metrics_check'] is None and w['max_abs_diff'] == 0
               for w in c['wavs'].values())
    assert c['wavs']['C4']['ref'] == str(screen.J / 'clustering_check' / 'lab_r4p' / CANDS['C4'])
    assert [x['passed'] for x in c['lists']] == [True]
    info = v['info']
    assert [(t['step'], t['s'], t['s_per_pair']) for t in info['timings']] == [('1. pair lists', 100, 33.33),
                                                                                ('2. conversions', 500, 166.67)]
    assert info['runs'][0]['n_errors'] == 1 and info['runs'][0]['total_s'] == 7
    assert info['sizes'][0]['n_files'] == 3 and set(info['sizes'][0]['mb_per_pair_by_group']) == {'crt_logits', 'other'}
    assert set(info['sizes'][1]['mb_per_pair_by_group']) == {'R1', 'R3_logits', 'G0.5', 'other'}
    assert capsys.readouterr().out.rstrip().endswith(f'check_d0: passed = True: {v["summary"]} -> '
                                                     f'{screen.tmp / "check_d0.json"}')


def test_dumps_without_a_mask_use_the_strict_conf_gate(screen, monkeypatch):
    """A faiss-only dump (D1) has no mask_tau0.4: S3 from crt_conf > 0.4, strictly, as run_conversions.gate. A token
    at conf == float32(0.4) whose argmax differs from the retrieved token stays retrieved."""
    pid = screen.ids[1]
    x = screen.data[pid].rc
    x['crt_conf'][4, 7] = np.float32(0.4)
    x['crt_argmax'][4, 7] = (x['retrieved_codes'][5, 7] + 1) % 2048
    x['mask_tau0.4'] = x['crt_conf'] > np.float32(0.4)
    write_rc(screen, screen.J / 'clustering_check')                 # the screen with the boundary token
    write_rc(screen, screen.tmp / 'd1' / 'rc', mask=False)
    code, v = check(monkeypatch, screen.tmp / 'v.json', '--pairs', screen.path, '--ref_screen', screen.J, '--rc',
                    screen.tmp / 'd1' / 'rc')
    assert code == 0 and v['checks']['codes']['n_equal'] == 3


# ── codes that differ ────────────────────────────────────────

def open_gate(a, closed=False):
    """(cb row, frame) of a refined token whose gate is open (closed: shut) and whose argmax != retrieved."""
    m = a['mask_tau0.4'] if not closed else ~a['mask_tau0.4']
    k, t = np.argwhere(m & (a['crt_argmax'] != a['retrieved_codes'][1:19]))[0]
    return int(k), int(t)


@pytest.mark.parametrize('case, s1, s3, passed', [('retrieved code under an open gate', True, False, False),
                                                   ('retrieved code of CB25', True, True, False),
                                                   ('argmax under an open gate', False, True, False),
                                                   ('argmax under a closed gate', False, False, True)])
def test_code_mismatches(screen, monkeypatch, case, s1, s3, passed):
    bad = screen.ids[2]

    def mutate(pid, a):
        if pid != bad:
            return
        if case == 'retrieved code of CB25':
            a['retrieved_codes'][25, 3] = (a['retrieved_codes'][25, 3] + 1) % 2048
            return
        k, t = open_gate(a, closed=case.endswith('closed gate'))
        key = 'retrieved_codes' if case.startswith('retrieved') else 'crt_argmax'
        row = k + 1 if key == 'retrieved_codes' else k
        a[key][row, t] = (a[key][row, t] + 1) % 2048
    code, v = check(monkeypatch, screen.tmp / 'v.json', *write_d0(screen, screen.tmp / 'd0', rc={'mutate': mutate}))
    c = v['checks']['codes']
    assert (c['s1_differ'] == [bad], c['s3_differ'] == [bad]) == (s1, s3)
    assert (code, v['passed'], c['passed'], c['n_equal']) == ((0, True, True, 3) if passed else (1, False, False, 2))
    assert v['summary'].split(': ', 1)[1].startswith('S1/S3 codes OK (3/3' if passed else 'S1/S3 codes FAIL (2/3')


def test_a_mask_that_differs_alone_is_reported_not_failed(screen, monkeypatch):
    """The verdict is on the codes: a gate opened where argmax == retrieved changes no code."""
    pid = screen.ids[0]
    k, t = open_gate(screen.data[pid].rc, closed=True)

    def mutate(p, a):
        if p == pid:
            a['crt_argmax'][k, t] = a['retrieved_codes'][k + 1, t]
            a['mask_tau0.4'][k, t] = True
    code, v = check(monkeypatch, screen.tmp / 'v.json', *write_d0(screen, screen.tmp / 'd0', rc={'mutate': mutate}))
    assert code == 0 and v['checks']['codes']['mask_differ_tokens'] == 1 and v['checks']['codes']['n_equal'] == 3


@pytest.mark.parametrize('which', ['anc', 'anc_r1r3'])
def test_an_out_code_that_differs_fails(screen, monkeypatch, which):
    bad = screen.ids[1]

    def mutate(pid, systems, oc):
        if pid == bad:
            oc[systems.index(ANC['A4']), 7, 2] ^= 1
    kw = {'anc' if which == 'anc' else 'r1r3': {'mutate': mutate}}
    code, v = check(monkeypatch, screen.tmp / 'v.json', *write_d0(screen, screen.tmp / 'd0', **kw))
    assert code == 1 and v['passed'] is False
    oc = {os.path.basename(x['test']): x for x in v['checks']['out_codes']}
    assert not oc[which]['passed'] and oc[which]['systems']['A4']['differ'] == [bad]
    assert all(y['n_equal'] == 3 for k, y in oc[which]['systems'].items() if k != 'A4')
    assert oc['anc_r1r3' if which == 'anc' else 'anc']['passed']
    assert f"A1-A5,C1 out_codes of {which} FAIL (missing 0, unreadable 0, pairs not equal {{'A4': 1}})" in v['summary']


def test_a_system_absent_from_the_dumps_fails(screen, monkeypatch):
    code, v = check(monkeypatch, screen.tmp / 'v.json',
                    *write_d0(screen, screen.tmp / 'd0', r1r3={'systems': R1R3_ANC[:-1]}))      # no C1
    oc = v['checks']['out_codes'][1]
    assert code == 1 and oc['systems']['C1']['absent'] == screen.ids and oc['systems']['C1']['n_equal'] == 0


@pytest.mark.parametrize('case', ['test dump missing', 'screen dump missing', 'truncated dump'])
def test_missing_or_unreadable_dumps_fail(screen, monkeypatch, case):
    pid = screen.ids[2]
    argv = write_d0(screen, screen.tmp / 'd0')
    if case == 'test dump missing':
        os.remove(screen.tmp / 'd0' / 'rc' / 'dumps' / 'dev' / '30' / f'{pid}.npz')
    elif case == 'screen dump missing':
        os.remove(screen.J / 'clustering_check' / 'dumps' / 'dev' / '30' / f'{pid}.npz')
    else:
        f = screen.tmp / 'd0' / 'rc' / 'dumps' / 'dev' / '30' / f'{pid}.npz'
        f.write_bytes(f.read_bytes()[:100])
    code, v = check(monkeypatch, screen.tmp / 'v.json', *argv)
    c = v['checks']['codes']
    assert code == 1 and c['n_equal'] == 2
    if case == 'truncated dump':
        assert list(c['unreadable']) == [pid] and c['missing'] == []
        assert 'unreadable 1' in v['summary']
    else:
        assert c['missing'] == [pid] and c['unreadable'] == {}
        assert 'S1/S3 codes FAIL (2/3 pairs equal, missing 1)' in v['summary']


# ── WAVs, and the metrics tolerance ──────────────────────────

def lsb(name, pid_bad):
    """A perturbation of one sample of `name` on pair `pid_bad` by one LSB."""
    def perturb(n, pid, a):
        if n == name and pid == pid_bad:
            a[100] = a[100] + 1 if a[100] < 32767 else a[100] - 1
        return a
    return perturb


def within(name, dsim=0.0009, dutmos=-0.009):
    def score(n, pid, r):
        if n == name:
            r['sim_wavlm'] += dsim
            r['utmos'] += dutmos
    return score


@pytest.mark.parametrize('kind', ['one LSB', 'shorter WAV'])
def test_wavs_that_differ_pass_within_the_metrics_tolerance(screen, monkeypatch, kind):
    bad = screen.ids[1]
    perturb = lsb(CANDS['C3'], bad) if kind == 'one LSB' else \
        (lambda n, pid, a: a[:-10] if (n, pid) == (CANDS['C3'], bad) else a)
    argv = write_d0(screen, screen.tmp / 'd0', cand={'perturb': perturb, 'score': within(CANDS['C3'])})
    code, v = check(monkeypatch, screen.tmp / 'v.json', *argv)
    w = v['checks']['wavs']['C3']
    assert code == 0 and v['passed'] is True
    assert (w['wavs_identical'], w['n_identical'], w['passed']) == (False, 2, True)
    assert w['max_abs_diff'] == (1 if kind == 'one LSB' else 0) and w['shape_differs'] == ([bad] if kind != 'one LSB'
                                                                                            else [])
    m = w['metrics_check']
    assert (m['n_within'], m['n_failed'], m['failed'], m['test_system'], m['ref_system']) == \
        (3, 0, {}, 'cand_gv1_pm_cont_T1', 'r3a_gv1_pm_cont_T1')
    assert m['max_dsim_wavlm'] == pytest.approx(0.0009, abs=1e-9) and m['max_dutmos'] == pytest.approx(0.009, abs=1e-9)
    assert all(v['checks']['wavs'][c]['wavs_identical'] for c in ('C1', 'C2', 'C4', 'C5'))
    assert f'C3 within tolerance (max sample diff {w["max_abs_diff"]})' in v['summary']


@pytest.mark.parametrize('case, reason', [
    ('SIM just outside', '|dsim_wavlm| = 0.0011 > 0.001'), ('UTMOS just outside', '|dutmos| = 0.0101 > 0.01'),
    ('hypothesis differs', 'Whisper hypothesis differs'), ('no test row', 'no metrics row'),
    ('error row', 'error row'), ('no hypothesis', 'no Whisper large-v3 hypothesis'), ('no SIM', 'no sim_wavlm'),
    ('UTMOS not finite', 'no utmos')])
def test_wavs_that_differ_fail_outside_the_metrics_tolerance(screen, monkeypatch, case, reason):
    """C5's WAV differs on pair 1; the scores of pair 0 (its WAV identical: the tolerance covers every pair) break
    the rule in one way."""
    name, wav_bad, score_bad = CANDS['C5'], screen.ids[1], screen.ids[0]

    def score(n, pid, r):
        within(name)(n, pid, r)
        if (n, pid) != (name, score_bad):
            return
        if case == 'SIM just outside':
            r['sim_wavlm'] += 0.0011 - 0.0009
        elif case == 'UTMOS just outside':
            r['utmos'] += 0.0101 + 0.009
        elif case == 'hypothesis differs':
            r['wer_large_v3']['hyp'] += ' again'
        elif case == 'no test row':
            r['system'] = 'cand_other'
        elif case == 'error row':
            r['error'] = 'RuntimeError: x'
        elif case == 'no hypothesis':
            del r['wer_large_v3']
        elif case == 'no SIM':
            r['sim_wavlm'] = None
        else:
            r['utmos'] = float('nan')
    argv = write_d0(screen, screen.tmp / 'd0', cand={'perturb': lsb(name, wav_bad), 'score': score})
    code, v = check(monkeypatch, screen.tmp / 'v.json', *argv)
    w = v['checks']['wavs']['C5']
    assert code == 1 and v['passed'] is False and w['passed'] is False and w['n_identical'] == 2
    assert w['metrics_check']['failed'] == {score_bad: reason} and w['metrics_check']['n_failed'] == 1
    assert all(v['checks']['wavs'][c]['passed'] for c in ('C1', 'C2', 'C3', 'C4'))
    assert 'C5 FAIL (missing 0, unreadable 0, metrics failed 1, max sample diff 1)' in v['summary']


@pytest.mark.parametrize('t, r, ok', [
    ({'sim_wavlm': 0.001, 'utmos': 2.0}, {'sim_wavlm': 0.0, 'utmos': 2.0}, True),            # |ΔSIM| = 0.001
    ({'sim_wavlm': 0.5, 'utmos': 0.01}, {'sim_wavlm': 0.5, 'utmos': 0.0}, True),             # |ΔUTMOS| = 0.01
    ({'sim_wavlm': 0.75, 'utmos': 2.0}, {'sim_wavlm': 0.75 - 2 ** -10, 'utmos': 2.0 + 2 ** -7}, True),
    ({'sim_wavlm': 0.8, 'utmos': 2.0}, {'sim_wavlm': 0.8011, 'utmos': 2.0}, False),
    ({'sim_wavlm': 0.8, 'utmos': 2.0}, {'sim_wavlm': 0.8, 'utmos': 1.9899}, False)])
def test_metric_tolerance_boundaries(t, r, ok):
    hyp = {'wer_large_v3': {'hyp': 'the same words'}}
    assert (check_d0.metric_diff({**t, **hyp}, {**r, **hyp}) == '') is ok


@pytest.mark.parametrize('case', ['test WAV missing', 'screen WAV missing', 'truncated WAV'])
def test_missing_or_unreadable_wavs_fail_even_within_tolerance(screen, monkeypatch, case):
    name, pid = CANDS['C2'], screen.ids[2]
    argv = write_d0(screen, screen.tmp / 'd0', cand={'score': within(name)})
    f = (screen.J / 'clustering_check' / 'lab_r3a' if case == 'screen WAV missing' else
         screen.tmp / 'd0' / 'lab' / 'cand') / name / f'{pid}.wav'
    if case == 'truncated WAV':
        f.write_bytes(f.read_bytes()[:20])
    else:
        os.remove(f)
    code, v = check(monkeypatch, screen.tmp / 'v.json', *argv)
    w = v['checks']['wavs']['C2']
    assert code == 1 and not w['passed'] and w['metrics_check']['n_failed'] == 0
    assert (list(w['unreadable']), w['missing']) == (([pid], []) if case == 'truncated WAV' else ([], [pid]))


# ── --ref_d0: D1 and D2 against D0 ──────────────────────────

def test_ref_d0_mode(screen, monkeypatch):
    d0 = screen.tmp / 'd0'
    write_d0(screen, d0)
    write_rc(screen, screen.tmp / 'd1' / 'rc', mask=False)                  # D1: faiss only, no masks
    write_anc(screen, screen.tmp / 'd1' / 'anc', D0_ANC[::-1])
    code, v = check(monkeypatch, screen.tmp / 'd1.json', '--pairs', screen.path, '--ref_d0', d0, '--rc',
                    screen.tmp / 'd1' / 'rc', '--anc', screen.tmp / 'd1' / 'anc', '--label', 'D1 vs D0')
    assert code == 0 and v['label'] == 'D1 vs D0' and set(v['checks']) == {'codes', 'out_codes'}
    lab = [str(d0 / 'lab' / 'cand'), str(d0 / 'lab' / 'metrics.jsonl'), 'cand_']
    assert v['reference'] == {'name': 'D0', 'rc': str(d0 / 'rc'), 'anc': str(d0 / 'anc'),
                              'cand': {c: lab for c in CANDS}}
    assert v['summary'] == 'D1 vs D0 (3 screen pairs): S1/S3 codes OK (3/3 pairs equal); A1-A5,C1 out_codes of anc OK'
    d2 = screen.tmp / 'd2' / 'lab'
    write_cands(screen, d2 / 'cand', d2 / 'metrics.jsonl', 'cand_', list(CANDS.values()))
    code, v = check(monkeypatch, screen.tmp / 'd2.json', '--pairs', screen.path, '--ref_d0', d0, '--cand', d2 / 'cand',
                    '--cand_metrics', d2 / 'metrics.jsonl', '--label', 'D2 vs D0')
    assert code == 0 and all(w['wavs_identical'] for w in v['checks']['wavs'].values())
    assert v['checks']['wavs']['C4']['ref'] == str(d0 / 'lab' / 'cand' / CANDS['C4'])
    write_cands(screen, d2 / 'cand', d2 / 'metrics.jsonl', 'cand_', list(CANDS.values()),
                perturb=lsb(CANDS['C1'], screen.ids[0]), score=within(CANDS['C1']))
    code, v = check(monkeypatch, screen.tmp / 'd2.json', '--pairs', screen.path, '--ref_d0', d0, '--cand', d2 / 'cand',
                    '--cand_metrics', d2 / 'metrics.jsonl')
    assert code == 0 and v['checks']['wavs']['C1']['metrics_check']['ref_system'] == 'cand_pm_rq_T1'
    assert v['summary'].startswith('vs D0 (3 screen pairs): C1-C5 WAVs: C1 within tolerance (max sample diff 1)')
    code, v = check(monkeypatch, screen.tmp / 'd2.json', '--pairs', screen.path, '--ref_d0', d0, '--cand', d2 / 'cand',
                    '--cand_metrics', d2 / 'nowhere.jsonl')
    assert code == 1
    assert v['checks']['wavs']['C1']['metrics_check']['failed'] == {p: 'no metrics row' for p in screen.ids}


# ── lists, refusals, exit status ─────────────────────────────

def test_pair_lists_check(screen, monkeypatch):
    good, bad = lists_dir(screen.tmp / 'good'), lists_dir(screen.tmp / 'bad', bad_dev=True)
    code, v = check(monkeypatch, screen.tmp / 'v.json', '--pairs', screen.path, '--ref_screen', screen.J, '--lists',
                    f'{good}:dev,vctk2k', '--lists', f'{bad}:dev', '--lists', f'{screen.tmp / "nowhere"}:vctk2k')
    assert code == 1 and [x['passed'] for x in v['checks']['lists']] == [True, False, False]
    assert v['summary'].endswith(f"pair lists FAIL ['{bad}:dev', '{screen.tmp / 'nowhere'}:vctk2k']")


def test_no_check_requested(screen, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['check_d0.py', '--pairs', screen.path, '--ref_screen', str(screen.J), '--out',
                                      str(screen.tmp / 'v.json')])
    with pytest.raises(SystemExit) as e:
        check_d0.main()
    assert e.value.code == 'ERROR: no check requested (--rc, --anc, --cand, --lists)'
    assert not os.path.exists(screen.tmp / 'v.json')


def test_exit_status_and_verdict_of_the_command(screen):
    argv = [str(x) for x in write_d0(screen, screen.tmp / 'd0')]
    out = screen.tmp / 'check_d0.json'
    p = subprocess.run([sys.executable, check_d0.__file__, *argv, '--out', str(out)], capture_output=True, text=True)
    assert p.returncode == 0 and json.load(open(out))['passed'] is True, p.stdout + p.stderr
    os.remove(screen.tmp / 'd0' / 'lab' / 'cand' / CANDS['C4'] / f'{screen.ids[0]}.wav')
    p = subprocess.run([sys.executable, check_d0.__file__, *argv, '--out', str(out)], capture_output=True, text=True)
    v = json.load(open(out))
    assert p.returncode == 1 and v['passed'] is False and v['checks']['wavs']['C4']['missing'] == [screen.ids[0]]
    assert '  FAIL wavs: ' in p.stdout and p.stdout.rstrip().endswith(f'passed = False: {v["summary"]} -> {out}')
