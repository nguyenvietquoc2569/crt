"""Tests of the options crt_anchor_lab.py gained for plan v1 (ANALYSIS_PLAN_v1.md §8.1 item 4): --mode R1+R3, --systems,
--source_e, --summarize_only, and the config fingerprint (cfg / cfg_sha), which must stay the old one when no new
option is given. main() runs end to end on fixtures_rows.FakePipe (no Mimi, CRT, torch or audio)."""
import glob
import hashlib
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

import crt_anchor_lab as cal
import run_conversions as rc
from fixtures_rows import (FakePipe, LTC_SHORT, dev_pairs, ltc_pairs, read_jsonl, stub_anchor_lab, vctk_pairs,
                           write_jsonl)

# A1-A5 and C1, the test-set systems (job_d0.slurm TEST_SYS), and the same in SYSTEMS order
TEST_SYS = 'R1_s0_pm_rq_T1,R3_r0_pm_rq_T1,R3_s0_argmax,R3_s0_tau0.4,src0_faiss,R3_s0_pm_rq_T1'
TEST_SYS_ORDERED = ['src0_faiss', 'R1_s0_pm_rq_T1', 'R3_r0_pm_rq_T1', 'R3_s0_tau0.4', 'R3_s0_argmax', 'R3_s0_pm_rq_T1']
# SYSTEMS of the script before the new options (HPC copy of 11 Oct 2026), the 36 systems D1 decodes
OLD_SYSTEMS = ['faiss', 'src0_faiss', 'R1_r0_tau0.4', 'R1_r0_argmax', 'R1_r0_pm_rq_T1', 'src0_tau0.4', 'R1_s0_argmax',
               'R1_s0_pm_rq_T1', 'R3_r0_tau0.4', 'R3_r0_argmax', 'R3_r0_pm_rq_T1', 'R3_s0_tau0.4', 'R3_s0_argmax',
               'R3_s0_pm_rq_T1', 'G0.5_r0_tau0.4', 'G0.5_r0_argmax', 'G0.5_r0_pm_rq_T1', 'G0.5_s0_tau0.4',
               'G0.5_s0_argmax', 'G0.5_s0_pm_rq_T1', 'R1_r0_tau0.1', 'R1_r0_tau0.2', 'R1_r0_tau0.3', 'R1_r0_tau0.5',
               'R1_r0_tau0.6', 'R1_r0_tau0.7', 'R1_r0_tau0.8', 'R1_r0_tau0.9', 'R3_r0_tau0.1', 'R3_r0_tau0.2',
               'R3_r0_tau0.3', 'R3_r0_tau0.5', 'R3_r0_tau0.6', 'R3_r0_tau0.7', 'R3_r0_tau0.8', 'R3_r0_tau0.9']
STUB_VERSIONS = {'torch': 'stub-torch', 'transformers': 'stub-transformers', 'faiss': 'stub-faiss',
                 'numpy': np.__version__, 'librosa': 'stub-librosa'}
NO_FULL_MODE_KEYS = ('R3s_', 'Psim_', 'psim_', 'js_R3_R3s')


def old_cfg(args, pipe, device, versions):
    """main()'s cfg before --mode / --systems / --source_e (HPC copy of 11 Oct 2026, l. 387-393), its constants
    (RECIPES, SYSTEMS, W_POOL; run_conversions' MIMI_ID, K, SPK_S, TOPK) written out."""
    return {'weights': args.ckpt, **pipe.crt_info, 'mimi': 'kyutai/mimi',
            'mimi_commit': getattr(pipe.mimi.config, '_commit_hash', None), 'mimi_dtype': str(pipe.dtype),
            'device_type': device, 'k': 8, 'spk_s': 15, 'topk': 16, 'budget': args.budget,
            'recipes': ('R1', 'R3', 'R3s', 'Psim', 'G0.5'), 'systems': OLD_SYSTEMS, 'w_pool': 0.5,
            'dump_logits': args.dump_logits, 'conv_dir': args.conv_dir, 'versions': versions}


def sha(cfg):
    """cfg_sha as main computes it."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def parse(monkeypatch, *extra):
    monkeypatch.setattr(sys, 'argv', ['crt_anchor_lab.py', '--pairs', 'p.jsonl', '--set', 'dev', '--out_dir', 'o',
                                      *map(str, extra)])
    return cal.parse_args()


def run_main(monkeypatch, *argv):
    """crt_anchor_lab.main() on argv: 0 if it returns, else the SystemExit code."""
    monkeypatch.setattr(sys, 'argv', ['crt_anchor_lab.py', *map(str, argv)])
    try:
        cal.main()
    except SystemExit as e:
        return e.code
    return 0


def manifest(out_dir, set_='dev', budget='30', name='manifest.jsonl'):
    return read_jsonl(os.path.join(out_dir, set_, budget, name))


def npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def dumps(out_dir, set_='dev', budget='30'):
    return {os.path.basename(f)[:-4]: npz(f) for f in sorted(glob.glob(f'{out_dir}/dumps/{set_}/{budget}/*.npz'))}


def no_models(monkeypatch):
    """torch, faiss and transformers cannot be imported and no Pipeline or Lab can be built."""
    for m in ('torch', 'faiss', 'transformers'):
        monkeypatch.setitem(sys.modules, m, None)

    def refuse(*a, **k):
        raise AssertionError('a model was loaded')
    monkeypatch.setattr(rc, 'Pipeline', refuse)
    monkeypatch.setattr(cal.cdl, 'Lab', refuse)


# ── --systems ────────────────────────────────────────────────

@pytest.mark.parametrize('spec', ['', ' ', ' , '])
def test_select_systems_default_is_every_system(spec):
    got = cal.select_systems(spec)
    assert got == OLD_SYSTEMS == cal.SYSTEMS and got is not cal.SYSTEMS


def test_select_systems_keeps_the_systems_order():
    assert cal.select_systems(TEST_SYS) == TEST_SYS_ORDERED
    assert cal.select_systems(' R3_r0_tau0.9 ,faiss') == ['faiss', 'R3_r0_tau0.9']
    assert cal.select_systems(','.join(reversed(cal.SYSTEMS))) == OLD_SYSTEMS


@pytest.mark.parametrize('spec', ['anc_src0_faiss', 'R1_s0_tau0.4', 'R3s_s0_argmax', 'Psim_r0_argmax', 'pm_rq_T1',
                                  'src0_faiss,src0_faiss', 'src0_faiss, src0_faiss ', 'faiss,R3_s0_argmax,faiss',
                                  'faiss,nope'])
def test_select_systems_refuses_unknown_or_repeated_names(spec):
    with pytest.raises(SystemExit) as e:
        cal.select_systems(spec)
    assert e.value.code.startswith(f'ERROR: --systems {spec!r}: unknown ')


@pytest.mark.parametrize('spec', ['src0_faiss,R3_s0_argmax,src0_faiss', 'anc_src0_faiss'])
def test_bad_systems_exit_with_status_1(tmp_path, spec):
    p = subprocess.run([sys.executable, cal.__file__, '--pairs', tmp_path / 'p.jsonl', '--set', 'dev', '--out_dir',
                        tmp_path / 'o', '--systems', spec], capture_output=True, text=True)
    assert p.returncode == 1 and p.stderr.startswith('ERROR: --systems'), p.stderr
    assert not os.path.exists(tmp_path / 'o')


def test_parse_args_defaults_and_new_options(monkeypatch):
    a = parse(monkeypatch)
    assert (a.mode, a.systems, a.source_e, a.summarize_only) == ('full', '', '', False)
    assert a.system_list == OLD_SYSTEMS
    a = parse(monkeypatch, '--mode', 'R1+R3', '--systems', TEST_SYS, '--source_e', 'se.json', '--summarize_only')
    assert (a.mode, a.system_list, a.source_e, a.summarize_only) == ('R1+R3', TEST_SYS_ORDERED, 'se.json', True)


# ── --mode ───────────────────────────────────────────────────

def test_modes_and_every_system_decodable_in_r1r3():
    assert cal.MODES == {'full': ('R1', 'R3', 'R3s', 'Psim', 'G0.5'), 'R1+R3': ('R1', 'R3', 'G0.5')}
    assert {a for a, _, _ in cal.SPEC.values()} - {None} <= set(cal.MODES['R1+R3'])
    assert set(TEST_SYS_ORDERED) <= set(cal.SPEC)


def test_r1r3_mode_reads_only_the_pairs_own_target_reference(tmp_path, monkeypatch, capsys):
    """VCTK-2k (no source speaker is a target): one reference fetch (the pair's target) and two CRT forwards per pair,
    no R3s / Psim output anywhere."""
    pairs = vctk_pairs()
    path = write_jsonl(tmp_path / 'vctk2k.jsonl', pairs)
    log = stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    assert run_main(monkeypatch, '--pairs', path, '--set', 'vctk2k', '--out_dir', out, '--device', 'cpu', '--mode',
                    'R1+R3', '--systems', TEST_SYS, '--dump_logits', '--dump_recipes', 'R3') == 0
    assert log.refs == [(p['pair_id'], p['target_speaker']) for p in pairs]
    assert len(log.crt) == 2 * len(pairs)
    rows = manifest(out, 'vctk2k')
    assert [(r['pair_id'], r['system']) for r in rows] == [(p['pair_id'], 'anc_' + s) for p in pairs
                                                           for s in TEST_SYS_ORDERED]
    assert all(r['psim_speaker'] is None and 'error' not in r for r in rows)
    for z in dumps(out, 'vctk2k').values():
        assert not [k for k in z if k.startswith(NO_FULL_MODE_KEYS)]
        assert {'R1_conf', 'R3_conf', 'G0.5_conf', 'js_R1_R3', 'R3_logits'} <= set(z) and 'R1_logits' not in z
        assert list(z['systems']) == TEST_SYS_ORDERED and z['out_codes'].shape[0] == 6
    s = json.load(open(out / 'vctk2k' / '30' / 'anchor_summary.json'))
    assert s['recipes'] == ['R1', 'R3', 'G0.5'] and s['pred1_confidence'] is None and s['pred2_embedding'] is None
    assert s['n_pairs'] == 12 and list(s['systems']) == TEST_SYS_ORDERED
    assert 'mode R1+R3' in capsys.readouterr().out


def test_full_mode_reads_the_source_and_a_third_speakers_reference(tmp_path, monkeypatch):
    pairs = dev_pairs()
    path = write_jsonl(tmp_path / 'dev.jsonl', pairs)
    log = stub_anchor_lab(monkeypatch, cal)
    assert run_main(monkeypatch, '--pairs', path, '--set', 'dev', '--max_pairs', '3', '--out_dir', tmp_path / 'anc',
                    '--device', 'cpu', '--systems', 'src0_faiss') == 0
    first = {}
    for p in pairs:
        first.setdefault(p['target_speaker'], p['pair_id'])
    assert len(log.refs) == 9 and len(log.crt) == 12
    psim = {r['pair_id']: r['psim_speaker'] for r in manifest(tmp_path / 'anc')}
    for i, p in enumerate(pairs[:3]):
        own, src, third = log.refs[3 * i:3 * i + 3]
        assert own == (p['pair_id'], p['target_speaker'])
        assert src == (first[p['source_speaker']], p['source_speaker'])
        assert third[1] not in (p['source_speaker'], p['target_speaker']) and third[0] == first[third[1]]
        assert psim[p['pair_id']] == third[1]
    z = dumps(tmp_path / 'anc')[pairs[0]['pair_id']]
    assert {'R3s_conf', 'Psim_conf', 'psim_codes', 'js_R3_R3s'} <= set(z)


def test_full_mode_exits_when_a_source_speaker_is_not_a_target(tmp_path, monkeypatch):
    path = write_jsonl(tmp_path / 'vctk2k.jsonl', vctk_pairs())
    stub_anchor_lab(monkeypatch, cal)
    code = run_main(monkeypatch, '--pairs', path, '--set', 'vctk2k', '--out_dir', tmp_path / 'anc', '--device', 'cpu')
    assert code == (f"ERROR: source speakers ['p301', 'p302', 'p303'] are not targets in {path}: R3s / Psim need "
                    f"their reference")


@pytest.mark.parametrize('mode, n_err', [('full', 3), ('R1+R3', 0)])
def test_a_source_speaker_shortfall_stops_only_the_full_mode(tmp_path, monkeypatch, capsys, mode, n_err):
    """LTC-2k at 180 s, target 6930 short: its 3 pairs are counted shortfalls in both modes; the 3 pairs with source
    6930 fail in full mode only (R3s / Psim need the source speaker's R), as errors, not shortfalls."""
    pairs = ltc_pairs()
    path = write_jsonl(tmp_path / 'ltc2k.jsonl', pairs)
    stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    code = run_main(monkeypatch, '--pairs', path, '--set', 'ltc2k', '--budget', '180', '--out_dir', out, '--device',
                    'cpu', '--mode', mode, '--systems', 'src0_faiss,R3_s0_pm_rq_T1')
    assert code == (1 if n_err else 0)
    assert f'{n_err} errors, 3 reference shortfalls' in capsys.readouterr().out
    rows = manifest(out, 'ltc2k', '180')
    short = {r['pair_id'] for r in rows if r.get('shortfall')}
    failed = {r['pair_id'] for r in rows if 'error' in r and not r['shortfall']}
    assert short == {p['pair_id'] for p in pairs if p['target_speaker'] == LTC_SHORT}
    assert failed == ({p['pair_id'] for p in pairs if p['source_speaker'] == LTC_SHORT} if n_err else set())
    assert all(r['error'].startswith('RuntimeError: source-speaker reference shortfall') for r in rows
               if r['pair_id'] in failed)


def test_r1r3_mode_decodes_the_test_systems_as_the_full_mode(tmp_path, monkeypatch):
    path = write_jsonl(tmp_path / 'dev.jsonl', dev_pairs())
    stub_anchor_lab(monkeypatch, cal)
    for mode in ('full', 'R1+R3'):
        assert run_main(monkeypatch, '--pairs', path, '--set', 'dev', '--max_pairs', '4', '--out_dir',
                        tmp_path / mode, '--device', 'cpu', '--mode', mode, '--systems', TEST_SYS) == 0
    full, r1r3 = dumps(tmp_path / 'full'), dumps(tmp_path / 'R1+R3')
    assert list(full) == list(r1r3) and len(full) == 4
    for pid, z in r1r3.items():
        for k in z:
            if k != 'cfg_sha':
                np.testing.assert_array_equal(z[k], full[pid][k], err_msg=k)
        assert set(full[pid]) - set(z) == {k for k in full[pid] if k.startswith(NO_FULL_MODE_KEYS)}
    strip = ('psim_speaker', 'conv_cfg_sha', 'run_id', 'wav', 'dump')
    a, b = manifest(tmp_path / 'full'), manifest(tmp_path / 'R1+R3')
    assert [{k: v for k, v in r.items() if k not in strip} for r in a] == \
        [{k: v for k, v in r.items() if k not in strip} for r in b]
    for ra, rb in zip(a, b):
        assert open(ra['wav'], 'rb').read() == open(rb['wav'], 'rb').read()


# ── --source_e ───────────────────────────────────────────────

def test_load_source_e_both_formats(tmp_path):
    f = tmp_path / 'vctk2k_source_E.json'
    f.write_text(json.dumps({'p301': {'paths': ['/a.flac', '/b.flac'], 'durs': [3.0, 4.0], 'total_s': 7.0},
                             'p302': ['/c.flac'], 'p303': []}, indent=1))
    assert cal.load_source_e(str(f)) == {'p301': ['/a.flac', '/b.flac'], 'p302': ['/c.flac'], 'p303': []}


@pytest.mark.parametrize('content', [['/a.flac'], {'p301': '/a.flac'}, {'p301': {'durs': [3.0]}},
                                     {'p301': [1, 2]}, {'set': 'vctk2k', 'speakers': {'p301': ['/a']}}])
def test_load_source_e_refuses_other_content(tmp_path, content):
    f = tmp_path / 'se.json'
    f.write_text(json.dumps(content))
    with pytest.raises(SystemExit) as e:
        cal.load_source_e(str(f))
    assert e.value.code.startswith(f'ERROR: --source_e {f}')


def test_source_e_of():
    spk_rows = {'p311': {'E': {'paths': ['/p311/e.flac']}}}
    se = {'p301': ['/p301/e.flac'], 'p302': [], 'p311': ['/p311/source_e.flac']}
    assert cal.source_e_of('p311', spk_rows, None) == ['/p311/e.flac']          # Dev, LTC-2k: the target E
    assert cal.source_e_of('p301', spk_rows, None) is None                      # not a target, no --source_e
    assert cal.source_e_of('p301', spk_rows, se) == ['/p301/e.flac']            # VCTK-2k: the file
    assert cal.source_e_of('p302', spk_rows, se) is None                        # empty entry
    assert cal.source_e_of('p303', spk_rows, se) is None                        # no entry
    assert cal.source_e_of('p311', spk_rows, se) == ['/p311/source_e.flac']     # the file wins when given
    assert cal.source_e_of('p311', spk_rows, {}) is None


def test_source_e_gives_sim_src_wavs_and_the_src_manifest(tmp_path, monkeypatch, capsys):
    pairs = vctk_pairs()
    path = write_jsonl(tmp_path / 'vctk2k.jsonl', pairs)
    se = tmp_path / 'vctk2k_source_E.json'
    e301, e302 = ['/v/p301/p301_900_mic1.flac', '/v/p301/p301_901_mic1.flac'], ['/v/p302/p302_900_mic1.flac']
    se.write_text(json.dumps({'p301': {'paths': e301, 'durs': [3.0, 3.0], 'total_s': 6.0}, 'p302': e302}))
    stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    assert run_main(monkeypatch, '--pairs', path, '--set', 'vctk2k', '--out_dir', out, '--device', 'cpu', '--mode',
                    'R1+R3', '--systems', TEST_SYS, '--source_e', se) == 0
    log = capsys.readouterr().out
    assert "WARN: source speakers ['p303'] have no E set (not in --source_e)" in log
    assert f'24 rows without a source-speaker E are not in {out}/vctk2k/30/manifest_src.jsonl' in log
    rows, src = manifest(out, 'vctk2k'), manifest(out, 'vctk2k', name='manifest_src.jsonl')
    want = {'p301': e301, 'p302': e302, 'p303': None}
    assert all(r['sim_src_wavs'] == want[r['source_speaker']] for r in rows)
    assert src == [dict(r, system=r['system'] + '@src', sim_ref_wavs=r['sim_src_wavs'],
                        target_speaker=r['source_speaker'], true_target_speaker=r['target_speaker'])
                   for r in rows if r['source_speaker'] != 'p303']
    cfg = json.load(open(out / 'vctk2k' / '30' / 'anchor_summary.json'))['cfg']
    assert cfg['source_e'] == str(se) and cfg['source_e_sha256'] == hashlib.sha256(se.read_bytes()).hexdigest()


def test_without_source_e_vctk_sources_get_no_src_rows(tmp_path, monkeypatch, capsys):
    path = write_jsonl(tmp_path / 'vctk2k.jsonl', vctk_pairs())
    stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    assert run_main(monkeypatch, '--pairs', path, '--set', 'vctk2k', '--out_dir', out, '--device', 'cpu', '--mode',
                    'R1+R3', '--systems', 'src0_faiss') == 0
    log = capsys.readouterr().out
    assert "WARN: source speakers ['p301', 'p302', 'p303'] have no E set (not targets, no --source_e)" in log
    assert all(r['sim_src_wavs'] is None for r in manifest(out, 'vctk2k'))
    assert manifest(out, 'vctk2k', name='manifest_src.jsonl') == []


def test_dev_sim_src_wavs_is_the_first_target_e_of_the_source(tmp_path, monkeypatch):
    """Dev, LTC-2k: a source speaker's E is its target E from the first pair row whose target it is (lab_rows takes the
    same row)."""
    pairs = dev_pairs()
    seen = set()
    for i, p in enumerate(pairs):            # every later row of a target carries another E
        if p['target_speaker'] in seen:
            pairs[i] = dict(p, E=dict(p['E'], paths=[f'/later/{p["target_speaker"]}_{i}.flac']))
        seen.add(p['target_speaker'])
    path = write_jsonl(tmp_path / 'dev.jsonl', pairs)
    stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    assert run_main(monkeypatch, '--pairs', path, '--set', 'dev', '--shard', '1/4', '--out_dir', out, '--device',
                    'cpu', '--mode', 'R1+R3', '--systems', 'src0_faiss,R3_s0_pm_rq_T1') == 0
    first = {}
    for p in pairs:
        first.setdefault(p['target_speaker'], p['E']['paths'])
    rows = manifest(out, name='manifest.shard1of4.jsonl')
    assert len(rows) == 12 and all(r['sim_src_wavs'] == first[r['source_speaker']] for r in rows)
    assert len(manifest(out, name='manifest_src.shard1of4.jsonl')) == 12


# ── cfg, cfg_sha ─────────────────────────────────────────────

@pytest.mark.parametrize('extra', [[], ['--mode', 'full'], ['--systems', ','.join(OLD_SYSTEMS)],
                                   ['--systems', ','.join(reversed(OLD_SYSTEMS))], ['--summarize_only']])
def test_cfg_without_new_options_is_the_old_cfg(monkeypatch, extra):
    for device, dump_logits in (('cuda', []), ('cpu', ['--dump_logits'])):
        args = parse(monkeypatch, *extra, *dump_logits, '--conv_dir', '/rc')
        pipe = FakePipe()
        new, old = cal.run_cfg(args, pipe, device, STUB_VERSIONS), old_cfg(args, pipe, device, STUB_VERSIONS)
        assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
        assert sha(new) == sha(old)


@pytest.mark.parametrize('extra, keys', [(['--mode', 'R1+R3'], {'recipes'}), (['--systems', 'src0_faiss'], {'systems'}),
                                         (['--source_e', 'SE'], {'source_e', 'source_e_sha256'})])
def test_new_options_change_the_cfg_through_their_values_only(tmp_path, monkeypatch, extra, keys):
    se = tmp_path / 'vctk2k_source_E.json'
    se.write_text(json.dumps({'p301': ['/a.flac']}))
    args = parse(monkeypatch, *[str(se) if x == 'SE' else x for x in extra])
    pipe = FakePipe()
    new, old = cal.run_cfg(args, pipe, 'cuda', STUB_VERSIONS), old_cfg(args, pipe, 'cuda', STUB_VERSIONS)
    assert {k for k in set(new) | set(old) if json.dumps(new.get(k)) != json.dumps(old.get(k))} == keys
    assert sha(new) != sha(old)
    if 'source_e' in keys:
        assert new['source_e'] == str(se) and new['source_e_sha256'] == hashlib.sha256(se.read_bytes()).hexdigest()
        se.write_text(json.dumps({'p301': ['/b.flac']}))                      # the file's content counts
        assert sha(cal.run_cfg(args, pipe, 'cuda', STUB_VERSIONS)) != sha(new)


def test_main_writes_under_the_old_cfg_sha_and_resumes(tmp_path, monkeypatch, capsys):
    """The default invocation (full mode, every system) keeps the old cfg_sha in rows, dumps and summary, so a
    directory written before the change resumes; a new option is a different cfg and is refused there."""
    path = write_jsonl(tmp_path / 'dev.jsonl', dev_pairs())
    stub_anchor_lab(monkeypatch, cal)
    out = tmp_path / 'anc'
    argv = ['--pairs', path, '--set', 'dev', '--max_pairs', '2', '--out_dir', out, '--device', 'cpu']
    assert run_main(monkeypatch, *argv) == 0
    args = types.SimpleNamespace(ckpt=rc.VARIANTS['v5.6.5']['slim'], budget='30', dump_logits=False, conv_dir='')
    want = sha(old_cfg(args, FakePipe(), 'cpu', STUB_VERSIONS))
    rows = manifest(out)
    assert len(rows) == 72 and {r['conv_cfg_sha'] for r in rows} == {want}
    assert {str(z['cfg_sha']) for z in dumps(out).values()} == {want}
    assert json.load(open(out / 'dev' / '30' / 'anchor_summary.json'))['cfg_sha'] == want
    capsys.readouterr()
    assert run_main(monkeypatch, *argv) == 0 and '(0 to do)' in capsys.readouterr().out
    se = tmp_path / 'dev_source_E.json'
    se.write_text(json.dumps({'p225': ['/p225/e.flac']}))
    for extra in (['--mode', 'R1+R3'], ['--systems', 'src0_faiss'], ['--source_e', se]):
        code = run_main(monkeypatch, *argv, *extra)
        assert isinstance(code, str) and code.endswith('use another --out_dir'), code


# ── --summarize_only ─────────────────────────────────────────

def r1r3_run(monkeypatch, path, out, *extra):
    return run_main(monkeypatch, '--pairs', path, '--set', 'dev', '--out_dir', out, '--device', 'cpu', '--mode',
                    'R1+R3', '--systems', 'src0_faiss,R3_s0_pm_rq_T1', *extra)


@pytest.fixture
def sharded(tmp_path, monkeypatch):
    """A Dev run in the R1+R3 mode, once unsharded (one/) and once in two shards (sh/)."""
    path = write_jsonl(tmp_path / 'dev.jsonl', dev_pairs())
    stub_anchor_lab(monkeypatch, cal)
    assert r1r3_run(monkeypatch, path, tmp_path / 'one') == 0
    for i in range(2):
        assert r1r3_run(monkeypatch, path, tmp_path / 'sh', '--shard', f'{i}/2') == 0
    return types.SimpleNamespace(path=path, one=tmp_path / 'one', sh=tmp_path / 'sh',
                                 run=tmp_path / 'sh' / 'dev' / '30')


def summarize_only(monkeypatch, s, *extra):
    no_models(monkeypatch)
    return run_main(monkeypatch, '--pairs', s.path, '--set', 'dev', '--out_dir', s.sh, '--summarize_only', *extra)


def test_summarize_only_over_the_shards_equals_the_unsharded_summary(sharded, monkeypatch, capsys):
    assert sorted(os.listdir(sharded.run)) == [
        'anc_R3_s0_pm_rq_T1', 'anc_src0_faiss', 'anchor_summary.shard0of2.json', 'anchor_summary.shard1of2.json',
        'manifest.shard0of2.jsonl', 'manifest.shard1of2.jsonl', 'manifest_src.shard0of2.jsonl',
        'manifest_src.shard1of2.jsonl']
    capsys.readouterr()
    assert summarize_only(monkeypatch, sharded) == 0
    one = open(sharded.one / 'dev' / '30' / 'anchor_summary.json').read()
    assert open(sharded.run / 'anchor_summary.json').read() == one
    cfg_sha = json.loads(one)['cfg_sha']
    assert (f'24 of 24 pairs summarized (24 dumps, 0 pairs with error rows) from 2 manifests, cfg {cfg_sha} -> '
            f'{sharded.run}/anchor_summary.json') in capsys.readouterr().out


def test_summarize_only_reads_manifest_jsonl_and_the_shard_files(sharded, monkeypatch, capsys):
    """An unsharded manifest.jsonl next to shard files is read too; manifest_src*, *.tmp and other files are not."""
    rows = read_jsonl(sharded.run / 'manifest.shard1of2.jsonl')
    write_jsonl(sharded.run / 'manifest.jsonl', rows[:4])                       # the same rows: kept once
    other = [dict(r, conv_cfg_sha='ffffffffffffffff') for r in rows]
    for name in ('manifest_src.jsonl', 'manifest.jsonl.tmp', 'manifest.shard1of2.jsonl.tmp', 'old_manifest.jsonl'):
        write_jsonl(sharded.run / name, other)
    capsys.readouterr()
    assert summarize_only(monkeypatch, sharded) == 0
    assert '24 of 24 pairs summarized (24 dumps, 0 pairs with error rows) from 3 manifests' in capsys.readouterr().out
    mans, kept, _ = cal.run_manifests(str(sharded.run))
    assert [os.path.basename(f) for f in mans] == ['manifest.jsonl', 'manifest.shard0of2.jsonl',
                                                   'manifest.shard1of2.jsonl']
    assert len(kept) == 48


@pytest.mark.parametrize('where', ['shard', 'manifest.jsonl'])
def test_summarize_only_refuses_two_cfgs(sharded, monkeypatch, where):
    rows = read_jsonl(sharded.run / 'manifest.shard1of2.jsonl')
    if where == 'shard':
        write_jsonl(sharded.run / 'manifest.shard1of2.jsonl', rows[:-1] + [dict(rows[-1], conv_cfg_sha='f' * 16)])
    else:
        write_jsonl(sharded.run / 'manifest.jsonl',
                    [dict(rows[0], pair_id='dev__p999_001__p225', conv_cfg_sha='f' * 16)])
    code = summarize_only(monkeypatch, sharded)
    assert code.startswith(f'ERROR: {sharded.run}: manifests ') and code.endswith('not exactly one')
    assert not os.path.exists(sharded.run / 'anchor_summary.json')


def test_summarize_only_refuses_a_run_without_manifest(tmp_path, monkeypatch):
    path = write_jsonl(tmp_path / 'dev.jsonl', dev_pairs())
    no_models(monkeypatch)
    code = run_main(monkeypatch, '--pairs', path, '--set', 'dev', '--out_dir', tmp_path / 'none', '--summarize_only')
    assert code == f"ERROR: {tmp_path}/none/dev/30: manifests [] hold cfg [], not exactly one"


def test_summarize_only_refuses_shard(sharded, monkeypatch):
    assert summarize_only(monkeypatch, sharded, '--shard', '0/2') == \
        'ERROR: --summarize_only reads every shard of the run; drop --shard'
    assert not os.path.exists(sharded.run / 'anchor_summary.json')


def test_summarize_only_leaves_out_pairs_with_an_error_row(sharded, monkeypatch, capsys):
    f = sharded.run / 'manifest.shard0of2.jsonl'
    rows = read_jsonl(f)
    write_jsonl(f, rows + [dict(rows[0], error='AssertionError: x', shortfall=False)])
    capsys.readouterr()
    assert summarize_only(monkeypatch, sharded) == 0
    assert '23 of 24 pairs summarized (23 dumps, 1 pairs with error rows) from 2 manifests' in capsys.readouterr().out
    assert json.load(open(sharded.run / 'anchor_summary.json'))['n_pairs'] == 23


def test_summarize_only_on_full_mode_shards(tmp_path, monkeypatch):
    """Full mode: predictions 1 and 2 are there and equal the unsharded run's."""
    path = write_jsonl(tmp_path / 'dev.jsonl', dev_pairs())
    stub_anchor_lab(monkeypatch, cal)
    argv = ['--pairs', path, '--set', 'dev', '--max_pairs', '6', '--device', 'cpu', '--systems', 'src0_faiss']
    assert run_main(monkeypatch, *argv, '--out_dir', tmp_path / 'one') == 0
    for i in range(3):
        assert run_main(monkeypatch, *argv, '--out_dir', tmp_path / 'sh', '--shard', f'{i}/3') == 0
    no_models(monkeypatch)
    assert run_main(monkeypatch, *argv, '--out_dir', tmp_path / 'sh', '--summarize_only') == 0
    a = json.load(open(tmp_path / 'one' / 'dev' / '30' / 'anchor_summary.json'))
    b = json.load(open(tmp_path / 'sh' / 'dev' / '30' / 'anchor_summary.json'))
    assert a == b and a['n_pairs'] == 6 and a['pred1_confidence'] and a['pred2_embedding']


# ── --conv_dir with the run_conversions dumps (D0 steps 2 and 4) ─

def test_conv_dir_faiss_rows_and_r1_reproduction(tmp_path, monkeypatch, capsys):
    pairs = dev_pairs()[:6]
    path = write_jsonl(tmp_path / 'dev.jsonl', pairs)
    stub_anchor_lab(monkeypatch, cal)
    dummy = tmp_path / 'weights.pt'
    dummy.write_text('stub')
    conv = tmp_path / 'rc'
    monkeypatch.setattr(sys, 'argv', ['run_conversions.py', '--pairs', str(path), '--set', 'dev', '--systems',
                                      'faiss,crt_tau:0.4', '--dump_logits', '--out_dir', str(conv), '--device', 'cpu',
                                      '--ckpt', str(dummy), '--src_ckpt', str(dummy), '--crt_script', str(dummy)])
    rc.main()
    argv = ['--pairs', path, '--set', 'dev', '--conv_dir', conv, '--device', 'cpu', '--mode', 'R1+R3', '--systems',
            TEST_SYS]
    assert run_main(monkeypatch, *argv, '--out_dir', tmp_path / 'anc') == 0
    faiss = {r['pair_id']: r for r in manifest(conv) if r['system'] == 'faiss'}
    for r in manifest(tmp_path / 'anc'):
        f = faiss[r['pair_id']]
        assert {k: r[k] for k in ('ref_s', 'R', 'sim_ref_wavs', 'n_ref_frames')} == \
            {k: f[k] for k in ('ref_s', 'R', 'sim_ref_wavs', 'n_ref_frames')}
        assert 'conv_time_s' not in r and r['conv_cfg_sha'] != f['conv_cfg_sha']
    rep = json.load(open(tmp_path / 'anc' / 'dev' / '30' / 'anchor_summary.json'))['r1_repro']
    assert rep == {'n_checked': 6, 'max_conf_diff': 0.0, 'max_logit_diff': 0.0, 'gate_flips_tau0.4': 0}
    bad = pairs[2]['pair_id']                       # a conv dump whose confidences differ: that pair fails
    f = conv / 'dumps' / 'dev' / '30' / f'{bad}.npz'
    z = npz(f)
    z['crt_conf'] = z['crt_conf'] + np.float32(0.01)
    np.savez(f, **z)
    capsys.readouterr()
    assert run_main(monkeypatch, *argv, '--out_dir', tmp_path / 'anc2') == 1
    assert f'ERROR {bad}: AssertionError: R1 does not reproduce {f}' in capsys.readouterr().out
    assert {r['pair_id'] for r in manifest(tmp_path / 'anc2') if 'error' in r} == {bad}
