"""Tests of mvc/mos/build_stimuli.py on synthetic pair lists, rows and WAVs (no network, no git repository): the seeded
draw (determinism, strata, the journal's order convention, the P preference, the eligibility rules at their
boundaries, the excluded pairs, the speaker and sentence rules, the pairs passed over, the practice items), the natural
clip, the SMOS reference and the source clip (never in R, never the same), the nvquoc -> ceai mapping, the audio
(24 kHz mono, loudness normalisation, cut, padding), the package (random opaque ids, nothing of a condition in
lt_data.json, rounded durations, the key and the cache outside the package and the repo, every stimulus and trial in the
key, the native rate of every source file, an atomic build), the second draw at build time, the manifest, the rsync
call, and the refusals of the freeze guard (addendum, builder, design, tag, origin) and of --dry_run on real inputs.
MOS_GIT_TESTS=1 also runs the guard against git repositories made in a temporary folder."""
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_stimuli as bs  # noqa: E402

np = pytest.importorskip('numpy')
sf = pytest.importorskip('soundfile')
pytest.importorskip('soxr')

SYSTEMS = {'mvc': ['mvc_v1', 'mvc_v1_q32'], 'journal': ['cand_pm18_rq', 'faiss'], 't4': ['seedvc', 'knnvc', 'facodec']}
SEED = '20260928:mvc_v1'
DESIGN = {'seed': SEED, 'n_per_stratum': 4, 'mos_conditions': ['nat', 'mvc_v1', 'crtal', 'seedvc', 'knnvc'],
          'smos_conditions': ['nat', 'mvc_v1', 'crtal', 'seedvc', 'knnvc', 'src'], 'halves': True,
          'prefer_parallel': {'vctk2k': True}, 'source_range': [2.0, 10.0], 'natural_range': [2.0, 10.0],
          'ref_range': [3.0, 10.0], 'ref_target_s': 5.0, 'ref_fallback': True, 'speaker_rule': True,
          'sentence_rule': True, 'pad_ms': [0, 250],
          'practice': [['vctk2k', 'F2F', 'nat'], ['ltc2k', 'M2M', 'crtal'], ['vctk2k', 'F2M', 'seedvc']],
          'attention': {'mos': [['vctk2k', 'F2M', 2], ['ltc2k', 'M2F', 4]],
                        'smos': [['ltc2k', 'F2F', 4], ['vctk2k', 'M2M', 2]]}}
N_PER_CELL = 12
N_MOS, N_SMOS = 5, 6


def make_pair(s, g, n, root, e_durs=(3.0, 4.5, 8.0), src_dur=4.0, p_dur=3.3, src_spk=None, tgt_spk=None, text=None):
    k = f'{s}__{g}_{n:02d}'
    with_p = s == 'vctk2k' and n % 2 == 1
    return {'pair_id': k, 'set': s, 'genders': g, 'source_speaker': src_spk or f'{s}{g}s{n}',
            'target_speaker': tgt_spk or f'{s}{g}t{n}', 'source_utt': f'{k}_src', 'source_dur': src_dur,
            'source_path': f'{root}/{k}_src.wav', 'source_text': text or f'Sentence {s} {g} number {n}.',
            'P': f'{root}/{k}_P.flac' if with_p else None, 'P_dur': p_dur if with_p else None,
            'E': {'paths': [f'{root}/{k}_E{i}.wav' for i in range(len(e_durs))], 'durs': list(e_durs)},
            'R': {'30': {'paths': [f'{root}/{k}_R{i}.wav' for i in range(2)]},
                  '180': {'paths': [f'{root}/{k}_R{i}.wav' for i in range(4)]}}}


def default_pair(s, g, n, root):
    """Every fifth LTC pair has only an E utterance over 10 s left for the SMOS reference (the fallback)."""
    if s == 'ltc2k' and n % 5 == 2:
        return make_pair(s, g, n, root, e_durs=(4.0, 12.0, 1.5))
    return make_pair(s, g, n, root)


def world(tmp_path, root=None, fail=(), drop=(), journal_root=None, pair=default_pair, n_per_cell=N_PER_CELL):
    """Pair lists of 2 sets x 4 cells x N_PER_CELL pairs and the row files of the three trees under tmp_path; the rows
    of (system, pair_id) in fail are error rows, those in drop are missing. Journal rows name journal_root (default
    root) as their prefix, as the journal rows on the Mac name /datastore/nvquoc."""
    root = root or str(tmp_path / 'audio')
    jroot = journal_root or root
    pairs = [pair(s, g, n, jroot) for s in bs.SETS for g in bs.CELLS for n in range(n_per_cell)]
    files = {}
    for s in bs.SETS:
        f = tmp_path / 'pairs' / f'{s}.jsonl'
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(''.join(json.dumps(p) + '\n' for p in pairs if p['set'] == s))
        files[s] = str(f)
    for tree, systems in SYSTEMS.items():
        for s in bs.SETS:
            path = Path(bs.rows_path(tree, s, str(tmp_path / 'mvc'), str(tmp_path / 'journal')))
            path.parent.mkdir(parents=True, exist_ok=True)
            pre = root if tree == 'mvc' else jroot
            with open(path, 'a') as fo:
                for p in pairs:
                    if p['set'] != s:
                        continue
                    for system in systems:
                        k = p['pair_id']
                        if (system, k) in drop:
                            continue
                        r = {'pair_id': k, 'set': s, 'budget': 30, 'system': system, 'wav': f'{pre}/{k}_{system}.wav',
                             'genders': p['genders'], 'target_speaker': p['target_speaker'],
                             'sim_ref_wavs': [x.replace(jroot, pre) for x in p['E']['paths']],
                             'parallel_wav': p['P'] and p['P'].replace(jroot, pre),
                             'source_wav': p['source_path'].replace(jroot, pre), 'utmos': 9.87654}
                        if (system, k) in fail:
                            r = {'pair_id': k, 'set': s, 'budget': 30, 'system': system, 'error': 'boom'}
                        fo.write(json.dumps(r) + '\n')
    return {p['pair_id']: p for p in pairs}, files


def do_draw(tmp_path, design=DESIGN, name='items.json', design_path=None, extra=(), **kw):
    if not (tmp_path / 'pairs').exists():
        world(tmp_path, **kw)
    files = {s: str(tmp_path / 'pairs' / f'{s}.jsonl') for s in bs.SETS}
    text = json.dumps(design)
    d = Path(design_path) if design_path else tmp_path / f'design_{bs.sha256_bytes(text.encode())[:8]}.json'
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(text)
    out = tmp_path / name
    bs.main(['draw', '--design', str(d), '--vctk_pairs', files['vctk2k'], '--ltc_pairs', files['ltc2k'],
             '--mvc_rows', str(tmp_path / 'mvc'), '--journal_rows', str(tmp_path / 'journal'), '--out', str(out),
             *extra])
    return json.loads(out.read_text()), out


def by_cell(doc):
    out = {}
    for it in doc['items']:
        out.setdefault((it['set'], it['genders']), []).append(it)
    return out


def order_of(s, g, seed=SEED, prefer=True, n_per_cell=N_PER_CELL):
    ids = sorted(f'{s}__{g}_{n:02d}' for n in range(n_per_cell))
    random.Random(f'{seed}:listening:{s}:{g}').shuffle(ids)
    if prefer and s == 'vctk2k':
        ids = [k for k in ids if int(k[-2:]) % 2 == 1] + [k for k in ids if int(k[-2:]) % 2 == 0]
    return ids


# ── draw ─────────────────────────────────────────────────────

def test_draw_is_deterministic_and_writes_its_sha256(tmp_path):
    doc, out = do_draw(tmp_path)
    first = out.read_bytes()
    _, out2 = do_draw(tmp_path, name='again.json')
    assert out2.read_bytes() == first
    assert (tmp_path / 'items.json.sha256').read_text() == f'{bs.sha256(out)}  items.json\n'
    other, _ = do_draw(tmp_path, design=dict(DESIGN, seed='1'), name='seed1.json')
    assert [it['pair_id'] for it in other['items']] != [it['pair_id'] for it in doc['items']]
    assert doc['throwaway'] is False and doc['seed_used'] == SEED and 'freeze' not in doc


def test_draw_is_stratified(tmp_path):
    doc, _ = do_draw(tmp_path)
    cells = by_cell(doc)
    assert sorted(cells) == sorted((s, g) for s in bs.SETS for g in bs.CELLS)
    for (s, g), its in cells.items():
        assert [it['rank'] for it in its] == [0, 1, 2, 3]
        assert all(it['pair_id'].startswith(f'{s}__{g}_') for it in its)
    assert len(doc['items']) == 32 and len({it['pair_id'] for it in doc['items']}) == 32


def test_order_is_the_journal_convention_and_p_comes_first_in_vctk(tmp_path):
    doc, _ = do_draw(tmp_path)
    for g in bs.CELLS:
        assert doc['order']['ltc2k'][g] == order_of('ltc2k', g)
        assert doc['order']['vctk2k'][g] == order_of('vctk2k', g)
        assert [it['pair_id'] for it in by_cell(doc)[('vctk2k', g)]] == order_of('vctk2k', g)[:4]
        assert all(it['has_P'] and it['audio']['nat']['kind'] == 'P' for it in by_cell(doc)[('vctk2k', g)])
    assert [it['pair_id'] for it in by_cell(doc)[('ltc2k', 'F2F')]] == order_of('ltc2k', 'F2F')[:4]


def test_without_the_p_preference_vctk_items_are_the_first_of_the_seeded_order(tmp_path):
    doc, _ = do_draw(tmp_path, design=dict(DESIGN, prefer_parallel={}))
    for g in bs.CELLS:
        assert [it['pair_id'] for it in by_cell(doc)[('vctk2k', g)]] == order_of('vctk2k', g, prefer=False)[:4]


def test_an_ineligible_pair_is_passed_over_and_reported(tmp_path):
    ids = order_of('ltc2k', 'F2M')
    doc, _ = do_draw(tmp_path, fail={('cand_pm18_rq', ids[1]), ('faiss', ids[0]), ('mvc_v1_q32', ids[2]),
                                     ('facodec', ids[2])}, drop={('knnvc', ids[3])})
    assert [it['pair_id'] for it in by_cell(doc)[('ltc2k', 'F2M')]] == [ids[0], ids[2], ids[4], ids[5]]
    assert [(r['pair_id'], r['rank'], r['position'], r['reason']) for r in doc['passed_over']] == [
        (ids[1], 1, 1, 'crtal: error row'), (ids[3], 2, 3, 'knnvc: no row')]   # faiss, q32, facodec: not in design


def test_too_few_eligible_pairs_stop_the_draw(tmp_path):
    fail = {('seedvc', f'ltc2k__M2M_{n:02d}') for n in range(N_PER_CELL - 3)}
    with pytest.raises(SystemExit, match='ltc2k M2M: 3 eligible pairs, 4 needed'):
        do_draw(tmp_path, fail=fail)


@pytest.mark.parametrize('field, change', [('sim_ref_wavs', lambda v: v[:1]), ('source_wav', lambda v: v + '.x')])
def test_a_row_that_disagrees_with_the_pair_list_stops_the_draw(tmp_path, field, change):
    world(tmp_path)
    f = Path(bs.rows_path('t4', 'ltc2k', str(tmp_path / 'mvc'), str(tmp_path / 'journal')))
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    for r in rows:
        if r['system'] == 'knnvc':
            r[field] = change(r[field])
    f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(SystemExit, match=rf"knnvc row of ltc2k__.* disagrees with the pair list on \['{field}"):
        do_draw(tmp_path)


def test_excluded_pairs_are_passed_over_and_unknown_ones_refused(tmp_path):
    ids = order_of('ltc2k', 'M2F')
    doc, _ = do_draw(tmp_path, design=dict(DESIGN, exclude_pairs=[ids[0], ids[2]]))
    assert [it['pair_id'] for it in by_cell(doc)[('ltc2k', 'M2F')]] == [ids[1], ids[3], ids[4], ids[5]]
    assert [(r['pair_id'], r['reason']) for r in doc['passed_over']] == [(ids[0], 'in exclude_pairs'),
                                                                          (ids[2], 'in exclude_pairs')]
    with pytest.raises(SystemExit, match='exclude_pairs names 1 pairs that are not in the pair lists'):
        do_draw(tmp_path, design=dict(DESIGN, exclude_pairs=['ltc2k__nope']), name='x.json')


def test_practice_items_come_after_the_items_and_use_no_target_of_an_item(tmp_path):
    def pair(s, g, n, root):     # targets repeat across the pairs of a cell: t0..t5
        return make_pair(s, g, n, root, tgt_spk=f'{s}{g[2]}t{n % 6}' if g in ('F2F', 'M2M') else None)
    doc, _ = do_draw(tmp_path, design=dict(DESIGN, speaker_rule=False), pair=pair)
    items = {it['pair_id'] for it in doc['items']}
    got = [(p['set'], p['genders'], p['condition']) for p in doc['practice']]
    assert got == [tuple(x) for x in DESIGN['practice']]
    for p in doc['practice']:
        o = doc['order'][p['set']][p['genders']]
        cell = by_cell(doc)[(p['set'], p['genders'])]
        assert p['pair_id'] not in items
        assert o.index(p['pair_id']) > max(o.index(it['pair_id']) for it in cell)
        assert p['target_speaker'] not in {it['target_speaker'] for it in doc['items'] if it['set'] == p['set']}
        skipped = o[max(o.index(it['pair_id']) for it in cell) + 1:o.index(p['pair_id'])]
        tg = {it['target_speaker'] for it in doc['items'] + doc['practice'] if it['set'] == p['set']}
        assert all(doc_pair_target(tmp_path, k) in tg for k in skipped)
        assert p['condition'] in p['audio'] and 'nat' in p['audio'] and 'ref' in p['audio']


def doc_pair_target(tmp_path, k):
    s = k.split('__')[0]
    return next(json.loads(x)['target_speaker'] for x in (tmp_path / 'pairs' / f'{s}.jsonl').read_text().splitlines()
                if json.loads(x)['pair_id'] == k)


def test_draw_reads_no_metric_and_maps_nvquoc_paths_to_ceai(tmp_path):
    doc, out = do_draw(tmp_path, root='/datastore/ceai/crt/synth', journal_root='/datastore/nvquoc/crt/synth')
    txt = out.read_text()
    assert '9.87654' not in txt and 'utmos' not in txt
    assert '/datastore/nvquoc/' not in txt
    it = doc['items'][0]
    assert it['audio']['crtal']['path'] == f'/datastore/ceai/crt/synth/{it["pair_id"]}_cand_pm18_rq.wav'
    assert it['audio']['ref']['path'].startswith('/datastore/ceai/crt/synth/')
    assert it['audio']['src']['path'] == f'/datastore/ceai/crt/synth/{it["pair_id"]}_src.wav'
    assert doc['inputs']['rows']['crtal']['vctk2k']['system'] == 'cand_pm18_rq'
    assert len(doc['inputs']['pairs']['ltc2k']['sha256']) == 64 and len(doc['design_sha256']) == 64


@pytest.mark.parametrize('change, match', [
    ({'mos_conditions': ['mvc_v1', 'crtal']}, 'must include nat'), ({'smos_conditions': ['nat', 'freevc']}, 'unknown'),
    ({'smos_conditions': []}, 'empty'), ({'n_per_stratum': 3}, 'even'),
    ({'practice': [['vctk2k', 'X2Y', 'nat']]}, 'practice'), ({'practice': [['vctk2k', 'F2F', 'src']]}, 'practice'),
    ({'attention': {'mos': [['ltc2k', 'F2F', 6]]}}, 'attention item'), ({'source_range': [10.0, 2.0]}, 'source_range'),
    ({'ref_target_s': 12.0}, 'ref_target_s'), ({'pad_ms': [0, 2000]}, 'pad_ms'), ({'pad_ms': [0.5, 9]}, 'pad_ms'),
    ({'speaker_rule': 1}, 'speaker_rule'), ({'extra_key': 1}, "unknown keys \\['extra_key'\\]"),
    ({'prefer_parallel': {'xx': True}}, 'prefer_parallel')])
def test_design_refusals(tmp_path, change, match):
    with pytest.raises(SystemExit, match=match):
        do_draw(tmp_path, design=dict(DESIGN, **change))


def test_a_design_without_a_required_key_is_refused(tmp_path):
    with pytest.raises(SystemExit, match=r"lacks \['ref_fallback'\]"):
        do_draw(tmp_path, design={k: v for k, v in DESIGN.items() if k != 'ref_fallback'})


# ── eligibility, natural clip, SMOS reference, source clip ───

D = bs.check_design(dict(DESIGN))


@pytest.mark.parametrize('src_dur, ok', [(1.99, False), (2.0, True), (10.0, True), (10.01, False)])
def test_source_duration_boundaries(src_dur, ok):
    p = make_pair('ltc2k', 'F2F', 0, '/a', src_dur=src_dur, e_durs=(3.0, 4.5, 8.0))
    audio, why = bs.item_audio(p, {}, ['nat', 'src'], D)
    assert (audio is not None) == ok and (ok or why == 'source duration outside source_range')


@pytest.mark.parametrize('p_dur, ok', [(1.99, False), (2.0, True), (10.0, True), (10.01, False), (None, False)])
def test_p_duration_boundaries_and_p_is_never_cut(p_dur, ok):
    p = make_pair('vctk2k', 'F2F', 1, '/a', p_dur=p_dur)
    nat, ref, why = bs.natural_and_ref(p, D)
    if ok:
        assert nat == {'path': '/a/vctk2k__F2F_01_P.flac', 'kind': 'P', 'dur': p_dur, 'cut_s': None}
        assert ref['path'] == '/a/vctk2k__F2F_01_E1.wav'                  # 4.5 s: within 3-10 s, closest to 5 s
    else:
        assert nat is None and why == 'P duration outside natural_range'


def test_natural_e_is_within_range_closest_to_the_source_and_never_cut():
    p = make_pair('ltc2k', 'M2F', 0, '/a', e_durs=(1.99, 10.01, 9.0, 5.0, 2.0), src_dur=2.1)
    nat, ref, _ = bs.natural_and_ref(p, D)
    assert nat['path'].endswith('_E4.wav') and nat['cut_s'] is None       # 2.0 s; 1.99 s is outside the range
    assert ref['path'].endswith('_E3.wav')                                  # 5.0 s
    p = make_pair('ltc2k', 'M2F', 0, '/a', e_durs=(1.99, 10.01), src_dur=4.0)
    assert bs.natural_and_ref(p, D)[2] == 'no E utterance within natural_range for the natural clip'
    p = make_pair('ltc2k', 'M2F', 0, '/a', e_durs=(6.0, 6.0, 2.5), src_dur=6.0)
    nat, ref, _ = bs.natural_and_ref(p, D)
    assert nat['path'].endswith('_E0.wav') and ref['path'].endswith('_E1.wav')   # ties: path


def test_reference_range_fallback_and_its_absence():
    p = make_pair('ltc2k', 'F2M', 0, '/a', e_durs=(4.0, 2.99, 12.0, 10.5), src_dur=4.0)
    nat, ref, _ = bs.natural_and_ref(p, D)
    assert nat['path'].endswith('_E0.wav')
    assert ref == {'path': '/a/ltc2k__F2M_00_E3.wav', 'kind': 'E', 'dur': 10.5, 'cut_s': 10.0}   # shortest > 10 s
    p = make_pair('ltc2k', 'F2M', 0, '/a', e_durs=(4.0, 3.0, 10.0, 12.0), src_dur=4.0)
    assert bs.natural_and_ref(p, D)[1]['path'].endswith('_E1.wav')           # 3.0 s: in range, so no fallback
    p = make_pair('ltc2k', 'F2M', 0, '/a', e_durs=(4.0, 12.0), src_dur=4.0)
    assert bs.natural_and_ref(p, dict(D, ref_fallback=False))[2] == 'no E utterance for the SMOS reference'
    p = make_pair('ltc2k', 'F2M', 0, '/a', e_durs=(4.0, 2.5), src_dur=4.0)
    assert bs.natural_and_ref(p, D)[2] == 'no E utterance for the SMOS reference'


def test_natural_and_reference_are_never_r_nor_the_same():
    p = make_pair('ltc2k', 'M2F', 0, '/a', e_durs=(3.5, 7.0, 9.0, 5.0), src_dur=6.5)
    nat, ref, _ = bs.natural_and_ref(p, D)
    assert nat['path'].endswith('_E1.wav') and ref['path'].endswith('_E3.wav')
    p['R']['180']['paths'].append(p['E']['paths'][1])                       # E1 also in R: never used
    p['R']['30']['paths'].append(p['E']['paths'][3])                        # E3 also in R: never used
    nat, ref, _ = bs.natural_and_ref(p, D)
    r = set(bs.conversion_ref(p))
    assert nat['path'] not in r and ref['path'] not in r and nat['path'] != ref['path']
    assert nat['path'].endswith('_E2.wav') and ref['path'].endswith('_E0.wav')
    p = make_pair('vctk2k', 'F2F', 1, '/a')
    p['R']['3'] = {'paths': [p['P']]}
    assert bs.natural_and_ref(p, D)[2] == 'P is in the conversion reference'


def test_the_source_clip_is_never_the_natural_clip_the_reference_or_r():
    p = make_pair('ltc2k', 'F2F', 0, '/a')
    assert bs.item_audio(p, {}, ['nat', 'src'], D)[0]['src'] == {'path': '/a/ltc2k__F2F_00_src.wav',
                                                                  'kind': 'source', 'dur': 4.0, 'cut_s': None}
    p['R']['30']['paths'].append(p['source_path'])
    assert bs.item_audio(p, {}, ['nat', 'src'], D) == (
        None, 'the source clip is missing, the natural clip, the reference or in R')
    assert bs.item_audio(p, {}, ['nat'], D)[0] is not None and 'src' not in bs.item_audio(p, {}, ['nat'], D)[0]


def test_drawn_items_keep_natural_reference_and_source_apart_from_r(tmp_path):
    doc, _ = do_draw(tmp_path)
    cut = 0
    for it in doc['items'] + doc['practice']:
        r = set(it['conversion_ref'])
        a = it['audio']
        nat = [a[k]['path'] for k in ('nat', 'ref', 'src') if k in a]
        assert r and len(set(nat)) == len(nat) and not set(nat) & r
        assert a['ref']['kind'] == 'E' and a['nat']['cut_s'] is None
        cut += bool(a['ref']['cut_s'])
    assert cut                                                            # the fallback reference is used


# ── speaker and sentence rules ───────────────────────────────

def shared_pair(s, g, n, root):
    """Source speakers of a gender shared by F2F/F2M (and M2F/M2M), target speakers of a gender shared by F2F/M2F
    (and F2M/M2M), and sentences repeated across the cells."""
    m = (n + 8 * bs.CELLS.index(g)) % 48
    return make_pair(s, g, n, root, src_spk=f'{s}{g[0]}s{n % 10}', tgt_spk=f'{s}{g[2]}t{(3 * n) % 10}',
                     text=f'SENTENCE, number {m}!' if g[0] == 'F' else f'sentence number {m}')


def test_the_speaker_and_sentence_rules_hold_within_a_set_and_the_draw_is_greedy(tmp_path):
    doc, _ = do_draw(tmp_path, pair=shared_pair, n_per_cell=24)
    pairs = {json.loads(x)['pair_id']: json.loads(x) for s in bs.SETS
             for x in (tmp_path / 'pairs' / f'{s}.jsonl').read_text().splitlines()}
    for s in bs.SETS:
        its = [it for it in doc['items'] if it['set'] == s]
        assert len(its) == 16
        for k in ('source_speaker', 'target_speaker'):
            assert len({it[k] for it in its}) == 16
        assert len({bs.sentence(pairs[it['pair_id']]) for it in its}) == 16
        used = {'source': set(), 'target': set(), 'sentence': set()}
        for g in bs.CELLS:                                               # the greedy rule, written out again
            want = []
            for k in doc['order'][s][g]:
                p = pairs[k]
                if (p['source_speaker'] in used['source'] or p['target_speaker'] in used['target']
                        or bs.sentence(p) in used['sentence']):
                    continue
                want.append(k)
                used['source'].add(p['source_speaker'])
                used['target'].add(p['target_speaker'])
                used['sentence'].add(bs.sentence(p))
                if len(want) == 4:
                    break
            assert [it['pair_id'] for it in by_cell(doc)[(s, g)]] == want
    reasons = {r['reason'].split(':')[0] for r in doc['passed_over']}
    assert reasons == {'speaker rule', 'sentence rule'}
    assert bs.sentence({'source_text': 'SENTENCE, number 3!', 'pair_id': 'x'}) == 'sentence number 3'


def test_without_the_rules_speakers_and_sentences_may_repeat_and_are_noted(tmp_path):
    doc, _ = do_draw(tmp_path, design=dict(DESIGN, speaker_rule=False, sentence_rule=False), pair=shared_pair,
                     n_per_cell=24)
    for s in bs.SETS:
        assert [it['pair_id'] for it in by_cell(doc)[(s, 'F2M')]] == order_of(s, 'F2M', n_per_cell=24)[:4]
    its = [it for it in doc['items'] if it['set'] == 'ltc2k']
    assert len({it['source_speaker'] for it in its}) < 16
    assert any('repeat the sentence' in n for n in doc['notes'])


# ── throwaway seed and the draw guard on the real lists ──────

def test_a_throwaway_draw_is_marked_and_cannot_be_built(tmp_path):
    doc, items = do_draw(tmp_path, extra=['--throwaway_seed', 'feasibility-1'])
    assert doc['throwaway'] is True and doc['seed_used'] == 'feasibility-1'
    assert doc['order']['ltc2k']['F2F'] == order_of('ltc2k', 'F2F', seed='feasibility-1')
    with pytest.raises(SystemExit, match='must differ from the seed'):
        do_draw(tmp_path, extra=['--throwaway_seed', SEED], name='x.json')
    with pytest.raises(SystemExit, match='drawn with --throwaway_seed'):
        dry_build(tmp_path, items)


def test_the_real_pair_lists_need_the_freeze_to_draw(tmp_path, monkeypatch):
    world(tmp_path)
    monkeypatch.setattr(bs, 'REAL_PAIR_LISTS', {bs.sha256(tmp_path / 'pairs' / 'ltc2k.jsonl')})
    with pytest.raises(SystemExit, match='needs --addendum and --addendum_sha256'):
        do_draw(tmp_path)
    doc, _ = do_draw(tmp_path, extra=['--throwaway_seed', 'feasibility-2'], name='t.json')   # allowed: not the seed
    assert doc['throwaway'] is True


# ── audio ────────────────────────────────────────────────────

def tone(sec, amp=0.1, sr=bs.SR, f0=220.0):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def test_normalise_reaches_the_target_active_level():
    x = np.concatenate([tone(1.0, 0.05), np.zeros(bs.SR, np.float32), tone(1.0, 0.05)])
    y, g = bs.normalise(x)
    assert bs.active_db(y) == pytest.approx(bs.TARGET_DBFS, abs=0.01)
    assert g == pytest.approx(bs.TARGET_DBFS - bs.active_db(x), abs=1e-6)
    assert bs.TARGET_DBFS == -26.0 and bs.PEAK_DBFS == -1.0 and bs.ACTIVE_DB == 40.0   # the journal's values
    assert bs.active_db(bs.pad(y, 250)) == pytest.approx(bs.TARGET_DBFS, abs=0.01)    # silence is not active


def test_normalise_keeps_the_peak_at_minus_1_dbfs():
    x = tone(1.0, 0.01)
    x[100] = 0.9
    y, _ = bs.normalise(x)
    assert 20 * np.log10(np.max(np.abs(y))) == pytest.approx(bs.PEAK_DBFS, abs=1e-4)
    assert bs.active_db(y) < bs.TARGET_DBFS


def test_cut_with_a_fade_and_pad_with_silence():
    y = bs.cut(np.ones(int(12.5 * bs.SR), np.float32), 10.0)
    assert len(y) == 10 * bs.SR and y[-1] == 0.0 and y[0] == 1.0
    assert len(bs.cut(np.ones(bs.SR, np.float32), 10.0)) == bs.SR
    z = bs.pad(np.ones(bs.SR, np.float32), 125)
    assert len(z) == bs.SR + 3000 and not z[bs.SR:].any()
    assert bs.browser_dur(3.04) == 3.0 and bs.browser_dur(3.06) == 3.1 and bs.browser_dur(0.02) == 0.1


def test_load_gives_24_khz_mono_and_the_native_format(tmp_path):
    f = tmp_path / 'x.flac'
    sf.write(f, np.stack([tone(2.0, 0.1, 48000), tone(2.0, 0.1, 48000)], axis=1), 48000)
    y, sr0, ch = bs.load(str(f))
    assert y.ndim == 1 and y.dtype == np.float32 and len(y) == 2 * bs.SR and (sr0, ch) == (48000, 2)


# ── build ────────────────────────────────────────────────────

def write_audio(doc, prefix='', skip=()):
    """A tone for every audio path of the items (at prefix + path): the duration of the pair list where it gives one
    (3 s otherwise), different levels, sample rates and channel counts."""
    rng = np.random.default_rng(0)
    durs = {a['path']: a.get('dur') or 3.0 for it in doc['items'] + doc['practice'] for a in it['audio'].values()}
    for i, p in enumerate(bs.audio_paths(doc)):
        if p in skip:
            continue
        sr = (16000, 22050, 24000, 48000)[i % 4]
        x = tone(durs[p], float(rng.uniform(0.003, 0.3)), sr, 150 + i % 50)
        if sr == 48000:
            x = np.stack([x, x], axis=1)
        os.makedirs(os.path.dirname(prefix + p), exist_ok=True)
        sf.write(prefix + p, x, sr)


def dry_build(tmp_path, items, out='pkg', key='private'):
    bs.main(['build', '--items', str(items), '--dry_run', '--out', str(tmp_path / out), '--key_dir',
             str(tmp_path / key)])
    return (json.loads((tmp_path / out / 'lt_data.json').read_text()),
            json.loads((tmp_path / key / 'key.json').read_text()))


def test_dry_build_writes_the_package_the_key_and_the_manifest(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    data, key = dry_build(tmp_path, items)
    assert data['dry_run'] is True and key['dry_run'] is True and key['freeze'] is None
    assert set(data) == {'version', 'dry_run', 'config', 'stimuli', 'practice', 'mos', 'smos', 'cmos'}
    assert data['version'] == 1 and data['cmos'] == [] and data['config']['halves'] is True
    assert len(data['mos']) == 32 * N_MOS + 2 and len(data['smos']) == 32 * N_SMOS + 2 and len(data['practice']) == 3
    files = sorted(os.listdir(tmp_path / 'pkg' / 'stimuli'))
    assert files == sorted(f'{c}.flac' for c in data['stimuli']) and set(data['stimuli']) == set(key['clips'])
    assert sorted(os.listdir(tmp_path / 'pkg')) == ['lt_data.json', 'stimuli']
    assert not [f for f in os.listdir(tmp_path) if f.startswith('.mos-build-')]
    assert os.stat(tmp_path / 'private' / 'key.json').st_mode & 0o777 == 0o600
    assert key['items_sha256'] == bs.sha256(items) and key['design_sha256'] == doc['design_sha256']
    assert key['lt_data_sha256'] == bs.sha256(tmp_path / 'pkg' / 'lt_data.json')
    for block, fields in (('practice', ['stim']), ('mos', ['stim']), ('smos', ['stim', 'ref'])):
        for t in data[block]:
            assert all(t[f] in data['stimuli'] for f in fields)               # the page's checkData
    tids = [t['tid'] for b in ('practice', 'mos', 'smos') for t in data[b]]
    assert len(tids) == len(set(tids)) and set(tids) == set(key['trials'])
    for t in key['trials'].values():
        assert {'block', 'kind', 'pair_id', 'set', 'condition'} <= set(t)
    for c in key['clips'].values():
        assert {'set', 'pair_id', 'condition', 'role', 'path', 'sha256', 'gain_db', 'dur', 'pad_ms', 'native_sr',
                'native_channels', 'source_sha256'} <= set(c)
    man = json.loads((tmp_path / 'private' / 'build_manifest.json').read_text())
    assert man['dry_run'] is True and man['key_sha256'] == bs.sha256(tmp_path / 'private' / 'key.json')
    assert man['clip_sha256'] == sorted(c['sha256'] for c in key['clips'].values()) and man['n_clips'] == len(files)
    assert man['items_sha256'] == bs.sha256(items) and man['lt_data_sha256'] == key['lt_data_sha256']
    txt = (tmp_path / 'private' / 'build_manifest.json').read_text()
    assert not any(c in txt for c in key['clips']) and not any(w in txt for w in ('crtal', 'seedvc', 'knnvc', 'pair'))


def test_the_key_records_native_rates_and_source_hashes(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    _, key = dry_build(tmp_path, items)
    rates = {c['native_sr'] for c in key['clips'].values()}
    assert rates == {16000, 22050, 24000, 48000}
    for c in key['clips'].values():
        assert c['source_sha256'] == bs.sha256(c['path'])
        assert c['native_channels'] == (2 if c['native_sr'] == 48000 else 1)


def test_gold_items_attention_halves_and_references(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    data, key = dry_build(tmp_path, items)
    trials, clips = key['trials'], key['clips']
    main = [t for t in data['mos'] if 'attention' not in t]
    gold = [t for t in main if trials[t['tid']]['kind'] == 'gold']
    assert len(gold) == 32 and all(clips[t['stim']]['condition'] == 'nat' for t in gold)
    smain = [t for t in data['smos'] if 'attention' not in t]
    gsrc = [t for t in smain if trials[t['tid']]['kind'] == 'gold_src']
    assert len(gsrc) == 16 and all(clips[t['stim']]['condition'] == 'src' and trials[t['tid']]['genders'] in bs.CROSS
                                   for t in gsrc)
    same_src = [t for t in smain if trials[t['tid']]['condition'] == 'src' and trials[t['tid']]['kind'] == 'main']
    assert len(same_src) == 16 and all(trials[t['tid']]['genders'] in ('F2F', 'M2M') for t in same_src)
    assert not [t for t in main if trials[t['tid']]['condition'] == 'src']                 # src: SMOS only
    for s in bs.SETS:
        for g in bs.CELLS:
            h = sorted(t['half'] for t in main if trials[t['tid']]['set'] == s and trials[t['tid']]['genders'] == g
                       and trials[t['tid']]['condition'] == 'mvc_v1')
            assert h == ['A', 'A', 'B', 'B']
    att = {b: [t for t in data[b] if 'attention' in t] for b in ('mos', 'smos')}
    assert sorted(t['attention'] for t in att['mos']) == [2, 4] and sorted(t['attention'] for t in att['smos']) == [2, 4]
    assert all('half' not in t and trials[t['tid']]['expected'] == t['attention'] for b in att.values() for t in b)
    for t in data['smos']:
        assert clips[t['ref']]['role'] == 'ref' and clips[t['ref']]['pair_id'] == clips[t['stim']]['pair_id']
        assert clips[t['ref']]['path'] != clips[t['stim']]['path']
    for t in main + smain:
        k = trials[t['tid']]
        assert clips[t['stim']]['condition'] == k['condition'] and clips[t['stim']]['pair_id'] == k['pair_id']
    gold_sha = {clips[t['stim']]['sha256'] for t in gold}
    assert not {clips[t['stim']]['sha256'] for b in att.values() for t in b} & gold_sha   # padded apart


def test_without_halves_no_trial_has_a_half(tmp_path):
    doc, items = do_draw(tmp_path, design=dict(DESIGN, halves=False, n_per_stratum=3))
    write_audio(doc)
    data, _ = dry_build(tmp_path, items)
    assert data['config']['halves'] is False and len(data['mos']) == 24 * N_MOS + 2
    assert not [t for b in ('practice', 'mos', 'smos') for t in data[b] if 'half' in t]


def test_ids_are_random_and_lt_data_says_nothing_of_a_condition(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    data, key = dry_build(tmp_path, items)
    data2, _ = dry_build(tmp_path, items, out='pkg2', key='private2')
    ids = list(data['stimuli']) + [t['tid'] for b in ('practice', 'mos', 'smos') for t in data[b]]
    assert all(re.fullmatch('[0-9a-f]{12}', i) for i in ids)
    assert not set(data2['stimuli']) & set(data['stimuli'])                  # not derived from the (public) seed
    txt = (tmp_path / 'pkg' / 'lt_data.json').read_text()
    for word in ['mvc', 'crtal', 'cand_pm18', 'seedvc', 'knnvc', 'facodec', 'vctk', 'ltc', 'F2M', 'audio', 'wav',
                 'condition', 'pair', 'src_', 'gold', 'instruction', 'pad']:
        assert word not in txt
    assert all(set(v) == {'src', 'dur'} for v in data['stimuli'].values())
    allowed = {'tid', 'stim', 'ref', 'half', 'attention'}
    assert all(set(t) <= allowed for b in ('practice', 'mos', 'smos') for t in data[b])
    seq = [key['clips'][t['stim']]['condition'] for t in data['mos'] if 'attention' not in t]
    assert seq != [c for it in doc['items'] for c in DESIGN['mos_conditions']]   # not written in item order


def test_durations_are_padded_and_rounded_so_that_they_do_not_tell_conditions_apart(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    data, key = dry_build(tmp_path, items)
    pads = [c['pad_ms'] for c in key['clips'].values()]
    assert all(0 <= p <= 250 for p in pads) and len(set(pads)) > 20
    for cid, c in list(key['clips'].items())[:60]:
        y, _ = sf.read(str(tmp_path / 'pkg' / 'stimuli' / f'{cid}.flac'), dtype='float32')
        assert len(y) / bs.SR == pytest.approx(c['dur'], abs=1e-3)
        assert abs(data['stimuli'][cid]['dur'] - c['dur']) <= 0.05 + 1e-9
        assert data['stimuli'][cid]['dur'] == round(data['stimuli'][cid]['dur'], 1)
        tail = int(round(c['pad_ms'] * bs.SR / 1000))
        assert tail == 0 or not y[len(y) - tail:].any()
    outputs = [c for c in key['clips'].values() if c['condition'] in ('mvc_v1', 'crtal', 'seedvc', 'knnvc')]
    assert len({c['dur'] for c in outputs}) > 1                               # same 3 s content, other lengths


def test_written_stimuli_are_24_khz_mono_at_the_target_level_and_cut(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    data, key = dry_build(tmp_path, items)
    cut = [c for c, v in key['clips'].items() if v['cut_s']]
    assert cut and all(key['clips'][c]['role'] == 'ref' for c in cut)
    assert all(key['clips'][c]['dur'] == pytest.approx(10.0 + key['clips'][c]['pad_ms'] / 1000, abs=1e-3)
               for c in cut)
    for c in list(data['stimuli'])[:40] + cut:
        y, sr = sf.read(tmp_path / 'pkg' / 'stimuli' / f'{c}.flac', dtype='float32')
        assert sr == bs.SR and y.ndim == 1
        assert bs.active_db(y) == pytest.approx(bs.TARGET_DBFS, abs=0.05)
        assert bs.sha256(tmp_path / 'pkg' / 'stimuli' / f'{c}.flac') == key['clips'][c]['sha256']


def test_the_key_and_the_cache_stay_out_of_the_package_and_the_repo(tmp_path):
    doc, items = do_draw(tmp_path)
    args = ['build', '--items', str(items), '--dry_run']
    with pytest.raises(SystemExit, match='must not contain one another'):
        bs.main(args + ['--out', str(tmp_path / 'pkg'), '--key_dir', str(tmp_path / 'pkg' / 'private')])
    with pytest.raises(SystemExit, match='must not contain one another'):
        bs.main(args + ['--out', str(tmp_path / 'k' / 'pkg'), '--key_dir', str(tmp_path / 'k')])
    in_repo = os.path.join(bs.ROOT, 'mvc', 'mos', '_test_package')
    with pytest.raises(SystemExit, match='inside the repository'):
        bs.main(args + ['--out', in_repo, '--key_dir', str(tmp_path / 'k')])
    with pytest.raises(SystemExit, match='inside the repository'):
        bs.main(args + ['--out', str(tmp_path / 'p'), '--key_dir', in_repo])
    assert not os.path.exists(in_repo) and not (tmp_path / 'pkg').exists()
    with pytest.raises(SystemExit, match='--cache .* inside the repository'):
        bs.check_places(str(tmp_path / 'p'), str(tmp_path / 'k'), in_repo, str(tmp_path / 'm.json'))
    with pytest.raises(SystemExit, match='--cache .* must not contain one another'):
        bs.check_places(str(tmp_path / 'p'), str(tmp_path / 'k'), str(tmp_path / 'p' / 'c'), str(tmp_path / 'm.json'))
    bs.check_places(str(tmp_path / 'p'), str(tmp_path / 'k'), str(tmp_path / 'k' / 'cache'), str(tmp_path / 'm.json'))


def test_a_second_build_into_the_same_place_is_refused(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    dry_build(tmp_path, items)
    with pytest.raises(SystemExit, match='already holds a build'):
        dry_build(tmp_path, items)
    with pytest.raises(SystemExit, match='key.json already exists'):
        dry_build(tmp_path, items, out='pkg3')


def test_a_failed_build_leaves_nothing_behind(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc, skip={doc['items'][-1]['audio']['knnvc']['path']})
    with pytest.raises(Exception):
        dry_build(tmp_path, items)
    assert not (tmp_path / 'pkg').exists() and not [f for f in os.listdir(tmp_path) if f.startswith('.mos-build-')]
    assert os.listdir(tmp_path / 'private') == []
    write_audio(doc)
    data, _ = dry_build(tmp_path, items)                                   # a re-run is not blocked
    assert len(data['stimuli']) == len(os.listdir(tmp_path / 'pkg' / 'stimuli'))


def test_build_draws_again_and_refuses_items_that_differ(tmp_path):
    doc, items = do_draw(tmp_path)
    write_audio(doc)
    edited = dict(doc, items=[dict(doc['items'][0], pair_id=doc['items'][1]['pair_id'])] + doc['items'][1:])
    bad = tmp_path / 'edited.json'
    bad.write_text(json.dumps(edited))
    with pytest.raises(SystemExit, match=r"does not give .*edited.json \(differs in \['items'\]\)"):
        dry_build(tmp_path, bad)
    design = Path(doc['design_file'])
    design.write_text(json.dumps(dict(DESIGN, seed='other')))
    with pytest.raises(SystemExit, match='missing or changed since the draw'):
        dry_build(tmp_path, items)
    assert not (tmp_path / 'pkg').exists()


def test_dry_run_refuses_real_inputs(tmp_path, monkeypatch):
    doc, items = do_draw(tmp_path, root='/datastore/ceai/crt/synth')
    with pytest.raises(SystemExit, match='synthetic inputs only'):
        dry_build(tmp_path, items)
    doc, items = do_draw(tmp_path / 'b')
    monkeypatch.setattr(bs, 'REAL_PAIR_LISTS', {doc['inputs']['pairs']['vctk2k']['sha256']})
    with pytest.raises(SystemExit, match=r"synthetic inputs only \(real pair lists \['vctk2k'\]"):
        dry_build(tmp_path / 'b', items)


# ── the freeze guard ─────────────────────────────────────────

COMMIT = 'c0ffee' * 6 + 'abcd'


def fake_git(top, commit=COMMIT, blobs=None, remote='same', ls_code=0):
    """A stand-in for bs.git: the repository top, the tag's commit (None: no tag), {(commit, path): bytes}, and the
    commit of the tag on origin ('same': the tag's commit; None: not on origin)."""
    def g(*args):
        if args == ('rev-parse', '--show-toplevel'):
            return 0, (str(top) + '\n').encode()
        if args[:2] == ('rev-parse', '--verify'):
            assert args[3] == f'refs/tags/{bs.TAG}^{{commit}}'
            return (0, (commit + '\n').encode()) if commit else (1, b'')
        if args[:2] == ('cat-file', 'blob'):
            c, rel = args[2].split(':', 1)
            b = (blobs or {}).get((c, rel))
            return (0, b) if b is not None else (128, b'')
        if args == ('ls-remote', '--tags', 'origin'):
            if ls_code:
                return ls_code, b''
            r = commit if remote == 'same' else remote
            lines = f'{"1" * 40}\trefs/tags/other\n{"2" * 40}\trefs/tags/other^{{}}\n'
            if r:
                lines += f'{"9" * 40}\trefs/tags/{bs.TAG}\n{r}\trefs/tags/{bs.TAG}^{{}}\n'
            return 0, lines.encode()
        raise AssertionError(args)
    return g


def guard_setup(tmp_path, monkeypatch, frozen=('addendum', 'builder', 'design'), changed=(), tagged=True,
                remote='same', ls_code=0, top=None):
    """A repository top in tmp_path/repo (bs.ROOT) holding the addendum, a copy of the builder (bs.SELF) and the
    design file; items drawn on synthetic lists whose paths name ceai; ceai as the local folder tmp_path/ceai."""
    repo = tmp_path / 'repo'
    monkeypatch.setattr(bs, 'ROOT', str(repo))
    files = {'addendum': repo / 'mvc' / 'LISTENING_ADDENDUM_mvc_v1.md', 'builder': repo / 'mvc' / 'mos' /
             'build_stimuli.py', 'design': repo / 'mvc' / 'mos' / 'design_mvc_v1.json'}
    files['builder'].parent.mkdir(parents=True)
    files['addendum'].write_text('# Addendum\n\nFrozen.\n')
    shutil.copy(bs.SELF, files['builder'])
    monkeypatch.setattr(bs, 'SELF', str(files['builder']))
    doc, items = do_draw(tmp_path, design_path=files['design'], root='/datastore/ceai/crt/synth',
                         journal_root='/datastore/nvquoc/crt/synth')
    blobs = {(COMMIT, f.relative_to(repo).as_posix()): f.read_bytes() for n, f in files.items() if n in frozen}
    for n in changed:
        with open(files[n], 'a') as f:
            f.write('\n' if n != 'design' else ' ')
    monkeypatch.setattr(bs, 'git', fake_git(top or repo, COMMIT if tagged else None, blobs, remote, ls_code))
    fetched = []

    def fake_fetch(paths, cache, remote):                       # ceai as a local folder
        fetched.append((tuple(paths), remote))
        for p in paths:
            os.makedirs(os.path.dirname(cache + p), exist_ok=True)
            shutil.copy(str(tmp_path / 'ceai') + p, cache + p)
    monkeypatch.setattr(bs, 'fetch', fake_fetch)
    write_audio(doc, prefix=str(tmp_path / 'ceai'))
    add = files['addendum']
    args = ['build', '--items', str(items), '--addendum', str(add), '--addendum_sha256', bs.sha256(add),
            '--out', str(tmp_path / 'pkg'), '--key_dir', str(tmp_path / 'private'), '--cache', str(tmp_path / 'cache')]
    return args, files, fetched, doc


def test_build_with_the_frozen_and_pushed_addendum(tmp_path, monkeypatch):
    args, files, fetched, doc = guard_setup(tmp_path, monkeypatch)
    bs.main(args)
    key = json.loads((tmp_path / 'private' / 'key.json').read_text())
    want = {'tag': bs.TAG, 'commit': COMMIT,
            'addendum': {'path': 'mvc/LISTENING_ADDENDUM_mvc_v1.md', 'sha256': bs.sha256(files['addendum'])},
            'builder': {'path': 'mvc/mos/build_stimuli.py', 'sha256': bs.sha256(files['builder'])},
            'design': {'path': 'mvc/mos/design_mvc_v1.json', 'sha256': bs.sha256(files['design'])}}
    assert key['dry_run'] is False and key['freeze'] == want
    assert doc['design_file'] == 'mvc/mos/design_mvc_v1.json'
    assert len(fetched) == 1 and fetched[0][1] == bs.REMOTE
    assert all(p.startswith('/datastore/ceai/crt/synth/') for p in fetched[0][0])
    assert len(os.listdir(tmp_path / 'pkg' / 'stimuli')) == len(key['clips'])
    man = json.loads((tmp_path / 'repo' / 'mvc' / 'LISTENING_BUILD_mvc_v1.json').read_text())
    assert man['freeze'] == want and man['dry_run'] is False and man['items_sha256'] == key['items_sha256']


def test_the_draw_on_the_real_lists_with_the_freeze_records_it(tmp_path, monkeypatch):
    args, files, _, _ = guard_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(bs, 'REAL_PAIR_LISTS', {bs.sha256(tmp_path / 'pairs' / 'vctk2k.jsonl')})
    add = ['--addendum', str(files['addendum']), '--addendum_sha256', bs.sha256(files['addendum'])]
    doc, _ = do_draw(tmp_path, design=json.loads(files['design'].read_text()), design_path=files['design'],
                     name='frozen.json', extra=add)
    assert doc['freeze']['commit'] == COMMIT and doc['freeze']['design']['path'] == 'mvc/mos/design_mvc_v1.json'


@pytest.mark.parametrize('setup, extra, match', [
    ({}, ['--addendum', None], 'needs --addendum and --addendum_sha256'),
    ({}, ['--addendum_sha256', '0' * 64], 'has sha256 .* not 0000'),
    ({'top': '/elsewhere'}, [], 'is not the top of a git repository'),
    ({'tagged': False}, [], 'tag listening-addendum-mvc-v1 does not exist'),
    ({'frozen': ('builder', 'design')}, [], 'the addendum mvc/LISTENING_ADDENDUM_mvc_v1.md is not in the commit'),
    ({'changed': ('addendum',)}, ['--addendum_sha256', 'EDITED'], 'the addendum .* differs from its copy at the tag'),
    ({'changed': ('builder',)}, [], 'the builder mvc/mos/build_stimuli.py differs from its copy at the tag'),
    ({'frozen': ('addendum', 'builder')}, [], 'the design mvc/mos/design_mvc_v1.json is not in the commit'),
    ({'remote': None}, [], 'on origin is missing'),
    ({'remote': 'f' * 40}, [], 'on origin is ffff'),
    ({'ls_code': 128}, [], 'ls-remote --tags origin failed')])
def test_build_refusals_of_the_guard(tmp_path, monkeypatch, setup, extra, match):
    args, files, fetched, _ = guard_setup(tmp_path, monkeypatch, **setup)
    for flag, value in zip(extra[::2], extra[1::2]):
        i = args.index(flag)
        if value is None:
            del args[i:i + 2]
        else:
            args[i + 1] = bs.sha256(files['addendum']) if value == 'EDITED' else value
    with pytest.raises(SystemExit, match=match):
        bs.main(args)
    assert not fetched and not (tmp_path / 'pkg').exists() and not (tmp_path / 'private').exists()
    assert not (tmp_path / 'repo' / 'mvc' / 'LISTENING_BUILD_mvc_v1.json').exists()


def test_a_design_changed_after_the_tag_is_refused(tmp_path, monkeypatch):
    args, files, fetched, _ = guard_setup(tmp_path, monkeypatch, changed=('design',))
    with pytest.raises(SystemExit, match='the design mvc/mos/design_mvc_v1.json differs from its copy at the tag'):
        bs.main(args)
    assert not fetched


def test_an_addendum_outside_the_repository_is_refused(tmp_path, monkeypatch):
    args, files, _, _ = guard_setup(tmp_path, monkeypatch)
    outside = tmp_path / 'ADDENDUM.md'
    shutil.copy(files['addendum'], outside)
    args[args.index('--addendum') + 1] = str(outside)
    with pytest.raises(SystemExit, match='the addendum .* is not inside the repository'):
        bs.main(args)


def test_remote_tag_commit_reads_lightweight_and_annotated_tags():
    t = bs.TAG
    assert bs.remote_tag_commit(f'aaa\trefs/tags/{t}\n') == 'aaa'
    assert bs.remote_tag_commit(f'aaa\trefs/tags/{t}\nbbb\trefs/tags/{t}^{{}}\n') == 'bbb'
    assert bs.remote_tag_commit(f'aaa\trefs/tags/{t}-old\nbbb\trefs/tags/x/{t}\n') is None


def test_fetch_runs_rsync_with_a_files_from_list(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, stdout=None):
        lst = next(a for a in cmd if a.startswith('--files-from=')).split('=', 1)[1]
        calls.append((cmd, open(lst).read()))
        for rel in open(lst).read().split():
            os.makedirs(os.path.dirname(cmd[-1] + rel), exist_ok=True)
            open(cmd[-1] + rel, 'w').close()
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(bs.subprocess, 'run', fake_run)
    bs.fetch(['/datastore/ceai/crt/a/x.wav', '/datastore/ceai/crt/b/y.flac'], str(tmp_path / 'cache'), 'u@host')
    cmd, lst = calls[0]
    assert cmd[:2] == ['rsync', '-a'] and cmd[-2:] == ['u@host:/', str(tmp_path / 'cache') + '/']
    assert lst == 'datastore/ceai/crt/a/x.wav\ndatastore/ceai/crt/b/y.flac\n'
    monkeypatch.setattr(bs.subprocess, 'run', lambda cmd, stdout=None: subprocess.CompletedProcess(cmd, 23))
    with pytest.raises(SystemExit, match='rsync exit 23'):
        bs.fetch(['/datastore/ceai/crt/z.wav'], str(tmp_path / 'cache'), 'u@host')


@pytest.mark.skipif(os.environ.get('MOS_GIT_TESTS') != '1', reason='set MOS_GIT_TESTS=1 to make git repositories in '
                    'a temporary folder')
def test_the_guard_against_real_git_repositories(tmp_path, monkeypatch):
    repo, origin = tmp_path / 'repo', tmp_path / 'origin.git'
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t',
               GIT_COMMITTER_EMAIL='t@t', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')

    def run(*a, cwd=repo):
        subprocess.run(['git', '-C', str(cwd), *a], check=True, env=env, capture_output=True)
    repo.mkdir()
    run('init', '-q')
    subprocess.run(['git', 'init', '-q', '--bare', str(origin)], check=True, env=env, capture_output=True)
    run('remote', 'add', 'origin', str(origin))
    add, builder, design = repo / 'mvc' / 'ADD.md', repo / 'mvc' / 'mos' / 'build_stimuli.py', repo / 'mvc' / 'd.json'
    builder.parent.mkdir(parents=True)
    add.write_text('addendum\n')
    design.write_text('{}\n')
    shutil.copy(bs.SELF, builder)
    monkeypatch.setattr(bs, 'ROOT', str(repo))
    monkeypatch.setattr(bs, 'SELF', str(builder))
    run('add', 'mvc/ADD.md', 'mvc/mos/build_stimuli.py', 'mvc/d.json')
    run('commit', '-q', '-m', 'freeze')
    sha = bs.sha256(add)
    with pytest.raises(SystemExit, match='does not exist'):
        bs.check_freeze(str(add), sha, str(design))
    run('tag', '-a', bs.TAG, '-m', 'freeze')
    with pytest.raises(SystemExit, match='on origin is missing'):
        bs.check_freeze(str(add), sha, str(design))
    run('push', '-q', 'origin', bs.TAG)
    got = bs.check_freeze(str(add), sha, str(design))
    assert got['addendum']['path'] == 'mvc/ADD.md' and got['design']['path'] == 'mvc/d.json'
    head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, env=env).stdout
    assert got['commit'] == head.decode().strip()
    add.write_text('addendum\nedited\n')
    with pytest.raises(SystemExit, match='differs from its copy at the tag'):
        bs.check_freeze(str(add), bs.sha256(add), str(design))
