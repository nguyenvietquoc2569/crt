"""Tests of benchmark/listening_stimuli.py, the listening-test stimuli of ANALYSIS_PLAN_v1.md §5.4: the natural target
clip and the SMOS reference chosen from P and E, an item replaced by the next pair of its cell when one of its outputs
failed, the practice items, the trial lists (counts, halves, gold, hidden identical pairs, attention items) and their
ids, the loudness normalisation and the cut, and build() with a stand-in for the Mimi copier on synthetic audio."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

import listening_stimuli as ls

CELLS = ('F2F', 'F2M', 'M2F', 'M2M')


def pair(set_, cell, n, with_p=False, e_durs=(3.0, 4.5, 8.0), src_dur=4.0, root='/a'):
    k = f'{set_}__{cell}_{n:02d}'
    return {'pair_id': k, 'set': set_, 'genders': cell, 'source_speaker': f'{set_}{cell}s{n}',
            'target_speaker': f'{set_}{cell}t{n}', 'source_dur': src_dur,
            'P': f'{root}/{k}_P.wav' if with_p else None, 'P_dur': 3.3 if with_p else None,
            'E': {'paths': [f'{root}/{k}_E{i}.wav' for i in range(len(e_durs))], 'durs': list(e_durs)}}


def world(fail=(), n_per_cell=10, root='/a'):
    """Pairs of 2 sets x 4 cells; items are ranks 0-3 of each cell's order; rows for every condition, with an error
    row for the (condition, pair_id) in fail."""
    pairs, items, sets = {}, [], {}
    for s in ('vctk2k', 'ltc2k'):
        sets[s] = {'order': {}}
        for c in CELLS:
            order = []
            for n in range(n_per_cell):
                p = pair(s, c, n, with_p=(s == 'vctk2k' and n % 2 == 1), root=root)
                pairs[p['pair_id']] = p
                order.append(p['pair_id'])
            sets[s]['order'][c] = order
            items += [{'set': s, 'genders': c, 'rank': r, 'pair_id': order[r]} for r in range(4)]
    rows = {c: {} for c in ('faiss', 'crt04', 'crtal', 'mimi_P', 'knnvc', 'seedvc')}
    for c in rows:
        for s in ('vctk2k', 'ltc2k'):
            rows[c][s] = {}
            for k, p in pairs.items():
                if p['set'] != s or (c == 'mimi_P' and not p['P']):
                    continue
                r = {'pair_id': k, 'system': c, 'wav': f'{root}/{k}_{c}.wav'}
                if (c, k) in fail:
                    r = {'pair_id': k, 'system': c, 'error': 'boom'}
                rows[c][s][k] = r
    return {'items': items, 'sets': sets}, pairs, rows


# ── natural target and SMOS reference ────────────────────────

def test_natural_is_p_when_it_exists():
    nat, ref = ls.natural_and_ref(pair('vctk2k', 'F2F', 1, with_p=True))
    assert nat['kind'] == 'P' and nat['path'].endswith('_P.wav') and not nat['cut']
    assert ref['path'].endswith('_E1.wav')                      # 4.5 s: within 3-10 s and closest to 5 s


def test_natural_e_is_the_one_closest_to_the_source_length():
    nat, ref = ls.natural_and_ref(pair('ltc2k', 'F2F', 0, e_durs=(2.5, 7.0, 9.0), src_dur=6.5))
    assert nat['path'].endswith('_E1.wav') and nat['kind'] == 'E'       # 7.0 s is closest to 6.5 s
    assert ref['path'].endswith('_E0.wav') is False and ref['path'].endswith('_E2.wav')  # 9.0 s (2.5 s < 3 s)


def test_long_utterances_are_cut():
    nat, ref = ls.natural_and_ref(pair('ltc2k', 'M2M', 0, e_durs=(26.6, 17.0), src_dur=4.0))
    assert nat['cut'] and ref['cut'] and nat['path'] != ref['path']
    assert nat['path'].endswith('_E1.wav')                      # none within 2-10 s: the one closest to 4 s


def test_no_second_e_utterance_gives_no_reference():
    nat, ref = ls.natural_and_ref(pair('ltc2k', 'F2M', 0, e_durs=(4.0,)))
    assert nat is not None and ref is None
    audio, why = ls.item_audio(pair('ltc2k', 'F2M', 0, e_durs=(4.0,)), world()[2], ls.CONDITIONS)
    assert audio is None and 'SMOS reference' in why


def test_nat_mimi_uses_t1s_copy_of_p_or_is_made():
    items, pairs, rows = world()
    with_p = next(p for p in pairs.values() if p['P'])
    without = next(p for p in pairs.values() if not p['P'])
    a, _ = ls.item_audio(with_p, rows, ls.CONDITIONS)
    b, _ = ls.item_audio(without, rows, ls.CONDITIONS)
    assert a['nat_mimi']['kind'] == 'mimi_copy_P' and a['nat_mimi']['path'].endswith('_mimi_P.wav')
    assert b['nat_mimi'] == {'path': None, 'from': b['nat']['path'], 'cut': b['nat']['cut'], 'kind': 'make'}


# ── plan: replacement and practice ───────────────────────────

def test_plan_keeps_the_drawn_items_when_nothing_failed():
    items, pairs, rows = world()
    p = ls.plan(items, pairs, rows, ls.CONDITIONS + ls.OPTIONAL)
    assert [it['pair_id'] for it in p['items']] == [it['pair_id'] for it in sorted(
        items['items'], key=lambda x: (x['set'], x['genders'], x['rank']))]
    assert p['replaced'] == [] and len(p['items']) == 32


def test_a_failed_output_replaces_the_item_with_the_next_pair_of_its_cell():
    items, pairs, rows = world(fail={('crtal', 'vctk2k__F2M_01'), ('faiss', 'vctk2k__F2M_04')})
    p = ls.plan(items, pairs, rows, ls.CONDITIONS)
    cell = [it for it in p['items'] if it['set'] == 'vctk2k' and it['genders'] == 'F2M']
    assert [it['pair_id'] for it in cell] == ['vctk2k__F2M_00', 'vctk2k__F2M_05', 'vctk2k__F2M_02', 'vctk2k__F2M_03']
    assert cell[1]['drawn'] == 'vctk2k__F2M_01' and cell[1]['rank'] == 1
    assert [(r['pair_id'], r['reason']) for r in p['replaced']] == [
        ('vctk2k__F2M_01', 'crtal failed or missing'), ('vctk2k__F2M_04', 'faiss failed or missing')]


def test_an_optional_condition_that_fails_also_replaces_when_planned():
    items, pairs, rows = world(fail={('seedvc', 'ltc2k__M2M_00')})
    assert ls.plan(items, pairs, rows, ls.CONDITIONS)['replaced'] == []
    assert len(ls.plan(items, pairs, rows, ls.CONDITIONS + ls.OPTIONAL)['replaced']) == 1


def test_practice_items_come_from_the_rest_of_the_order():
    items, pairs, rows = world()
    p = ls.plan(items, pairs, rows, ls.CONDITIONS)
    got = [(x['set'], x['genders'], x['pair_id'], x['condition']) for x in p['practice']]
    assert sorted(got) == sorted([('vctk2k', 'F2F', 'vctk2k__F2F_04', 'nat'),
                                  ('ltc2k', 'M2M', 'ltc2k__M2M_04', 'faiss'),
                                  ('vctk2k', 'F2M', 'vctk2k__F2M_04', 'crtal')])


# ── trials ───────────────────────────────────────────────────

def make_trials(conditions=ls.CONDITIONS):
    items, pairs, rows = world()
    return ls.trials(ls.plan(items, pairs, rows, conditions))


def test_trial_counts_halves_and_gold():
    pages, clips, key = make_trials()
    assert len(pages['mos']) == 32 * 5 + 2 and len(pages['smos']) == 32 * 5 + 1
    assert len(pages['cmos']) == 32 * 2 + 8 + 1 and len(pages['practice']) == 3
    main = [t for t in pages['mos'] if 'attention' not in t]
    assert sum(t['half'] == 'A' for t in main) == sum(t['half'] == 'B' for t in main) == 80
    for s in ('vctk2k', 'ltc2k'):
        for c in CELLS:
            halves = sorted(t['half'] for t in main if key[t['tid']]['set'] == s and key[t['tid']]['genders'] == c
                            and key[t['tid']]['condition'] == 'nat')
            assert halves == ['A', 'A', 'B', 'B']
    gold = [t for t in main if key[t['tid']]['kind'] == 'gold']
    assert len(gold) == 32 and all(key[t['tid']]['condition'] == 'nat' for t in gold)


def test_cmos_blocks_and_hidden_identical_pairs():
    pages, clips, key = make_trials()
    groups = {}
    for t in pages['cmos']:
        if 'attention' not in t:
            groups.setdefault(t['group'], set()).add(key[t['tid']]['kind'])
    assert sorted(len(v) for v in groups.values()) == [1, 1, 1] and len(groups) == 3   # one opaque id per comparison
    assert all(len(g) == 10 and 'P8' not in g for g in groups)
    kinds = [key[t['tid']]['kind'] for t in pages['cmos']]
    assert kinds.count('P8') == 32 and kinds.count('K8') == 32 and kinds.count('hidden') == 8
    for t in pages['cmos']:
        k = key[t['tid']]
        if k['kind'] in ('P8', 'K8'):
            assert clips[t['x']]['condition'] == 'crtal' and clips[t['y']]['condition'] == {'P8': 'faiss', 'K8': 'crt04'}[k['kind']]
        if k['kind'] == 'hidden':
            assert t['x'] != t['y'] and clips[t['x']]['path'] == clips[t['y']]['path']


def test_attention_items():
    pages, clips, key = make_trials()
    att = {b: [t for t in pages[b] if 'attention' in t] for b in ('mos', 'smos', 'cmos')}
    assert [t['attention'] for t in att['mos']] == [3, 3] and [t['attention'] for t in att['smos']] == [2]
    assert [t['attention'] for t in att['cmos']] == [0]
    assert all('please select' in t['instruction'] for b in att.values() for t in b)
    assert all(key[t['tid']]['kind'] == 'attention' and key[t['tid']]['expected'] == t['attention']
               for b in att.values() for t in b)


def test_ids_are_unique_stable_and_say_nothing():
    pages, clips, key = make_trials()
    tids = [t['tid'] for b in pages.values() for t in b]
    assert len(tids) == len(set(tids))
    again, _, _ = make_trials()
    assert [t['tid'] for b in again.values() for t in b] == tids
    for i in list(clips) + tids:
        assert len(i) == 10 and all(ch in '0123456789abcdef' for ch in i)
    smos = [t for t in pages['smos'] if 'attention' not in t]
    refs = {}
    for t in smos:
        refs.setdefault(key[t['tid']]['pair_id'], set()).add(t['ref'])
    assert all(len(v) == 1 for v in refs.values())               # one SMOS reference per item


def test_optional_conditions_enter_mos_and_smos_only_when_asked():
    pages, _, _ = make_trials(ls.CONDITIONS + ('knnvc',))
    assert len(pages['mos']) == 32 * 6 + 2


# ── audio ────────────────────────────────────────────────────

def tone(sec, amp=0.1, sr=ls.SR):
    t = np.arange(int(sec * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_normalise_reaches_the_target_active_level():
    x = np.concatenate([tone(1.0, 0.05), np.zeros(ls.SR, np.float32), tone(1.0, 0.05)])
    y, g = ls.normalise(x)
    assert ls.active_db(y) == pytest.approx(ls.TARGET_DBFS, abs=0.01)
    assert g == pytest.approx(ls.TARGET_DBFS - ls.active_db(x), abs=1e-6)


def test_normalise_keeps_the_peak_below_minus_1_dbfs():
    x = tone(1.0, 0.01)
    x[100] = 0.9                                                # a click far above the speech level
    y, g = ls.normalise(x)
    assert 20 * np.log10(np.max(np.abs(y))) == pytest.approx(ls.PEAK_DBFS, abs=1e-4)
    assert ls.active_db(y) < ls.TARGET_DBFS


def test_cut_to_10_s_with_a_fade():
    x = np.ones(int(12.5 * ls.SR), np.float32)
    y = ls.cut(x)
    assert len(y) == 10 * ls.SR and y[-1] == 0.0 and y[0] == 1.0
    assert ls.cut(x[:ls.SR]) is not None and len(ls.cut(x[:ls.SR])) == ls.SR


# ── build ────────────────────────────────────────────────────

def test_build_writes_the_package(tmp_path):
    root = str(tmp_path / 'audio')
    items, pairs, rows = world(root=root)
    p = ls.plan(items, pairs, rows, ls.CONDITIONS)
    os.makedirs(root)
    needed = set()
    for it in p['items'] + p['practice']:
        for a in it['audio'].values():
            needed.add(a['path'] or a['from'])
    for f in needed:
        sf.write(f, tone(3.0, 0.05), ls.SR)
    calls = []

    def copier(x):
        calls.append(len(x))
        return np.concatenate([x * 0.5, np.zeros(1920, np.float32)])   # Mimi pads to whole frames
    out = tmp_path / 'pkg'
    data, key = ls.build(p, str(out), str(tmp_path / 'key.json'), copier, ls.DEFAULT_CONFIG)
    assert (tmp_path / 'key.json').exists() and not (out / 'key.json').exists()
    files = os.listdir(out / 'stimuli')
    assert len(files) == len(data['stimuli']) == len(key['clips'])
    assert all(f.endswith('.flac') for f in files)
    made = [c for c in key['clips'].values() if c['kind'] == 'make']
    assert made and len(calls) == len({c['from'] for c in made})        # one copy per natural clip
    js = (out / 'lt_data.js').read_text()
    assert js.startswith('window.LT_DATA = ') and js.rstrip().endswith(';')
    loaded = json.loads(js[len('window.LT_DATA = '):].rstrip().rstrip(';'))
    assert loaded['config']['headphone_check'] == {'trials': 6, 'pass': 5, 'attempts': 2}
    assert set(loaded) == {'version', 'config', 'stimuli', 'practice', 'mos', 'smos', 'cmos'}
    for c in key['clips'].values():
        assert len(c['sha256']) == 64 and c['dur'] == pytest.approx(3.0, abs=0.01)
    y, sr = sf.read(out / 'stimuli' / files[0])
    assert sr == ls.SR
    assert 'condition' not in json.dumps(loaded['stimuli'])             # the page never sees what a clip is


def test_the_key_may_not_go_into_the_package(tmp_path):
    items, pairs, rows = world()
    p = ls.plan(items, pairs, rows, ls.CONDITIONS)
    with pytest.raises(SystemExit, match='must not be inside the package'):
        ls.build(p, str(tmp_path / 'pkg'), str(tmp_path / 'pkg' / 'key.json'), None, ls.DEFAULT_CONFIG)


# ── config of a built package ────────────────────────────────

def _package(tmp_path):
    data = {'version': 1, 'config': dict(ls.DEFAULT_CONFIG), 'stimuli': {'s1': {'src': 'stimuli/s1.flac', 'dur': 2.0}},
            'practice': [], 'mos': [{'tid': 'm1', 'stim': 's1'}], 'smos': [], 'cmos': []}
    (tmp_path / 'lt_data.js').write_text(ls.LT_PREFIX + json.dumps(data, indent=1) + ';\n')
    return data


def test_set_config_replaces_only_consent_contact_and_title(tmp_path):
    before = _package(tmp_path)
    consent = '<p><b>Thông tin.</b> Voluntary – you may stop at any time.</p>\n'
    data, sha = ls.set_config(str(tmp_path), consent, '  organiser@uit.example  ')
    js = (tmp_path / 'lt_data.js').read_text()
    assert js.startswith('window.LT_DATA = ') and js.endswith(';\n')
    again = json.loads(js[len(ls.LT_PREFIX):].rstrip()[:-1])
    assert again == data and again['config']['consent_html'] == consent.strip()
    assert again['config']['contact'] == 'organiser@uit.example' and again['config']['title'] == before['config']['title']
    assert {k: v for k, v in again.items() if k != 'config'} == {k: v for k, v in before.items() if k != 'config'}
    assert {k: v for k, v in again['config'].items() if k not in ('consent_html', 'contact')} == \
        {k: v for k, v in before['config'].items() if k not in ('consent_html', 'contact')}
    assert len(sha) == 64
    data2, _ = ls.set_config(str(tmp_path), consent, 'x@y', title='Listening test')
    assert data2['config']['title'] == 'Listening test'


@pytest.mark.parametrize('consent, contact, match', [
    ('', 'x@y', 'empty'), (ls.DEFAULT_CONFIG['consent_html'], 'x@y', 'placeholder'),
    ('<p>ok</p><SCRIPT>alert(1)</script>', 'x@y', 'script'), ('<p>ok</p>', '  ', 'contact')])
def test_set_config_refusals(tmp_path, consent, contact, match):
    _package(tmp_path)
    with pytest.raises(SystemExit, match=match):
        ls.set_config(str(tmp_path), consent, contact)


def test_set_config_refuses_a_file_that_is_not_lt_data(tmp_path):
    (tmp_path / 'lt_data.js').write_text('var x = 1;\n')
    with pytest.raises(SystemExit, match='not an lt_data.js'):
        ls.set_config(str(tmp_path), '<p>ok</p>', 'x@y')


def test_set_config_refuses_the_draft_fields_of_consent_v1(tmp_path):
    _package(tmp_path)
    draft = open(os.path.join(os.path.dirname(ls.__file__), 'listening', 'consent_v1.html'), encoding='utf-8').read()
    with pytest.raises(SystemExit, match='draft fields'):
        ls.set_config(str(tmp_path), draft, 'x@y')
    filled = draft.replace('[CONTACT]', 'x@y')
    for m in ('[COMPENSATION', '[ETHICS', '[THÙ LAO', '[ĐẠO ĐỨC'):
        start = filled.index(m)
        filled = filled[:start] + filled[filled.index(']', start) + 1:]
    data, _ = ls.set_config(str(tmp_path), filled, 'x@y')
    assert 'Tham gia là tự nguyện' in data['config']['consent_html'] and '[' not in data['config']['consent_html']
