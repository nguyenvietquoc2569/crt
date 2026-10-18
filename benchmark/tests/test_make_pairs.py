"""Tests of benchmark/make_pairs.py: the pair lists, the VCTK-2k source-E step, --source_e_only, the 10-target variant,
the Dev-speaker check of a run without dev and --check (ANALYSIS_PLAN_v1.md §1.1, §1.2, §8.1 items 3 and 9).

Everything runs on a synthetic corpus built in tmp_path, laid out as make_pairs reads it. The VCTK part has 20 Dev
speakers (eval_pairs.json) and 42 others as silent 1 kHz *_mic1.flac / *_mic2.flac files, with transcripts and
speaker-info.txt. It includes p315 and s5 (excluded by rule), a speaker with mic2 only, one without txt, one whose txt
and mic1 never match, 12 shared-block sentences with omissions, a non-canonical text under a block number, a block text
under another number, texts read by several speakers, a text read twice, too-short utterances, mic2-only utterances and
mic1 without txt. The LibriTTS-R part (silent 1 kHz WAV) has 12 test-clean speakers, one more with too few 3-10 s
utterances, and a wav without a transcript. Default counts apply, so the sets have 1,000 / 2,000 / 600 pairs and the
10-target variant 4,000.

data/make_pairs_orig.py is make_pairs.py as it was before the additions of 18 Oct 2026 (pinned by ORIG_SHA256). It is
the Oct 9 HPC copy that wrote pairs/{dev,vctk2k,ltc2k}.jsonl, except for the default data root D, which these tests
never use because every path is passed. The byte-identity tests run it and the current script in subprocesses with the
same arguments and compare the lists byte for byte. Do not edit it. Neither version writes pairs/self.jsonl (the truth
set is built outside make_pairs.py), so it has nothing to compare.
"""
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf

import make_pairs

TESTS = os.path.dirname(os.path.abspath(__file__))
NEW = os.path.join(os.path.dirname(TESTS), 'make_pairs.py')
ORIG = os.path.join(TESTS, 'data', 'make_pairs_orig.py')
ORIG_SHA256 = '9e4334d0fba16bf4aaddddb6ebc9cc7d747d346293f419c39406a9f5ea3ffd1f'
SEED = 20260928
SR = 1000
WORDS = 'harbour lantern meadow orchard pebble quarry ribbon saddle thistle velvet walnut willow'.split()

DEV = ['p%03d' % i for i in range(225, 245)]                 # the conference speakers, F / M alternating
OTHER = ['p%03d' % i for i in range(245, 287)]               # 21 F + 21 M
GENDER = {s: 'FM'[i % 2] for spk in (DEV, OTHER) for i, s in enumerate(spk)}
GENDER.update({'p315': 'F', 's5': 'F', 'p362': 'M', 'p376': 'M', 'p380': 'F'})
BAD = {'p362': 'no mic1 audio', 'p376': 'no txt', 'p380': 'no utterance with both txt and mic1'}
BLOCK = {'%03d' % (i + 1): f'Please call Stella and ask her to bring the {w} from shop number {i + 1}.'
         for i, w in enumerate(WORDS)}
OMIT = {('p227', '004'), ('p232', '009'), ('p247', '002'), ('p247', '011'), ('p252', '006'), ('p261', '001'),
        ('p261', '012'), ('p270', '008')}
COMMON = [('Every Dev speaker also read this line about the lantern.', DEV),            # below the 80 % block share
          ('Every other speaker also read this line about the orchard.', OTHER),
          ('Three speakers read this line about the quarry.', ('p226', 'p251', 'p266'))]
LTC = [str(1000 + 37 * i) for i in range(13)]                # LTC[12] has too few 3-10 s utterances
LTC_GENDER = dict({s: 'FM'[i % 2] for i, s in enumerate(LTC[:12])}, **{LTC[12]: 'F'})

MATRIX = {                                                   # run name: make_pairs arguments besides the paths
    'full': [],                                              # the command that wrote pairs/ on the HPC
    't10': ['--sets', 'vctk2k', '--vctk_targets', '10'],     # job D0 step 1: pairs_vctk2k_t10/
    'vctk2k': ['--sets', 'vctk2k'],
    'dev': ['--sets', 'dev'],
    'ltc2k': ['--sets', 'ltc2k'],
    'seed7': ['--seed', '7'],
    'small_e': ['--sets', 'dev,vctk2k', '--n_e', '5', '--e_max_s', '12', '--min_ref_dur', '1.5', '--budgets', '3,30'],
    'missing': ['--sets', 'vctk2k', '--min_ref_dur', '1.5'],  # SHORT cannot fill a source E
}
NEW_KEYS = {'args': ('dev_pairs', 'source_e_only', 'check'), 'vctk2k': ('dev_speakers_checked', 'source_E')}


def vctk2k_split(seed=SEED, k=10):
    """VCTK-2k source and target speakers as make_pairs draws them: per gender the sorted non-Dev speakers, shuffled
    by random.Random('<seed>:vctk2k:speakers:<g>'); the first k are sources, the next k targets."""
    src, tgt = [], []
    for g in 'FM':
        pool = sorted(s for s in OTHER if GENDER[s] == g)
        random.Random(f'{seed}:vctk2k:speakers:{g}').shuffle(pool)
        src, tgt = src + pool[:k], tgt + pool[k:2 * k]
    return src, tgt


V2K_SRC, V2K_TGT = vctk2k_split()
SHORT = sorted(min(s for s in V2K_SRC if GENDER[s] == g) for g in 'FM')   # own recordings 1.05-1.45 s, two of 2-2.5 s


# ── corpus ───────────────────────────────────────────────────

def write_audio(path, secs, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, np.zeros(int(round(secs * SR)), dtype='int16'), SR, format=fmt, subtype='PCM_16')


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text + '\n')


def build_corpus(root):
    """Writes the corpus under root. Its paths are relative to root, the cwd of every run, which keeps the lists
    small. Returns them and utts {VCTK speaker: [{uid, path (mic1), text, dur, txt, mic1, in_block}]}, the ground
    truth the source-E rule is checked against."""
    r = random.Random(20261018)
    at = lambda *p: os.path.join(root, *p)
    vdir, lroot = 'vctk', 'ltc'
    write_text(at(vdir, 'speaker-info.txt'), 'ID  AGE  GENDER  ACCENTS  REGION COMMENTS ')
    with open(at(vdir, 'speaker-info.txt'), 'a') as f:
        for s in sorted(GENDER, key=lambda x: (x[0], int(x[1:]))):
            f.write(f'{s}  {r.randint(18, 38)}  {GENDER[s]}    English    Southern  England\n')
    utts = {}
    for i, s in enumerate(sorted(GENDER)):
        recs = [(n, 'An odd line read under a block number.' if (s, n) == ('p230', '005') else t,
                 round(r.uniform(2.0, 5.0), 3)) for n, t in BLOCK.items() if (s, n) not in OMIT]
        if s == 'p250':
            recs.append(('310', BLOCK['003'], 3.21))
        texts = [f'Speaker {s} reads own line {j} about the {r.choice(WORDS)} and the {r.choice(WORDS)}.'
                 for j in range(26 if s in DEV else 30 if s in OTHER else 8)]
        for k, (t, who) in enumerate(COMMON):
            if s in who:
                texts[4 + k] = t
        if s in ('p233', 'p255'):
            texts[-1] = texts[0]
        lo, hi = (1.05, 1.45) if s in SHORT else (1.2, 2.8) if i % 2 else (2.6, 4.8)
        for j, t in enumerate(texts):
            if j in (7, 19):
                d = r.uniform(2.0, 2.5) if s in SHORT else r.uniform(0.4, 0.9)
            else:
                d = r.uniform(lo, hi)
            recs.append(('%03d' % (25 + j), t, round(d, 3)))
        wdir, utts[s] = os.path.join(vdir, 'wav48_silence_trimmed', s), []
        for j, (n, t, d) in enumerate(recs):
            uid = f'{s}_{n}'
            has_txt, has_mic1 = s != 'p376', s not in ('p362', 'p380') and not (i % 5 == 0 and j == 20)
            if has_txt:
                write_text(at(vdir, 'txt', s, uid + '.txt'), t)
            if has_mic1:
                write_audio(at(wdir, uid + '_mic1.flac'), d, 'FLAC')
            if not has_mic1 or j % 6 == 1:
                write_audio(at(wdir, uid + '_mic2.flac'), d, 'FLAC')
            if s == 'p380':
                write_audio(at(wdir, f'{s}_{int(n) + 500:03d}_mic1.flac'), d, 'FLAC')
            utts[s].append({'uid': uid, 'path': os.path.join(wdir, uid + '_mic1.flac'), 'text': t, 'dur': d,
                            'txt': has_txt, 'mic1': has_mic1, 'in_block': n in BLOCK or t in BLOCK.values()})
        if s in OTHER and i % 4 == 0:
            write_audio(at(wdir, f'{s}_900_mic1.flac'), 2.0, 'FLAC')       # mic1 without txt
    conf = [{'source_speaker': DEV[i % 20], 'target_speaker': DEV[(i + 1 + i // 20) % 20],
             'source_file': f'{DEV[i % 20]}_{30 + i:03d}', 'target_file': f'{DEV[(i + 1 + i // 20) % 20]}_{40 + i:03d}'}
            for i in range(100)]
    write_text(at(vdir, 'eval_pairs.json'), json.dumps(conf))

    tdir = os.path.join(lroot, 'LibriTTS_R', 'test-clean')
    for s in LTC:
        for c in range(2):
            chap = str(4000 + 13 * int(s) % 900 + c)
            for j in range(8):
                if s == LTC[12]:
                    d = 4.5 if c == 0 and j < 3 else r.choice((r.uniform(1.0, 2.8), r.uniform(10.5, 13.0)))
                else:
                    d = r.uniform(3.2, 9.6) if j < 4 else r.uniform(1.0, 14.0)
                stem = os.path.join(tdir, s, chap, f'{s}_{chap}_{j:06d}_{(3 * j + c) % 10:06d}')
                write_audio(at(stem + '.wav'), round(d, 3), 'WAV')
                if not (s == LTC[3] and c == 0 and j == 0):
                    write_text(at(stem + '.normalized.txt'), f'Reader {s} chapter {c} line {j}: {r.choice(WORDS)}.')
                    write_text(at(stem + '.original.txt'), f'Reader {s}, chapter {c}: line {j}.')
    write_text(at(lroot, 'LibriTTS_R', 'speakers.tsv'), 'READER\tGENDER\tSUBSET\tNAME\n' + ''.join(
        f'{s}\t{LTC_GENDER.get(s, "M")}\ttest-clean\tReader {s}\n' for s in LTC + ['9999']).rstrip('\n'))
    write_text(at(lroot, 'test_clean.done'), '')
    return types.SimpleNamespace(root=str(root), at=at, vctk_dir=vdir, conf=os.path.join(vdir, 'eval_pairs.json'),
                                 ltc_root=lroot, ltc_speakers=os.path.join(lroot, 'none.tsv'), utts=utts)


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp('corpus'))


# ── running the scripts ──────────────────────────────────────

def command(script, corpus, out_dir, args):
    return [sys.executable, script, '--vctk_dir', corpus.vctk_dir, '--conf_pairs_json', corpus.conf,
            '--ltc_root', corpus.ltc_root, '--ltc_speakers', corpus.ltc_speakers, '--output_dir', str(out_dir), *args]


def execute(cmd, cwd, hashseed='0'):
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env, timeout=600)


def run(corpus, out_dir, args, script=NEW):
    return execute(command(script, corpus, out_dir, args), corpus.root)


def run_check(out_dir, sets):
    return execute([sys.executable, NEW, '--check', '--sets', sets, '--output_dir', str(out_dir)],
                   os.path.dirname(str(out_dir)))


def run_parallel(jobs, cwd):
    """{key: (cmd, PYTHONHASHSEED)} -> {key: CompletedProcess}, the commands run concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 2)) as ex:
        futs = {k: ex.submit(execute, c, cwd, h) for k, (c, h) in jobs.items()}
        return {k: f.result() for k, f in futs.items()}


def sets_of(args):
    return (args[args.index('--sets') + 1] if '--sets' in args else 'dev,vctk2k,ltc2k').split(',')


def opt(args, name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


@pytest.fixture(scope='module')
def runs(corpus, tmp_path_factory):
    """Every MATRIX run with the original (PYTHONHASHSEED 1) and the current script (PYTHONHASHSEED 2). The current
    script's runs with vctk2k but not dev in --sets read --dev_pairs from the main run (new/full, = the HPC pairs/),
    so they run after it; the main directory is snapshot before and after them. small_e has dev in --sets and passes a
    --dev_pairs that does not exist."""
    base = str(tmp_path_factory.mktemp('runs'))
    out = lambda kind, name: os.path.join(base, kind, name)
    main = out('new', 'full')
    alone = [n for n, a in MATRIX.items() if 'vctk2k' in sets_of(a) and 'dev' not in sets_of(a)]
    extra = {'small_e': ['--dev_pairs', os.path.join(base, 'absent', 'dev.jsonl')]}
    jobs = {('orig', n): (command(ORIG, corpus, out('orig', n), a), '1') for n, a in MATRIX.items()}
    jobs.update({('new', n): (command(NEW, corpus, out('new', n), a + extra.get(n, [])), '2')
                 for n, a in MATRIX.items() if n not in alone})
    res = run_parallel(jobs, corpus.root)
    r = res['new', 'full']
    assert r.returncode == 0, r.stderr[-3000:]
    before = snapshot(main)
    res.update(run_parallel({('new', n): (command(NEW, corpus, out('new', n), MATRIX[n] + [
        '--dev_pairs', os.path.join(main, 'dev.jsonl')]), '2') for n in alone}, corpus.root))
    return types.SimpleNamespace(res=res, dir=out, main=main, main_before=before, main_after=snapshot(main))


# ── helpers ──────────────────────────────────────────────────

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def snapshot(d):
    """{file: (size, mtime_ns, sha256)} of the files in d."""
    out = {}
    for f in sorted(os.listdir(d)):
        st = os.stat(os.path.join(d, f))
        out[f] = (st.st_size, st.st_mtime_ns, sha256(os.path.join(d, f)))
    return out


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in rows))


def load_json(*parts):
    with open(os.path.join(*parts)) as f:
        return json.load(f)


def ok(r):
    assert r.returncode == 0, f'exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}'
    return r


def copy_dir(src, dst, only=None, skip=()):
    os.makedirs(dst)
    for f in os.listdir(src):
        if (only is None or f in only) and f not in skip:
            shutil.copy2(os.path.join(src, f), dst)
    return dst


def check_line(stdout, name):
    lines = [x for x in stdout.splitlines() if x.startswith(f'[check] {name}:')]
    assert len(lines) == 1, stdout
    return lines[0]


def expected_source_E(corpus, rows, seed, n_e, e_max_s, min_dur):
    """The source-E rule (plan v1 §1.2, make_pairs docstring) restated on the corpus as written, not on make_pairs'
    catalogue: per source speaker of the list, its mic1 recordings with a transcript that are neither shared-block
    recordings nor source recordings of the list, sorted by uid and shuffled by random.Random('<seed>:vctk2k:
    source_E:<speaker>'). E takes the utterances of at least min_dur until it holds n_e or reaches e_max_s seconds,
    whichever comes first; a speaker whose pool gets to neither is missing. Returns (E, missing)."""
    src_paths = {r['source_path'] for r in rows}
    E, missing = {}, {}
    for s in sorted({r['source_speaker'] for r in rows}):
        pool = sorted((u for u in corpus.utts[s] if u['txt'] and u['mic1'] and not u['in_block']
                       and u['path'] not in src_paths), key=lambda u: u['uid'])
        random.Random(f'{seed}:vctk2k:source_E:{s}').shuffle(pool)
        sel, tot = [], 0.0
        for u in pool:
            if len(sel) == n_e or tot >= e_max_s:
                break
            if u['dur'] >= min_dur:
                sel.append(u)
                tot += u['dur']
        if len(sel) == n_e or tot >= e_max_s:
            E[s] = {'paths': [u['path'] for u in sel], 'durs': [u['dur'] for u in sel],
                    'total_s': round(sum(u['dur'] for u in sel), 3)}
        else:
            missing[s] = {'n_utts': len(sel), 'total_s': round(tot, 3)}
    return E, missing


def assert_source_E_rules(corpus, rows, E, n_e, e_max_s, min_dur):
    """Each rule on its own, so a failure names the rule."""
    src_paths = {r['source_path'] for r in rows}
    owner = {u['path']: (s, u) for s, us in corpus.utts.items() for u in us if u['mic1'] and u['txt']}
    assert list(E) == sorted(E)
    for s, e in E.items():
        assert list(e) == ['paths', 'durs', 'total_s'], s
        assert len(set(e['paths'])) == len(e['paths']) == len(e['durs']), s
        for p, d in zip(e['paths'], e['durs']):
            assert p in owner and owner[p][0] == s, f'{s}: {p} is not its own mic1 recording with a transcript'
            assert not owner[p][1]['in_block'], f'{s}: shared-block recording {p}'
            assert p not in src_paths, f'{s}: source recording {p}'
            assert d == owner[p][1]['dur'] == round(sf.info(corpus.at(p)).duration, 3) and d >= min_dur, (s, p, d)
        assert e['total_s'] == round(sum(e['durs']), 3), s
        assert len(e['paths']) == n_e or sum(e['durs']) >= e_max_s, f'{s}: E not filled'
        assert len(e['paths']) <= n_e and sum(e['durs'][:-1]) < e_max_s, f'{s}: E goes past n_e or e_max_s'


def strip_summary(s, out_dir, new):
    """summary.json without 'created', with the output dir replaced, and without the keys the change added (new)."""
    s = json.loads(json.dumps(s).replace(out_dir, '<OUT>'))
    del s['created']
    if new:
        for k in NEW_KEYS['args']:
            del s['args'][k]
        for k in NEW_KEYS['vctk2k']:
            s['sets'].get('vctk2k', {}).pop(k, None)
    return s


# ── the original script ──────────────────────────────────────

def test_original_copy_is_the_pristine_pre_change_script():
    assert sha256(ORIG) == ORIG_SHA256, 'tests/data/make_pairs_orig.py must stay the unmodified pre-change script'
    orig, new = open(ORIG).read(), open(NEW).read()
    for added in ('--source_e_only', "'--check'", '--dev_pairs', 'def build_source_E', 'def run_source_E',
                  'def check_lists', 'source_E'):
        assert added not in orig and added in new, added


# ── byte identity with the original ──────────────────────────

@pytest.mark.parametrize('name', MATRIX)
def test_lists_byte_identical_to_the_original(runs, name):
    for kind in ('orig', 'new'):
        r = runs.res[kind, name]
        assert r.returncode == 0, f'{kind} {name}: exit {r.returncode}\n{r.stderr[-3000:]}'
    o, n = runs.dir('orig', name), runs.dir('new', name)
    sets = sets_of(MATRIX[name])
    lists = sorted(f'{x}.jsonl' for x in sets)
    assert sorted(os.listdir(o)) == sorted(lists + ['summary.json'])         # neither writes a self.jsonl
    assert sorted(os.listdir(n)) == sorted(lists + ['summary.json'] + ['vctk2k_source_E.json'] * ('vctk2k' in sets))
    for f in lists:
        with open(os.path.join(o, f), 'rb') as a, open(os.path.join(n, f), 'rb') as b:
            assert a.read() == b.read(), f'{name}: {f} differs from the original script'


@pytest.mark.parametrize('name', MATRIX)
def test_summary_differs_from_the_original_only_by_the_new_keys(runs, name):
    so, sn = (load_json(runs.dir(k, name), 'summary.json') for k in ('orig', 'new'))
    sets = sets_of(MATRIX[name])
    v = sn['sets'].get('vctk2k', {})
    assert ('source_E' in v) == ('vctk2k' in sets)
    assert ('dev_speakers_checked' in v) == ('vctk2k' in sets and 'dev' not in sets)
    assert json.dumps(strip_summary(sn, runs.dir('new', name), True)) == \
        json.dumps(strip_summary(so, runs.dir('orig', name), False))      # same keys, order and values
    for x in sets:
        assert sn['sets'][x]['sha256'] == sha256(sn['sets'][x]['jsonl'])


def test_synthetic_corpus_exercises_the_edge_cases(runs):
    s = load_json(runs.main, 'summary.json')
    v = s['vctk']
    assert v['excluded_missing_data'] == BAD and v['excluded_by_rule'] == ['p315', 's5'] and v['n_indexed'] == 62
    b = v['shared_text_block']
    assert b['n_sentences'] == 12 and b['ids'] == sorted(BLOCK) and b['n_in_block_recordings_non_canonical_text'] == 1
    assert b['next_largest_text_speaker_count'] == len(OTHER)
    sets = s['sets']
    assert [sets[x]['n_pairs'] for x in ('dev', 'vctk2k', 'ltc2k')] == [1000, 2000, 600]
    assert sets['vctk2k']['source_speakers'] == {x: GENDER[x] for x in sorted(V2K_SRC)}
    assert sets['vctk2k']['target_speakers'] == {x: GENDER[x] for x in sorted(V2K_TGT)}
    assert list(sets['ltc2k']['excluded_too_few_source_utts']) == [LTC[12]]
    assert sets['dev']['shared_slots_fallen_back_to_specific'] + sets['vctk2k']['shared_slots_fallen_back_to_specific']
    dropped = sum((collections.Counter(sets[x]['ref_pool_dropped_total']) for x in sets), collections.Counter())
    assert all(dropped[k] for k in ('shared_block', 'is_a_source', 'same_text_as_a_source', 'too_short')), dropped
    assert all(sets[x]['R_per_target']['180']['n_shortfall'] for x in sets)
    se = sets['vctk2k']['source_E']
    assert se['pool_dropped_total']['too_short'] and se['pool_dropped_total']['shared_block']
    E = load_json(runs.main, 'vctk2k_source_E.json')
    assert {len(e['paths']) == 10 for e in E.values()} == {True, False}, 'both stops (n_e, e_max_s) occur'


# ── the source-E file ────────────────────────────────────────

@pytest.mark.parametrize('name', ['full', 't10', 'seed7', 'small_e', 'missing'])
def test_source_E_follows_the_rules(runs, corpus, name):
    a, d = MATRIX[name], runs.dir('new', name)
    seed, n_e, e_max_s, min_dur = (opt(a, '--seed', SEED), opt(a, '--n_e', 10), opt(a, '--e_max_s', 30.0),
                                   opt(a, '--min_ref_dur', 1.0))
    rows = read_jsonl(os.path.join(d, 'vctk2k.jsonl'))
    E = load_json(d, 'vctk2k_source_E.json')
    v = load_json(d, 'summary.json')['sets']['vctk2k']
    want_E, want_missing = expected_source_E(corpus, rows, seed, n_e, e_max_s, min_dur)
    assert_source_E_rules(corpus, rows, E, n_e, e_max_s, min_dur)
    assert E == want_E                                        # the exact draw, on its own seeded stream
    assert v['source_E']['missing'] == want_missing
    assert set(E) | set(want_missing) == {r['source_speaker'] for r in rows}
    path = os.path.join(d, 'vctk2k_source_E.json')
    assert v['source_E']['json'] == path and v['source_E']['sha256'] == sha256(path)
    lst = os.path.join(d, 'vctk2k.jsonl')
    assert v['source_E']['pairs_jsonl'] == v['jsonl'] == lst                 # the list is not rewritten
    assert v['source_E']['pairs_sha256'] == v['sha256'] == sha256(lst)
    if name in ('full', 't10', 'seed7'):
        assert not want_missing
    if name == 'seed7':
        assert E != load_json(runs.main, 'vctk2k_source_E.json')


def test_speakers_who_cannot_fill_E_are_left_out_and_counted(runs, corpus):
    d = runs.dir('new', 'missing')
    rows = read_jsonl(os.path.join(d, 'vctk2k.jsonl'))
    E = load_json(d, 'vctk2k_source_E.json')
    se = load_json(d, 'summary.json')['sets']['vctk2k']['source_E']
    assert set(SHORT) <= {r['source_speaker'] for r in rows}
    assert sorted(se['missing']) == SHORT and not set(E) & set(SHORT) and len(E) == 18
    assert all(m['n_utts'] < 10 and m['total_s'] < 30 for m in se['missing'].values())
    assert se['n_source_speakers'] == 20 and se['n_with_E'] == 18
    assert se['n_pairs_without_E'] == sum(r['source_speaker'] in SHORT for r in rows) == 200
    assert set(SHORT) <= set(load_json(runs.main, 'vctk2k_source_E.json'))   # with min_ref_dur 1 they have one
    r = run_check(d, 'vctk2k')
    assert r.returncode == 1 and 'check: FAIL' in r.stderr
    line = check_line(r.stdout, 'vctk2k')
    assert f'without one {SHORT} (200 pairs)' in line and line.endswith('-> FAIL')


def test_source_E_step_draws_only_its_own_streams(runs, corpus, monkeypatch):
    monkeypatch.chdir(corpus.root)
    cat, _ = make_pairs.load_vctk(corpus.vctk_dir, [s for s in GENDER if s not in ('p315', 's5')])
    make_pairs.detect_shared_block(cat, 0.8)
    rows = read_jsonl(os.path.join(runs.main, 'vctk2k.jsonl'))
    tags, rng = [], make_pairs.rng
    monkeypatch.setattr(make_pairs, 'rng', lambda seed, *t: tags.append((seed,) + t) or rng(seed, *t))
    state = random.getstate()
    E, missing, _ = make_pairs.build_source_E('vctk2k', cat, rows, 10, 30.0, 1.0, SEED)
    assert random.getstate() == state
    assert tags == [(SEED, 'vctk2k', 'source_E', s) for s in sorted(V2K_SRC)]
    assert E == load_json(runs.main, 'vctk2k_source_E.json') and not missing


def test_build_source_E_on_a_hand_made_catalogue(tmp_path):
    def utt(s, n, d, block=False):
        p = str(tmp_path / s / f'{s}_{n:03d}_mic1.flac')
        write_audio(p, d, 'FLAC')
        return {'uid': f'{s}_{n:03d}', 'path': p, 'text': f'{s} {n}', 'norm': f'{s} {n}', 'shared': None,
                'in_block': block}

    cat = {'pa': [utt('pa', 30 + i, 2.0) for i in range(3)],                     # 2 left, 4 s: missing
           'pb': [utt('pb', 30 + i, 7.0) for i in range(30)],                    # e_max_s first: 5 utterances
           'pc': [utt('pc', 30 + i, 1.5) for i in range(30)],                    # n_e first: 10 utterances
           'pd': [utt('pd', 30 + i, 0.5) for i in range(20)] + [utt('pd', 60 + i, 2.0) for i in range(12)]
           + [utt('pd', 1 + i, 9.0, True) for i in range(20)],                   # too short and block skipped
           'pe': [utt('pe', 30 + i, 4.0) for i in range(15)],                    # 12 are sources: 3 left, missing
           'pf': [utt('pf', 30 + i, 6.0) for i in range(8)],                     # exactly e_max_s at 5 utterances
           'pg': [utt('pg', 1 + i, 9.0, True) for i in range(24)] + [utt('pg', 40, 3.0)]}   # nothing left
    pairs = [{'source_speaker': s, 'source_path': cat[s][0]['path']} for s in ('pa', 'pb', 'pc', 'pd', 'pf')]
    pairs += [{'source_speaker': 'pe', 'source_path': u['path']} for u in cat['pe'][:12]]
    pairs += [{'source_speaker': 'pg', 'source_path': cat['pg'][-1]['path']}]
    E, missing, dropped = make_pairs.build_source_E('vctk2k', cat, pairs, 10, 30.0, 1.0, SEED)
    assert missing == {'pa': {'n_utts': 2, 'total_s': 4.0}, 'pe': {'n_utts': 3, 'total_s': 12.0},
                       'pg': {'n_utts': 0, 'total_s': 0.0}}
    assert {s: (len(e['paths']), e['total_s']) for s, e in E.items()} == {
        'pb': (5, 35.0), 'pc': (10, 15.0), 'pd': (10, 20.0), 'pf': (5, 30.0)}
    src = {p['source_path'] for p in pairs}
    for s, e in E.items():      # the uid-sorted pool shuffled by its own stream; E = its long-enough head
        pool = [u['path'] for u in sorted(cat[s], key=lambda u: u['uid']) if not u['in_block'] and u['path'] not in src]
        random.Random(f'{SEED}:vctk2k:source_E:{s}').shuffle(pool)
        seen = pool[:pool.index(e['paths'][-1]) + 1]
        assert e['paths'] == [p for p in seen if make_pairs.dur(p) >= 1.0], s
        assert e['durs'] == [make_pairs.dur(p) for p in e['paths']], s
        assert dropped[s].get('too_short', 0) == sum(make_pairs.dur(p) < 1.0 for p in seen), s
    assert dropped['pd']['shared_block'] == 20 and dropped['pd']['is_a_source'] == 1 and dropped['pd']['too_short']
    assert dropped['pe'] == {'is_a_source': 12} and dropped['pg'] == {'shared_block': 24, 'is_a_source': 1}
    assert (E, missing) == make_pairs.build_source_E('vctk2k', cat, pairs, 10, 30.0, 1.0, SEED)[:2]
    assert E['pc']['paths'] != make_pairs.build_source_E('vctk2k', cat, pairs, 10, 30.0, 1.0, 7)[0]['pc']['paths']


# ── --source_e_only ──────────────────────────────────────────

@pytest.mark.parametrize('kind, name', [('orig', 'full'), ('new', 'full'), ('new', 't10'), ('new', 'missing')])
def test_source_e_only_writes_only_the_source_E_file(runs, corpus, tmp_path, kind, name):
    """('orig', 'full') is job D0 step 1 on the HPC pairs/, which the original script wrote: the file equals the one a
    current full run writes inline, and D0's --check then passes."""
    d = copy_dir(runs.dir(kind, name), str(tmp_path / 'pairs'), skip=('vctk2k_source_E.json',))
    write_jsonl(os.path.join(d, 'self.jsonl'), [{'set': 'self'}])        # the HPC pairs/ also holds the truth set
    before = snapshot(d)
    r = ok(run(corpus, d, MATRIX[name] + ['--source_e_only']))
    after = snapshot(d)
    assert sorted(after) == sorted(list(before) + ['vctk2k_source_E.json'])
    assert {f: after[f] for f in before} == before                     # the list and the rest are not rewritten
    assert sha256(os.path.join(d, 'vctk2k_source_E.json')) == sha256(os.path.join(runs.dir('new', name),
                                                                                 'vctk2k_source_E.json'))
    assert f'list sha256 {sha256(os.path.join(d, "vctk2k.jsonl"))}' in r.stdout
    if kind == 'orig':
        assert ok(run_check(d, 'dev,vctk2k,ltc2k')).stdout.count('-> OK') == 3


def crafted_list(runs, corpus, case):
    """100 rows of the main lists (the offending row last) or an empty list."""
    rows = read_jsonl(os.path.join(runs.main, 'vctk2k.jsonl'))
    if case in ('dev_rows', 'ltc2k_rows'):
        return read_jsonl(os.path.join(runs.main, case[:-5] + '.jsonl'))[:100]
    if case == 'one_dev_row':
        return rows[:99] + read_jsonl(os.path.join(runs.main, 'dev.jsonl'))[:1]
    if case == 'mic2_path':           # a recording on disk, but mic2
        i = next(i for i, r in enumerate(rows) if i >= 99 and os.path.exists(corpus.at(r['source_path'].replace(
            '_mic1.', '_mic2.'))))
        rows[i]['source_path'] = rows[i]['source_path'].replace('_mic1.', '_mic2.')
        return rows[i - 99:i + 1]
    if case == 'other_vctk_dir':
        hpc = '/datastore/nvquoc/crt/mimi_vc/benchmark/vctk/'
        return [dict(r, source_path=r['source_path'].replace('vctk/', hpc, 1)) for r in rows[:100]]
    assert case == 'empty'
    return []


@pytest.mark.parametrize('case', ['dev_rows', 'ltc2k_rows', 'one_dev_row', 'mic2_path', 'other_vctk_dir', 'empty'])
def test_source_e_only_refuses_a_list_that_is_not_vctk2k_mic1(runs, corpus, tmp_path, case):
    d = str(tmp_path / 'pairs')
    os.makedirs(d)
    write_jsonl(os.path.join(d, 'vctk2k.jsonl'), crafted_list(runs, corpus, case))
    before = snapshot(d)
    r = run(corpus, d, ['--source_e_only'])
    assert r.returncode == 1 and 'is not a VCTK-2k list of recordings in --vctk_dir' in r.stderr, r.stderr[-2000:]
    assert snapshot(d) == before


# ── the 10-target variant and separate output dirs ───────────

def test_10_target_variant_goes_to_its_own_dir_with_its_own_summary(runs):
    t10 = runs.dir('new', 't10')
    assert sorted(os.listdir(t10)) == ['summary.json', 'vctk2k.jsonl', 'vctk2k_source_E.json']
    s, m = load_json(t10, 'summary.json'), load_json(runs.main, 'summary.json')
    assert s['args']['output_dir'] == t10 and s['args']['vctk_targets'] == 10 and list(s['sets']) == ['vctk2k']
    v, v5 = s['sets']['vctk2k'], m['sets']['vctk2k']
    assert v['jsonl'] == os.path.join(t10, 'vctk2k.jsonl')
    assert v['source_E']['json'] == os.path.join(t10, 'vctk2k_source_E.json')
    assert v['source_E']['pairs_jsonl'] == v['jsonl']
    assert v5['n_pairs'] == 2000 and v['n_pairs'] == 4000 and v['pairs_per_target'] == {'min': 200, 'max': 200}
    assert v['gender_pairs'] == {g: 1000 for g in ('F2F', 'F2M', 'M2F', 'M2M')}
    assert (v['source_speakers'], v['target_speakers']) == (v5['source_speakers'], v5['target_speakers'])
    targets = collections.defaultdict(list)
    for r in read_jsonl(v['jsonl']):
        targets[r['source_path']].append(r['target_gender'] + r['target_speaker'])
    assert len(targets) == 400 == v['n_source_utts']
    assert all(len(set(t)) == 10 and collections.Counter(x[0] for x in t) == {'F': 5, 'M': 5} for t in targets.values())
    assert v['dev_speakers_checked'] == {'dev_pairs': os.path.join(runs.main, 'dev.jsonl'), 'n_dev_speakers': 20,
                                         'shared': []}
    assert v['source_E']['n_source_speakers'] == v['source_E']['n_with_E'] == 20


def test_runs_into_another_output_dir_leave_the_main_lists_alone(runs):
    assert runs.main_after == runs.main_before
    assert snapshot(runs.main) == runs.main_before
    assert sorted(runs.main_before) == ['dev.jsonl', 'ltc2k.jsonl', 'summary.json', 'vctk2k.jsonl',
                                        'vctk2k_source_E.json']
    for name in ('t10', 'vctk2k', 'missing'):
        assert load_json(runs.dir('new', name), 'summary.json')['args']['output_dir'] == runs.dir('new', name)


# ── the Dev-speaker check of a run without dev ───────────────

def test_dev_check_passes_with_the_dev_list(runs):
    dev_pairs = os.path.join(runs.main, 'dev.jsonl')
    rows = read_jsonl(dev_pairs)
    assert {r['source_speaker'] for r in rows} | {r['target_speaker'] for r in rows} == set(DEV)
    for name in ('vctk2k', 't10', 'missing'):
        r = runs.res['new', name]
        assert f'[vctk2k] speakers checked against the 20 speakers of {dev_pairs}: none shared' in ok(r).stdout
        v = load_json(runs.dir('new', name), 'summary.json')['sets']['vctk2k']
        assert v['dev_speakers_checked'] == {'dev_pairs': dev_pairs, 'n_dev_speakers': 20, 'shared': []}
        assert not (set(v['source_speakers']) | set(v['target_speakers'])) & set(DEV)


@pytest.mark.parametrize('case', ['overlap', 'one_dev_speaker_short', 'ltc2k_list', 'absent'])
def test_dev_check_refuses_a_list_that_shares_speakers_or_is_not_dev(runs, corpus, tmp_path, case):
    rows = read_jsonl(os.path.join(runs.main, 'dev.jsonl'))
    msg = f"VCTK-2k shares ['{V2K_TGT[0]}']" if case == 'overlap' else 'is not the Dev list'
    if case == 'overlap':
        rows[0]['target_speaker'] = V2K_TGT[0]
    elif case == 'one_dev_speaker_short':
        rows = [r for r in rows if DEV[0] not in (r['source_speaker'], r['target_speaker'])]
    elif case == 'ltc2k_list':
        rows = read_jsonl(os.path.join(runs.main, 'ltc2k.jsonl'))
    else:
        msg = 'FileNotFoundError'
    dev_pairs, out = str(tmp_path / 'dev.jsonl'), str(tmp_path / 'out')
    if case != 'absent':
        write_jsonl(dev_pairs, rows)
    r = run(corpus, out, MATRIX['t10'] + ['--dev_pairs', dev_pairs])
    assert r.returncode == 1 and msg in r.stderr, r.stderr[-2000:]
    assert os.listdir(out) == []


def test_dev_pairs_is_not_read_when_dev_is_in_sets(runs):
    a = runs.res['new', 'small_e'].args
    assert not os.path.exists(a[a.index('--dev_pairs') + 1])
    ok(runs.res['new', 'small_e'])
    assert 'dev_speakers_checked' not in load_json(runs.dir('new', 'small_e'), 'summary.json')['sets']['vctk2k']


# ── --check ──────────────────────────────────────────────────

def test_check_ok_on_good_lists_and_writes_nothing(runs):
    for d, sets in ((runs.main, 'dev,vctk2k,ltc2k'), (runs.dir('new', 't10'), 'vctk2k')):
        before = snapshot(d)
        r = ok(run_check(d, sets))
        assert r.stdout.count('-> OK') == len(sets.split(',')) and r.stdout.splitlines()[-1] == 'check: OK'
        assert snapshot(d) == before
    r = run_check(runs.main, 'dev,vctk2k,ltc2k')
    assert check_line(r.stdout, 'dev').startswith('[check] dev: 1000 pairs, 20 source / 20 target speakers, 20 '
                                                  'sources are targets, 20 sources have an E set (target E)')
    assert check_line(r.stdout, 'vctk2k').startswith('[check] vctk2k: 2000 pairs, 20 source / 20 target speakers, 0 '
                                                     'sources are targets, 20 sources have an E set')
    assert check_line(r.stdout, 'ltc2k').startswith('[check] ltc2k: 600 pairs, 12 source / 12 target speakers, 12 '
                                                    'sources are targets, 12 sources have an E set (target E)')


def test_check_creates_nothing_for_a_missing_dir(tmp_path):
    d = tmp_path / 'absent'
    r = run_check(d, 'dev,vctk2k,ltc2k')
    assert r.returncode == 1 and not d.exists() and r.stdout.count('-> FAIL') == 3 and 'check: FAIL' in r.stderr


@pytest.mark.parametrize('name', ['dev', 'vctk2k', 'ltc2k'])
def test_check_fails_on_an_empty_list(runs, tmp_path, name):
    d = copy_dir(runs.main, str(tmp_path / 'pairs'), skip=('summary.json',))
    open(os.path.join(d, f'{name}.jsonl'), 'w').close()
    r = run_check(d, 'dev,vctk2k,ltc2k')
    assert r.returncode == 1 and 'check: FAIL' in r.stderr
    for x in ('dev', 'vctk2k', 'ltc2k'):
        line = check_line(r.stdout, x)
        assert line.endswith('-> FAIL') if x == name else line.endswith('-> OK'), line
    assert check_line(r.stdout, name).startswith(f'[check] {name}: 0 pairs')


@pytest.mark.parametrize('case', ['file_absent', 'speaker_absent', 'empty_entry'])
def test_check_fails_when_a_vctk2k_source_has_no_E(runs, tmp_path, case):
    d = copy_dir(runs.main, str(tmp_path / 'pairs'), only=('vctk2k.jsonl',))
    E = load_json(runs.main, 'vctk2k_source_E.json')
    s = sorted(E)[3]
    if case == 'speaker_absent':
        del E[s]
    elif case == 'empty_entry':
        E[s] = {'paths': [], 'durs': [], 'total_s': 0}
    if case != 'file_absent':
        with open(os.path.join(d, 'vctk2k_source_E.json'), 'w') as f:
            json.dump(E, f)
    r = run_check(d, 'vctk2k')
    assert r.returncode == 1 and 'check: FAIL' in r.stderr
    line = check_line(r.stdout, 'vctk2k')
    assert line.endswith('-> FAIL')
    assert 'missing' in line if case == 'file_absent' else f"without one ['{s}'] (100 pairs)" in line


@pytest.mark.parametrize('case', ['vctk2k_own_source', 'vctk2k_other_speaker', 'dev_source_not_target',
                                  'dev_E_holds_a_source', 'ltc2k_E_holds_another_speaker'])
def test_check_flags_an_E_set_that_breaks_the_rules(runs, tmp_path, capsys, case):
    name = case.split('_')[0]
    d = copy_dir(runs.main, str(tmp_path / 'pairs'), only=(f'{name}.jsonl', 'vctk2k_source_E.json'))
    rows = read_jsonl(os.path.join(d, f'{name}.jsonl'))
    src = sorted({r['source_speaker'] for r in rows})
    s = src[0]
    own_source = next(r['source_path'] for r in rows if r['source_speaker'] == s)
    want = f'source or foreign recording {[s]}'
    if name == 'vctk2k':                       # its own source recording (same dir) / another source's E recording
        E = load_json(d, 'vctk2k_source_E.json')
        E[s]['paths'].append(own_source if case.endswith('own_source') else E[src[1]]['paths'][0])
        with open(os.path.join(d, 'vctk2k_source_E.json'), 'w') as f:
            json.dump(E, f)
    elif case == 'dev_source_not_target':
        rows = [r for r in rows if r['target_speaker'] != s]
        want = f"without one ['{s}']"
    else:                                      # Dev: its own source recording; LTC: another target's E recording
        x = own_source if name == 'dev' else next(r['E']['paths'][0] for r in rows if r['target_speaker'] != s)
        for r in rows:
            if r['target_speaker'] == s:
                r['E']['paths'][0] = x
    write_jsonl(os.path.join(d, f'{name}.jsonl'), rows)
    capsys.readouterr()
    assert make_pairs.check_lists(d, [name]) is False
    line = check_line(capsys.readouterr().out, name)
    assert want in line and line.endswith('-> FAIL'), line
