#!/usr/bin/env python3
"""
Stimuli of the listening test of ANALYSIS_PLAN_v1.md §5.4, in two steps.

  plan   (CPU) --items DEV_SELECTION_RECORD_v1.listening_items.json --vctk_pairs --ltc_pairs --v1 $J/v1 --out plan.json
         The record's 32 items (16 VCTK-2k, 16 LTC-2k, 4 per set and gender combination). An item whose output fails
         for any condition is replaced by the next pair of its set and gender combination in the record's seeded order,
         and the replacement is reported (§5.4). Per item, the audio of each condition:
           nat       natural target speech: P where it exists, else the target's E utterance of 2-10 s closest in
                     length to the source utterance (none of 2-10 s: the shortest of at least 1 s, cut to 10 s)
           nat_mimi  its Mimi Q = 32 copy-synthesis: T1's mimi_copy_P where nat is P, else made by `build`
           faiss, crt04, crtal   S1 FAISS, S3 CRT τ = 0.4 (run_conversions rows faiss, crt_tau0.4) and CRT-AL (the
                     decode-lab row lab_<code> of the record's choice), at 30 s
           knnvc, seedvc         S5, S7 (T4 rows), kept so that the pilot can decide whether they enter (§5.4)
         and the SMOS reference: another E utterance of the target, of 3-10 s closest to 5 s (none: the one closest
         to 5 s, cut to 10 s). Natural target speech and the reference are never in R (make_pairs keeps R, P and E
         disjoint). Three practice items: the next pair (rank 4) of three cells.
  build  (GPU node) --plan plan.json --out <package dir> [--device cuda]
         Makes the missing Mimi copies with the Mimi model and calls of run_conversions.py (Q = 32, float16 on the
         GPU), loads every clip as 24 kHz mono, cuts it to 10 s where the plan says so (50 ms fade-out), normalises it
         to an active-speech RMS of -26 dBFS (20 ms frames within 40 dB of the loudest; the gain is lowered when the
         peak would pass -1 dBFS), and writes stimuli/<sid>.flac (16-bit) under an id that does not reveal the
         condition. Writes lt_data.js (window.LT_DATA for the page in benchmark/listening/page/: the trials of the
         practice, MOS, SMOS and CMOS blocks; each CMOS pair carries an opaque group id per comparison, which the
         page uses to balance the A/B order within each comparison) and --key, outside the package (what every
         trial and clip is: kept by the organiser, never given to raters).
  config (CPU) --package <package dir> --consent <html file> --contact <text> [--title <text>]
         Puts the approved information-and-consent text (an HTML fragment), the contact and optionally the title
         into the config of a built package's lt_data.js; the clips and the trials are untouched. Refuses an empty
         consent text, the placeholder of DEFAULT_CONFIG, a <script> in it, a draft field of consent_v1.html left
         in it ([CONTACT], [COMPENSATION...], ...), or an empty contact, and prints the sha256 of the new lt_data.js.
Trials: MOS and SMOS rate every item under every condition of --conditions (default nat, nat_mimi, faiss, crt04,
crtal), each with a half label (A: ranks 0 and 2, B: ranks 1 and 3 of each cell) for the case where a session would
pass 60 min; CMOS has the 32 CRT-AL vs FAISS pairs (P8), the 32 CRT-AL vs CRT τ = 0.4 pairs (K8) and 8 hidden
identical FAISS-FAISS pairs (rank 0 of each cell, the second copy under its own id). Attention items: two in MOS
(expected 3), one in SMOS (expected 2), one in CMOS (expected 0), on natural clips under their own ids.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys

SEED = 20260928
SR = 24000
CONDITIONS = ('nat', 'nat_mimi', 'faiss', 'crt04', 'crtal')
OPTIONAL = ('knnvc', 'seedvc')
CAND_CODE = {'C1': 'anc_R3_s0_pm_rq_T1', 'C2': 'pm18_rq', 'C3': 'gv1_pm_cont_T1', 'C4': 'gv1_shrink0.85_cont',
             'C5': 'gv0.5_shrink0.75_cont'}
MAX_S, NAT_RANGE, REF_RANGE, REF_TARGET_S = 10.0, (2.0, 10.0), (3.0, 10.0), 5.0
TARGET_DBFS, PEAK_DBFS, ACTIVE_DB, FRAME_S, FADE_S = -26.0, -1.0, 40.0, 0.02, 0.05
CELLS = ('F2F', 'F2M', 'M2F', 'M2M')
PRACTICE = (('vctk2k', 'F2F', 'nat'), ('ltc2k', 'M2M', 'faiss'), ('vctk2k', 'F2M', 'crtal'))
ATTENTION = {'mos': [('vctk2k', 'F2M', 3), ('ltc2k', 'M2F', 3)], 'smos': [('ltc2k', 'F2M', 2)],
             'cmos': [('vctk2k', 'M2M', 0)]}
INSTRUCTION = {'mos': 'To show you are paying attention, please select {r} for this sample.',
               'smos': 'To show you are paying attention, please select {r} for this pair.',
               'cmos': 'To show you are paying attention, please select {r} for this pair.'}
LABEL = {'mos': {1: '1 (Bad)', 2: '2 (Poor)', 3: '3 (Fair)', 4: '4 (Good)', 5: '5 (Excellent)'},
         'smos': {1: '1', 2: '2 (Probably different)', 3: '3', 4: '4', 5: '5'},
         'cmos': {0: '0 (About the same)'}}


def sid(*parts):
    """A clip or trial id that does not reveal what it is."""
    return hashlib.sha256(':'.join(map(str, (SEED,) + parts)).encode()).hexdigest()[:10]


def read_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


# ── plan ─────────────────────────────────────────────────────

def manifest_rows(path, system):
    """{pair_id: row} of one system in a manifest (a missing file gives {})."""
    if not os.path.exists(path):
        return {}
    return {r['pair_id']: r for r in read_jsonl(path) if r['system'] == system}


def output_rows(v1, choice):
    """{condition: {set: {pair_id: row}}} of the converted systems at 30 s (T1, T2 and T4 outputs under v1)."""
    out = {c: {} for c in ('faiss', 'crt04', 'crtal', 'mimi_P', 'knnvc', 'seedvc')}
    for s, part in (('vctk2k', 't1'), ('ltc2k', 't2')):
        rc = os.path.join(v1, part, 'rc', s, '30', 'manifest.jsonl')
        out['faiss'][s] = manifest_rows(rc, 'faiss')
        out['crt04'][s] = manifest_rows(rc, 'crt_tau0.4')
        if choice == 'C1':
            out['crtal'][s] = manifest_rows(os.path.join(v1, part, 'anc', s, '30', 'manifest.jsonl'),
                                            'anc_' + CAND_CODE['C1'])
        else:
            out['crtal'][s] = manifest_rows(os.path.join(v1, part, 'lab', 'cand', 'manifest.jsonl'),
                                            'lab_' + CAND_CODE[choice])
        out['mimi_P'][s] = manifest_rows(os.path.join(v1, part, 'mc', s, '30', 'manifest.jsonl'), 'mimi_copy_P')
        for b in OPTIONAL:
            out[b][s] = manifest_rows(os.path.join(v1, 't4', 'bl', s, '30', b, 'manifest.jsonl'), b)
    return out


def ok(row):
    return row is not None and 'error' not in row and bool(row.get('wav'))


def pick_e(paths, durs, exclude, lo_hi, target_s):
    """(path, cut) of the E utterance with duration in lo_hi closest to target_s; else the closest one of at least
    1 s, cut when longer than MAX_S. None when no E utterance is left."""
    cand = [(p, d) for p, d in zip(paths, durs) if p not in exclude and d >= 1.0]
    if not cand:
        return None
    inside = [(p, d) for p, d in cand if lo_hi[0] <= d <= lo_hi[1]]
    p, d = min(inside or cand, key=lambda x: (abs(x[1] - target_s), x[0]))
    return p, d > MAX_S


def natural_and_ref(pair):
    """The natural target clip and the SMOS reference of a pair (§5.4): (nat, ref), each {path, cut, kind}."""
    e = pair.get('E') or {}
    paths, durs = list(e.get('paths') or []), list(e.get('durs') or [])
    if pair.get('P'):
        nat = {'path': pair['P'], 'cut': (pair.get('P_dur') or 0) > MAX_S, 'kind': 'P'}
    else:
        got = pick_e(paths, durs, (), NAT_RANGE, pair.get('source_dur') or REF_TARGET_S)
        nat = got and {'path': got[0], 'cut': got[1], 'kind': 'E'}
    if nat is None:
        return None, None
    got = pick_e(paths, durs, {nat['path']}, REF_RANGE, REF_TARGET_S)
    return nat, (got and {'path': got[0], 'cut': got[1], 'kind': 'E'})


def item_audio(pair, rows, conditions):
    """{condition: {path, cut, ...}} of one pair, or (None, reason) when a condition is missing or failed."""
    nat, ref = natural_and_ref(pair)
    if nat is None:
        return None, 'no natural target clip'
    if ref is None:
        return None, 'no second E utterance for the SMOS reference'
    s, k = pair['set'], pair['pair_id']
    out = {'nat': nat, 'ref': ref}
    for c in conditions:
        if c in ('nat',):
            continue
        if c == 'nat_mimi':
            r = rows['mimi_P'][s].get(k)
            if nat['kind'] == 'P':
                if not ok(r):
                    return None, 'mimi_copy_P failed or missing'
                out[c] = {'path': r['wav'], 'cut': nat['cut'], 'kind': 'mimi_copy_P'}
            else:
                out[c] = {'path': None, 'from': nat['path'], 'cut': nat['cut'], 'kind': 'make'}
            continue
        r = rows[c][s].get(k)
        if not ok(r):
            return None, f'{c} failed or missing'
        out[c] = {'path': r['wav'], 'cut': False, 'kind': 'output'}
    return out, None


def plan(items_doc, pairs, rows, conditions):
    """The items after replacement, their audio, the practice items and the replacements made (§5.4)."""
    by_cell = {}
    for it in items_doc['items']:
        by_cell.setdefault((it['set'], it['genders']), []).append(it)
    items, replaced, practice = [], [], []
    for (s, g), its in sorted(by_cell.items()):
        its = sorted(its, key=lambda x: x['rank'])
        drawn = {it['pair_id'] for it in its}
        queue = [k for k in items_doc['sets'][s]['order'][g] if k not in drawn]   # the seeded replacement order
        for it in its:
            k = it['pair_id']
            audio, why = item_audio(pairs[k], rows, conditions)
            while audio is None:
                replaced.append({'set': s, 'genders': g, 'rank': it['rank'], 'pair_id': k, 'reason': why})
                if not queue:
                    raise SystemExit(f'ERROR: no replacement left for {s} {g} rank {it["rank"]}')
                k = queue.pop(0)
                audio, why = item_audio(pairs[k], rows, conditions)
            items.append({'set': s, 'genders': g, 'rank': it['rank'], 'pair_id': k, 'drawn': it['pair_id'],
                          'source_speaker': pairs[k]['source_speaker'], 'target_speaker': pairs[k]['target_speaker'],
                          'audio': audio})
        for ps, pg, pc in PRACTICE:
            if (ps, pg) != (s, g):
                continue
            for k in queue:
                audio, _ = item_audio(pairs[k], rows, ('nat',) if pc == 'nat' else ('nat', pc))
                if audio is not None:
                    practice.append({'set': s, 'genders': g, 'pair_id': k, 'condition': pc, 'audio': audio})
                    queue.remove(k)
                    break
    return {'items': items, 'replaced': replaced, 'practice': practice, 'conditions': list(conditions)}


def trials(p):
    """The trial lists of the page (ids only) and the key (what each trial and clip is)."""
    clips, key_trials = {}, {}

    def clip(item, cond, role='stim'):
        a = item['audio']['ref' if role == 'ref' else cond]
        c = sid(item['pair_id'], cond, role)
        clips[c] = {'pair_id': item['pair_id'], 'condition': cond, 'role': role, **a}
        return c
    mos, smos, cmos, practice = [], [], [], []
    half = {0: 'A', 2: 'A', 1: 'B', 3: 'B'}
    for it in p['items']:
        for c in p['conditions']:
            t = sid('mos', it['pair_id'], c)
            mos.append({'tid': t, 'stim': clip(it, c), 'half': half[it['rank']]})
            key_trials[t] = {'block': 'mos', 'kind': 'gold' if c == 'nat' else 'main', 'pair_id': it['pair_id'],
                             'set': it['set'], 'genders': it['genders'], 'rank': it['rank'], 'condition': c}
            t = sid('smos', it['pair_id'], c)
            smos.append({'tid': t, 'stim': clip(it, c), 'ref': clip(it, 'ref', 'ref'), 'half': half[it['rank']]})
            key_trials[t] = dict(key_trials[sid('mos', it['pair_id'], c)], block='smos', kind='main')
        for other, block in (('faiss', 'P8'), ('crt04', 'K8')):
            t = sid('cmos', it['pair_id'], other)
            cmos.append({'tid': t, 'x': clip(it, 'crtal'), 'y': clip(it, other), 'group': sid('group', block)})
            key_trials[t] = {'block': 'cmos', 'kind': block, 'pair_id': it['pair_id'], 'set': it['set'],
                             'genders': it['genders'], 'rank': it['rank'], 'x': 'crtal', 'y': other}
        if it['rank'] == 0:
            t = sid('cmos', it['pair_id'], 'hidden')
            cmos.append({'tid': t, 'x': clip(it, 'faiss'), 'y': clip(it, 'faiss', 'dup'), 'group': sid('group', 'hidden')})
            key_trials[t] = {'block': 'cmos', 'kind': 'hidden', 'pair_id': it['pair_id'], 'set': it['set'],
                             'genders': it['genders'], 'rank': 0, 'x': 'faiss', 'y': 'faiss'}
    cell = {(it['set'], it['genders'], it['rank']): it for it in p['items']}
    for block, specs in ATTENTION.items():
        for n, (s, g, r) in enumerate(specs):
            it = cell[(s, g, 1)]
            t = sid(block, 'attention', n)
            entry = {'tid': t, 'attention': r, 'instruction': INSTRUCTION[block].format(r=LABEL[block][r])}
            if block == 'mos':
                entry['stim'] = clip(it, 'nat', f'att{n}')
            elif block == 'smos':
                entry.update(stim=clip(it, 'nat', f'att{n}'), ref=clip(it, 'ref', 'ref'))
            else:
                entry.update(x=clip(it, 'nat', f'att{n}'), y=clip(it, 'nat', f'att{n}b'))
            {'mos': mos, 'smos': smos, 'cmos': cmos}[block].append(entry)
            key_trials[t] = {'block': block, 'kind': 'attention', 'expected': r, 'pair_id': it['pair_id']}
    for n, pr in enumerate(p['practice']):
        t = sid('practice', n)
        c = sid(pr['pair_id'], pr['condition'], 'practice')
        clips[c] = {'pair_id': pr['pair_id'], 'condition': pr['condition'], 'role': 'practice',
                    **pr['audio'][pr['condition']]}
        practice.append({'tid': t, 'stim': c})
        key_trials[t] = {'block': 'practice', 'kind': 'practice', 'pair_id': pr['pair_id'],
                         'condition': pr['condition']}
    return {'practice': practice, 'mos': mos, 'smos': smos, 'cmos': cmos}, clips, key_trials


# ── build ────────────────────────────────────────────────────

def active_db(x, sr=SR):
    """Active-speech RMS level (dBFS) of a mono float signal: 20 ms frames within ACTIVE_DB of the loudest."""
    import numpy as np
    n = max(1, int(FRAME_S * sr))
    frames = x[:len(x) // n * n].reshape(-1, n) if len(x) >= n else x.reshape(1, -1)
    e = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-20)
    act = frames[e >= e.max() - ACTIVE_DB]
    return float(10 * np.log10(np.mean(act.astype(np.float64) ** 2) + 1e-20))


def normalise(x, sr=SR):
    """(signal at TARGET_DBFS active RMS, gain dB); the gain is lowered so that the peak stays at PEAK_DBFS."""
    import numpy as np
    g = TARGET_DBFS - active_db(x, sr)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0 and 20 * math.log10(peak) + g > PEAK_DBFS:
        g = PEAK_DBFS - 20 * math.log10(peak)
    return (x * 10 ** (g / 20)).astype(np.float32), g


def cut(x, sr=SR, max_s=MAX_S, fade_s=FADE_S):
    import numpy as np
    n = int(max_s * sr)
    if len(x) <= n:
        return x
    y = x[:n].copy()
    f = int(fade_s * sr)
    y[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    return y


class MimiCopier:
    """Mimi Q = 32 copy-synthesis with the model and calls of run_conversions.Pipeline (encode / decode)."""

    def __init__(self, device):
        import torch
        from transformers import AutoFeatureExtractor, MimiModel
        from run_conversions import MIMI_ID
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == 'cuda' else torch.float32
        self.mimi = MimiModel.from_pretrained(MIMI_ID, torch_dtype=self.dtype).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(MIMI_ID)
        assert self.fe.sampling_rate == SR

    def __call__(self, wav):
        import numpy as np
        inp = self.fe(raw_audio=wav, sampling_rate=SR, return_tensors='pt')
        inp = {k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else None) for k, v in inp.items()}
        with self.torch.no_grad():
            codes = self.mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes
            return self.mimi.decode(codes).audio_values.cpu().squeeze().numpy().astype(np.float32)


def build(plan_doc, out, key_path, copier, config):
    pkg, kdir = os.path.abspath(out), os.path.abspath(os.path.dirname(key_path) or '.')
    if kdir == pkg or kdir.startswith(pkg + os.sep):
        raise SystemExit(f'ERROR: the key {key_path} must not be inside the package {out}, which raters receive')
    import librosa
    import numpy as np
    import soundfile as sf
    pages, clips, key_trials = trials(plan_doc)
    os.makedirs(os.path.join(out, 'stimuli'), exist_ok=True)
    made, stimuli = {}, {}
    for c, a in sorted(clips.items()):
        src = a['path'] if a['kind'] != 'make' else a['from']
        x = librosa.load(src, sr=SR, mono=True)[0].astype(np.float32)
        if a['cut']:
            x = cut(x)
        if a['kind'] == 'make':
            if src not in made:
                made[src] = copier(x)
            x = made[src][:len(x)]
        y, g = normalise(x)
        f = os.path.join(out, 'stimuli', f'{c}.flac')
        sf.write(f, y, SR, subtype='PCM_16', format='FLAC')
        a.update(source=src, gain_db=round(g, 2), dur=round(len(y) / SR, 3),
                 sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest())
        stimuli[c] = {'src': f'stimuli/{c}.flac', 'dur': a['dur']}
    data = {'version': 1, 'config': config, 'stimuli': stimuli, **pages}
    with open(os.path.join(out, 'lt_data.js'), 'w') as fo:
        fo.write('window.LT_DATA = ' + json.dumps(data, indent=1) + ';\n')
    key = {'version': 1, 'plan': {k: plan_doc[k] for k in ('replaced', 'conditions')}, 'items': [
        {k: v for k, v in it.items() if k != 'audio'} for it in plan_doc['items']], 'trials': key_trials,
        'clips': clips, 'normalisation': {'target_dbfs': TARGET_DBFS, 'peak_dbfs': PEAK_DBFS, 'active_db': ACTIVE_DB,
                                           'frame_s': FRAME_S, 'max_s': MAX_S, 'sr': SR}}
    with open(key_path, 'w') as fo:
        json.dump(key, fo, indent=1)
    return data, key


DEFAULT_CONFIG = {'title': 'Listening test: voice conversion', 'contact': 'the organiser',
                  'consent_html': '<p>[Consent text approved for this study goes here.]</p>', 'halves': False,
                  'require_full_play': True, 'headphone_check': {'trials': 6, 'pass': 5, 'attempts': 2},
                  'submit_url': None}


LT_PREFIX = 'window.LT_DATA = '
DRAFT_MARKERS = ('[CONTACT]', '[COMPENSATION', '[ETHICS', '[THÙ LAO', '[ĐẠO ĐỨC')   # listening/consent_v1.html


def set_config(package, consent_html, contact, title=None):
    """Replaces consent_html, contact (and title, when given) in <package>/lt_data.js; returns (data, sha256)."""
    path = os.path.join(package, 'lt_data.js')
    txt = open(path, encoding='utf-8').read().rstrip()
    if not txt.startswith(LT_PREFIX) or not txt.endswith(';'):
        raise SystemExit(f'ERROR: {path} is not an lt_data.js written by build')
    data = json.loads(txt[len(LT_PREFIX):-1])
    if not consent_html.strip() or consent_html.strip() == DEFAULT_CONFIG['consent_html'] \
            or re.search(r'<\s*script', consent_html, re.I):
        raise SystemExit('ERROR: the consent text is empty, still the placeholder, or holds a <script>')
    left = [m for m in DRAFT_MARKERS if m in consent_html]
    if left:
        raise SystemExit(f'ERROR: the consent text still has the draft fields {left}: fill them in or delete them')
    if not contact.strip():
        raise SystemExit('ERROR: the contact is empty')
    data['config'] = dict(data['config'], consent_html=consent_html.strip(), contact=contact.strip(),
                          **({'title': title.strip()} if title else {}))
    with open(path + '.tmp', 'w', encoding='utf-8') as fo:
        fo.write(LT_PREFIX + json.dumps(data, indent=1) + ';\n')
    os.replace(path + '.tmp', path)
    return data, hashlib.sha256(open(path, 'rb').read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--items', required=True)
    p.add_argument('--record', required=True, help='DEV_SELECTION_RECORD_v1.json (CRT-AL choice)')
    p.add_argument('--vctk_pairs', required=True)
    p.add_argument('--ltc_pairs', required=True)
    p.add_argument('--v1', required=True, help='$J/v1 with t1, t2 and t4')
    p.add_argument('--conditions', default=','.join(CONDITIONS + OPTIONAL))
    p.add_argument('--out', required=True)
    b = sub.add_parser('build')
    b.add_argument('--plan', required=True)
    b.add_argument('--out', required=True, help='the package folder raters receive (with the page files)')
    b.add_argument('--key', required=True, help='key.json, outside the package: what every trial and clip is')
    b.add_argument('--page_conditions', default=','.join(CONDITIONS),
                   help='conditions of the MOS and SMOS blocks (§5.4: kNN-VC and Seed-VC only after the pilot)')
    b.add_argument('--device', default='cuda')
    c = sub.add_parser('config')
    c.add_argument('--package', required=True, help='the built package folder (holds lt_data.js)')
    c.add_argument('--consent', required=True, help='HTML fragment: the approved information and consent text')
    c.add_argument('--contact', required=True)
    c.add_argument('--title', default=None)
    a = ap.parse_args(argv)
    if a.cmd == 'config':
        data, sha = set_config(a.package, open(a.consent, encoding='utf-8').read(), a.contact, a.title)
        print(f'config: consent ({len(data["config"]["consent_html"])} characters), contact {data["config"]["contact"]!r}'
              f', title {data["config"]["title"]!r} -> {os.path.join(a.package, "lt_data.js")} (sha256 {sha})')
        return 0
    if a.cmd == 'plan':
        rec = json.load(open(a.record))
        pairs = {r['pair_id']: r for f in (a.vctk_pairs, a.ltc_pairs) for r in read_jsonl(f)}
        doc = plan(json.load(open(a.items)), pairs, output_rows(a.v1, rec['selection']['choice']),
                   a.conditions.split(','))
        with open(a.out, 'w') as fo:
            json.dump(doc, fo, indent=1)
        print(f'plan: {len(doc["items"])} items, {len(doc["replaced"])} replaced, {len(doc["practice"])} practice '
              f'-> {a.out}')
        return 0
    doc = json.load(open(a.plan))
    page = a.page_conditions.split(',')
    if not set(page) <= set(doc['conditions']):
        raise SystemExit(f'ERROR: {page} not all in the plan conditions {doc["conditions"]}')
    doc = dict(doc, conditions=page)
    data, key = build(doc, a.out, a.key, MimiCopier(a.device), DEFAULT_CONFIG)
    print(f'build: {len(data["stimuli"])} clips, MOS {len(data["mos"])}, SMOS {len(data["smos"])}, '
          f'CMOS {len(data["cmos"])}, practice {len(data["practice"])} -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
