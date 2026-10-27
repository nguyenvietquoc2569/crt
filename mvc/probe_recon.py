#!/usr/bin/env python3
"""Same-speaker reconstruction on held-out speakers (LibriTTS-R dev-clean): the in-domain proxy of PLAN.md 5.2, used
for checkpoint selection and the P6 gate (reconstruction capped WER <= 5 %).

For n utterances (seeded, spread over the speakers) the model converts the utterance with a reference made of OTHER
utterances of the same speaker (up to --ref_s seconds), so the ideal output is the input itself. Enrolment for SIM is a
third, disjoint set of up to 10 utterances of the speaker. Rows are written in the journal's native metric-row form
(pair_id, system, wav, ref_text, sim_ref_wavs, source_wav, parallel_wav=None, wer_primary=True), to be scored by
$J/metrics.py --metrics wer,sim,utmos and summarised by summarize.py.

    python probe_recon.py --shards .../mvc/mimi/dev-clean --raw .../LibriTTS_R --name rec_v1 \
        --cfg '{"run_dir": ".../runs/p6_V"}' --out .../eval/recon --n 300
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import encode_mimi as em
import encode_perturbed as ep


def plan(rows, n, ref_s, seed=0, n_enrol=10, max_s=None):
    """[(utterance row, reference rows, enrolment rows)]: references and enrolment never hold the utterance, and are
    disjoint from each other. Utterances are drawn round-robin over speakers; max_s: only utterances of at most max_s
    seconds are reconstructed (references and enrolment may be longer)."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r['speaker'], []).append(r)
    spk = sorted(s for s, v in by.items() if len(v) >= 4)
    order = {s: list(rng.permutation(len(by[s]))) for s in spk}
    out, k = [], 0
    while len(out) < n and any(order.values()):
        s = spk[k % len(spk)]
        k += 1
        if not order[s]:
            continue
        u = by[s][order[s].pop()]
        if max_s and u['samples'] / em.SR > max_s:
            continue
        others = [r for r in by[s] if r['id'] != u['id']]
        rng.shuffle(others)
        ref, tot = [], 0.0
        for r in others:
            if tot >= ref_s:
                break
            ref.append(r)
            tot += r['samples'] / em.SR
        rest = [r for r in others if r not in ref][:n_enrol]
        if not rest:
            continue
        out.append((u, ref, rest))
    return out


def row(name, u, wav, enrol, raw):
    return dict(pair_id=f"recon__{u['id']}", system=name, wav=wav, ref_text=u['text'],
                sim_ref_wavs=[ep.wav_path(raw, r) for r in enrol], source_wav=ep.wav_path(raw, u), parallel_wav=None,
                wer_primary=True, set='recon', source_speaker=u['speaker'], target_speaker=u['speaker'])


def run(adapter, items, raw, out, name, reader=em.read_wav, ref_s=20.0, log=print):
    import soundfile as sf
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    rows_path = d / 'rows.jsonl'
    done = {json.loads(l)['pair_id'] for l in open(rows_path)} if rows_path.exists() else set()
    with open(rows_path, 'a') as f:
        for i, (u, refs, enrol) in enumerate(items):
            pid = f"recon__{u['id']}"
            if pid in done:
                continue
            src = reader(ep.wav_path(raw, u))
            ref = np.concatenate([reader(ep.wav_path(raw, r)) for r in refs])[:int(ref_s * em.SR)]
            np.random.seed(int.from_bytes(pid.encode()[:4], 'little'))
            wav, sr = adapter.convert(src, ref, em.SR)
            path = str(d / f'{pid}.wav')
            tmp = path[:-4] + '.tmp.wav'
            sf.write(tmp, wav, sr, subtype='PCM_16')
            os.replace(tmp, path)
            f.write(json.dumps(row(name, u, path, enrol, raw)) + '\n')
            f.flush()
            if (i + 1) % 50 == 0:
                log(f'  {i + 1}/{len(items)}')
    return rows_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', required=True, help='vc_adapter settings JSON')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--ref_s', type=float, default=20.0)
    ap.add_argument('--max_s', type=float, default=0.0, help='reconstruct only utterances of at most this length')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    rows = []
    for js in sorted(Path(a.shards).glob('shard_*.json')):
        rows += json.load(open(js))
    items = plan(rows, a.n, a.ref_s, max_s=a.max_s or None)
    import vc_adapter
    ad = vc_adapter.MimiVC()
    ad.load(a.device, **json.loads(a.cfg))
    p = run(ad, items, a.raw, a.out, a.name, ref_s=a.ref_s, log=lambda m: print(m, flush=True))
    print(f'{len(items)} reconstructions -> {p}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
