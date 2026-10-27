#!/usr/bin/env python3
"""Contamination test of mvc_v1's training data for the selection record (ANALYSIS_PLAN_mvc_v1 §4.8; PLAN §3.1).

Rebuilds R1's training and validation corpora with the code R1 ran (data.Corpus with the audit exclusion list, the same
min_frames) from a copy of the shard metadata (shard_*.json only; Corpus reads no arrays at construction), then applies
train.check_no_test_data and train.check_speakers against the evaluation pair lists, exactly as train.train does at
start. Writes the training ID manifest (one utterance ID per line, sorted) and a JSON
{"passed", "manifest", "manifest_sha256", "command", counts, "error"} for select_mvc.py --contamination.

    python mvc/analysis/contamination_r1.py --shards_root MR/r1_meta/mimi --exclude MR/r1_meta/exclude.txt \
        --pairs MR/r1_meta/pairs/dev.jsonl,MR/r1_meta/pairs/vctk2k.jsonl,MR/r1_meta/pairs/ltc2k.jsonl \
        --out_manifest MR/r1_meta/r1_train_ids.txt --out_json MR/r1_meta/contamination.json
"""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data as dt  # noqa: E402
import train as tr  # noqa: E402

TRAIN = ('train-clean-100', 'train-clean-360', 'train-other-500')
VAL = ('dev-clean',)


def run(shards_root, exclude, pairs, out_manifest, min_frames=25):
    root = Path(shards_root)
    excl = set(Path(exclude).read_text().split()) if exclude else set()
    tdirs, vdirs = [str(root / s) for s in TRAIN], [str(root / s) for s in VAL]
    res = dict(passed=False, error=None)
    try:
        tr.check_no_test_data(tdirs + vdirs)
        trc = dt.Corpus(tdirs, min_frames=min_frames, exclude=excl)
        vac = dt.Corpus(vdirs, min_frames=min_frames, exclude=excl)
        msgs = []
        tr.check_speakers(trc, vac, pairs, log=msgs.append)
        res.update(passed=True, log=msgs)
    except SystemExit as e:                               # the checks refuse by SystemExit, as at R1's start
        res['error'] = str(e)
        trc = dt.Corpus(tdirs, min_frames=min_frames, exclude=excl)
        vac = dt.Corpus(vdirs, min_frames=min_frames, exclude=excl)
    ids = sorted(u['id'] for u in trc.utts)
    Path(out_manifest).write_text('\n'.join(ids) + '\n')
    res.update(manifest=str(out_manifest), manifest_sha256=hashlib.sha256(Path(out_manifest).read_bytes()).hexdigest(),
               train_utts=len(trc.utts), train_speakers=len({u['speaker'] for u in trc.utts}),
               train_hours=round(trc.hours(), 1),
               val_utts=len(vac.utts), val_speakers=len({u['speaker'] for u in vac.utts}),
               excluded_train=trc.n_excluded, excluded_val=vac.n_excluded)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--shards_root', required=True, help='folder holding train-clean-100/360, train-other-500, dev-clean')
    ap.add_argument('--exclude', default='', help="the audit exclusion list R1 used (results/audit_v1/exclude.txt)")
    ap.add_argument('--pairs', required=True, help='comma list of the evaluation pair lists (dev, vctk2k, ltc2k)')
    ap.add_argument('--out_manifest', required=True)
    ap.add_argument('--out_json', required=True)
    a = ap.parse_args(argv)
    res = run(a.shards_root, a.exclude, a.pairs, a.out_manifest)
    cmd = sys.argv if argv is None else [__file__, *argv]
    res['command'] = ' '.join(shlex.quote(x) for x in [sys.executable, *cmd])
    Path(a.out_json).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'log'}), flush=True)
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
