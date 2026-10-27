"""Tests of mvc/analysis/contamination_r1.py: a clean corpus passes and writes a sorted manifest with its sha256; an
evaluation speaker in the training data fails; the audit exclusions are honoured."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import contamination_r1 as cr  # noqa: E402


def make(root, subset, speakers, n=3):
    d = root / subset
    d.mkdir(parents=True)
    rows = [dict(id=f'{s}_c_{u}', speaker=s, chapter='c', subset=subset, frames=40, samples=40 * 1920, offset=0)
            for s in speakers for u in range(n)]
    (d / 'shard_0000.json').write_text(json.dumps(rows))


def corpus(tmp_path, eval_speaker='p225'):
    root = tmp_path / 'mimi'
    make(root, 'train-clean-100', ['11', '12'])
    make(root, 'train-clean-360', ['13'])
    make(root, 'train-other-500', ['14'])
    make(root, 'dev-clean', ['90'])
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(json.dumps(dict(source_speaker=eval_speaker, target_speaker='p226')) + '\n')
    return root, pairs


def test_clean_corpus_passes_and_writes_the_manifest(tmp_path):
    root, pairs = corpus(tmp_path)
    ex = tmp_path / 'exclude.txt'
    ex.write_text('11_c_0\n')
    man, js = tmp_path / 'ids.txt', tmp_path / 'c.json'
    assert cr.main(['--shards_root', str(root), '--exclude', str(ex), '--pairs', str(pairs), '--out_manifest', str(man),
                    '--out_json', str(js)]) == 0
    r = json.loads(js.read_text())
    ids = man.read_text().split()
    assert r['passed'] and ids == sorted(ids) and '11_c_0' not in ids and len(ids) == 11 and r['excluded_train'] == 1
    assert r['manifest_sha256'] == hashlib.sha256(man.read_bytes()).hexdigest() and r['train_speakers'] == 4
    assert 'contamination_r1.py' in r['command']


def test_an_evaluation_speaker_in_training_fails(tmp_path):
    root, pairs = corpus(tmp_path, eval_speaker='13')
    man, js = tmp_path / 'ids.txt', tmp_path / 'c.json'
    assert cr.main(['--shards_root', str(root), '--pairs', str(pairs), '--out_manifest', str(man),
                    '--out_json', str(js)]) == 1
    r = json.loads(js.read_text())
    assert not r['passed'] and 'evaluation speakers' in r['error'] and man.exists()
