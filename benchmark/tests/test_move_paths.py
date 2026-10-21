"""Tests of benchmark/move_paths.py, the path move after the CRT tree was copied from nvquoc to ceai (DEVIATIONS_v1.md
item 4): which files and links `rewrite` changes and which it leaves (binary files, job logs, --keep folders), its
all-or-nothing refusal, the log, `check` (logged hashes, the frozen section-2 hashes mapped back, leftovers) and
`compare` of metrics rows. Trees are built under tmp_path."""
import hashlib
import json
import math
import os

import pytest

import move_paths

OLD, NEW = '/datastore/nvquoc/crt', '/datastore/ceai/crt'
J = 'mimi_vc/benchmark/journal'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def put(root, rel, data, mode=0o644):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    p.chmod(mode)
    return p


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'crt'
    pairs = ''.join(json.dumps({'pair_id': f'p{i}', 'source_path': f'{OLD}/mimi_vc/benchmark/vctk/p{i}.flac'}) + '\n'
                    for i in range(3))
    put(root, f'{J}/pairs/dev.jsonl', pairs)
    put(root, f'{J}/job_m.slurm', f'#!/bin/bash\n#SBATCH --output={OLD}/mimi_vc/benchmark/logs/m_%j.out\n'
                                  f'source {OLD}/envs/mimi_vc/bin/activate\n', 0o755)
    put(root, 'envs/mimi_vc/bin/activate', f"VIRTUAL_ENV='{OLD}/envs/mimi_vc'\n")
    put(root, 'envs/mimi_vc/bin/pip', f'#!{OLD}/envs/mimi_vc/bin/python3.12\nimport pip\n', 0o755)
    put(root, f'{J}/models/w.pt', b'\0PK' + OLD.encode() + b'\0')                     # binary suffix: not read
    put(root, f'{J}/v1/t1/rc/dump.bin2', b'\0\0' + OLD.encode())                    # NUL in the head: binary
    put(root, 'mimi_vc/benchmark/logs/crt_t3_1.out', f'wrote {OLD}/x\n')             # job log
    put(root, f'{J}/v1/t1/run.log', f'{OLD}/y\n')                                    # job log
    put(root, '_transfer_logs/submit.log', f'{OLD}/job.slurm\n')                     # --keep folder
    put(root, f'{J}/notes.md', 'no paths here\n')
    put(root, f'{J}/other.txt', '/datastore/nvquoc/qwen stays\n')                  # another folder of nvquoc
    os.symlink(f'{OLD}/mimi_vc/data_storage/cache', root / 'mimi_vc/cache_link')     # absolute link into the tree
    os.symlink('../x', root / 'envs/rel_link')                                         # relative link
    os.symlink('/usr/bin/python3', root / 'envs/mimi_vc/bin/python')
    (root / 'mimi_vc/data_storage/cache').mkdir(parents=True)
    os.symlink(f'{OLD}/mimi_vc/data_storage/cache', root / 'mimi_vc/dir_link')        # link to a folder
    return root


def run(*argv):
    return move_paths.main(list(argv))


def rewrite(root, log, *extra):
    return run('rewrite', '--root', str(root), '--old', OLD, '--new', NEW, '--log', str(log),
               '--keep', '_transfer_logs', *extra)


def check(root, *extra):
    return run('check', '--root', str(root), '--old', OLD, '--new', NEW, '--keep', '_transfer_logs', *extra)


def log_rows(log):
    return {line.split('\t')[0]: line.rstrip('\n').split('\t')[1:] for line in open(log)}


# ── rewrite ──────────────────────────────────────────────────

def test_rewrite_changes_text_files_and_links_and_logs_each(tree, tmp_path):
    before = {r: (tree / r).read_bytes() for r in (f'{J}/pairs/dev.jsonl', f'{J}/job_m.slurm',
                                                    'envs/mimi_vc/bin/activate', 'envs/mimi_vc/bin/pip')}
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    rows = log_rows(log)
    assert set(rows) == set(before) | {'mimi_vc/cache_link', 'mimi_vc/dir_link'}
    for rel, data in before.items():
        now = (tree / rel).read_bytes()
        assert now == data.replace(OLD.encode(), NEW.encode()) and OLD.encode() not in now
        kind, count, b, a = rows[rel]
        assert (kind, int(count), b, a) == ('file', data.count(OLD.encode()), sha(data), sha(now))
    assert rows[f'{J}/pairs/dev.jsonl'][1] == '3' and rows[f'{J}/job_m.slurm'][1] == '2'
    assert os.readlink(tree / 'mimi_vc/cache_link') == f'{NEW}/mimi_vc/data_storage/cache'
    assert rows['mimi_vc/cache_link'] == ['link', '1', f'{OLD}/mimi_vc/data_storage/cache',
                                          f'{NEW}/mimi_vc/data_storage/cache']
    assert os.readlink(tree / 'mimi_vc/dir_link') == f'{NEW}/mimi_vc/data_storage/cache'


def test_rewrite_leaves_binary_files_logs_keep_folders_and_other_paths(tree, tmp_path):
    keep = {r: (tree / r).read_bytes() for r in (f'{J}/models/w.pt', f'{J}/v1/t1/rc/dump.bin2',
                                                  'mimi_vc/benchmark/logs/crt_t3_1.out', f'{J}/v1/t1/run.log',
                                                  '_transfer_logs/submit.log', f'{J}/notes.md', f'{J}/other.txt')}
    assert rewrite(tree, tmp_path / 'move.tsv') == 0
    for rel, data in keep.items():
        assert (tree / rel).read_bytes() == data, rel
    assert os.readlink(tree / 'envs/rel_link') == '../x'
    assert os.readlink(tree / 'envs/mimi_vc/bin/python') == '/usr/bin/python3'


def test_rewrite_keeps_the_mode_and_leaves_no_temporary_files(tree, tmp_path):
    assert rewrite(tree, tmp_path / 'move.tsv') == 0
    assert (tree / f'{J}/job_m.slurm').stat().st_mode & 0o777 == 0o755
    assert (tree / f'{J}/pairs/dev.jsonl').stat().st_mode & 0o777 == 0o644
    left = [p for p in tree.rglob('*') if p.name.startswith('.move_paths.')]
    assert left == []


def test_rewrite_refuses_everything_when_a_file_already_holds_the_new_prefix(tree, tmp_path):
    put(tree, f'{J}/mixed.json', json.dumps({'a': f'{OLD}/x', 'b': f'{NEW}/y'}))
    snapshot = {p: p.read_bytes() for p in tree.rglob('*') if p.is_file() and not p.is_symlink()}
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 1
    assert not log.exists()
    assert {p: p.read_bytes() for p in snapshot} == snapshot
    assert os.readlink(tree / 'mimi_vc/cache_link').startswith(OLD)


def test_rewrite_refuses_an_existing_log(tree, tmp_path):
    log = tmp_path / 'move.tsv'
    log.write_text('x\n')
    with pytest.raises(SystemExit):
        rewrite(tree, log)


def test_second_rewrite_finds_nothing_to_change(tree, tmp_path):
    assert rewrite(tree, tmp_path / 'a.tsv') == 0
    assert rewrite(tree, tmp_path / 'b.tsv') == 0
    assert (tmp_path / 'b.tsv').read_text() == ''


def test_old_and_new_must_be_distinct_absolute_folders(tree, tmp_path):
    for old, new in ((OLD, OLD + '2'), ('datastore/nvquoc/crt', NEW), (OLD, NEW + '/'), ('/a/b', '/a/b/c')):
        with pytest.raises(SystemExit):
            run('rewrite', '--root', str(tree), '--old', old, '--new', new, '--log', str(tmp_path / 'x.tsv'))


# ── check ────────────────────────────────────────────────────

def freeze(tmp_path, entries):
    p = tmp_path / 'FREEZE_v1.sha256'
    p.write_text('# Section 1, the Mac\n' + 'a' * 64 + '  benchmark/ANALYSIS_PLAN_v1.md\n'
                 + '# Section 2, the HPC\n' + ''.join(f'{h}  {r}\n' for h, r in entries)
                 + '# Section 3\n# CRT weights state-dict sha256: ' + 'b' * 64 + '\n')
    return p


def test_check_passes_after_a_rewrite_with_frozen_files_mapped_back(tree, tmp_path):
    put(tree, 'mimi_vc/data_storage/cache/whisper/base.pt', b'\0weights')
    frozen = freeze(tmp_path, [(sha((tree / f'{J}/pairs/dev.jsonl').read_bytes()), 'J/pairs/dev.jsonl'),
                               (sha((tree / f'{J}/models/w.pt').read_bytes()), 'J/models/w.pt'),
                               (sha(b'\0weights'), 'W/data_storage/cache/whisper/base.pt')])
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    assert f'{J}/pairs/dev.jsonl' in log_rows(log)                    # the frozen pair list was rewritten
    assert check(tree, '--log', str(log), '--freeze', str(frozen)) == 0


def test_check_fails_when_a_frozen_file_differs_or_is_missing(tree, tmp_path, capsys):
    log = tmp_path / 'move.tsv'
    data = (tree / f'{J}/pairs/dev.jsonl').read_bytes()
    frozen = freeze(tmp_path, [(sha(data + b'x'), 'J/pairs/dev.jsonl'), (sha(b''), 'J/pairs/missing.jsonl')])
    assert rewrite(tree, log) == 0
    assert check(tree, '--log', str(log), '--freeze', str(frozen)) == 1
    out = capsys.readouterr().out
    assert 'frozen J/pairs/dev.jsonl: sha256' in out and 'frozen J/pairs/missing.jsonl: missing' in out


def test_check_fails_when_a_rewritten_file_changed_later(tree, tmp_path, capsys):
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    (tree / f'{J}/job_m.slurm').write_text('#!/bin/bash\necho changed\n')
    assert check(tree, '--log', str(log)) == 1
    assert f'{J}/job_m.slurm: missing or changed since the rewrite' in capsys.readouterr().out


def test_check_fails_when_mapping_back_does_not_give_the_logged_hash(tree, tmp_path, capsys):
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    rows = log.read_text().splitlines()
    i = next(k for k, r in enumerate(rows) if r.startswith(f'{J}/pairs/dev.jsonl\t'))
    f = rows[i].split('\t')
    rows[i] = '\t'.join(f[:3] + ['0' * 64, f[4]])
    log.write_text('\n'.join(rows) + '\n')
    assert check(tree, '--log', str(log)) == 1
    assert 'mapped back, it does not have its sha256 from before' in capsys.readouterr().out


def test_check_fails_on_a_leftover_old_path_or_a_moved_link(tree, tmp_path, capsys):
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    put(tree, f'{J}/late.json', json.dumps({'wav': f'{OLD}/z.wav'}))
    os.remove(tree / 'mimi_vc/cache_link')
    os.symlink('/elsewhere', tree / 'mimi_vc/cache_link')
    assert check(tree, '--log', str(log)) == 1
    out = capsys.readouterr().out
    assert f'{J}/late.json still holds {OLD}' in out and 'link mimi_vc/cache_link does not point to' in out


def test_check_ignores_job_logs_and_keep_folders(tree, tmp_path):
    log = tmp_path / 'move.tsv'
    assert rewrite(tree, log) == 0
    assert check(tree, '--log', str(log)) == 0           # crt_t3_1.out, run.log and _transfer_logs still hold OLD


def test_check_reads_several_logs(tree, tmp_path, capsys):
    first, second = tmp_path / 'a.tsv', tmp_path / 'b.tsv'
    assert rewrite(tree, first, '--keep', 'envs') == 0
    assert rewrite(tree, second) == 0
    assert set(log_rows(second)) == {'envs/mimi_vc/bin/activate', 'envs/mimi_vc/bin/pip'}
    capsys.readouterr()
    assert check(tree, '--log', str(first), '--log', str(second)) == 0
    assert f'{len(log_rows(first)) + 2} logged changes' in capsys.readouterr().out


def test_walk_lists_links_to_folders_without_entering_them(tree):
    put(tree, 'data/real/a.json', f'{OLD}/a')
    os.symlink(str(tree / 'data/real'), tree / 'data/alias')
    got = list(move_paths.walk(str(tree / 'data')))
    assert sorted(got) == [('alias', 'link'), ('real/a.json', 'file')]     # a.json once: alias is not entered


# ── compare ──────────────────────────────────────────────────

def mrow(pid, system='faiss', **kw):
    r = {'pair_id': pid, 'system': system, 'wav': f'{NEW}/{pid}.wav', 'mcd_db': 8.5, 'sim_wavlm': 0.91,
         'utmos': 3.2, 'wer_primary': True, 'mcd_frames': 700,
         'wer_large_v3': {'hyp': 'a b c', 'S': 1, 'D': 0, 'I': 0, 'N': 3}, 'time_s': {'utmos': 0.01},
         'cfg': {'npow_db': -20.0, 'f0_method': 'harvest'}}
    r.update(kw)
    return r


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def test_compare_passes_identical_rows_and_counts_hypotheses(tmp_path, capsys):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1'), mrow('p2'), mrow('p3', 'anc_x')])
    new = write(tmp_path / 'new.jsonl', [mrow('p1'), mrow('p3', 'anc_x')])
    assert run('compare', '--new', new, '--ref', ref) == 0
    out = capsys.readouterr().out
    assert '2 rows, 0 without a reference row, 0 error rows; identical Whisper hypotheses 2/2' in out


def test_compare_fails_on_a_metric_over_the_tolerance(tmp_path, capsys):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1')])
    new = write(tmp_path / 'new.jsonl', [mrow('p1', sim_wavlm=0.9102)])
    assert run('compare', '--new', new, '--ref', ref) == 1
    assert 'max |diff| sim_wavlm: 0.0002 OVER' in capsys.readouterr().out
    assert run('compare', '--new', new, '--ref', ref, '--tol', '0.001') == 0


def test_compare_reports_whisper_and_timing_differences_only(tmp_path, capsys):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1')])
    new = write(tmp_path / 'new.jsonl', [mrow('p1', wer_large_v3={'hyp': 'a b d', 'S': 2, 'D': 0, 'I': 0, 'N': 3},
                                              time_s={'utmos': 5.0})])
    assert run('compare', '--new', new, '--ref', ref) == 0
    out = capsys.readouterr().out
    assert 'identical Whisper hypotheses 0/1' in out and 'wer_large_v3.S: 1 (reported only)' in out


def test_compare_treats_nan_against_a_number_as_a_difference(tmp_path, capsys):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1', mcd_db=float('nan')), mrow('p2', mcd_db=float('nan'))])
    new = write(tmp_path / 'new.jsonl', [mrow('p1', mcd_db=float('nan')), mrow('p2', mcd_db=8.5)])
    assert run('compare', '--new', new, '--ref', ref) == 1
    assert 'max |diff| mcd_db: inf OVER' in capsys.readouterr().out


def test_compare_fails_on_missing_reference_error_rows_or_no_rows(tmp_path):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1')])
    assert run('compare', '--new', write(tmp_path / 'a.jsonl', [mrow('p9')]), '--ref', ref) == 1
    assert run('compare', '--new', write(tmp_path / 'b.jsonl', [mrow('p1', error='boom')]), '--ref', ref) == 1
    assert run('compare', '--new', write(tmp_path / 'c.jsonl', []), '--ref', ref) == 1


def test_compare_uses_the_last_reference_row_of_a_pair(tmp_path):
    ref = write(tmp_path / 'ref.jsonl', [mrow('p1', utmos=1.0), mrow('p1', utmos=3.2)])
    assert run('compare', '--new', write(tmp_path / 'n.jsonl', [mrow('p1')]), '--ref', ref) == 0


def test_numbers_flattens_dicts_and_drops_bools():
    got = move_paths.numbers({'a': 1, 'b': True, 'c': {'d': 2.5, 'e': 'x', 'f': {'g': 3}}, 'h': 'y'})
    assert got == {'a': 1.0, 'c.d': 2.5, 'c.f.g': 3.0}
    assert math.isnan(move_paths.numbers({'x': float('nan')})['x'])


def test_keep_takes_single_files_too(tree, tmp_path):
    both = put(tree, f'{J}/job_move_ceai.slurm', f'OLD={OLD}\nNEW={NEW}\n')
    assert rewrite(tree, tmp_path / 'a.tsv') == 1                      # names both folders: everything refused
    assert rewrite(tree, tmp_path / 'b.tsv', '--keep', f'{J}/job_move_ceai.slurm') == 0
    assert both.read_text() == f'OLD={OLD}\nNEW={NEW}\n'
    assert check(tree, '--log', str(tmp_path / 'b.tsv'), '--keep', f'{J}/job_move_ceai.slurm') == 0
    assert check(tree, '--log', str(tmp_path / 'b.tsv')) == 1          # without --keep it counts as a leftover


def test_rewrite_changes_site_packages_path_files_but_not_pth_checkpoints(tmp_path):
    """A venv's .pth path file is text (the seedvc env's mimi_vc_base.pth names the mimi_vc site-packages); a torch
    checkpoint named .pth is binary and holds a NUL byte in its first 8 KiB."""
    root = tmp_path / 'crt'
    sp = 'envs/seedvc/lib/python3.12/site-packages'
    put(root, f'{sp}/mimi_vc_base.pth', f'{OLD}/envs/mimi_vc/lib/python3.12/site-packages\n')
    ckpt = b'PK\x03\x04\x14\x00\x00\x00' + OLD.encode() + b'\x00' * 16
    put(root, 'mimi_vc/benchmark/FreeVC/checkpoints/freevc.pth', ckpt)
    log = tmp_path / 'move.tsv'
    move_paths.rewrite(str(root), OLD, NEW, str(log))
    assert (root / sp / 'mimi_vc_base.pth').read_text() == f'{NEW}/envs/mimi_vc/lib/python3.12/site-packages\n'
    assert (root / 'mimi_vc/benchmark/FreeVC/checkpoints/freevc.pth').read_bytes() == ckpt
    assert move_paths.check(str(root), OLD, NEW, [str(log)]) == 0
