# Runbook v1: the CRT journal jobs on the UIT HPC (account nvquoc)

The job layer of `ANALYSIS_PLAN_v1.md` §8.1 items 8–9 and §8.2. `J=/datastore/nvquoc/crt/mimi_vc/benchmark/journal`. Every job writes only into its own new directory under `$J/v1/`, and every job is submitted with `submit_nvquoc.sh`, inside tmux on the login node, as `nvquoc`.

## Rules for the account

- Fewer than 3 jobs of `nvquoc`, of any name, are running or pending when a job is submitted. The account runs at most 3, and the Qwen service that shared it until 18 Oct 2026 has moved off.
- A job is submitted only after the jobs whose outputs it reads have finished, by hand or with `chain` (below). Never use `--dependency`, because a dependent job would sit pending.
- No submission between 03:00 and 04:10 Asia/Ho_Chi_Minh. Until 18 Oct 2026 a 04:00 cron submitted the next `qwen-serve`; the window is kept in case it returns.
- At least 3660 s must pass after the last Submit of any job of `nvquoc`, `qwen-serve` included.
- Never cancel or touch a `qwen-serve` job.
- Tell the interpreter session before each submission.

`submit_nvquoc.sh <script> [sbatch args]` enforces the first three rules. It sleeps until they hold, then submits and appends `time, job id, Submit, script, args` to `/datastore/nvquoc/crt/_transfer_logs/submit.log`. It refuses to submit if `squeue` or `sacct` fails, if `date` has no zone data for Asia/Ho_Chi_Minh, if an extra argument is not a `--name=value` option, or if the script breaks the job rules (`--account=uit --partition=normal --gres=gpu:2g.35gb:1`, `--time` ≤ 72 h, arrays only with `%1`). It unsets `SBATCH_*` variables first, because sbatch lets them override the `#SBATCH` lines.

A job removes its own marker once its own prerequisite checks pass, and writes it again only when it succeeds. The next job refuses to start without that marker:

| Job | Starts only with | Writes |
|---|---|---|
| D0 | – | `d0/check_d0.json` |
| D1 conv | `d0/check_d0.json` with `passed` true | `d1/conv.done` |
| D1 metrics | `d1/conv.done` | `d1/metrics.done` |
| D2 | `d1/conv.done` | `d2/d2.done` |
| T1 | `selection.env` | `t1/t1.done` |
| T2 | `t1.done` | `t2/t2.done` |
| T3 | `t2.done` | `t3/t3.done` |
| T4 | `t3.done` | `t4/t4.done` |
| M | `t4.done`, or `t3.done` when no baseline is listed | `m/<part>/m.done` |

From T2 onwards, the marker must also have been written under the same `selection.env` (its sha256 is stored in the marker).

All conversion and metrics steps resume, and `crt_decode_lab.py` reruns whole, so resubmitting the same script after a failure or a time-out is safe as long as no script has changed. After a fix to any script (logged in `DEVIATIONS_v1.md`), move the job's directory aside before resubmitting, e.g. `mv $J/v1/d0 $J/v1/d0_try1`: `run_conversions.py` and `crt_anchor_lab.py` resume under a configuration hash that does not cover their own code, and `metrics.py` reuses a row whose input fields are unchanged even when the WAV behind it was rewritten, so a resumed run would mix the old outputs with the new ones.

## 0. Once, before D0 (login node)

Set `J=/datastore/nvquoc/crt/mimi_vc/benchmark/journal` in the shell first; the commands below and `chain` use it.

1. Push the frozen-to-be files from the Mac into `$J`:
   - `check_d0.py`, `lab_rows.py`, `make_pairs.py`, `run_conversions.py`, `crt_anchor_lab.py`, `crt_decode_lab.py`, `metrics.py`, `codeword_clustering.py`, `run_baselines.py`;
   - the `baselines/` package;
   - `submit_nvquoc.sh` and `job_*.slurm`.
2. Create the directories: `mkdir -p /datastore/nvquoc/crt/mimi_vc/benchmark/logs $J/logs $J/v1 /datastore/nvquoc/crt/_transfer_logs`. The v1 jobs log to `benchmark/logs`; `job_baselines_smoke.slurm` logs to `$J/logs`.
3. Rewrite the pair-list paths. The lists were relayed byte for byte from `ceai` and still point to `/datastore/ceai/nvquoc`, which `nvquoc` cannot read. D0 refuses to start until this is done. Keep the old bytes and both hashes:
   ```
   cd $J && mkdir -p pairs_pre_nvquoc && cp -p pairs/{dev,vctk2k,ltc2k}.jsonl pairs_pre_nvquoc/
   sha256sum pairs/{dev,vctk2k,ltc2k}.jsonl > pairs_pre_nvquoc/sha256_before.txt
   sed -i 's#/datastore/ceai/nvquoc/#/datastore/nvquoc/crt/#g' pairs/{dev,vctk2k,ltc2k}.jsonl
   grep -c -e /datastore/ceai -e /datastore/eaic pairs/{dev,vctk2k,ltc2k}.jsonl     # 0 for each
   sha256sum pairs/{dev,vctk2k,ltc2k}.jsonl > pairs_pre_nvquoc/sha256_after.txt
   ls -l $(grep -o '/datastore/nvquoc/crt/[^"]*\.flac' pairs/dev.jsonl | head -3)   # readable
   ```
4. Check the screen outputs that D0 is compared with. Each directory should hold 50 files, and both metrics files must exist:
   ```
   for d in clustering_check/dumps/dev/30 anchor_lab/dumps/dev/30 clustering_check/lab_r3a/{pm_rq_T1,pm18_rq,gv1_pm_cont_T1} \
            clustering_check/lab_r4p/{gv1_shrink0.85_cont,gv0.5_shrink0.75_cont}; do echo "$d $(ls $J/$d | wc -l)"; done
   ls -l $J/clustering_check/lab_r3_metrics.jsonl $J/clustering_check/lab_r4_metrics.jsonl
   ```
5. Check the Whisper checkpoints: `ls -l /datastore/nvquoc/crt/mimi_vc/data_storage/cache/whisper/` must list `large-v3.pt` (D0, D1, D2 and M) and `base.pt` (job M only, §3.2). `metrics.py` stops at once when a checkpoint is missing or its sha256 differs, and for Whisper-base that would first show in job M.
6. The author commits v0 (§8.4 step 1).

## 1. D0 (`job_d0.slurm`, about 2–3 h, before the freeze)

Submit with `tmux new -s crt; cd $J; bash submit_nvquoc.sh job_d0.slurm`. Output: `$J/v1/d0`, log `/datastore/nvquoc/crt/mimi_vc/benchmark/logs/crt_d0_<id>.out`.

Check:
- `check_d0.json`: `passed` is true. `summary` gives one line per check:
  - S1/S3 codes against `clustering_check`;
  - A1–A5 and C1 `out_codes` against job 2634, for `anc/` and for the test-set mode `anc_r1r3/`;
  - C1–C5 WAVs against `lab_r3a` and `lab_r4p`, either identical or within the metrics tolerance (`checks.wavs.<C>.metrics_check`, with the largest sample difference);
  - the pair lists.
- `make_pairs.log`:
  - the source-E provenance line `[vctk2k] source E: ... (sha256 …; list sha256 …)`, which `pairs/summary.json` does not record, so keep this log;
  - the 10-target run (4,000 pairs);
  - two `check: OK`;
  - the sha256 of every list.
- `check_d0.json` `info`:
  - `timings` gives seconds per pair and h per 1,000 pairs for each step. These replace the §8.2 estimates. `crt_decode_lab.py` cannot resume, so a job that times out repeats its decode calls whole. D2 must fit (7a + 7b + 3) × 20 plus about 4 h of scoring, and T1–T3 their decode calls scaled to their pairs (7a / 5 per candidate and 7b / 3 per front system, per 50 pairs), within the 72 h limit with a wide margin; otherwise sharding is added to `crt_decode_lab.py` before the freeze (§8.2). D1 conv resumes, so (2 + 4) × 20 only has to fit its 48 h per submission.
  - `sizes` gives MB per pair per array group (`R3_logits`, `crt_logits`, the other arrays per recipe), for the storage of D1 (× 1,000) and of T1–T3 (§8.2).
- The log's metrics lines show no error rows.
- `rows/conv_metrics.jsonl`, the Mimi copies (no screen counterpart, not in `check_d0.json`): `mimi_copy_P`'s dB-MCD against P is small, and both copies have about the duration and UTMOS of `source_copy` / `target_copy`.

The make_pairs files are made only if absent. To rebuild `pairs/vctk2k_source_E.json` or `pairs_vctk2k_t10/`, delete them first.

## 2. Freeze (the author, §8.4)

Steps 2–7 of §8.4: approve the plan, write `FREEZE_v1.sha256`, commit, tag, and make the public timestamp. On the login node:
- `cd $J && sha256sum pairs/{dev,vctk2k,ltc2k}.jsonl pairs/summary.json pairs/vctk2k_source_E.json pairs_vctk2k_t10/{vctk2k.jsonl,vctk2k_source_E.json,summary.json} models/wavlm-base-plus-sv/*`;
- the UTMOS checkpoint under `$WORK_DIR/data_storage/torch_cache/hub`;
- `/datastore/nvquoc/crt/envs/mimi_vc/bin/pip freeze`.

Before D1, check the `$J` copies of the frozen scripts against `FREEZE_v1.sha256`, and run `grep -l -e /datastore/eaic -e /datastore/ceai` over the frozen list, which must print nothing (§8.2). `pairs/summary.json`, which §8.4 step 5 hashes, holds 11 `/datastore/eaic` strings in `args` and the `jsonl` fields (metadata only; no job reads a path from it), so it will be printed: the author decides before the freeze whether it leaves the grep list or the exception goes into `DEVIATIONS_v1.md`.

## 3. D1 (after the public timestamp)

1. `bash submit_nvquoc.sh job_d1_conv.slurm` (1,000 Dev pairs; FAISS with logits; anchor lab with 36 systems and R3 logits; summarise-only). Then check:
   - `d1/conv.done` exists;
   - the log shows `0 errors` for both conversion scripts;
   - `d1/repro_d1_vs_d0.json` has `passed` true. A mismatch is reported (§4.4) and does not block the metrics job.
2. `bash submit_nvquoc.sh job_d1_metrics.slurm` (about 12 h). Then check that `d1/metrics.done` exists and that the row counts are 1,000 per system with no error rows.

D1 has two jobs so that each gets its own time limit (48 h; both resume), and so that conversion errors and the screen reproduction are seen before 12 h of scoring. The reasoning is in `job_d1_conv.slurm`. D1's job ID for §8.4 step 8 is the conversion job.

## 4. D2

Submit with `bash submit_nvquoc.sh job_d2.slurm`. It runs:
- the decode lab: `cand` for C1–C5 and `front` for S1u and E17, each checked for 1,000 pairs before it is scored;
- the Mimi copies;
- the natural controls;
- `codeword_clustering.py`;
- the metrics.

Check:
- `d2/d2.done` exists;
- `d2/repro_d2_vs_d0.json` has `passed` true;
- `d2/codeword_clustering.md`;
- the counts: 1,000 per system, 500 for `mimi_copy_P` and `target_copy`.

D2 reads only the D1 conversions, so it can run alongside the D1 metrics: up to 3 jobs of `nvquoc` may be queued. To run D1 and D2 unattended, use `chain` for the conversions, then queue both. `chain` submits each job after the previous one has left the queue and written its marker, and stops otherwise. The second `submit_nvquoc.sh` waits for the first one's lock and then for the 1 h gap, so D2 starts about an hour after the D1 metrics:
```
chain() {  # chain '<script> [sbatch args]|<marker under $J/v1>' ...
    local x
    for x in "$@"; do
        bash $J/submit_nvquoc.sh $J/${x%%|*} || return 1
        sleep 120
        while squeue -h -u nvquoc -o %j | grep -qv '^qwen-serve$'; do sleep 600; done
        [ -f $J/v1/${x#*|} ] || { echo "no $J/v1/${x#*|}: stopping after ${x%%|*}"; return 1; }
    done
}
chain 'job_d1_conv.slurm|d1/conv.done' && { bash $J/submit_nvquoc.sh $J/job_d1_metrics.slurm; bash $J/submit_nvquoc.sh $J/job_d2.slurm; }
```

## 5. Back to the Mac: the selection record

1. On the login node, run `sacct -u nvquoc -S 2026-10-18 -P -X -o JobID,JobName,Submit,End,State > $J/v1/sacct_nvquoc.txt`.
2. Job 2634 ran under `ceai`. From the Mac, run `ssh <old-account>@<hpc-login-host> "sacct -j 2634 -P -X -o JobID,JobName,Submit,End,State" > sacct_2634.txt`.
3. Copy these files into `V=paper/journal_extension/results/v1`, keeping each list next to its `summary.json` and `vctk2k_source_E.json`, because `select_crt_al.py` finds those two files next to the list:

| HPC (`$J/...`) | Mac (`$V/...`) |
|---|---|
| `pairs/{dev,vctk2k,ltc2k}.jsonl`, `pairs/summary.json`, `pairs/vctk2k_source_E.json` | `pairs/` |
| `pairs_vctk2k_t10/{vctk2k.jsonl,summary.json,vctk2k_source_E.json}` | `pairs_vctk2k_t10/` |
| `v1/d0/check_d0.json` | `d0/check_d0.json` |
| `v1/d1/anc/dev/30/metrics.jsonl`, `metrics_src.jsonl` (about 0.9 GB together) | `d1/` |
| `v1/d1/repro_d1_vs_d0.json`, `v1/d1/anc/dev/30/anchor_summary.json` | `d1/` |
| `v1/d2/lab/metrics.jsonl`, `metrics_src.jsonl` | `d2/lab_metrics.jsonl`, `d2/lab_metrics_src.jsonl` |
| `v1/d2/controls/metrics.jsonl`, `metrics_src.jsonl` | `d2/controls_metrics.jsonl`, `d2/controls_metrics_src.jsonl` |
| `v1/d2/repro_d2_vs_d0.json`, `v1/d2/codeword_clustering.{json,md}`, `v1/d2/lab/{cand,front}/lab_summary.json` | `d2/` (the `lab_summary.json` files renamed by tag) |
| `v1/sacct_nvquoc.txt` | `sacct_nvquoc.txt` |

4. Write `$V/baselines.json` from the smoke test's `$J/v1/baselines_smoke/baselines_record.json` (`job_baselines_smoke.slurm`, §8.1 item 10, outside this job layer). It is one entry per baseline that will run, with the keys `id` (S5–S8), `system`, `repo_commit`, `checkpoint_sha256`, `sample_rate` and `command`.
5. Run the selection:
   ```
   cd /Users/harry/duplex
   .venv-stats/bin/python benchmark/select_crt_al.py --pairs $V/pairs/dev.jsonl \
     --metrics $V/d1/metrics.jsonl,$V/d1/metrics_src.jsonl,$V/d2/lab_metrics.jsonl,$V/d2/lab_metrics_src.jsonl,$V/d2/controls_metrics.jsonl,$V/d2/controls_metrics_src.jsonl \
     --vctk_pairs $V/pairs/vctk2k.jsonl --vctk10_pairs $V/pairs_vctk2k_t10/vctk2k.jsonl --ltc_pairs $V/pairs/ltc2k.jsonl \
     --repro $V/d0/check_d0.json,$V/d1/repro_d1_vs_d0.json,$V/d2/repro_d2_vs_d0.json \
     --sacct $V/sacct_nvquoc.txt,$V/sacct_2634.txt --jobs 2634=2634,D0=<id>,D1=<d1_conv id>,D1m=<d1_metrics id>,D2=<id> \
     --baselines $V/baselines.json --out benchmark/DEV_SELECTION_RECORD_v1.md
   ```
   The status must be `complete`. `STOPPED` means a reproduction file failed (§4.4). The author commits the record and timestamps it before T1 (§4.6, §8.4 step 8).

## 6. `selection.env`

These are the values that only the record fixes. Write them on the Mac from the committed record and copy the file to `$J/v1/selection.env`. Do not change it once T1 has started: T2–M compare its sha256 with the markers.
```
.venv-stats/bin/python - > selection.env <<'PY'
import hashlib, json
md = 'benchmark/DEV_SELECTION_RECORD_v1.md'
d = json.load(open('benchmark/DEV_SELECTION_RECORD_v1.json'))
assert d['status'] == 'complete', d['status']
bl = {'S5': 'knnvc', 'S6': 'freevc', 'S7': 'seedvc', 'S8': 'facodec'}
print(f"RECORD_SHA256={hashlib.sha256(open(md, 'rb').read()).hexdigest()}")
print(f"TAU_DEV={d['tau_rule']['tau']:g}")                       # S2 τ_dev; 0.4 = S3, left out of --systems
print(f"CRT_AL={d['selection']['choice'] or 'none'}")             # C1..C5, none under fallback F3
print(f"S1U={d['s1u']['setting'].replace('front_', '', 1)}")       # us32_w0.2_0.2_0.1 | us32_w0.5_0.3_0.3
print(f"E17={'yes' if d['e17']['enters'] else 'no'}")
print(f"VCTK_TARGETS={d['size']['rebuild']['use']}")              # 5 | 10
print(f"BASELINES={','.join(bl[b['id']] for b in d['baselines'] or []) or 'none'}")
PY
```
The jobs refuse a missing value, a placeholder, or a value outside the plan.

## 7. Test sets: T1 → T2 → T3 → T4 → M

```
chain 'job_t1.slurm|t1/t1.done' 'job_t2.slurm|t2/t2.done' 'job_t3.slurm|t3/t3.done' 'job_t4.slurm|t4/t4.done' \
      'job_m.slurm --export=ALL,PART=t1|m/t1/m.done' 'job_m.slurm --export=ALL,PART=t2|m/t2/m.done' \
      'job_m.slurm --export=ALL,PART=t3|m/t3/m.done' 'job_m.slurm --export=ALL,PART=t4|m/t4/m.done'
```
Leave out T4 and M part t4 when `BASELINES=none`.

Check after each job:
- the marker exists;
- the log ends in `failed steps: none`;
- the `rows_ok` lines show the expected rows per system and no error rows other than the 180 s LTC-2k shortfalls in T3, T4 and M;
- `du` and the quota.

What each job runs:
- **T1** (VCTK-2k, the record's variant):
  - S1–S4 with logits;
  - the Mimi copies in `mc/`;
  - A1–A5 (and C1) in the R1+R3 mode with `--source_e`;
  - S1u (and E17);
  - CRT-AL if it is C2–C5.
- **T2**: the same on LTC-2k, without P.
- **T3**: the budget sweep at 3 and 180 s.
- **T4**: the listed baselines, at 30 s and, for kNN-VC, also at 3 and 180 s, in the smoke test's environment. The `conv_cfg_sha` of its rows must equal the smoke test's (`$J/v1/baselines_smoke/dev/30/<system>/manifest.jsonl`). If T4 hits 72 h, resubmit it, or split it with `--export=ALL,ONLY=seedvc`.
- **M**: one part per submission. No test-set metric exists before T4 (§9). The outputs are `$J/v1/m/<part>/metrics/<set>_<budget>{,_src}.jsonl`, the inputs of the test-set statistics.

Row names on the test sets:
- `faiss`, `crt_tau0.4`, `crt_tau<τ_dev>`, `crt_argmax`;
- `mimi_copy_src`, `mimi_copy_P`;
- `source_copy`, `target_copy`;
- `anc_<A1–A5 names>`, with CRT-AL as `anc_R3_s0_pm_rq_T1` if it is C1;
- `front_<S1u>`, `front_lmbr_k2_g1_T1`;
- `cand_<CRT-AL code name>` if CRT-AL is C2–C5;
- the baselines `knnvc`, `freevc`, `seedvc`, `facodec`;
- each with `@src`.

## 8. Since 19 Oct 2026: the account ceai (DEVIATIONS_v1.md item 4)

CRT moved to `ceai`, at `/datastore/ceai/crt` with a 1.2 TB quota. `nvquoc` now serves Qwen and support models.

T3 and T4 finish on `nvquoc` with the frozen scripts. They cannot move first: the anchor lab's resume check and the baselines' `conv_cfg_sha` both hold absolute paths. Then:

1. Delete T3's logit dumps on `nvquoc` once its decodes are checked (item 3), so that they are not relayed.
2. Relay from the Mac with `relay_copy.sh sync` (nvquoc → Mac → ceai). Repeat until `relay_copy.sh verify` shows no difference.
3. Push the job layer for ceai: `job_*.slurm`, `submit_ceai.sh` and `move_paths.py`. They use paths under `/datastore/ceai/crt`, and their guards refuse pair lists that still hold `nvquoc`, `ceai/nvquoc` or `eaic` paths.
4. Run `bash submit_ceai.sh job_move_ceai.slurm`. It rewrites the paths and logs each change, maps the frozen hashes back, checks the environment, then re-scores 32 Dev rows and compares them with D1. It writes `v1/move/move.done`.
5. Run M on ceai: `bash submit_ceai.sh job_m.slurm --export=ALL,PART=t1`, then t2, t3 and t4. On ceai, CRT runs at most 2 jobs and the interpreter's P1 runs 1. Both use `submit_ceai.sh` (lock `/tmp/submit_ceai.lock`, log `_transfer_logs/submit_ceai.log`).
6. Delete `/datastore/nvquoc/crt` once the copy on ceai is verified, as the author instructed.

From here on, use `J=/datastore/ceai/crt/mimi_vc/benchmark/journal`.
