#!/usr/bin/env python3
"""
Source-anchored CRT (T2.1 / idea M22 in paper/journal_extension/crt_math_ideas_verified.md §4).

CRT was trained on (anchor = speaker S's own clean CB0, CB1-18 hints retrieved from ANOTHER speaker's corpus with that
CB0 as queries by the published greedy path, embedding = S; train_crt.py sample_batch / corrupt_faiss) but the
published inference feeds the retrieved target CB0 as the anchor. Per pair, with the run_conversions.py objects
(Pipeline, get_reference and its per-reference cache, retrieve, crt_logits, crt_stats, gate, decode, write_wav,
base_row, paths_for; nothing re-implemented), four CRT forwards (R1 and R3 only with --mode R1+R3, below):
  R1    anchor = retrieved CB0, hints = retrieved CB1-18, embedding = target (published). With --conv_dir the source
        and retrieved codes must equal the run_conversions dump and the logits / max-softmax must match it within
        max(LOGIT_TOL, 2 fp16 ulp) / CONF_TOL (fp16 rounding, kernel order), else the pair fails.
  R3    anchor = src_codes[0]; hints and embedding as R1
  R3s   R3 with the SOURCE speaker's embedding: Resemblyzer on the first 15 s of the source speaker's own R[budget]
        reference, computed as a target's (a Dev source speaker is a Dev target; its R entry comes from any pair row
        whose target_speaker is that speaker, and R never holds a source recording of the set). Its E set is not
        used: E is the enrollment set that SIM to the source speaker is scored against.
  Psim  anchor = src_codes[0], hints = CB1-18 retrieved from a third Dev speaker D's R[budget] reference by the
        published greedy path (source CB0 queries), source embedding: the training input configuration, i.e. the
        in-distribution confidence reference. D = the Dev targets other than source and target and without a
        reference shortfall, sorted, index crc32(pair_id) % n (training drew the other speaker uniformly).
  G0.5  geometric pooling log p ∝ 0.5 log p_R3 + 0.5 log p_R1 (no forward)
Decoded systems (anc_<name> in the manifest; Mimi decode of 32 x T codes as the harness, identical codes decoded once):
  faiss, src0_faiss   retrieved codes; src0_faiss with CB0 := source CB0 (T1.1 control, no CRT)
  <A>_<c0>_<rule>     A in R1, R3, G0.5; output CB0 c0 = r0 (retrieved) or s0 (source); rule = tau0.4 / argmax
                      (run_conversions.gate) or pm_rq_T1 (crt_decode_lab's rule on its Lab: RQ of the posterior-mean
                      latent over all 31 acoustic levels, so CB19-31 are re-quantised). R1_s0_tau0.4 is the T1.1
                      control and is named src0_tau0.4; R1_r0_tau0.4 / R1_r0_argmax are the published crt_tau0.4 /
                      crt_argmax.
  <A>_r0_tau<τ>       A in R1, R3, τ in DEC_TAUS: with tau0.4 the decoded τ grid 0.1-0.9, so that R1-τ_dev and
                      R3-τ_dev are chosen by Dev WER within this job (prediction 3).
Registered (manifest field registered = True) are the controls, every r0 system and the s0 pm_rq_T1 systems. The s0
gate systems (<A>_s0_tau0.4 / _argmax but src0_tau0.4) keep CB19-31 retrieved under a source CB0 although the plan
says mixed-CB0 outputs use pm_rq: they are exploratory, reported but never used for selection.
--mode R1+R3 (every test-set run, ANALYSIS_PLAN_v1 §1.3, §8.1 item 4): the R1 and R3 forwards (and G0.5) only, no R3s
and no Psim, so no source-speaker or third-speaker reference is read: a source speaker need not be a target (VCTK-2k)
and a source-speaker shortfall cannot stop a pair (LTC-2k at 180 s). Every system can still be decoded (none uses R3s
or Psim). --systems decodes a subset (names without anc_; test sets: the ablations A1-A5 R1_s0_pm_rq_T1,
R3_r0_pm_rq_T1, R3_s0_argmax, R3_s0_tau0.4, src0_faiss and, if selected, C1 R3_s0_pm_rq_T1). sim_src_wavs is the
source speaker's own target E (Dev, LTC-2k) or, with --source_e, the paths of its entry in make_pairs.py's source-E
file (VCTK-2k: vctk2k_source_E.json, {source speaker: {paths, durs, total_s}}; a plain list of paths is read too). These
options enter cfg only through the values they change (recipes, systems; source_e and its sha256 only when given), so
the default invocation keeps its cfg_sha and resume check.
Outputs under --out_dir (layout, resume and error rows as run_conversions):
  <set>/<budget>/anc_<name>/<pair_id>.wav
  dumps/<set>/<budget>/<pair_id>.npz  cfg_sha, src_codes, retrieved_codes, ref_codes, cand_idx, cand_sim,
      crt_first_cb, path, path_rank, path_sim (R1), psim_speaker, psim_codes (the Psim input, 32 x T); per recipe
      <R>_argmax (the gate's tokens), <R>_conf (max softmax, float32), <R>_entropy, <R>_topk_idx / <R>_topk_prob
      (18 x T x 16; topk_idx[..., 0] can differ from argmax on exact ties), <R>_cand_logp
      (18 x 8 x T, not Psim, whose candidates are D's), with --dump_logits <R>_logits (18 x T x 2048 float16; not G0.5,
      which follows from R1 and R3); js_R3_R3s, js_R1_R3 (per-token Jensen-Shannon divergence, bits); systems,
      out_codes (n_sys x 32 x T); r1_repro (max |Δconf|, max |Δlogit|, τ = 0.4 gate flips against the --conv_dir
      dump; nan without one). Written after the pair's audio. With <R>_logits / <R>_cand_logp / <R>_topk_idx renamed
      to crt_logits / cand_logp / topk_idx a dump is a crt_decode_lab.run_pair input. The R1+R3 mode writes no R3s /
      Psim array, psim_speaker, psim_codes or js_R3_R3s; systems and out_codes follow --systems.
  <set>/<budget>/manifest[.shard<i>of<n>].jsonl  metrics.py rows: the pair's faiss row from --conv_dir (else
      run_conversions.base_row) with system, wav, dump, tau replaced, plus sim_src_wavs (the source speaker's E, null
      without one), anchor, out_cb0, rule, registered, psim_speaker (null in the R1+R3 mode), same_codes_as,
      changed_cb1_18, changed_cb19_31, cb0_changed.
  <set>/<budget>/manifest_src[.shard<i>of<n>].jsonl  the manifest's rows without error that have a sim_src_wavs, with
      system := <system>@src, sim_ref_wavs := sim_src_wavs, target_speaker := source_speaker (true_target_speaker keeps
      the target, so metrics.sv_trials scores EER to the source): metrics.py --metrics sim,secs on it gives SIM / SECS
      to the SOURCE speaker's E for every pair with one (E never holds a source recording, so check_row passes); the
      rows left out for want of a source E are counted in the log.
  <set>/<budget>/anchor_summary[.shard<i>of<n>].json  pre-registered predictions over the selected pairs whose rows
      have no error, frames pooled: 1. median top-1 at CB1-4 per recipe, R3 - R1 (pair bootstrap 95 % CI) and the
      share of the R1 -> Psim gap that R3 closes, reported only when Psim is above R1 (the plan's premise; else
      gap_closed is null), its CI only when Psim is above R1 in >= 97.5 % of the resamples (the gap is undefined in
      the others); the prediction holds iff Psim > R1, R3 > R1 and the share >= 0.5; 2. JS(R3, R3s) against
      JS(R1, R3), argmax and τ = 0.4 token change rates; plus per-recipe τ-grid gate-open / change rates, copy rate
      (argmax = hint) and per-system change rates. In the R1+R3 mode only the R1, R3 and G0.5 statistics:
      pred1_confidence and pred2_embedding are null.
      Prediction 3 (WER, SIM_target) is scored by metrics.py on the manifest, SIM_source on manifest_src.
--summarize_only (T9 when the run is sharded; no conversion, no model loaded): summarize over the dumps of every shard
of <out_dir>/<set>/<budget>, i.e. the pairs of --pairs, in order, without an error row in any
manifest[.shard<i>of<n>].jsonl there, under the one cfg_sha of those manifests; written to
<set>/<budget>/anchor_summary.json (an unsharded run's own file, rewritten with the same content).
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import socket
import sys
import time
import traceback
import zlib

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_decode_lab as cdl  # noqa: E402
import run_conversions as rc  # noqa: E402

VARIANT = 'v5.6.5'
RECIPES = ('R1', 'R3', 'R3s', 'Psim', 'G0.5')
MODES = {'full': RECIPES, 'R1+R3': ('R1', 'R3', 'G0.5')}     # --mode -> recipes
W_POOL = 0.5
TAU = 0.4
TAU_GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))
DEC_TAUS = tuple(t for t in TAU_GRID if t != TAU)
LOGIT_TOL, CONF_TOL = 0.05, 5e-3
SPEC = {'faiss': (None, 'r0', None), 'src0_faiss': (None, 's0', None)}
SPEC.update({('src0_tau0.4' if (a, c, r) == ('R1', 's0', 'tau0.4') else f'{a}_{c}_{r}'): (a, c, r)
             for a in ('R1', 'R3', 'G0.5') for c in ('r0', 's0') for r in ('tau0.4', 'argmax', 'pm_rq_T1')})
SPEC.update({f'{a}_r0_tau{t:g}': (a, 'r0', f'tau{t:g}') for a in ('R1', 'R3') for t in DEC_TAUS})
SYSTEMS = list(SPEC)
REGISTERED = {s for s, (a, c, r) in SPEC.items() if c == 'r0' or r in (None, 'pm_rq_T1') or s == 'src0_tau0.4'}


def tau_of(rule):
    return float(rule[3:]) if rule and rule.startswith('tau') else None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help="make_pairs.py JSONL; every row is read for the source speakers' E "
                   "and, in full mode, the source and third speakers' references")
    p.add_argument('--set', required=True, help='must match the set field of the pair rows')
    p.add_argument('--budget', default='30', help='reference budget in s: 3, 30 or 180')
    p.add_argument('--shard', default='', help='i/n with 0 <= i < n: pair rows k with k %% n == i')
    p.add_argument('--max_pairs', type=int, default=0, help='first N pair rows (before sharding)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--dump_logits', action='store_true', help='also dump the full R1, R3, R3s, Psim logits (float16)')
    p.add_argument('--conv_dir', default='', help='run_conversions --out_dir of the same pairs: faiss rows are '
                   'copied and R1 is checked against its dumps (published path only)')
    p.add_argument('--path_mode', choices=('pub', 'us32'), default='pub',
                   help="hint frames: the published greedy path, or crt_decode_lab's us32 unit selection (T1.4, M23); "
                        'the third-speaker Psim control always uses the published path')
    p.add_argument('--us32_w', default='0.5,0.3,0.3', help='us32 (repeat, move, join) weights')
    p.add_argument('--dump_recipes', default='R1,R3,R3s,Psim', help='recipes whose full logits --dump_logits saves')
    p.add_argument('--mode', choices=list(MODES), default='full',
                   help='full: R1, R3, R3s, Psim (Dev); R1+R3 (test sets): R1 and R3 only, no source-speaker reference')
    p.add_argument('--systems', default='', help='comma list of the systems to decode (names without anc_, e.g. '
                   'R1_s0_pm_rq_T1,src0_faiss); default every system')
    p.add_argument('--source_e', default='', help='make_pairs.py source-E file (VCTK-2k vctk2k_source_E.json, JSON '
                   '{source speaker: {paths, durs, total_s}} or {source speaker: [E wav paths]}): sim_src_wavs; '
                   "default the source speaker's own target E")
    p.add_argument('--summarize_only', action='store_true', help='no conversion, no model: summarize the dumps of '
                   'every shard of the run into anchor_summary.json')
    a = p.parse_args()
    v = rc.VARIANTS[VARIANT]
    a.compat_spie, a.pair_ids, a.variant, a.ckpt, a.crt_script = False, '', VARIANT, v['slim'], v['script']
    a.system_list = select_systems(a.systems)
    return a


def select_systems(spec):
    """--systems: the named systems in SYSTEMS order; empty = every system."""
    want = [s.strip() for s in spec.split(',') if s.strip()]
    if not want:
        return list(SYSTEMS)
    bad = sorted(set(want) - set(SPEC))
    if bad or len(set(want)) < len(want):
        raise SystemExit(f'ERROR: --systems {spec!r}: unknown {bad} or repeated names; the systems are {SYSTEMS}')
    return [s for s in SYSTEMS if s in want]


def load_source_e(path):
    """--source_e: make_pairs.py's {source speaker: E entry {'paths': [...], 'durs', 'total_s'}} -> {source speaker:
    E paths}; an entry may also be the plain list of paths."""
    d = json.load(open(path))
    if not isinstance(d, dict):
        raise SystemExit(f'ERROR: --source_e {path} is not a JSON object')
    out = {}
    for s, v in d.items():
        v = v['paths'] if isinstance(v, dict) and 'paths' in v else v
        if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            raise SystemExit(f'ERROR: --source_e {path}: entry {s!r} is not a list of wav paths')
        out[s] = v
    return out


def source_e_of(spk, spk_rows, src_e):
    """sim_src_wavs: the source speaker's entry in --source_e if given, else its own target E (Dev, LTC-2k); None
    without one."""
    if src_e is not None:
        return src_e.get(spk) or None
    return spk_rows[spk]['E']['paths'] if spk in spk_rows else None


def js_bits(a, b):
    """Jensen-Shannon divergence in bits per token of two logit tensors [1, n_cb, T, V] -> [n_cb, T]."""
    la, lb = a[0].log_softmax(-1), b[0].log_softmax(-1)
    lm = la.logaddexp(lb) - np.log(2)
    return ((la.exp() * (la - lm)).sum(-1) + (lb.exp() * (lb - lm)).sum(-1)) / (2 * np.log(2))


def pm_rq(lab, retr, logits):
    """crt_decode_lab pm_rq_T1: retr [32, T] (lab.device) with CB1-31 := RQ(mu), mu = sum_{k<=n_cb} p_k E_k +
    sum_{k>n_cb} E_k[retr_k], p = softmax of logits [n_cb, T, V]."""
    n = logits.shape[0]
    p = cdl.softmax(logits.cpu().numpy(), 1.0)
    mu = sum(lab.torch.as_tensor(p[k], device=lab.device, dtype=lab.torch.float32) @ lab.E[k] for k in range(n))
    c = retr.clone()
    c[1:] = lab.rq(mu + lab.latent(retr[1 + n:], start=n))
    return c


def pair_base(args, row, conv, spk_rows, src_e=None):
    """The pair's faiss row of --conv_dir (else the same fields from run_conversions.base_row) + the source's E."""
    b = conv.get(row['pair_id']) or rc.base_row(args, row, 'faiss', None)
    b = {k: v for k, v in b.items() if k not in ('override_coverage', 'change_coverage', 'n_changed', 'same_codes_as',
                                                 'conv_time_s', 'tau', 'error')}
    b['sim_src_wavs'] = source_e_of(row['source_speaker'], spk_rows, src_e)
    return b


def out_row(args, base, name):
    wav, dump = rc.paths_for(args, base['pair_id'], 'anc_' + name)
    return dict(base, system='anc_' + name, wav=wav, dump=dump, tau=tau_of(SPEC[name][2]))


def anchor_pair(pipe, lab, row, args, spk_rows, base, cache, cfg_sha):
    """The mode's CRT forwards, statistics, the selected systems' codes and audio. Returns (manifest rows, print
    line)."""
    torch = pipe.torch
    full = args.mode == 'full'
    pid, s_spk, t_spk = row['pair_id'], row['source_speaker'], row['target_speaker']
    src_wav = librosa.load(row['source_path'], sr=pipe.sr, mono=True)[0].astype(np.float32)
    src_codes = pipe.encode(src_wav)
    ref, _ = rc.get_reference(pipe, row, args, cache)
    d_spk = None
    if full:
        if spk_rows[s_spk]['R'][args.budget]['shortfall']:
            raise RuntimeError(f'source-speaker reference shortfall: {s_spk} R[{args.budget}] (R3s / Psim need it)')
        sref, _ = rc.get_reference(pipe, spk_rows[s_spk], args, cache)
        others = sorted(s for s, r in spk_rows.items()
                        if s not in (s_spk, t_spk) and not r['R'][args.budget]['shortfall'])
        d_spk = others[zlib.crc32(pid.encode()) % len(others)]
        dref, _ = rc.get_reference(pipe, spk_rows[d_spk], args, cache)
    path, rank, cand_idx, cand_sim = pipe.retrieve(src_codes, ref['index'])
    if args.path_mode == 'us32':
        s0 = torch.as_tensor(src_codes[0, 0].numpy().astype(np.int64), device=lab.device)
        Sm = lab.C0n[s0] @ lab.C0n[torch.as_tensor(ref['codes'][:, 0].astype(np.int64), device=lab.device)].T
        path = cdl.us32_frames(Sm, cdl.ref_latents(lab, ref['codes'].astype(np.int64)),
                              cdl.ref_next_ok(len(ref['codes']), row, lab.chunk), *map(float, args.us32_w.split(',')))
        hit = cand_idx == path[:, None]
        rank = np.where(hit.any(1), hit.argmax(1), -1)
        sim_t = Sm[torch.arange(len(path), device=lab.device), torch.as_tensor(path, device=lab.device)].cpu().numpy()
    faiss_codes = torch.from_numpy(np.ascontiguousarray(ref['codes'][path].T))[None]
    anch = faiss_codes.clone()
    anch[0, 0] = src_codes[0, 0]
    inp = {'R1': (faiss_codes, ref['spk'], ref['codes'], cand_idx), 'R3': (anch, ref['spk'], ref['codes'], cand_idx)}
    if full:
        d_path, _, d_cand, _ = pipe.retrieve(src_codes, dref['index'])
        psim = torch.from_numpy(np.ascontiguousarray(dref['codes'][d_path].T))[None]
        psim[0, 0] = src_codes[0, 0]
        inp.update(R3s=(anch, sref['spk'], ref['codes'], cand_idx), Psim=(psim, sref['spk'], dref['codes'], d_cand))
    logits = {k: pipe.crt_logits(c, s) for k, (c, s, _, _) in inp.items()}
    logits['G0.5'] = W_POOL * logits['R3'].log_softmax(-1) + (1 - W_POOL) * logits['R1'].log_softmax(-1)
    T = faiss_codes.shape[2]
    arrays = {'cfg_sha': cfg_sha, 'src_codes': src_codes[0].numpy().astype(np.int16),
              'retrieved_codes': faiss_codes[0].numpy().astype(np.int16), 'ref_codes': ref['codes'].astype(np.int16),
              'cand_idx': cand_idx.astype(np.int32), 'cand_sim': cand_sim.astype(np.float32),
              'crt_first_cb': np.int8(pipe.cbs.start), 'path': path.astype(np.int32),
              'path_rank': rank.astype(np.int8),
              'path_sim': (sim_t if args.path_mode == 'us32' else cand_sim[np.arange(T), rank]).astype(np.float32)}
    if full:
        arrays.update(psim_speaker=d_spk, psim_codes=psim[0].numpy().astype(np.int16),
                      js_R3_R3s=js_bits(logits['R3'], logits['R3s']).cpu().numpy().astype(np.float16))
    arrays['js_R1_R3'] = js_bits(logits['R1'], logits['R3']).cpu().numpy().astype(np.float16)
    st = {}
    for k, lg in logits.items():
        _, _, rcodes, ci = inp.get(k, inp['R1'])
        st[k] = tokens, conf, extra = pipe.crt_stats(lg, rcodes, ci)
        arrays.update({f'{k}_argmax': tokens.numpy().astype(np.int16), f'{k}_conf': conf.numpy().astype(np.float32),
                       f'{k}_entropy': extra['crt_entropy'].astype(np.float16),
                       f'{k}_topk_idx': extra['topk_idx'], f'{k}_topk_prob': extra['topk_prob'].astype(np.float16)})
        if k != 'Psim':
            arrays[f'{k}_cand_logp'] = extra['cand_logp'].astype(np.float32)
        if args.dump_logits and k in args.dump_recipes.split(','):
            arrays[f'{k}_logits'] = lg[0].half().cpu().numpy()

    rep = [np.nan] * 3
    cdump = os.path.join(args.conv_dir, 'dumps', args.set, args.budget, f'{pid}.npz') if args.conv_dir else ''
    if cdump and os.path.exists(cdump) and args.path_mode == 'pub':
        c1 = arrays['R1_conf']
        with np.load(cdump) as z:
            assert (z['src_codes'] == arrays['src_codes']).all(), f'source codes differ from {cdump}'
            assert (z['retrieved_codes'] == arrays['retrieved_codes']).all(), f'retrieved codes differ from {cdump}'
            dl, bad = np.array(np.nan), False
            if 'crt_logits' in z.files:     # fp16 values: one ulp is 0.03125 at |logit| in [32, 64)
                zl = z['crt_logits']
                dl = np.abs(zl.astype(np.float32) - logits['R1'][0].half().float().cpu().numpy())
                bad = (dl > np.maximum(LOGIT_TOL, 2 * np.spacing(np.abs(zl)).astype(np.float32))).any()
            rep = [float(np.abs(z['crt_conf'] - c1).max()), float(dl.max()),
                   float((z[f'mask_tau{TAU:g}'] != (c1 > TAU)).sum()) if f'mask_tau{TAU:g}' in z.files else np.nan]
        if rep[0] > CONF_TOL or bad:
            raise AssertionError(f'R1 does not reproduce {cdump}: max |dconf| {rep[0]:.3g}, max |dlogit| {rep[1]:.3g}')

    rule = {(None, None): faiss_codes}
    codes = {}
    for name in args.system_list:
        k, c0, r = SPEC[name]
        if (k, r) not in rule:
            tokens, conf, _ = st[k]
            rule[(k, r)] = (pm_rq(lab, faiss_codes[0].to(lab.device), logits[k][0]).cpu()[None] if r == 'pm_rq_T1'
                            else rc.gate(faiss_codes, tokens, conf, tau_of(r), pipe.cbs)[0])
        codes[name] = rule[(k, r)].clone()
        if c0 == 's0':
            codes[name][0, 0] = src_codes[0, 0]
    rows, decoded, first = [], {}, {}
    for name in args.system_list:
        k, c0, r = SPEC[name]
        c = codes[name]
        key = c.numpy().tobytes()
        first.setdefault(key, name)
        if key not in decoded:
            decoded[key] = pipe.decode(c)
        o = out_row(args, base, name)
        rc.write_wav(o['wav'], decoded[key], pipe.sr)
        ch = (c[0] != faiss_codes[0]).float()
        o.update(n_frames=T, n_ref_frames=len(ref['codes']), anchor=k or 'none', out_cb0=c0, rule=r or 'copy',
                 registered=name in REGISTERED, psim_speaker=d_spk,
                 same_codes_as=first[key] if first[key] != name else None,
                 changed_cb1_18=float(ch[1:19].mean()), changed_cb19_31=float(ch[19:].mean()),
                 cb0_changed=float(ch[0].mean()))
        rows.append(o)
    arrays.update(systems=np.array(args.system_list), r1_repro=np.array(rep, np.float32),
                  out_codes=np.stack([codes[s][0].numpy() for s in args.system_list]).astype(np.int16))
    _, dump = rc.paths_for(args, pid, 'faiss')
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    tmp = f'{dump[:-4]}.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, dump)
    med = ' '.join(f'{k}={np.median(arrays[k + "_conf"][0]):.3f}' for k in MODES[args.mode])
    return rows, f'T={T} D={d_spk} median conf CB1 {med} | repro {np.round(rep, 4).tolist()}'


def summarize(files, cfg_sha, boot=1000):
    """Prediction statistics over the pairs' npz files: frames pooled; CI by a pair bootstrap (speaker clusters
    ignored). The recipes are the dumps' own: R1, R3 and G0.5 in the R1+R3 mode, which has no prediction 1 (it needs
    Psim) or 2 (it needs R3s)."""
    P = collections.defaultdict(list)
    systems, rec = [], RECIPES
    for f in files:
        with np.load(f) as z:
            if str(z['cfg_sha']) != cfg_sha:
                continue
            rec = tuple(k for k in RECIPES if k + '_conf' in z.files)
            for k in rec:
                P[k + '_conf'].append(z[k + '_conf'])
                P[k + '_ent'].append(z[k + '_entropy'].astype(np.float32))
                P[k + '_top1'].append(z[k + '_argmax'])
                P[k + '_hint'].append((z['psim_codes'] if k == 'Psim' else z['retrieved_codes'])[1:19])
            for k in ('js_R3_R3s', 'js_R1_R3'):
                if k in z.files:
                    P[k].append(z[k].astype(np.float32))
            oc, rt = z['out_codes'], z['retrieved_codes']
            P['chg'].append(np.stack([(oc[:, 1:19] != rt[1:19]).mean((1, 2)), (oc[:, 0] != rt[0]).mean(1),
                                      (oc[:, 19:] != rt[19:]).mean((1, 2))], 1))
            P['repro'].append(z['r1_repro'])
            systems = [str(s) for s in z['systems']]
    n = len(P['repro'])
    if not n:
        return {'n_pairs': 0}
    C = {k: np.concatenate(v, -1) for k, v in P.items() if k not in ('chg', 'repro')}

    def rnd(x):
        return [round(float(v), 4) for v in x]

    def m14(idx):
        return {k: float(np.median(np.concatenate([P[k + '_conf'][i][:4] for i in idx], -1)))
                for k in ('R1', 'R3', 'Psim')}

    def gap(m):     # defined only when Psim is above R1 (the plan's premise)
        return (m['R3'] - m['R1']) / (m['Psim'] - m['R1']) if m['Psim'] > m['R1'] else float('nan')

    def gated(k, t=TAU):
        return np.where(C[k + '_conf'] > t, C[k + '_top1'], C[k + '_hint'])

    med = {k: np.median(C[k + '_conf'], 1) for k in rec}
    R = np.array(P['repro'], np.float64)
    ok = ~np.isnan(R[:, 0])
    out = {
        'n_pairs': n, 'n_frames': int(C['R1_conf'].shape[1]), 'recipes': rec,
        'median_top1': {k: rnd(med[k]) for k in rec},
        'median_entropy': {k: rnd(np.median(C[k + '_ent'], 1)) for k in rec},
        'copy_rate': {k: rnd((C[k + '_top1'] == C[k + '_hint']).mean(1)) for k in rec},
        'pred1_confidence': None, 'pred2_embedding': None,
        'tau_grid': {'tau': list(TAU_GRID), **{k: {
            'open_cb1_18': rnd([(C[k + '_conf'] > t).mean() for t in TAU_GRID]),
            'changed_cb1_18': rnd([(gated(k, t) != C[k + '_hint']).mean() for t in TAU_GRID]),
            'open': [rnd((C[k + '_conf'] > t).mean(1)) for t in TAU_GRID],
            'changed': [rnd((gated(k, t) != C[k + '_hint']).mean(1)) for t in TAU_GRID]} for k in rec}},
        'systems': {s: dict(zip(('changed_cb1_18', 'cb0_changed', 'changed_cb19_31'), rnd(x)),
                            registered=s in REGISTERED) for s, x in zip(systems, np.mean(P['chg'], 0))},
        'r1_repro': {'n_checked': int(ok.sum()), 'max_conf_diff': float(R[ok, 0].max()) if ok.any() else None,
                     'max_logit_diff': float(np.nanmax(R[ok, 1])) if ok.any() and (~np.isnan(R[ok, 1])).any() else None,
                     'gate_flips_tau0.4': int(np.nansum(R[ok, 2]))}}
    if not {'R3s', 'Psim'} <= set(rec):
        return out
    m = m14(range(n))
    up = m['Psim'] > m['R1']
    rng = np.random.default_rng(0)
    bm = [m14(rng.integers(0, n, n)) for _ in range(boot)]
    bs = [gap(x) for x in bm]
    bup = float(np.mean([x['Psim'] > x['R1'] for x in bm]))
    d = med['Psim'][:4] - med['R1'][:4]
    out['pred1_confidence'] = {
        'median_top1_cb1_4': m, 'r3_minus_r1': m['R3'] - m['R1'],
        'r3_minus_r1_ci95': rnd(np.percentile([x['R3'] - x['R1'] for x in bm], [2.5, 97.5])),
        'psim_minus_r1': m['Psim'] - m['R1'], 'premise_psim_above_r1': bool(up),
        'boot_share_psim_above_r1': bup, 'gap_closed_cb1_4': gap(m) if up else None,
        'ci95_pair_bootstrap': rnd(np.nanpercentile(bs, [2.5, 97.5])) if up and bup >= 0.975 else None,
        'psim_minus_r1_per_cb': rnd(d),
        'gap_closed_per_cb': [round(float(x), 4) if v > 0 else None
                              for x, v in zip((med['R3'][:4] - med['R1'][:4]) / np.where(d > 0, d, 1), d)],
        'holds': bool(up and m['R3'] > m['R1'] and gap(m) >= 0.5)}
    out['pred2_embedding'] = {
        'js_bits_R3_R3s': rnd(C['js_R3_R3s'].mean(1)), 'js_bits_R1_R3': rnd(C['js_R1_R3'].mean(1)),
        'js_ratio_R3s_to_anchor_cb1_18': float(C['js_R3_R3s'].mean() / C['js_R1_R3'].mean()),
        'argmax_change_R3_R3s': rnd((C['R3_top1'] != C['R3s_top1']).mean(1)),
        'argmax_change_R1_R3': rnd((C['R1_top1'] != C['R3_top1']).mean(1)),
        'tau0.4_token_change_R3_R3s': float((gated('R3') != gated('R3s')).mean()),
        'tau0.4_token_change_R1_R3': float((gated('R1') != gated('R3')).mean())}
    return out


def src_rows(rows):
    """manifest_src: the rows without error that have a source-speaker E, with system := <system>@src, sim_ref_wavs :=
    sim_src_wavs, target_speaker := source_speaker (true_target_speaker keeps the target)."""
    return [dict(r, system=r['system'] + '@src', sim_ref_wavs=r['sim_src_wavs'], target_speaker=r['source_speaker'],
                 true_target_speaker=r['target_speaker']) for r in rows if 'error' not in r and r.get('sim_src_wavs')]


def summary_files(args, rows, kept):
    """The dumps summarize() reads: those of the pairs of `rows` without an error row in `kept`, where they exist."""
    bad = {r['pair_id'] for r in kept.values() if 'error' in r}
    files = [rc.paths_for(args, r['pair_id'], 'faiss')[1] for r in rows if r['pair_id'] not in bad]
    return [f for f in files if os.path.exists(f)]


def run_manifests(run_dir):
    """Every shard's manifest of a run directory, their rows merged, and the one cfg_sha they were written under."""
    mans = sorted(glob.glob(os.path.join(run_dir, 'manifest.jsonl')) +
                  glob.glob(os.path.join(run_dir, 'manifest.shard*of*.jsonl')))
    kept = {}
    for f in mans:
        kept.update(rc.load_manifest(f))
    shas = sorted({str(r.get('conv_cfg_sha')) for r in kept.values()})
    if len(shas) != 1:
        raise SystemExit(f'ERROR: {run_dir}: manifests {[os.path.basename(f) for f in mans]} hold cfg {shas}, '
                         f'not exactly one')
    return mans, kept, shas[0]


def run_cfg(args, pipe, device, versions):
    """The run's configuration; cfg_sha (resume check, dumps) is its hash. --mode, --systems and --source_e enter only
    through the values they change (recipes, systems; source_e and its sha256 only when given), so the default
    invocation keeps its cfg_sha."""
    cfg = {'weights': args.ckpt, **pipe.crt_info, 'mimi': rc.MIMI_ID,
           'mimi_commit': getattr(pipe.mimi.config, '_commit_hash', None), 'mimi_dtype': str(pipe.dtype),
           'device_type': device, 'k': rc.K, 'spk_s': rc.SPK_S, 'topk': rc.TOPK, 'budget': args.budget,
           'recipes': MODES[args.mode], 'systems': args.system_list, 'w_pool': W_POOL, 'dump_logits': args.dump_logits,
           'conv_dir': args.conv_dir, 'versions': versions}
    if args.source_e:
        cfg.update(source_e=args.source_e, source_e_sha256=rc.sha256_file(args.source_e))
    return cfg


def print_summary(summ):
    p1, p2 = summ.get('pred1_confidence') or {}, summ.get('pred2_embedding') or {}
    if 'Psim' not in summ.get('recipes', RECIPES):     # R1+R3 mode: no prediction 1 or 2
        print(f'  R1 repro {summ.get("r1_repro")}')
        return
    print(f'  pred1: median top-1 CB1-4 {p1.get("median_top1_cb1_4")}, R3-R1 {p1.get("r3_minus_r1")} CI '
          f'{p1.get("r3_minus_r1_ci95")}, Psim above R1 {p1.get("premise_psim_above_r1")}, R1->Psim gap closed by R3 '
          f'{p1.get("gap_closed_cb1_4")} CI {p1.get("ci95_pair_bootstrap")} holds={p1.get("holds")}')
    print(f'  pred2: JS(R3,R3s)/JS(R1,R3) {p2.get("js_ratio_R3s_to_anchor_cb1_18")}, τ=0.4 token change R3 vs R3s '
          f'{p2.get("tau0.4_token_change_R3_R3s")}, R1 vs R3 {p2.get("tau0.4_token_change_R1_R3")}; '
          f'R1 repro {summ.get("r1_repro")}')


def summarize_run(args):
    """--summarize_only: summarize() over the dumps of every shard of the run, without loading any model."""
    rows = rc.load_pairs(args)
    run_dir = os.path.join(args.out_dir, args.set, args.budget)
    mans, kept, cfg_sha = run_manifests(run_dir)
    files = summary_files(args, rows, kept)
    shard_summ = (json.load(open(f)) for f in sorted(glob.glob(os.path.join(run_dir, 'anchor_summary*.json'))))
    cfg = next((s['cfg'] for s in shard_summ if s.get('cfg_sha') == cfg_sha and s.get('cfg')), None)
    summ = {'cfg_sha': cfg_sha, 'cfg': cfg, **summarize(files, cfg_sha)}
    out = os.path.join(run_dir, 'anchor_summary.json')
    with open(out, 'w') as f:
        json.dump(summ, f, indent=1)
    print_summary(summ)
    n_bad = len({r['pair_id'] for r in kept.values() if 'error' in r})
    print(f'crt_anchor_lab --summarize_only: {summ["n_pairs"]} of {len(rows)} pairs summarized ({len(files)} dumps, '
          f'{n_bad} pairs with error rows) from {len(mans)} manifests, cfg {cfg_sha} -> {out}', flush=True)


def main():
    args = parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    if args.shard:
        args.shard = rc.parse_shard(args.shard)
    if args.budget not in ('3', '30', '180'):
        raise SystemExit(f'ERROR: --budget {args.budget} not in 3, 30, 180')
    if args.summarize_only:
        if args.shard:
            raise SystemExit('ERROR: --summarize_only reads every shard of the run; drop --shard')
        return summarize_run(args)
    t_start = time.time()
    import faiss
    import torch
    import transformers
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    faiss.omp_set_num_threads(1)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('ERROR: --device cuda, but CUDA is not available')

    rows = rc.load_pairs(args)
    spk_rows = {}
    for line in open(args.pairs):
        if line.strip():
            r = json.loads(line)
            spk_rows.setdefault(r['target_speaker'], r)
    miss = sorted({r['source_speaker'] for r in rows} - set(spk_rows))
    if miss and args.mode == 'full':
        raise SystemExit(f'ERROR: source speakers {miss} are not targets in {args.pairs}: R3s / Psim need their '
                         f'reference')
    src_e = None
    if args.source_e:
        args.source_e = os.path.abspath(args.source_e)
        src_e = load_source_e(args.source_e)
    no_e = sorted({r['source_speaker'] for r in rows if not source_e_of(r['source_speaker'], spk_rows, src_e)})
    if no_e:
        print(f'  WARN: source speakers {no_e} have no E set '
              f'({"not in --source_e" if src_e is not None else "not targets, no --source_e"}): their pairs get no '
              f'manifest_src row', flush=True)
    conv = {}
    if args.conv_dir:
        args.conv_dir = os.path.abspath(args.conv_dir)
        for f in sorted(glob.glob(os.path.join(args.conv_dir, args.set, args.budget, 'manifest*.jsonl'))):
            conv.update({r['pair_id']: r for r in rc.load_manifest(f).values()
                         if r['system'] == 'faiss' and not r.get('error')})
    run_dir = os.path.join(args.out_dir, args.set, args.budget)
    os.makedirs(run_dir, exist_ok=True)
    sfx = '.shard{}of{}'.format(*args.shard) if args.shard else ''
    manifest = os.path.join(run_dir, f'manifest{sfx}.jsonl')

    pipe = rc.Pipeline(args, device)
    lab = cdl.Lab(device)            # latent algebra only; audio comes from pipe.decode as the harness
    cfg = run_cfg(args, pipe, device, {'torch': torch.__version__, 'transformers': transformers.__version__,
                                       'faiss': faiss.__version__, 'numpy': np.__version__,
                                       'librosa': librosa.__version__})
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    run_id = f'{time.strftime("%Y%m%d-%H%M%S")}-{socket.gethostname()}-{os.getpid()}'
    prev = rc.load_manifest(manifest)
    other = {r.get('conv_cfg_sha') for r in prev.values()} - {cfg_sha}
    if other:
        raise SystemExit(f'ERROR: {manifest} holds rows of cfg {sorted(map(str, other))}, now {cfg_sha}; '
                         f'use another --out_dir')

    def finished(pid):
        rs = [prev.get((pid, 'anc_' + s)) for s in args.system_list]
        return (all(r is not None and 'error' not in r and os.path.exists(r['wav']) for r in rs)
                and os.path.exists(rs[0]['dump']))
    todo = [r for r in rows if not finished(r['pair_id'])]
    print(f'crt_anchor_lab {run_id}: {args.set} budget {args.budget}, {len(rows)} pairs ({len(todo)} to do), '
          f'{len(spk_rows)} speakers, {len(conv)} conv faiss rows, {len(args.system_list)} systems, mode {args.mode}, '
          f'device {device}, cfg {cfg_sha} -> {manifest}', flush=True)

    cache, n_err, n_short = {}, 0, 0
    with open(manifest, 'a') as out:
        for i, row in enumerate(todo):
            t0 = time.time()
            base = pair_base(args, row, conv, spk_rows, src_e)
            try:
                new, line = anchor_pair(pipe, lab, row, args, spk_rows, base, cache, cfg_sha)
            except Exception as e:
                msg = f'{type(e).__name__}: {e}'
                short = msg.startswith('ValueError: reference shortfall')
                n_err += not short
                n_short += short
                print(f'  ERROR {row["pair_id"]}: {msg}\n{traceback.format_exc()}', flush=True)
                new, line = [], 'ERROR'
                for name in args.system_list:
                    r = out_row(args, base, name)
                    if os.path.exists(r['wav']):
                        os.remove(r['wav'])
                    new.append(dict(r, error=msg, shortfall=short))
            for r in new:
                r.update(run_id=run_id, conv_cfg_sha=cfg_sha)
                out.write(json.dumps(r) + '\n')
            out.flush()
            print(f'  [{i + 1}/{len(todo)}] {row["pair_id"]} {line} | {time.time() - t0:.1f}s', flush=True)

    kept = rc.load_manifest(manifest)
    with open(manifest + '.tmp', 'w') as f:
        for r in kept.values():
            f.write(json.dumps(r) + '\n')
    os.replace(manifest + '.tmp', manifest)
    msrc = os.path.join(run_dir, f'manifest_src{sfx}.jsonl')
    srows = src_rows(kept.values())
    with open(msrc + '.tmp', 'w') as f:
        for r in srows:
            f.write(json.dumps(r) + '\n')
    os.replace(msrc + '.tmp', msrc)
    summ = {'cfg_sha': cfg_sha, 'cfg': cfg, **summarize(summary_files(args, rows, kept), cfg_sha)}
    with open(os.path.join(run_dir, f'anchor_summary{sfx}.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    print_summary(summ)
    n_no_e = sum('error' not in r for r in kept.values()) - len(srows)
    if n_no_e:
        print(f'  {n_no_e} rows without a source-speaker E are not in {msrc}', flush=True)
    print(f'done in {time.time() - t_start:.0f}s: {len(todo) - n_err - n_short} pairs, {n_err} errors, {n_short} '
          f'reference shortfalls -> {manifest} ({len(kept)} rows)', flush=True)
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
