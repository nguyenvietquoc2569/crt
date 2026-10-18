#!/usr/bin/env python3
"""
CRT decoding lab (CRT journal extension, 10 Oct 2026): many decoding rules on the SAME run_conversions.py outputs,
no new CRT forward. Reads a conversion directory made with --dump_logits (dumps/<set>/<budget>/<pair>.npz and the
manifest) and writes <out_dir>/<system>/<pair>.wav plus manifest.jsonl rows (the pair's faiss row with system and
wav replaced) that metrics.py scores directly.

Why: Mimi's residual tokens are unstable (CB8-18 flip 87-98 % under small edits, mostly as a residual cascade;
paper/journal_extension/results/token_stability_32.md) while the decoder only sees the projected codeword SUMS.
So decode for the latent, not for the tokens. Notation: acoustic RVQ levels CB1..CB31 with codewords E_k (256-d,
before output_proj); a_F = sum_k E_k[retrieved_k] the FAISS acoustic latent; p_k = softmax(CRT logits_k / T) for the
refined levels CB1-18; mu = sum_{k<=18} p_k E_k + sum_{k>18} E_k[retrieved_k] (linearity of expectation: an estimate
of the acoustic latent even though the heads ignore the cascade). RQ(z) = Mimi's own greedy residual quantisation of
z over all 31 acoustic levels (a consistent cascade by construction; for one level, the minimum-Bayes-risk token
under squared latent error given p is the codeword nearest to the level's posterior mean, and RQ(mu) is the greedy
joint version). CB0 (semantic) is always the retrieved token.

Systems:
  faiss, tau<t>, argmax          published rules (decoded through MimiModel.decode exactly as the harness)
  faiss_rq                       RQ(a_F): the control for every *_rq rule (is a gain CRT's or re-quantisation's?)
  nbr<K>_tau<t>                  gate on the mass of the argmax and its K-1 nearest codewords (decoder space)
  pm_cont_T<T> / pm_rq_T<T>      mu decoded as a continuous latent / RQ(mu)
  shrink<l>_cont / shrink<l>_rq  a_F + l (mu - a_F)
  pm18_rq                        RQ(sum_{k<=18} p_k E_k): drops the retrieved CB19-31 detail
  spk<b>_pm_rq                   p_k ∝ p_k * p_ref,k^b (add-1 unigram of the target reference's tokens), then RQ(mu)
  rerank_g<g>                    per frame, the FAISS candidate frame maximising mean_k CRT log p of its tokens plus
                                 g * continuity (Viterbi over the 8 candidates); all 32 codes from that real frame
  knn4_cont / knn4_rq            mean acoustic latent of the 4 most similar FAISS candidates (kNN-VC style)
  knncrt_rq                      candidate latents weighted by softmax over candidates of mean_k CRT log p
  trunc<n>                       FAISS codes decoded with the first n codebooks only

Round 2 (paper/journal_extension/crt_math_ideas_verified.md section 3; --systems_set r2). W = output_proj (decoder
space), r = retrieved tokens, t = CRT argmax, d = W(mu_T - a_F), s0 = the source's CB0; CB0 is the retrieved token unless
stated, and frame-valued rules output all 32 codes of the chosen reference frame (CB0 included):
  src0_faiss / src0_tau<t>       faiss / tau<t> codes with CB0 = s0 (T1.1, no CRT information in the swap)
  mnn<e>_src0 / mnn<e>_cfr       on frames failing the CB0 mutual-nearest-neighbour test (tolerance e): CB0 = s0 / all 32
                                 codes of the most CB0-similar candidate frame (both streams from one real frame)
  thr<th>_src0                   CB0 = s0 where path_sim < th (the baseline the MNN test has to beat)
  lr<d>                          r -> t where logit_t - logit_r > d (log-ratio gate; d = 0 is argmax)
  hs<l>                          r -> t where 2 l <W(e_k[t] - e_k[r]), d> > ||W(e_k[t] - e_k[r])||^2 (per-level half space)
  risk<l>_g<g>_M<M>[_T<T>]       ICM (3 sweeps over CB1-18) on ||s(c) - a_F - l (mu_T - a_F)||^2 + g * gamma0 * #edits in
                                 decoder space, candidates {r} + CRT top-M + the 8 candidate frames' tokens; gamma0 = the
                                 pair's median ||W(e_k[t] - e_k[r])||^2
  chow                           r -> t where p1 > tau_k, tau_k = per-level FAISS token accuracy (--tau_chow)
  lmbr_k<k>_g<g>_T<T>            candidate frame scored -||B_j - W mu_T||^2 / 2 sigma^2 + k * cand_sim (B_j = CB1-18 decoder
                                 latent of candidate j), Viterbi g as rerank_g; controls with W mu_T replaced:
                                 medoid8_k<k>_g<g> (mean of the 8 B_j), cent_k<k>_g<g> (codebook means), minE_k<k>_g<g> (0)
  faiss_succ                     published greedy with continuity 0.5 / (1 + |cand - prev - 1| / 12.5) - 0.1 [cand < prev]
  faiss_vit                      exact Viterbi of the published objective 0.5 sim + 0.5 / (1 + |cand - prev| / 12.5)
  ctx<a>                         S + a (S_{t-1,i-1} + S_{t+1,i+1}) on the top-16 reference frames by CB0 cosine S, its
                                 top-8 (scaled by 1 / (1 + 2a)) into the published greedy
  us32_w<rep>_<w0>_<wJ>          Viterbi over the top-32 reference frames by S: emission S, advance 0, repeat -rep, other
                                 moves -w0 - wJ ||y_{i+1} - y_j||^2 / median ||y_{i+1} - y_i||^2 (y = full decoder input)
  gv<g>_<base>                   variance restoration of pm_cont_T<T>, knn4_cont or shrink<l>_cont: per-utterance gains
                                 clip((v_ref / v_x)^(g/2), 1, 3) on the reference's top-16 decoder-space PCs, both
                                 statistics on speech frames (||W a|| above its 20th percentile; for the output, of
                                 the FAISS latent, so the mask is the same across bases), speech-frame mean kept
  pois<a>_faiss / pois<a>_tau<t> screened-Poisson blending (fidelity a) of the acoustic latent within 2 frames of path joins
Round 7 (18 Oct 2026, pre-freeze screen): Gaussian optimal transport in decoder space (gaussian_ot):
  src_cont                       control: the source's own acoustic latent and CB0, decoded continuously (the Mimi copy)
  ot<k>[nm]_<base>               the Monge map between Gaussians from the base's speech-frame statistics to the reference's,
                                 in the reference's top-k decoder-space PCs (k = 0: mean shift only; nm: no mean shift),
                                 output covariance shrunk toward its diagonal (OT_BETA); base src_cont (the source's own
                                 latent and CB0: CRT-free conversion), pm_cont_T<T>, shrink<l>_cont or knn4_cont
  otf<k>[nm]_<base>              the same map on the whole decoder input (semantic + acoustic projections, Lab.full_input)
                                 against the reference's decoder inputs, decoded as a continuous input (Lab.decode_input):
                                 tests whether the speaker identity the semantic stream carries moves with its moments
Not implemented: the hmm_b<b> extension of T1.3 (only if lmbr beats its controls) and the matched low-pass control of
T1.6. The r2 list omits argmax (top of the coverage curve) and faiss_rq (T1.6 control): both come from round 1 on the
same dumps with the same code. Every pair also gets the T1.0 zero-decode diagnostics, averaged into lab_summary.json
'diag' (--diag_only: no decoding, written to lab_diag.json, manifest untouched); --icm_check adds the exhaustive 2^18
optimality gap of ICM.
"""
import argparse
import functools
import glob
import json
import os
import re
import time

import numpy as np
import soundfile as sf

SR = 24000
DEFAULT_SYSTEMS = ('faiss,faiss_rq,tau0.4,argmax,nbr8_tau0.4,pm_cont_T1,pm_rq_T1,pm_cont_T0.5,pm_rq_T0.5,shrink0.5_cont,'
                   'shrink0.5_rq,pm18_rq,spk0.5_pm_rq,rerank_g0,rerank_g1,knn4_cont,knn4_rq,knncrt_rq,trunc8,trunc16')
DEFAULT_SYSTEMS_R2 = ('faiss,tau0.1,tau0.2,tau0.3,tau0.4,tau0.5,tau0.6,'               # baselines + coverage curve (T1.2)
                      'src0_faiss,src0_tau0.4,mnn0_src0,mnn0.05_src0,mnn0_cfr,mnn0.05_cfr,thr0.9_src0,thr0.95_src0,'  # T1.1
                      'lr0.5,lr1,lr2,lr3,hs0.5,hs1,risk1_g0_M4_T1,risk0.5_g0.25_M4_T1,risk1_g0.25_M4_T1,risk1_g1_M4_T1,'
                      'risk1_g0.25_M4_T0.5,chow,'
                      'lmbr_k0_g1_T1,lmbr_k2_g1_T1,lmbr_k2_g0_T1,medoid8_k2_g1,cent_k2_g1,minE_k2_g1,'        # T1.3
                      'faiss_succ,faiss_vit,ctx0.5,us32_w0.2_0.2_0.1,us32_w0.5_0.3_0.3,'                      # T1.4
                      'gv0.5_pm_cont_T1,gv1_knn4_cont,pois1_faiss,pois0.25_tau0.4')                         # T1.5-6
DISP_GROUPS = {'CB1': (0, 1), 'CB2-4': (1, 4), 'CB5-18': (4, 18), 'CB19-31': (18, 31), 'CB1-31': (0, 31)}
ICM_FRAMES = 200


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--conv_dir', required=True, help='run_conversions --out_dir/<set>/<budget> (manifest*.jsonl)')
    p.add_argument('--dumps', required=True, help='run_conversions --out_dir/dumps/<set>/<budget>')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--systems', default='', help='comma list (default: the --systems_set list)')
    p.add_argument('--systems_set', choices=('r1', 'r2'), default='r1', help='r1 DEFAULT_SYSTEMS, r2 DEFAULT_SYSTEMS_R2')
    p.add_argument('--tau_chow', default='', help='JSON {"tau": [per-level FAISS token accuracy]} for chow (skipped if absent)')
    p.add_argument('--diag_only', action='store_true', help='T1.0 diagnostics only, no decoding (-> lab_diag.json)')
    p.add_argument('--icm_check', action='store_true', help=f'ICM vs exhaustive 2^18 search on {ICM_FRAMES} frames')
    p.add_argument('--logits_from', default='', help='crt_anchor_lab dumps dir: take CRT logits from there (same pairs)')
    p.add_argument('--logits_key', default='R3', help='anchor recipe whose logits replace crt_logits (R1, R3, R3s, Psim)')
    p.add_argument('--pool_w', type=float, default=1.0,
                   help='log-linear pooling w * logits_key + (1 - w) * R1 (anchor dumps; 1 = logits_key alone)')
    p.add_argument('--guide_g', type=float, default=0.0,
                   help='speaker guidance: logits = (1 + g) logits_key - g guide_key (anchor dumps; 0 = off)')
    p.add_argument('--guide_key', default='R3s', help='anchor recipe for the negative (source-speaker) condition')
    p.add_argument('--cb0', choices=('retr', 'src'), default='retr',
                   help="output CB0 of every rule that passes the retrieved CB0 through: retrieved or the source's")
    p.add_argument('--device', default='cuda')
    return p.parse_args()


class Lab:
    def __init__(self, device):
        import torch
        from transformers import MimiModel
        from run_conversions import CHUNK_S, FRAME_RATE, cb0_table
        self.torch, self.device = torch, device
        self.dtype = torch.float16 if device == 'cuda' else torch.float32     # as the harness
        self.mimi = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=self.dtype).to(device).eval()
        q = self.mimi.quantizer
        self.sem, self.ac = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer
        with torch.no_grad():
            self.E = torch.stack([(l.codebook.embed_sum.float() / l.codebook.cluster_usage.float()
                                   .clamp(min=l.codebook.epsilon)[:, None]) for l in self.ac.layers]).to(device)
            W = self.ac.output_proj.weight[:, :, 0].float()                              # [512, 256]
            self.Edec = self.E @ W.T                                                     # [31, 2048, 512]
            sc = self.sem.layers[0].codebook
            self.E0 = sc.embed_sum.float() / sc.cluster_usage.float().clamp(min=sc.epsilon)[:, None]   # [2048, 256]
            C0h = torch.as_tensor(cb0_table(self.mimi), device=device)                  # the FAISS key table (fp16)
            self.C0n = C0h / C0h.norm(dim=1, keepdim=True)                               # retrieval geometry (cosine)
            self.Wsem = self.sem.output_proj.weight[:, :, 0].float()                    # [512, 256]
            self.E0dec = self.E0 @ self.Wsem.T                                           # [2048, 512]
            u = torch.stack([l.codebook.cluster_usage.float() for l in self.ac.layers])  # [31, 2048]
            self.ebar_dec = (u[:, :, None] * self.Edec).sum(1) / u.sum(1, keepdim=True)  # [31, 512] usage-weighted mean
        self.chunk = int(CHUNK_S * FRAME_RATE)          # reference frames per independently encoded chunk
        self.tau_chow = None
        self.nn_cache = {}

    def nbrs(self, k, K):
        if (k, K) not in self.nn_cache:
            e = self.Edec[k]
            d = self.torch.cdist(e[None], e[None])[0]
            d.fill_diagonal_(-1.0)
            self.nn_cache[(k, K)] = d.argsort(1)[:, :K]
        return self.nn_cache[(k, K)]

    # ---- latent algebra (acoustic levels index 0..30 = CB1..CB31) ----
    def latent(self, codes, start=0):
        """sum_k E_{start+k}[codes_k] for acoustic codes [n, T] (rows = levels start.., 0 = CB1) -> [T, 256]."""
        return sum(self.E[start + k][codes[k]] for k in range(codes.shape[0]))

    def rq(self, z):
        """Greedy residual quantisation of z [T, 256] over all 31 acoustic levels -> codes [31, T]."""
        r, out = z.clone(), []
        for k in range(self.E.shape[0]):
            idx = self.torch.cdist(r[None], self.E[k][None])[0].argmin(1)
            out.append(idx)
            r = r - self.E[k][idx]
        return self.torch.stack(out)

    # ---- decoding ----
    def decode_codes(self, codes32):
        with self.torch.no_grad():
            return self.mimi.decode(codes32[None].to(self.device)).audio_values.float().cpu().squeeze().numpy()

    def decode_latent(self, cb0, ac):
        """Continuous acoustic latent ac [T, 256] + semantic token cb0 [T] -> waveform."""
        m = self.mimi
        with self.torch.no_grad():
            emb = self.sem.decode(cb0[None, None, :].to(self.device))
            emb = emb + self.ac.output_proj(ac.T[None].to(self.dtype))
            emb = m.upsample(emb)
            h = m.decoder_transformer(emb.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
            return m.decoder(h).float().cpu().squeeze().numpy()

    def full_input(self, cb0, ac):
        """The decoder input [T, 512] of semantic tokens cb0 [T] and an acoustic latent ac [T, 256], computed with the
        quantizer's own modules (sem.decode + ac.output_proj), float32."""
        with self.torch.no_grad():
            emb = self.sem.decode(cb0[None, None, :].to(self.device))
            emb = emb + self.ac.output_proj(ac.T[None].to(self.dtype))
            return emb[0].T.float()

    def decode_input(self, x):
        """Decoder input x [T, 512] -> waveform (upsample, decoder transformer, SEANet decoder)."""
        m = self.mimi
        with self.torch.no_grad():
            emb = m.upsample(x.T[None].contiguous().to(self.dtype))
            h = m.decoder_transformer(emb.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
            return m.decoder(h).float().cpu().squeeze().numpy()


def softmax(x, T):
    x = x / T
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def viterbi(score, cand_ref, g, abs_delta=False):
    """score [T, C] (higher better), cand_ref [T, C] reference frame indices; continuity bonus
    g / (1 + |ref_t - ref_{t-1} - 1| / 12.5): successor continuity (the next reference frame scores highest), unlike the
    published harness path, whose |ref_t - ref_{t-1}| scores a repeat highest (run_conversions.retrieve); rerank_g<g> of
    round 1 used the successor form. abs_delta=True gives the published form. Returns the chosen candidate per frame."""
    T, C = score.shape
    off = 0 if abs_delta else 1
    acc, back = score[0].copy(), np.zeros((T, C), np.int64)
    for t in range(1, T):
        cont = g / (1.0 + np.abs(cand_ref[t][None, :] - cand_ref[t - 1][:, None] - off) / 12.5)   # [prev, cur]
        tot = acc[:, None] + cont
        back[t] = tot.argmax(0)
        acc = tot.max(0) + score[t]
    path = [int(acc.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])


def viterbi_tr(em, tr):
    """Max-sum path for emissions em [T, C] and transitions tr [T - 1, C, C] (prev, cur). Returns the state per frame."""
    T, C = em.shape
    acc, back = em[0].copy(), np.zeros((T, C), np.int64)
    for t in range(1, T):
        tot = acc[:, None] + tr[t - 1]
        back[t] = tot.argmax(0)
        acc = tot.max(0) + em[t]
    path = [int(acc.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    return np.array(path[::-1])


def greedy(cand, sim, succ=False):
    """The published path (run_conversions.retrieve) over candidates cand [T, C] sorted by sim: frame 0 = top-1, then
    argmax 0.5 sim + 0.5 / (1 + |cand - prev| / 12.5). succ: |cand - prev - 1| and -0.1 for stepping back (M17)."""
    path = np.zeros(len(cand), np.int64)
    path[0] = cand[0, 0]
    for t in range(1, len(cand)):
        dl = cand[t] - path[t - 1]
        path[t] = cand[t, np.argmax(0.5 * sim[t] + 0.5 / (1 + np.abs(dl - succ) / 12.5) - 0.1 * succ * (dl < 0))]
    return path


def ref_next_ok(n_ref, row, chunk):
    """[n_ref] bool: reference frame i + 1 continues frame i, i.e. the same R[budget] file (from keep_s; a frame belongs
    to the file its centre falls in) and the same encoding chunk. Without a manifest row only chunks are known."""
    cut = list(range(chunk, n_ref, chunk))
    if row and 'R' in row:
        cut += list(np.cumsum(next(iter(row['R'].values()))['keep_s'])[:-1] * 12.5)
    f = np.searchsorted(np.sort(cut), np.arange(n_ref) + 0.5)
    return np.r_[f[1:] == f[:-1], False]


def moves(path, ok):
    """Shares of path moves, a partition (sum 1): advance (next frame of the same file), repeat, backward (any delta < 0),
    jump (forward only: delta > 1 or an advance across a file boundary; the plan's |delta| > 1 also counts backward
    jumps, which are in back here); adv_run = mean length in frames of runs of consecutive advances."""
    d = np.diff(path)
    adv = (d == 1) & ok[path[:-1]]
    e = np.diff(np.r_[0, adv, 0].astype(np.int8))
    n = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)                              # advances per run
    return {'adv': float(adv.mean()), 'rep': float((d == 0).mean()), 'back': float((d < 0).mean()),
            'jump': float(((d > 0) & ~adv).mean()), 'adv_run': float((n + 1).mean()) if len(n) else 1.0}


def icm(Ed, C, r, u, gam, sweeps=3):
    """Iterated conditional modes for min_c ||u + sum_k (Ed[k][c_k] - Ed[k][r_k])||^2 + gam #{k: c_k != r_k} over
    c_k in C[k] ([n_cb, T, Kc], r_k among them); u = s(r) - target [T, 512]. J falls monotonically (the current choice is
    always a candidate); duplicate candidates are harmless (argmin takes the first). Returns c [n_cb, T]."""
    c = r.clone()
    for _ in range(sweeps):
        for k in range(len(C)):
            cur = Ed[k][c[k]]
            dl = Ed[k][C[k]] - cur[:, None]                                              # [T, Kc, 512]
            dJ = (dl * (dl + 2 * u[:, None])).sum(-1) + gam * ((C[k] != r[k][:, None]).float()
                                                               - (c[k] != r[k]).float()[:, None])
            new = C[k].gather(1, dJ.argmin(1, keepdim=True))[:, 0]
            u = u + Ed[k][new] - cur
            c[k] = new
    return c


def poisson_blend(a, path, ok, y, alpha, w=2):
    """Screened-Poisson join blending (T1.6, M25) of a latent a [T, D] (numpy): frames farther than w from a join keep a
    exactly; the others minimise alpha ||x_t - a_t||^2 + sum_t ||x_t - x_{t-1} - g_t||^2 (one tridiagonal system, all
    dimensions at once), g_t = a_t - a_{t-1} on advances, 0 on repeats and on joins the symmetric reference velocity
    1/2 [(y_{p_t} - y_{p_t - 1}) + (y_{p_{t-1} + 1} - y_{p_{t-1}})] of the reference latents y [T_ref, D] (a term that
    would cross a file boundary or precede frame 0 is left out). Returns (x, free-frame mask)."""
    T = len(a)
    p0, p1 = path[:-1], path[1:]
    adv, rep = (p1 - p0 == 1) & ok[p0], p1 == p0
    join = ~adv & ~rep
    v1 = ((p1 >= 1) & ok[np.maximum(p1 - 1, 0)])[:, None] * 1.0
    v2 = ok[p0][:, None] * 1.0
    vel = v1 * (y[p1] - y[np.maximum(p1 - 1, 0)]) + v2 * (y[np.minimum(p0 + 1, len(y) - 1)] - y[p0])
    g = np.where(adv[:, None], a[1:] - a[:-1], 0.0)                                   # g[t - 1]: edge (t - 1, t)
    g[join] = (vel / np.maximum(v1 + v2, 1))[join]
    free = np.convolve(np.r_[0, join], np.ones(2 * w), 'full')[w:w + T] > 0           # frames e - w .. e + w - 1
    A = np.diag(np.where(free, alpha, 1.0))
    rhs = np.where(free[:, None], alpha, 1.0) * a
    f = np.flatnonzero(free)
    lo, hi = f[f >= 1], f[f <= T - 2]
    A[lo, lo] += 1
    A[lo, lo - 1] -= 1
    rhs[lo] += g[lo - 1]
    A[hi, hi] += 1
    A[hi, hi + 1] -= 1
    rhs[hi] -= g[hi]
    x = np.linalg.solve(A, rhs)
    x[~free] = a[~free]
    return x, free


def us32_frames(S, y, ok, w_rep, w0, wJ, n=32):
    """us32 unit selection (T1.4, M23): Viterbi over the top-n reference frames by CB0 cosine S [T, T_ref] with emission
    S, advance to the natural successor 0, repeat -w_rep, any other move -w0 - wJ ||y_{i+1} - y_j||^2 / median
    ||y_{i+1} - y_i||^2 (y [T_ref, 512] = full decoder input, ok [T_ref] = frame i+1 continues frame i). Returns frames [T]."""
    import torch
    i = np.arange(len(y))
    ynx = y[torch.as_tensor(np.where(ok, i + 1, i), device=y.device)]                    # natural successor (or itself)
    s2 = ((y[1:] - y[:-1]) ** 2).sum(1)[torch.as_tensor(ok[:-1], device=y.device)].median()
    sc, top = S.topk(min(n, S.shape[1]), 1)
    topn = top.cpu().numpy()
    tr = -w0 - wJ * (torch.cdist(ynx[top[:-1]], y[top[1:]]) ** 2 / s2).cpu().numpy()   # [T-1, prev, cur]
    prv, cur = topn[:-1, :, None], topn[1:, None, :]
    tr = np.where(cur == prv, -w_rep, np.where((cur == prv + 1) & ok[prv], 0.0, tr))
    return topn[np.arange(len(topn)), viterbi_tr(sc.cpu().numpy(), tr)]


OT_BETA = 0.1   # shrinkage of the output covariance toward its diagonal (a 5 s source has about 60 frames)


def parse_ot(s):
    """ot[f]<k>[nm]_<base> -> (k, keep_mean, base, full); base 'src_cont' or a cont_latent name; full (otf): the map
    acts on the whole decoder input (semantic + acoustic projections) instead of the acoustic projection alone."""
    m = re.fullmatch(r'ot(f?)(\d+)(nm)?_(src_cont|pm_cont_T[\d.]+|shrink[\d.]+_cont|knn4_cont)', s)
    if not m:
        raise SystemExit(f'unknown system {s}')
    return int(m.group(2)), bool(m.group(3)), m.group(4), bool(m.group(1))


def gaussian_ot(x, sp, yr, k, keep_mean=False, beta=OT_BETA):
    """Gaussian optimal transport of decoder-space frames x [T, D] toward the reference's speech frames yr [T_ref, D]:
    the Monge map between the two Gaussians, in the reference's top-k principal subspace U. With m, mr the speech-frame
    means (sp [T] bool) and Sx, Sr the covariances of the U coordinates (Sr = diag of the top-k eigenvalues),
    A = Sx^-1/2 (Sx^1/2 Sr Sx^1/2)^1/2 Sx^-1/2 is the symmetric map with A Sx A = Sr, and
    x' = x + (mr - m) + (x - m) U (A - I) U^T (statistics on the speech frames, the map applied to all frames).
    keep_mean drops the mean shift; k = 0 is the mean shift alone; a diagonal Sx gives A = diag(sqrt(vr / vx)), the
    gv1 gains without their clamp. Sx is shrunk toward its diagonal by beta. Returns (x' - x [T, D], A [k, k])."""
    x, yr = np.asarray(x, np.float64), np.asarray(yr, np.float64)
    m, mr = x[sp].mean(0), yr.mean(0)
    out = np.zeros_like(x) if keep_mean else np.broadcast_to(mr - m, x.shape).copy()
    if k == 0:
        return out, np.zeros((0, 0))
    ev, U = np.linalg.eigh((yr - mr).T @ (yr - mr) / len(yr))
    U, vr = U[:, -k:], np.clip(ev[-k:], 0.0, None)
    P = (x - m) @ U
    Sx = P[sp].T @ P[sp] / sp.sum()
    Sx = (1 - beta) * Sx + beta * np.diag(np.diag(Sx))
    Sx = Sx + 1e-9 * max(np.trace(Sx), 1e-30) / k * np.eye(k)
    w, V = np.linalg.eigh(Sx)
    sq, isq = (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T
    wm, Vm = np.linalg.eigh(sq @ np.diag(vr) @ sq)
    A = isq @ ((Vm * np.sqrt(np.clip(wm, 0.0, None))) @ Vm.T) @ isq
    A = (A + A.T) / 2
    return out + (P @ (A - np.eye(k))) @ U.T, A


def ref_latents(lab, ref):
    """S-free inputs of us32_frames for a reference ref [T_ref, 32]: the full decoder input y [T_ref, 512]."""
    torch = lab.torch
    rc = torch.as_tensor(ref.T.copy(), device=lab.device)
    ar = torch.arange(lab.E.shape[0], device=lab.device)[:, None]
    return lab.E0dec[rc[0]] + lab.Edec[ar, rc[1:]].sum(0)


def run_pair(lab, z, systems, row=None):
    """Returns {system: (waveform, stats)} for one dump. row = the pair's manifest row (reference file boundaries from
    R[budget].keep_s); without it only the 30 s encoding chunks are known as boundaries."""
    torch = lab.torch
    dev = lab.device
    first = int(z['crt_first_cb'])
    assert first == 1, 'lab assumes CRT refines CB1..CB(n_cb)'
    retr = torch.as_tensor(z['retrieved_codes'].astype(np.int64), device=dev)            # [32, T]
    logits = z['crt_logits'].astype(np.float32)                                         # [n_cb, T, 2048]
    n_cb = logits.shape[0]
    cbs = slice(1, 1 + n_cb)
    a_F = lab.latent(retr[1:])
    ref = z['ref_codes'].astype(np.int64)                                               # [T_ref, 32]
    cand = z['cand_idx'].astype(np.int64)                                               # [T, 8]
    cand_sim = z['cand_sim'].astype(np.float32)
    cand_logp = z['cand_logp'].astype(np.float32)                                       # [n_cb, 8, T]
    p1 = softmax(logits, 1.0)
    t1 = torch.as_tensor(p1.argmax(-1), device=dev)
    conf = p1.max(-1)
    out = {}
    # round 2 (decoder space; acoustic level index 0 = CB1)
    nT = logits.shape[1]
    path = z['path'].astype(np.int64)
    s0n = z['src_codes'][0].astype(np.int64)
    s0 = torch.as_tensor(s0n, device=dev)
    ar = torch.arange(lab.E.shape[0], device=dev)[:, None]
    Ed = lab.Edec[:n_cb]
    er = Ed[ar[:n_cb], retr[cbs]]                                                       # [n_cb, T, 512]
    dt = Ed[ar[:n_cb], t1] - er                                                         # argmax swap per level
    aFd = lab.Edec[ar, retr[1:]].sum(0)
    e_F = (aFd ** 2).sum(1).mean()
    ctok = torch.as_tensor(ref[cand][:, :, cbs].transpose(2, 0, 1).copy(), device=dev)  # [n_cb, T, 8]

    def mu_from(p):
        m = sum(torch.as_tensor(p[k], device=dev, dtype=torch.float32) @ lab.E[k] for k in range(n_cb))
        return m + lab.latent(retr[1 + n_cb:], start=n_cb)

    def codes_with(acodes):
        c = retr.clone()
        c[1:] = acodes
        return c

    def stats_codes(c):
        ch = (c[cbs] != retr[cbs]).float().mean(1).cpu().numpy()
        cc = c[cbs] != retr[cbs]
        orph = cc[1:] & ~cc[:-1] & (t1[:-1] != retr[cbs][:-1])     # child changed, parent kept against CRT's argmax
        return {'changed_cb1_18': float(ch.mean()), 'changed_per_cb': [round(float(x), 4) for x in ch],
                'latent_dist_to_faiss': float((lab.latent(c[1:]) - a_F).norm(dim=1).mean()),
                'cb0_changed': float((c[0] != retr[0]).float().mean()),
                'energy_rel': float((lab.Edec[ar, c[1:]].sum(0) ** 2).sum(1).mean() / e_F),
                'orphan_share': float(orph.sum() / cc[1:].sum()) if cc[1:].any() else float('nan')}

    @functools.lru_cache(None)
    def Wac():
        return lab.ac.output_proj.weight[:, :, 0].float()                                # [512, 256]

    def stats_lat(a):
        return {'latent_dist_to_faiss': float((a - a_F).norm(dim=1).mean()),
                'energy_rel': float(((a @ Wac().T) ** 2).sum(1).mean() / e_F)}

    @functools.lru_cache(None)
    def dec_T(T):
        """(per-level posterior-mean decoder latents muk [n_cb, T, 512], d = W(mu_T - a_F) [T, 512])."""
        muk = torch.einsum('ktc,kcd->ktd', torch.as_tensor(softmax(logits, T), device=dev), Ed)
        return muk, (muk - er).sum(0)

    @functools.lru_cache(None)
    def nxt():
        return ref_next_ok(len(ref), row, lab.chunk)

    @functools.lru_cache(None)
    def S():
        """CB0 cosine of every source frame to every reference frame [T, T_ref] (the FAISS geometry)."""
        return lab.C0n[s0] @ lab.C0n[torch.as_tensor(ref[:, 0], device=dev)].T

    @functools.lru_cache(None)
    def Y():
        """Full decoder input of every reference frame [T_ref, 512]."""
        rc = torch.as_tensor(ref.T.copy(), device=dev)
        return lab.E0dec[rc[0]] + lab.Edec[ar, rc[1:]].sum(0)

    @functools.lru_cache(None)
    def cand_lat():
        return torch.stack([lab.latent(torch.as_tensor(ref[cand[:, j]][:, 1:].T.copy(), device=dev))
                            for j in range(cand.shape[1])])                              # [8, T, 256]

    def gate_codes(mask):
        c = retr.clone()
        sel = c[cbs]
        sel[mask] = t1[mask]
        c[cbs] = sel
        return c

    def frame_codes(fr):
        return torch.as_tensor(ref[fr].T.copy(), device=dev)

    def path_stats(fr):
        return {'same_frame_as_faiss': float((fr == path).mean()), **moves(fr, nxt()),
                'cb0_match_src': float((ref[fr, 0] == s0n).mean()),
                'mean_sim': float(S()[torch.arange(nT, device=dev), torch.as_tensor(fr, device=dev)].mean())}

    def cont_latent(name):
        """Acoustic latent [T, 256] of a continuous round-1 system: pm_cont_T<T>, shrink<l>_cont or knn4_cont."""
        if name.startswith('pm_cont_T'):
            return mu_from(softmax(logits, float(name[9:])))
        if name.startswith('shrink'):
            return a_F + float(name[6:].split('_')[0]) * (mu_from(p1) - a_F)
        assert name == 'knn4_cont', name
        w = np.zeros_like(cand_sim)
        np.put_along_axis(w, np.argsort(-cand_sim, 1)[:, :4], 0.25, 1)
        return (torch.as_tensor(w.T[:, :, None], device=dev, dtype=torch.float32) * cand_lat()).sum(0)

    def sel_frames(score, g, abs_delta=False):
        j = viterbi(score, cand, g, abs_delta) if g > 0 else score.argmax(1)
        return cand[np.arange(nT), j]

    for s in systems:
        if s == 'faiss':
            c = retr
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s == 'faiss_rq':
            c = codes_with(lab.rq(a_F))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('tau') or s == 'argmax':
            c = retr.clone()
            mask = torch.ones_like(t1, dtype=torch.bool) if s == 'argmax' else torch.as_tensor(
                conf > float(s[3:]), device=dev)
            sel = c[cbs]
            sel[mask] = t1[mask]
            c[cbs] = sel
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('nbr'):
            K, tau = s[3:].split('_tau')
            K, tau = int(K), float(tau)
            c = retr.clone()
            sel = c[cbs]
            for k in range(n_cb):
                nb = lab.nbrs(k, K)[t1[k]]                                               # [T, K]
                mass = torch.as_tensor(p1[k], device=dev).gather(1, nb).sum(1)
                m = mass > tau
                sel[k][m] = t1[k][m]
            c[cbs] = sel
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith(('pm_cont_T', 'pm_rq_T')):
            T = float(s.split('_T')[1])
            mu = mu_from(softmax(logits, T))
            if s.startswith('pm_cont'):
                out[s] = (lab.decode_latent(retr[0], mu), stats_lat(mu))
            else:
                c = codes_with(lab.rq(mu))
                out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('shrink'):
            lam, kind = s[6:].split('_')
            a = a_F + float(lam) * (mu_from(p1) - a_F)
            if kind == 'cont':
                out[s] = (lab.decode_latent(retr[0], a), stats_lat(a))
            else:
                c = codes_with(lab.rq(a))
                out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('pmk'):   # CRT posterior mean on CB1..K only (truth-set Shapley: CRT's information is there)
            K, kind = s[3:].split('_')                                                   # pmk<K>_{rq,cont,keep}
            K = int(K)
            m = sum(torch.as_tensor(p1[k], device=dev, dtype=torch.float32) @ lab.E[k] for k in range(K))
            m = m + lab.latent(retr[1 + K:], start=K)                                    # retrieved CB(K+1)..CB31
            if kind == 'cont':
                out[s] = (lab.decode_latent(retr[0], m), stats_lat(m))
            else:   # rq: re-quantise all 31 levels; keep: re-quantise CB1..K only, the retrieved CB(K+1).. kept as is
                c = codes_with(lab.rq(m)) if kind == 'rq' else codes_with(torch.cat([lab.rq(m)[:K], retr[1 + K:]]))
                out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s == 'pm18_rq':
            m = sum(torch.as_tensor(p1[k], device=dev, dtype=torch.float32) @ lab.E[k] for k in range(n_cb))
            c = codes_with(lab.rq(m))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('spk'):
            b = float(s[3:].split('_')[0])
            pr = np.stack([np.log((np.bincount(ref[:, 1 + k], minlength=2048) + 1.0)
                                  / (len(ref) + 2048.0)) for k in range(n_cb)]).astype(np.float32)  # [n_cb, 2048]
            c = codes_with(lab.rq(mu_from(softmax(logits + b * pr[:, None, :], 1.0))))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('rerank_g'):
            g = float(s[8:])
            score = cand_logp.mean(0).T                                                  # [T, 8]
            j = viterbi(score, cand, g) if g > 0 else score.argmax(1)
            frames = cand[np.arange(len(j)), j]
            c = torch.as_tensor(ref[frames].T.copy(), device=dev)
            out[s] = (lab.decode_codes(c), {**stats_codes(c), 'same_frame_as_faiss': float(
                (frames == z['path'].astype(np.int64)).mean())})
        elif s.startswith('knn'):
            if s.startswith('knn4'):
                top = np.argsort(-cand_sim, 1)[:, :4]
                w = np.zeros_like(cand_sim)
                np.put_along_axis(w, top, 0.25, 1)
            else:
                lp = cand_logp.mean(0).T                                                 # [T, 8]
                w = softmax(lp, 1.0)
            ac_c = torch.stack([lab.latent(torch.as_tensor(ref[cand[:, j]][:, 1:].T.copy(), device=dev))
                                for j in range(cand.shape[1])])                          # [8, T, 256]
            a = (torch.as_tensor(w.T[:, :, None], device=dev, dtype=torch.float32) * ac_c).sum(0)
            if s.endswith('_cont'):
                out[s] = (lab.decode_latent(retr[0], a), stats_lat(a))
            else:
                c = codes_with(lab.rq(a))
                out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('trunc'):
            n = int(s[5:])
            with torch.no_grad():
                w = lab.mimi.decode(retr[None, :n]).audio_values.float().cpu().squeeze().numpy()
            out[s] = (w, {'n_codebooks': n})
        # ---- round 2 (T1.1-T1.6) ----
        elif s.startswith('src0_'):
            b = s[5:]
            c = retr.clone() if b == 'faiss' else gate_codes(torch.as_tensor(conf > float(b[3:]), device=dev))
            c[0] = s0
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('mnn'):
            eps, kind = s[3:].split('_')
            eps = float(eps)
            fwd = cand_sim[np.arange(nT), z['path_rank'].astype(np.int64)] >= cand_sim[:, 0] - eps
            Sb = (lab.C0n[retr[0]] @ lab.C0n[s0].T).cpu().numpy()                     # [T, T] retrieved vs source CB0
            bwd = np.diag(Sb) >= Sb.max(1) - eps
            fail = torch.as_tensor(~((fwd & bwd) | (ref[path, 0] == s0n)), device=dev)
            c = retr.clone()
            if kind == 'src0':
                c[0, fail] = s0[fail]
            else:
                c[:, fail] = frame_codes(cand[np.arange(nT), cand_sim.argmax(1)])[:, fail]
            out[s] = (lab.decode_codes(c), {**stats_codes(c), 'mnn_fail': float(fail.float().mean())})
        elif s.startswith('thr'):
            m = torch.as_tensor(z['path_sim'] < float(s[3:].split('_')[0]), device=dev)
            c = retr.clone()
            c[0, m] = s0[m]
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('lr'):
            lgap = logits.max(-1) - np.take_along_axis(logits, z['retrieved_codes'][cbs].astype(np.int64)[..., None],
                                                       -1)[..., 0]
            c = gate_codes(torch.as_tensor(lgap > float(s[2:]), device=dev))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('hs'):
            c = gate_codes(2 * float(s[2:]) * (dt * dec_T(1.0)[1]).sum(-1) - (dt ** 2).sum(-1) > 0)
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s.startswith('risk'):
            lam, g, M, Tm = re.fullmatch(r'risk([\d.]+)_g([\d.]+)_M(\d+)(?:_T([\d.]+))?', s).groups()
            topk = torch.as_tensor(z['topk_idx'][:, :, :int(M)].astype(np.int64), device=dev)
            C = torch.cat([retr[cbs][..., None], topk, ctok], -1)                        # [n_cb, T, 1 + M + 8]
            c = retr.clone()
            c[cbs] = icm(Ed, C, retr[cbs], -float(lam) * dec_T(float(Tm or 1))[1],
                         float(g) * float((dt ** 2).sum(-1).median()))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif s == 'chow':
            c = gate_codes(torch.as_tensor(conf > lab.tau_chow[:, None], device=dev))
            out[s] = (lab.decode_codes(c), stats_codes(c))
        elif re.match(r'(lmbr|medoid8|cent|minE)_k', s):
            kind, kap, g, Tm = re.fullmatch(r'(lmbr|medoid8|cent|minE)_k([\d.]+)_g([\d.]+)(?:_T([\d.]+))?', s).groups()
            B = sum(Ed[k][ctok[k]] for k in range(n_cb))                                 # [T, 8, 512] CB1-18 latents
            if kind == 'lmbr':
                tgt = dec_T(float(Tm or 1))[0].sum(0)
            elif kind == 'medoid8':
                tgt = B.mean(1)
            elif kind == 'cent':
                tgt = lab.ebar_dec[:n_cb].sum(0)
            else:
                tgt = torch.zeros_like(aFd)
            D = ((B - tgt[..., None, :]) ** 2).sum(-1)                                   # [T, 8]
            score = -D / (2 * D.min(1).values.median()) + float(kap) * torch.as_tensor(cand_sim, device=dev)
            fr = sel_frames(score.cpu().numpy(), float(g))
            c = frame_codes(fr)
            out[s] = (lab.decode_codes(c), {**stats_codes(c), **path_stats(fr)})
        elif s in ('faiss_succ', 'faiss_vit'):
            fr = greedy(cand, cand_sim, succ=True) if s == 'faiss_succ' else sel_frames(0.5 * cand_sim, 0.5, True)
            c = frame_codes(fr)
            out[s] = (lab.decode_codes(c), {**stats_codes(c), **path_stats(fr)})
        elif s.startswith('ctx'):
            a = float(s[3:])
            Sm = S()
            okt = torch.as_tensor(nxt()[:-1], device=dev, dtype=Sm.dtype)
            ctx = torch.zeros_like(Sm)
            ctx[1:, 1:] += Sm[:-1, :-1] * okt                                            # S_{t-1,i-1} within a file
            ctx[:-1, :-1] += Sm[1:, 1:] * okt                                            # S_{t+1,i+1}
            top = Sm.topk(16, 1).indices                                                 # centre gate
            st, o = ((Sm + a * ctx).gather(1, top) / (1 + 2 * a)).sort(1, descending=True)
            fr = greedy(top.gather(1, o[:, :8]).cpu().numpy(), st[:, :8].cpu().numpy())
            c = frame_codes(fr)
            out[s] = (lab.decode_codes(c), {**stats_codes(c), **path_stats(fr)})
        elif s.startswith('us32_w'):
            fr = us32_frames(S(), Y(), nxt(), *map(float, s[6:].split('_')))
            c = frame_codes(fr)
            out[s] = (lab.decode_codes(c), {**stats_codes(c), **path_stats(fr)})
        elif s.startswith('gv'):
            gam, base = s[2:].split('_', 1)
            a = cont_latent(base)
            yr = lab.Edec[ar, torch.as_tensor(ref[:, 1:].T.copy(), device=dev)].sum(0)  # W a_ref [T_ref, 512]
            nr = yr.norm(dim=1)
            yr = yr[nr > nr.quantile(0.2)]                                               # speech frames
            ev, U = torch.linalg.eigh((yr - yr.mean(0)).T @ (yr - yr.mean(0)) / len(yr))
            U, vR = U[:, -16:], ev[-16:]                                                 # top-16 PCs, variances
            nx = aFd.norm(dim=1)
            sp = nx >= nx.quantile(0.2)     # output speech frames: the rule on the FAISS latent W a_F (>=: never empty)
            y = a @ Wac().T
            pc = (y - y[sp].mean(0)) @ U                                                 # [T, 16]
            gain = ((vR / (pc[sp] ** 2).mean(0)) ** (float(gam) / 2)).clamp(1, 3)      # stats on speech, map on all
            a = a + ((pc * (gain - 1)) @ U.T) @ torch.linalg.pinv(Wac()).T              # PCs lie in range(W): exact
            out[s] = (lab.decode_latent(retr[0], a), {**stats_lat(a), 'gv_gain_mean': float(gain.mean())})
        elif s.startswith('pois'):
            al, base = s[4:].split('_', 1)
            a = a_F if base == 'faiss' else lab.latent(gate_codes(torch.as_tensor(conf > float(base[3:]), device=dev))[1:])
            yref = lab.latent(torch.as_tensor(ref[:, 1:].T.copy(), device=dev)).cpu().numpy().astype(np.float64)
            x, free = poisson_blend(a.cpu().numpy().astype(np.float64), path, nxt(), yref, float(al))
            a = torch.as_tensor(x, device=dev, dtype=torch.float32)
            out[s] = (lab.decode_latent(retr[0], a), {**stats_lat(a), 'free_share': float(free.mean())})
        # ---- round 7: Gaussian optimal transport in decoder space ----
        elif s == 'src_cont':       # control: the source's own latent decoded continuously (= the Mimi copy)
            a = lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64), device=dev))
            out[s] = (lab.decode_latent(s0, a), stats_lat(a))
        elif s.startswith('ot'):
            k, keep_mean, base, full = parse_ot(s)
            if base == 'src_cont':  # the source's own latent with its own CB0, whatever --cb0 says
                a, c0 = lab.latent(torch.as_tensor(z['src_codes'][1:].astype(np.int64), device=dev)), s0
            else:
                a, c0 = cont_latent(base), retr[0]
            yr = lab.Edec[ar, torch.as_tensor(ref[:, 1:].T.copy(), device=dev)].sum(0)  # W a_ref [T_ref, 512]
            nr = yr.norm(dim=1)
            yr = yr[nr > nr.quantile(0.2)]                                               # speech frames, as gv
            Wd = Wac().detach()                             # a module parameter: leave autograd before numpy
            y = a @ Wd.T
            ny = y.norm(dim=1)
            sp = (ny >= ny.quantile(0.2)).cpu().numpy()     # the output's own speech frames (>=: never empty)
            if full:    # the whole decoder input (semantic + acoustic), the same speech frames on both sides
                x = lab.full_input(c0, a)
                rr = torch.as_tensor(ref.T.copy(), device=dev)
                xr = lab.full_input(rr[0], lab.latent(rr[1:]))[nr > nr.quantile(0.2)]
                dx, A = gaussian_ot(x.cpu().numpy(), sp, xr.cpu().numpy(), k, keep_mean)
                wave, st_ = lab.decode_input(x + torch.as_tensor(dx, device=dev, dtype=torch.float32)), stats_lat(a)
                shift = float(np.linalg.norm(xr.cpu().numpy().mean(0) - x.cpu().numpy()[sp].mean(0)))
            else:
                dy, A = gaussian_ot(y.cpu().numpy(), sp, yr.cpu().numpy(), k, keep_mean)
                a = a + torch.as_tensor(dy, device=dev, dtype=torch.float32) @ torch.linalg.pinv(Wd).T  # in range(W)
                wave, st_ = lab.decode_latent(c0, a), stats_lat(a)
                shift = float(np.linalg.norm(yr.cpu().numpy().mean(0) - y.cpu().numpy()[sp].mean(0)))
            out[s] = (wave, {**st_, 'ot_k': k, 'ot_full': full, 'ot_scale': float(np.trace(A) / k) if k else 1.0,
                             'ot_shift': 0.0 if keep_mean else shift})
        else:
            raise SystemExit(f'unknown system {s}')
    return out


def diagnostics(lab, z, row=None):
    """T1.0 zero-decode diagnostics of one dump: path moves (reference file boundaries masked), FAISS ties, CB0
    mismatch, CRT's copy probability per level and dispersion ratios tr Cov_t(x) / tr Cov(a_ref) per level group in
    decoder space. c0n_vs_cand_sim = max |C0n[s0] . C0n[ref0[cand]] - cand_sim| (must be about 1e-3 or less)."""
    torch, dev = lab.torch, lab.device
    retr = torch.as_tensor(z['retrieved_codes'].astype(np.int64), device=dev)
    ref = z['ref_codes'].astype(np.int64)
    s0 = z['src_codes'][0].astype(np.int64)
    cand, cand_sim = z['cand_idx'].astype(np.int64), z['cand_sim'].astype(np.float32)
    path = z['path'].astype(np.int64)
    logits = z['crt_logits'].astype(np.float32)
    n_cb, nT = logits.shape[:2]
    p1 = softmax(logits, 1.0)
    r = z['retrieved_codes'][1:1 + n_cb].astype(np.int64)[..., None]
    sim = (lab.C0n[torch.as_tensor(s0, device=dev)][:, None] * lab.C0n[torch.as_tensor(ref[cand, 0], device=dev)]).sum(-1)
    out = {**moves(path, ref_next_ok(len(ref), row, lab.chunk)),
           'path_repro': float((greedy(cand, cand_sim) == path).mean()),        # the lab's greedy = the harness path
           'ties': float(((cand_sim >= cand_sim[:, :1] - 1e-6).sum(1) >= 2).mean()),
           'mism': float((s0 != ref[path, 0]).mean()),
           'missing': float((~np.isin(s0, ref[:, 0])).mean()),
           'cont_override': float((cand_sim[np.arange(nT), z['path_rank'].astype(np.int64)] < cand_sim[:, 0] - 1e-6).mean()),
           'p_copy_median': np.median(np.take_along_axis(p1, r, -1)[..., 0], 1).tolist(),
           'logit_gap_median': np.median(logits.max(-1) - np.take_along_axis(logits, r, -1)[..., 0], 1).tolist(),
           'c0n_vs_cand_sim': float((sim.cpu() - torch.as_tensor(cand_sim)).abs().max())}
    ar = torch.arange(lab.E.shape[0], device=dev)[:, None]
    lv = lambda codes: lab.Edec[ar, codes]                                              # [31, N, 512] per level
    P = torch.as_tensor(p1, device=dev)
    aF = lv(retr[1:])
    mu = torch.cat([torch.einsum('ktc,kcd->ktd', P, lab.Edec[:n_cb]), aF[n_cb:]])
    mu256 = sum(P[k] @ lab.E[k] for k in range(n_cb)) + lab.latent(retr[1 + n_cb:], start=n_cb)
    w = np.zeros_like(cand_sim)
    np.put_along_axis(w, np.argsort(-cand_sim, 1)[:, :4], 0.25, 1)
    knn = sum(float(1) * torch.as_tensor(w[:, j], device=dev)[:, None] * lv(torch.as_tensor(ref[cand[:, j], 1:].T.copy(),
                                                                                   device=dev)) for j in range(cand.shape[1]))
    xs = {'faiss': aF, 'pm_T1': mu, 'pm_rq_T1': lv(lab.rq(mu256)), 'knn4': knn, 'shrink0.5': 0.5 * (aF + mu)}

    def trcov(L):
        G = torch.stack([L[i:j].sum(0) for i, j in DISP_GROUPS.values()])           # [groups, N, 512]
        return ((G - G.mean(1, keepdim=True)) ** 2).sum(-1).mean(1)
    tr_ref = trcov(lv(torch.as_tensor(ref[:, 1:].T.copy(), device=dev)))
    for x, L in xs.items():
        out.update({f'disp_{x}_{g}': float(v) for g, v in zip(DISP_GROUPS, trcov(L) / tr_ref)})
    return out


def icm_check(lab, z, n, lam=1.0, gs=(0.0, 0.25, 1.0)):
    """T1.2 check on the first n frames of a dump: ICM with M = 1 (candidates {r, argmax}, gamma = g * gamma0 as risk)
    against all 2^n_cb subsets, J(x) = x'Qx - 2 lam x'b + gamma x'e per frame (Q = Delta Delta', b = Delta d, e = argmax
    differs; J(none) = 0). Returns {g: (J_icm - J_opt) / -J_opt per frame}."""
    torch, dev = lab.torch, lab.device
    retr = torch.as_tensor(z['retrieved_codes'].astype(np.int64), device=dev)
    logits = z['crt_logits'].astype(np.float32)
    n_cb = len(logits)
    P = torch.as_tensor(softmax(logits, 1.0), device=dev)
    ar = torch.arange(n_cb, device=dev)[:, None]
    Ed, r, t1 = lab.Edec[:n_cb], retr[1:1 + n_cb], P.argmax(-1)
    er = Ed[ar, r]
    dt = Ed[ar, t1] - er
    d = (torch.einsum('ktc,kcd->ktd', P, Ed) - er).sum(0)
    gam0 = float((dt ** 2).sum(-1).median())
    r, t1, dt, d = r[:, :n], t1[:, :n], dt[:, :n], d[:n]
    X = ((torch.arange(2 ** n_cb, device=dev)[:, None] >> torch.arange(n_cb, device=dev)) & 1).double()
    xs = {g: (icm(Ed, torch.stack([r, t1], -1), r, -lam * d, g * gam0) != r).double() for g in gs}
    out = {g: [] for g in gs}
    for t in range(r.shape[1]):
        D = dt[:, t].double()
        Q, b, e = D @ D.T, D @ d[t].double(), (t1[:, t] != r[:, t]).double()
        quad, lin, ne = ((X @ Q) * X).sum(1), X @ b, X @ e
        for g in gs:
            Jo = float((quad - 2 * lam * lin + g * gam0 * ne).min())
            x = xs[g][:, t]
            Ji = float(x @ Q @ x - 2 * lam * x @ b + g * gam0 * x @ e)
            out[g].append((Ji - Jo) / max(-Jo, 1e-12))
    return out


def with_anchor(z, pid, a):
    """z with CRT logits from crt_anchor_lab's dump of the same pair (logits_key, optionally pooled with R1) and the
    statistics derived from them recomputed (argmax, conf, top-16, candidate log-probs), and/or the source's CB0 in place
    of the retrieved one (rules that copy whole reference frames keep the frames' own CB0)."""
    if a.logits_from:
        with np.load(os.path.join(a.logits_from, pid + '.npz')) as an:
            assert (an['src_codes'] == z['src_codes']).all() and (an['retrieved_codes'] == z['retrieved_codes']).all(), pid
            lg = an[f'{a.logits_key}_logits'].astype(np.float32)
            if a.pool_w != 1.0:
                lg = a.pool_w * lg + (1 - a.pool_w) * an['R1_logits'].astype(np.float32)
            if a.guide_g:      # contrastive product of experts: p_tgt^(1+g) / p_src^g (classifier-free guidance form)
                lg = (1 + a.guide_g) * lg - a.guide_g * an[f'{a.guide_key}_logits'].astype(np.float32)
        lp = lg - lg.max(-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))                                  # log-softmax [n_cb, T, V]
        n_cb, T = lg.shape[:2]
        tok = z['ref_codes'].astype(np.int64)[z['cand_idx']][:, :, 1:1 + n_cb]               # [T, 8, n_cb]
        z.update(crt_logits=lg, crt_argmax=lg.argmax(-1), crt_conf=np.exp(lp.max(-1)),
                 topk_idx=np.argsort(-lg, -1)[:, :, :16],
                 cand_logp=np.take_along_axis(lp, tok.transpose(2, 0, 1), 2).transpose(0, 2, 1).astype(np.float32))
    if a.cb0 == 'src':
        r = z['retrieved_codes'].copy()
        r[0] = z['src_codes'][0]
        z['retrieved_codes'] = r
    return z


def main():
    a = parse_args()
    systems = [] if a.diag_only else (a.systems or {'r1': DEFAULT_SYSTEMS, 'r2': DEFAULT_SYSTEMS_R2}[a.systems_set]).split(',')
    rows = {}
    for f in sorted(glob.glob(os.path.join(a.conv_dir, 'manifest*.jsonl'))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get('system') == 'faiss' and not r.get('error'):
                    rows[r['pair_id']] = r
    dumps = sorted(glob.glob(os.path.join(a.dumps, '*.npz')))
    lab = Lab(a.device)
    if a.tau_chow and os.path.exists(a.tau_chow):
        lab.tau_chow = np.asarray(json.load(open(a.tau_chow))['tau'], np.float32)
    elif 'chow' in systems:
        systems.remove('chow')
        print('  chow skipped: no --tau_chow file (it comes from crt_truth_fit)', flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    man = None if a.diag_only else open(os.path.join(a.out_dir, 'manifest.jsonl'), 'w')   # diag_only: keep the manifest
    stats_all, sanity, latent_check, diag, icm_gaps = {}, [], [], [], {}
    t0 = time.time()
    for i, f in enumerate(dumps):
        pid = os.path.basename(f)[:-4]
        if pid not in rows:
            print(f'  skip {pid}: no faiss manifest row', flush=True)
            continue
        z = np.load(f)
        if a.logits_from or a.cb0 == 'src':
            z = with_anchor(dict(z), pid, a)
        base = rows[pid]
        diag.append(diagnostics(lab, z, row=base))
        if a.icm_check:
            for g, gaps in icm_check(lab, z, ICM_FRAMES).items():
                icm_gaps.setdefault(g, []).extend(gaps)
        if a.diag_only:
            continue
        res = run_pair(lab, z, systems, row=base)
        if not latent_check:   # the continuous-latent decode path must reproduce MimiModel.decode on real codes
            r32 = lab.torch.as_tensor(z['retrieved_codes'].astype(np.int64), device=lab.device)
            w1, w2 = lab.decode_codes(r32), lab.decode_latent(r32[0], lab.latent(r32[1:]))
            latent_check.append(float(np.abs(w1 - w2).max()))
        if os.path.exists(base['wav']) and 'faiss' in res:      # the lab's faiss must equal the harness's faiss
            h, _ = sf.read(base['wav'], dtype='float32')
            w = res['faiss'][0]
            n = min(len(h), len(w))
            sanity.append(float(np.abs(h[:n] - np.clip(w[:n], -1, 1)).max()))
        for s, (wav, st) in res.items():
            p = os.path.join(a.out_dir, s, pid + '.wav')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            sf.write(p, wav, SR, subtype='PCM_16')
            row = {k: v for k, v in base.items() if k not in ('override_coverage', 'change_coverage', 'n_changed',
                                                              'same_codes_as', 'conv_time_s', 'tau', 'error')}
            row.update(system='lab_' + s, wav=p)
            man.write(json.dumps(row) + '\n')
            stats_all.setdefault(s, []).append(st)
        print(f'  [{i + 1}/{len(dumps)}] {pid} {time.time() - t0:.0f}s', flush=True)
    if man:
        man.close()
    summ = {'systems': systems, 'n_pairs': len(diag),
            'diag': {k: np.mean(np.array([d[k] for d in diag if k in d], dtype=float), axis=0).tolist()
                     for k in (diag[0] if diag else {})},
            'faiss_vs_harness_max_abs_diff': max(sanity) if sanity else None,
            'latent_path_vs_mimi_decode_max_abs_diff': latent_check[0] if latent_check else None, 'stats': {}}
    if icm_gaps:
        summ['icm_check'] = {str(g): {'mean_rel_gap': float(np.mean(v)), 'share_suboptimal': float(np.mean(np.array(v) > 1e-6))}
                             for g, v in icm_gaps.items()}
    for s, lst in stats_all.items():
        keys = [k for k in lst[0] if isinstance(lst[0][k], float)]
        summ['stats'][s] = {}
        for k in keys:   # mean over the pairs where k is defined (orphan_share is NaN without CB2-18 changes)
            v = [x[k] for x in lst if not np.isnan(x[k])]
            summ['stats'][s][k] = float(np.mean(v)) if v else None
    json.dump(summ, open(os.path.join(a.out_dir, 'lab_diag.json' if a.diag_only else 'lab_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
