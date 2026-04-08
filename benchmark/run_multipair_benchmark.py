#!/usr/bin/env python3
"""
Multi-pair benchmark: Evaluate CRT, kNN-VC, FreeVC on VCTK evaluation pairs.
Reports mean +/- std across all pairs for SECS, MCD, WER, UTMOS.
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
import sys
import json
import time
import argparse
import tempfile
import shutil
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs_json', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/vctk/eval_pairs.json')
    p.add_argument('--max_pairs', type=int, default=20, help='Max pairs to evaluate (0=all)')
    p.add_argument('--crt_ckpt', default='/datastore/eaic/nvquoc/mimi_vc/v5.6.5-secs-high/checkpoints/crt_xlarge_best.pt')
    p.add_argument('--crt_script', default='/datastore/eaic/nvquoc/mimi_vc/v5.6.5-secs-high/train_crt.py')
    p.add_argument('--knnvc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/knn-vc')
    p.add_argument('--freevc_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/FreeVC')
    p.add_argument('--data_processed', default='/datastore/eaic/nvquoc/mimi_vc/data_storage/processed_merged_v3')
    p.add_argument('--output_dir', default='/datastore/eaic/nvquoc/mimi_vc/benchmark/results_multipair')
    p.add_argument('--sr', type=int, default=24000)
    p.add_argument('--skip_knnvc', action='store_true')
    p.add_argument('--skip_freevc', action='store_true')
    p.add_argument('--skip_crt', action='store_true')
    return p.parse_args()


# ── Metrics ──────────────────────────────────────────────────

def compute_mcd(ref, syn, sr):
    r = librosa.feature.mfcc(y=ref.astype(np.float32), sr=sr, n_mfcc=13)
    s = librosa.feature.mfcc(y=syn.astype(np.float32), sr=sr, n_mfcc=13)
    L = min(r.shape[1], s.shape[1])
    if L == 0:
        return float('nan')
    return float(np.mean(np.sqrt(2 * np.sum((r[1:, :L] - s[1:, :L]) ** 2, axis=0))))


def compute_secs(wav_path, spk_enc, tgt_emb):
    from resemblyzer import preprocess_wav
    w = preprocess_wav(wav_path)
    ea = spk_enc.embed_utterance(w)
    return float(np.dot(ea, tgt_emb) / (np.linalg.norm(ea) * np.linalg.norm(tgt_emb)))


def compute_tgt_embedding(tgt_files, spk_enc, sr=16000):
    """Compute target speaker embedding from multiple reference files."""
    from resemblyzer import preprocess_wav
    embeddings = []
    for f in tgt_files[:3]:  # Use first 3 files
        w = preprocess_wav(f)
        w = w[:sr * 5]  # 5s max per file
        embeddings.append(spk_enc.embed_utterance(w))
    return np.mean(embeddings, axis=0)


# ── kNN-VC ───────────────────────────────────────────────────

def run_knnvc_pair(src_path, tgt_files, output_path, knnvc_dir):
    """Run kNN-VC on a single source-target pair."""
    import torch, torchaudio

    sys.path.insert(0, knnvc_dir)
    orig_dir = os.getcwd()
    os.chdir(knnvc_dir)

    from hubconf import knn_vc
    if not hasattr(run_knnvc_pair, '_model'):
        print('    Loading kNN-VC model...', flush=True)
        run_knnvc_pair._model = knn_vc(pretrained=True, progress=False, prematched=False, device='cuda')
    model = run_knnvc_pair._model

    # Prepare source 16kHz
    tmp_dir = os.path.dirname(output_path)
    src_16k = os.path.join(tmp_dir, '_knnvc_src.wav')
    w, sr = torchaudio.load(src_path)
    if sr != 16000:
        w = torchaudio.functional.resample(w, sr, 16000)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)
    torchaudio.save(src_16k, w, 16000)

    # Prepare target 16kHz (concatenate reference files, max 30s)
    tgt_wavs = []
    for tf in tgt_files:
        tw, tsr = torchaudio.load(tf)
        if tsr != 16000:
            tw = torchaudio.functional.resample(tw, tsr, 16000)
        if tw.shape[0] > 1:
            tw = tw.mean(0, keepdim=True)
        tgt_wavs.append(tw)
    tgt_cat = torch.cat(tgt_wavs, dim=1)[:, :30*16000]
    tgt_16k = os.path.join(tmp_dir, '_knnvc_tgt.wav')
    torchaudio.save(tgt_16k, tgt_cat, 16000)

    with torch.no_grad():
        query_seq = model.get_features(src_16k)
        matching_set = model.get_matching_set([tgt_16k])
        out_wav = model.match(query_seq, matching_set, topk=4)

    out_np = out_wav.squeeze().cpu().numpy()
    sf.write(output_path, out_np, 16000)
    os.chdir(orig_dir)
    return output_path


# ── FreeVC ───────────────────────────────────────────────────

def run_freevc_pair(src_path, tgt_files, output_path, freevc_dir):
    """Run FreeVC in subprocess on a single pair."""
    import subprocess

    tgt_path = tgt_files[0]  # Use first target file
    ckpt_path = os.path.join(freevc_dir, 'checkpoints', 'freevc.pth')
    config_path = os.path.join(freevc_dir, 'configs', 'freevc.json')
    spk_enc_path = os.path.join(freevc_dir, 'speaker_encoder', 'ckpt', 'pretrained_bak_5805000.pt')

    worker = os.path.join(os.path.dirname(output_path), '_freevc_mp_worker.py')
    with open(worker, 'w') as f:
        f.write(f'''import os, sys
os.chdir({freevc_dir!r})
sys.path.insert(0, {freevc_dir!r})
import torch, librosa, numpy as np, soundfile as sf
import utils
from models import SynthesizerTrn
from speaker_encoder.voice_encoder import SpeakerEncoder
from wavlm import WavLM, WavLMConfig
device = 'cuda'
hps = utils.get_hparams_from_file({config_path!r})
net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, **hps.model).to(device).eval()
utils.load_checkpoint({ckpt_path!r}, net_g, None)
smodel = SpeakerEncoder({spk_enc_path!r}, device=device)
wavlm_ckpt = torch.load({os.path.join(freevc_dir, 'wavlm', 'WavLM-Large.pt')!r}, map_location='cpu')
wavlm_cfg = WavLMConfig(wavlm_ckpt['cfg'])
cmodel = WavLM(wavlm_cfg).to(device)
cmodel.load_state_dict(wavlm_ckpt['model'])
cmodel.eval()
target_sr = hps.data.sampling_rate
src, _ = librosa.load({src_path!r}, sr=target_sr)
tgt, _ = librosa.load({tgt_path!r}, sr=target_sr)
tgt16 = librosa.resample(tgt, orig_sr=target_sr, target_sr=16000)
src16 = librosa.resample(src, orig_sr=target_sr, target_sr=16000)
g_tgt = smodel.embed_utterance(tgt16)
g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)
src_tensor = torch.from_numpy(src16).unsqueeze(0).to(device)
with torch.no_grad():
    c = cmodel.extract_features(src_tensor)[0]
    c = c.transpose(1, 2)
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()
sf.write({output_path!r}, audio, target_sr)
print('done', flush=True)
''')

    result = subprocess.run([sys.executable, worker], capture_output=True, timeout=120, env=os.environ.copy())
    if result.returncode != 0:
        return None
    return output_path if os.path.exists(output_path) else None


# ── CRT Pipeline ─────────────────────────────────────────────

_crt_state = {}

def init_crt(args, device):
    """Load CRT model and Mimi codec once."""
    if _crt_state:
        return

    import torch, faiss
    from transformers import MimiModel, AutoFeatureExtractor

    dtype = torch.float16
    mimi = MimiModel.from_pretrained('kyutai/mimi', torch_dtype=dtype).to(device).eval()
    feat_ext = AutoFeatureExtractor.from_pretrained('kyutai/mimi')
    SR = feat_ext.sampling_rate

    # Load CRT
    import importlib.util
    spec = importlib.util.spec_from_file_location('crt_mod', args.crt_script)
    crt_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crt_mod)

    ckpt = torch.load(args.crt_ckpt, weights_only=False, map_location='cpu')
    cfg = crt_mod.MODEL_CONFIGS[ckpt.get('model_size', 'xlarge')]
    crt_model = crt_mod.CRT(n_cb=18, cb_size=2048, spk_emb_dim=256, **cfg).to(device).eval()
    crt_model.load_state_dict(ckpt['model_state'], strict=False)

    # CB0 embeddings
    q = mimi.quantizer
    layer = q.semantic_residual_vector_quantizer.layers[0] if hasattr(q, 'semantic_residual_vector_quantizer') else q.layers[0]
    cb0_emb = None
    for attr in ['codebook.embed', 'codebook.embed_sum', 'codebook.embedding.weight']:
        obj = layer
        try:
            for part in attr.split('.'):
                obj = getattr(obj, part)
            w = obj.detach().cpu().float()
            if w.dim() == 3:
                w = w.squeeze(0)
            if w.shape[0] >= 2048:
                cb0_emb = w.numpy()
                break
        except AttributeError:
            continue

    from resemblyzer import VoiceEncoder
    spk_enc_cpu = VoiceEncoder(device='cpu')

    _crt_state.update({
        'mimi': mimi, 'feat_ext': feat_ext, 'SR': SR, 'dtype': dtype,
        'crt_model': crt_model, 'cb0_emb': cb0_emb, 'spk_enc': spk_enc_cpu,
        'device': device,
    })
    print('  CRT pipeline initialized.', flush=True)


def run_crt_pair(src_path, tgt_files, output_dir, pair_name, device):
    """Run CRT pipeline on a single pair. Returns dict of variant paths."""
    import torch, faiss

    s = _crt_state
    mimi, feat_ext, SR, dtype = s['mimi'], s['feat_ext'], s['SR'], s['dtype']
    crt_model, cb0_emb, spk_enc = s['crt_model'], s['cb0_emb'], s['spk_enc']

    def encode(wav):
        inp = feat_ext(raw_audio=wav, sampling_rate=SR, return_tensors='pt')
        inp = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else None) for k, v in inp.items()}
        with torch.no_grad():
            return mimi.encode(inp['input_values'], inp.get('padding_mask')).audio_codes.cpu()

    def decode(codes):
        with torch.no_grad():
            return mimi.decode(codes.to(device)).audio_values.cpu().squeeze().numpy().astype(np.float32)

    # Source
    src_wav = librosa.load(src_path, sr=SR, mono=True)[0].astype(np.float32)
    src_codes = encode(src_wav)
    T_src = src_codes.shape[2]

    # Target
    tgt_wavs = [librosa.load(str(f), sr=SR, mono=True)[0].astype(np.float32) for f in tgt_files]
    tgt_concat = np.concatenate(tgt_wavs)
    CHUNK = int(30 * SR)
    tgt_codes = torch.cat([
        encode(tgt_concat[i:i+CHUNK])
        for i in range(0, len(tgt_concat), CHUNK)
        if len(tgt_concat[i:i+CHUNK]) > SR
    ], dim=2)
    tgt_full = tgt_codes[0].numpy().T
    tgt_sem = tgt_codes[0, 0].numpy()

    # FAISS index
    tgt_emb = cb0_emb[tgt_sem].astype(np.float32).copy()
    faiss.normalize_L2(tgt_emb)
    tgt_index = faiss.IndexFlatIP(cb0_emb.shape[1])
    tgt_index.add(tgt_emb)

    # Speaker embedding
    tgt16 = librosa.resample(tgt_concat[:SR*15], orig_sr=SR, target_sr=16000)
    tgt_spk_emb = torch.tensor(spk_enc.embed_utterance(tgt16), dtype=torch.float32)

    # FAISS retrieval
    src_sem_np = src_codes[0, 0].numpy()
    src_emb_q = cb0_emb[src_sem_np].astype(np.float32).copy()
    faiss.normalize_L2(src_emb_q)
    dists_k, indices_k = tgt_index.search(src_emb_q, 8)

    selected = np.zeros(T_src, dtype=np.int64)
    selected[0] = indices_k[0, 0]
    for t in range(1, T_src):
        cands = indices_k[t]
        combined = 0.5 * dists_k[t] + 0.5 / (1 + np.abs(cands - selected[t-1]) / 12.5)
        selected[t] = cands[np.argmax(combined)]

    # FAISS baseline
    faiss_codes = src_codes.clone()
    for t in range(T_src):
        faiss_codes[0, :, t] = torch.tensor(tgt_full[selected[t]])

    # CRT inference
    ret_cb0 = np.array([tgt_full[selected[t]][0] for t in range(T_src)])
    ret_cb = np.array([tgt_full[selected[t]][1:19] for t in range(T_src)])
    cb0_d = torch.tensor(ret_cb0, dtype=torch.long).unsqueeze(0).to(device)
    cb_d = torch.tensor(ret_cb.T, dtype=torch.long).unsqueeze(0).to(device)
    spk_d = tgt_spk_emb.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits, _ = crt_model(cb0_d, cb_d, spk_d)
        probs = logits.float().softmax(-1)
        crt_tokens = probs.argmax(-1)[0].cpu()
        crt_conf = probs.max(-1).values[0].cpu()

    faiss_tokens = torch.tensor(ret_cb.T, dtype=torch.long)

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # FAISS baseline
    faiss_wav = decode(faiss_codes)
    p = os.path.join(output_dir, f'{pair_name}_faiss.wav')
    sf.write(p, faiss_wav, SR)
    results['FAISS'] = p

    # CRT conf>0.4
    sel = faiss_tokens.clone()
    mask = crt_conf > 0.4
    sel[mask] = crt_tokens[mask]
    codes = faiss_codes.clone()
    codes[0, 1:19] = sel
    wav = decode(codes)
    p = os.path.join(output_dir, f'{pair_name}_crt04.wav')
    sf.write(p, wav, SR)
    results['CRT-conf>0.4'] = p

    return results


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('='*70)
    print(' Multi-Pair VC Benchmark (VCTK)')
    print(f' Device: {device}')
    print('='*70, flush=True)

    with open(args.pairs_json) as f:
        all_pairs = json.load(f)

    if args.max_pairs > 0:
        pairs = all_pairs[:args.max_pairs]
    else:
        pairs = all_pairs
    print(f'  Evaluating {len(pairs)} pairs (of {len(all_pairs)} total)', flush=True)

    # Load resemblyzer once
    from resemblyzer import VoiceEncoder, preprocess_wav
    spk_enc = VoiceEncoder(device='cpu')

    # Initialize CRT if needed
    if not args.skip_crt:
        init_crt(args, device)

    all_results = {m: {'secs': [], 'mcd': []} for m in ['FAISS', 'CRT-conf>0.4', 'kNN-VC', 'FreeVC']}

    for i, pair in enumerate(pairs):
        src_spk = pair['source_speaker']
        tgt_spk = pair['target_speaker']
        src_file = pair['source_file']
        tgt_files = pair['target_files']
        pair_name = f'{src_spk}_to_{tgt_spk}'

        print(f'\n[{i+1}/{len(pairs)}] {pair_name}', flush=True)

        # Check files exist
        if not os.path.exists(src_file):
            print(f'  SKIP: source not found: {src_file}', flush=True)
            continue

        tgt_files = [f for f in tgt_files if os.path.exists(f)]
        if not tgt_files:
            print(f'  SKIP: no target files found', flush=True)
            continue

        # Target speaker embedding
        tgt_emb = compute_tgt_embedding(tgt_files, spk_enc)

        # Source audio for MCD
        src_wav, _ = librosa.load(src_file, sr=args.sr, mono=True)

        pair_dir = os.path.join(args.output_dir, pair_name)
        os.makedirs(pair_dir, exist_ok=True)

        # ── CRT ──
        if not args.skip_crt:
            try:
                crt_paths = run_crt_pair(src_file, tgt_files, pair_dir, pair_name, device)
                for method, path in crt_paths.items():
                    s = compute_secs(path, spk_enc, tgt_emb)
                    wav, _ = librosa.load(path, sr=args.sr, mono=True)
                    m = compute_mcd(src_wav, wav, args.sr)
                    all_results[method]['secs'].append(s)
                    all_results[method]['mcd'].append(m)
                    print(f'  {method:<20} SECS={s:.4f} MCD={m:.2f}', flush=True)
            except Exception as e:
                print(f'  CRT FAILED: {e}', flush=True)

        # ── kNN-VC ──
        if not args.skip_knnvc:
            try:
                knnvc_out = os.path.join(pair_dir, f'{pair_name}_knnvc.wav')
                run_knnvc_pair(src_file, tgt_files, knnvc_out, args.knnvc_dir)
                s = compute_secs(knnvc_out, spk_enc, tgt_emb)
                wav, _ = librosa.load(knnvc_out, sr=args.sr, mono=True)
                m = compute_mcd(src_wav, wav, args.sr)
                all_results['kNN-VC']['secs'].append(s)
                all_results['kNN-VC']['mcd'].append(m)
                print(f'  {"kNN-VC":<20} SECS={s:.4f} MCD={m:.2f}', flush=True)
            except Exception as e:
                print(f'  kNN-VC FAILED: {e}', flush=True)

        # ── FreeVC ──
        if not args.skip_freevc:
            try:
                freevc_out = os.path.join(pair_dir, f'{pair_name}_freevc.wav')
                run_freevc_pair(src_file, tgt_files, freevc_out, args.freevc_dir)
                if freevc_out and os.path.exists(freevc_out):
                    s = compute_secs(freevc_out, spk_enc, tgt_emb)
                    wav, _ = librosa.load(freevc_out, sr=args.sr, mono=True)
                    m = compute_mcd(src_wav, wav, args.sr)
                    all_results['FreeVC']['secs'].append(s)
                    all_results['FreeVC']['mcd'].append(m)
                    print(f'  {"FreeVC":<20} SECS={s:.4f} MCD={m:.2f}', flush=True)
            except Exception as e:
                print(f'  FreeVC FAILED: {e}', flush=True)

    # ── Summary ──
    print(f'\n{"="*70}')
    print(f' MULTI-PAIR BENCHMARK RESULTS ({len(pairs)} pairs)')
    print(f'{"="*70}')
    print(f'{"Method":<20} {"SECS":>12} {"MCD":>12} {"N pairs":>8}')
    print('-'*55)
    for method in ['CRT-conf>0.4', 'FAISS', 'kNN-VC', 'FreeVC']:
        r = all_results[method]
        if r['secs']:
            sm, ss = np.mean(r['secs']), np.std(r['secs'])
            mm, ms = np.mean(r['mcd']), np.std(r['mcd'])
            print(f'{method:<20} {sm:.3f}+/-{ss:.3f} {mm:.1f}+/-{ms:.1f} {len(r["secs"]):>8}')
        else:
            print(f'{method:<20} {"N/A":>12} {"N/A":>12} {0:>8}')
    print(f'{"="*70}')

    # Save
    save_data = {}
    for method, r in all_results.items():
        if r['secs']:
            save_data[method] = {
                'secs_mean': float(np.mean(r['secs'])),
                'secs_std': float(np.std(r['secs'])),
                'mcd_mean': float(np.mean(r['mcd'])),
                'mcd_std': float(np.std(r['mcd'])),
                'n_pairs': len(r['secs']),
                'secs_all': [float(x) for x in r['secs']],
                'mcd_all': [float(x) for x in r['mcd']],
            }
    with open(os.path.join(args.output_dir, 'multipair_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'\nResults saved to {args.output_dir}/multipair_results.json')


if __name__ == '__main__':
    main()
