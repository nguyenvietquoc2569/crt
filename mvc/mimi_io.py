"""Mimi encode / decode for Mimi-VC: wav -> (latent [T, 512], codes [T, 32]) and latent -> wav.

decode_latent(x) feeds the decoder the unquantised path that Mimi's training used part of the time:
    dec_in = sem.output_proj(sem.input_proj(x)) + ac.output_proj(ac.input_proj(x))
decode_codes(codes) is MimiModel's own decoder on 32 codes; quantize(x) gives the codes of a latent (Mimi's RVQ), so a
predicted latent can also be emitted as tokens.
"""
import numpy as np

import encode_mimi as em


class MimiIO:
    def __init__(self, device='cuda', model=None, decoder_ckpt=None):
        """decoder_ckpt: a decoder_ft.py checkpoint; decode_latent then uses that fine-tuned copy of the upsample,
        decoder transformer and SEANet decoder, while decode_codes keeps Mimi's own decoder."""
        import copy
        import torch
        self.torch, self.device = torch, device
        self.enc = em.MimiEncoder(device, model=model)
        self.m = self.enc.m
        q = self.m.quantizer
        self.sem, self.ac = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer
        self.dec = dict(upsample=self.m.upsample, decoder_transformer=self.m.decoder_transformer, decoder=self.m.decoder)
        self.decoder_ckpt = decoder_ckpt
        if decoder_ckpt:
            sd = torch.load(decoder_ckpt, map_location=device, weights_only=False)['decoder']
            self.dec = {k: copy.deepcopy(v) for k, v in self.dec.items()}
            for k, v in self.dec.items():
                v.load_state_dict(sd[k])
                v.eval()

    def encode(self, wav):
        lat, codes = self.enc([wav])[0]
        return lat, codes

    def decoder_input(self, x):
        """x [T, 512] (numpy or tensor) -> unquantised decoder input [1, 512, T]."""
        torch = self.torch
        z = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device).T[None]
        return self.sem.output_proj(self.sem.input_proj(z)) + self.ac.output_proj(self.ac.input_proj(z))

    def _decode_input(self, emb, d=None):
        d = d or self.dec
        emb = d['upsample'](emb)
        h = d['decoder_transformer'](emb.transpose(1, 2), return_dict=False)[0].transpose(1, 2)
        return d['decoder'](h).float().cpu().squeeze().numpy()

    def decode_latent(self, x):
        with self.torch.no_grad():
            return self._decode_input(self.decoder_input(x))

    def relax(self, x, alpha, k=2):
        """Damped Mimi round trip of a generated latent x [T, 512]: k times x <- x + alpha (E(D(x)) - x), with Mimi's own
        decoder and encoder (never a fine-tuned decoder). It pulls the acoustic part of generated latents toward the
        encoder's manifold (26 Oct, local probes: alpha 0.15-0.25, k 2, +0.16 to +0.20 UTMOS on generated latents;
        it hurts real latents, -0.07, so it is for converted latents only)."""
        stock = dict(upsample=self.m.upsample, decoder_transformer=self.m.decoder_transformer, decoder=self.m.decoder)
        x = np.asarray(x, dtype=np.float32)
        for _ in range(k):
            with self.torch.no_grad():
                w = self._decode_input(self.decoder_input(x), stock)
            x = x + alpha * (np.asarray(self.encode(w)[0], dtype=np.float32)[:len(x)] - x)
        return x

    def quantize(self, x):
        torch = self.torch
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device).T[None]
            return self.m.quantizer.encode(z).transpose(0, 1)[0].T.cpu().numpy()     # [T, 32]

    def cb0_table(self):
        """Mimi's CB0 codewords [2048, 256] (embed_sum / cluster_usage of the semantic codebook), as crt_decode_lab."""
        cb = self.sem.layers[0].codebook
        return (cb.embed_sum.float() / cb.cluster_usage.float().clamp(min=cb.epsilon)[:, None]).cpu().numpy()

    def decode_codes(self, codes):
        torch = self.torch
        with torch.no_grad():
            c = torch.as_tensor(np.asarray(codes), dtype=torch.long, device=self.device).T[None]
            return self.m.decode(c).audio_values.float().cpu().squeeze().numpy()
