"""Tests of the model-free logic of the baseline adapters benchmark/baselines/{knnvc,freevc,seedvc,facodec}.py
(ANALYSIS_PLAN_v1 §2.4): the fixed settings, the reference handling (kNN-VC's matching set and FreeVC's trimmed
speaker embedding cached per reference object, Seed-VC's first 25 s, FACodec's padding to a multiple of 200 samples),
the git commit / changed-tree checks (subprocess.check_output stubbed), and describe() as run_baselines.py consumes it.

No torch and no repository are installed here, so load() runs against stand-ins: a fake torch module, and fake
repository modules written under tmp_path. Those of kNN-VC (matcher.py, hubconf.knn_vc), Seed-VC (inference.py's
modules) and FACodec (Amphion ns3_codec V2 classes) copy the upstream signatures, so a keyword the adapter passes that
upstream does not take fails here too; FreeVC's follow the calls of its convert.py. The fakes record their calls.
sys.modules and sys.path are restored after every test.
"""
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import types

import librosa
import librosa.effects  # noqa: F401  (loaded before the tests snapshot sys.modules)
import numpy as np
import pytest

import baselines
import run_baselines as rb

STUB_ROOTS = ('torch', 'torchaudio', 'torchvision', 'matcher', 'hubconf', 'utils', 'models', 'wavlm',
              'speaker_encoder', 'modules', 'Amphion', 'einops', 'yaml', 'huggingface_hub', 'transformers')
ADAPTERS = tuple(f'baselines.{s}' for s in baselines.SYSTEMS)
COMMIT = '0123456789abcdef0123456789abcdef01234567'


def _ours(name):
    return name.split('.')[0] in STUB_ROOTS or name in ADAPTERS


@pytest.fixture
def isolated(monkeypatch):
    """Fresh adapter modules and stand-ins; sys.modules entries of these names and sys.path restored afterwards."""
    monkeypatch.setattr(sys, 'path', list(sys.path))
    saved = {k: v for k, v in sys.modules.items() if _ours(k)}
    attrs = {s: getattr(baselines, s) for s in baselines.SYSTEMS if hasattr(baselines, s)}
    for k in saved:
        del sys.modules[k]
    for s in attrs:
        delattr(baselines, s)
    yield
    for k in [k for k in sys.modules if _ours(k)]:
        del sys.modules[k]
    sys.modules.update(saved)
    for s in baselines.SYSTEMS:
        if hasattr(baselines, s):
            delattr(baselines, s)
    for s, v in attrs.items():
        setattr(baselines, s, v)


class Git:
    """subprocess.check_output for `git -C <dir> ... rev-parse HEAD` and `... status --porcelain`."""

    def __init__(self, changed=''):
        self.changed, self.calls = changed, []

    def __call__(self, cmd, text=False, **kw):
        assert cmd[0] == 'git' and text is True, cmd
        self.calls.append(cmd)
        if cmd[-2:] == ['rev-parse', 'HEAD']:
            return COMMIT + '\n'
        if cmd[-3:] == ['status', '--porcelain', '--untracked-files=no']:
            return self.changed
        raise AssertionError(f'unexpected git call {cmd}')


class FakeTensor:
    """A numpy array behind the torch.Tensor methods the adapters call."""

    def __init__(self, a, device='cpu', tag=None):
        self.a, self.device, self.tag = np.asarray(a), str(device), tag

    def unsqueeze(self, d):
        return FakeTensor(np.expand_dims(self.a, d), self.device, self.tag)

    def to(self, device):
        return FakeTensor(self.a, device, self.tag)

    def float(self):
        return FakeTensor(self.a.astype(np.float32), self.device, self.tag)

    def cpu(self):
        return FakeTensor(self.a, 'cpu', self.tag)

    def numpy(self):
        return self.a

    @property
    def data(self):
        return self

    def size(self, d=None):
        return self.a.shape if d is None else self.a.shape[d]

    def __getitem__(self, i):
        return FakeTensor(self.a[i], self.device, self.tag)


def fake_torch():
    torch = types.ModuleType('torch')
    torch.__version__ = 'fake'
    torch.calls = []
    torch.float16, torch.float32 = 'float16', 'float32'
    torch.no_grad = contextlib.nullcontext
    torch.from_numpy = lambda a: FakeTensor(a)

    def load(f, map_location=None, weights_only=None):
        torch.calls.append(('load', str(f), map_location, weights_only))
        return {'state_dict_of': os.path.basename(f)}
    torch.load = load
    torch.device = FakeDevice
    return torch


class FakeDevice:
    def __init__(self, spec):
        self.spec, self.type = str(spec), str(spec).split(':')[0]

    def __str__(self):
        return self.spec


def write_tree(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text)


def contract(mod, name):
    """describe() through run_baselines.describe (contract keys, sha256 of every checkpoint), as conv_cfg uses it."""
    d = rb.describe(mod, name)
    json.dumps(d, sort_keys=True)
    for path, sha in d['checkpoints'].items():
        assert os.path.isabs(path) and sha == hashlib.sha256(open(path, 'rb').read()).hexdigest()
    return d


def test_systems_are_the_four_baselines_of_the_plan():
    assert baselines.SYSTEMS == ('knnvc', 'freevc', 'seedvc', 'facodec')
    bench = os.path.dirname(os.path.abspath(baselines.__file__))
    assert all(os.path.exists(os.path.join(bench, f'{s}.py')) for s in baselines.SYSTEMS)


# ── S5 kNN-VC ────────────────────────────────────────────────

class FakeKnnVC:
    """KNeighborsVC of knn-vc matcher.py: the methods the adapter calls, with their signatures."""
    sr = 16000   # self.h.sampling_rate of hifigan/config_v1_wavlm.json

    def __init__(self):
        self.calls = []

    def get_matching_set(self, wavs, weights=None, vad_trigger_level=7):
        self.calls.append(('get_matching_set', [w.a.copy() for w in wavs], [w.a.flags.writeable for w in wavs],
                           weights, vad_trigger_level))
        return FakeTensor(np.ones((len(wavs[0].a) // 320, 1024), np.float32), tag=len(self.calls))

    def get_features(self, path, weights=None, vad_trigger_level=0):
        self.calls.append(('get_features', path.a.copy(), weights, vad_trigger_level))
        return FakeTensor(np.ones((len(path.a) // 320, 1024), np.float32))

    def match(self, query_seq, matching_set, synth_set=None, topk=4, tgt_loudness_db=-16, target_duration=None,
              device=None):
        self.calls.append(('match', matching_set.tag, synth_set, topk, tgt_loudness_db, target_duration, device))
        return FakeTensor(np.full(len(query_seq.a) * 320, 0.1, np.float32))


def knn_vc_entry(pretrained=True, progress=True, prematched=True, device='cuda'):
    """The signature of hubconf.knn_vc."""


@pytest.fixture
def knnvc(isolated, monkeypatch, tmp_path):
    hub = tmp_path / 'torch_cache' / 'hub'
    write_tree(hub / 'checkpoints', {f: f.encode() * 10 for f in ('WavLM-Large.pt', 'prematch_g_02500000.pt',
                                                                  'g_02500000.pt')})
    torch = fake_torch()
    knn = FakeKnnVC()
    sys.modules['torch'] = torch                              # (isolated)

    def hub_load(repo_or_dir, model, *args, source='github', trust_repo=None, force_reload=False, verbose=True,
                 skip_validation=False, **kwargs):
        inspect.signature(knn_vc_entry).bind(*args, **kwargs)
        torch.calls.append(('hub.load', repo_or_dir, model, args, source, kwargs))
        sys.modules['matcher'] = types.SimpleNamespace(SPEAKER_INFORMATION_LAYER=6)   # what hubconf's import leaves
        return knn
    torch.hub = types.SimpleNamespace(get_dir=lambda: str(hub), load=hub_load)
    mod = importlib.import_module('baselines.knnvc')
    repo = tmp_path / 'knn-vc'
    write_tree(repo, {f: f'# {f} of bshall/knn-vc\n' for f in mod.CODE})
    monkeypatch.setattr(mod, 'REPO_DIR', str(repo))
    return types.SimpleNamespace(mod=mod, torch=torch, knn=knn, hub=hub, repo=repo)


def test_knnvc_settings_are_those_of_the_plan(knnvc):
    m = knnvc.mod
    assert m.DEFAULTS == {'prematched': True, 'topk': 4, 'ref_vad_trigger_level': 7, 'tgt_loudness_db': -16}
    assert m.SR == 16000 and m.REPO == 'https://github.com/bshall/knn-vc' and m.COMMIT.startswith('c616845')


def test_knnvc_load_uses_the_local_checkout_and_the_hub_checkpoints(knnvc):
    m = knnvc.mod
    m.load('cpu')
    [call] = [c for c in knnvc.torch.calls if c[0] == 'hub.load']
    assert call == ('hub.load', str(knnvc.repo), 'knn_vc', (), 'local',
                    {'prematched': True, 'pretrained': True, 'progress': False, 'device': 'cpu'})
    ck = knnvc.hub / 'checkpoints'
    assert m._ckpts == [str(ck / 'WavLM-Large.pt'), str(ck / 'prematch_g_02500000.pt')]


def test_knnvc_load_without_prematching_takes_the_other_vocoder(knnvc):
    knnvc.mod.load('cuda', prematched=False)
    call = [c for c in knnvc.torch.calls if c[0] == 'hub.load'][0]
    assert call[5]['prematched'] is False and call[5]['device'] == 'cuda'
    assert knnvc.mod._ckpts[1] == str(knnvc.hub / 'checkpoints' / 'g_02500000.pt')


def test_knnvc_load_refuses_unknown_settings_and_missing_weights(knnvc, monkeypatch):
    with pytest.raises(SystemExit, match=r"knnvc cfg keys \['k'\] not in"):
        knnvc.mod.load('cpu', k=4)
    os.remove(knnvc.hub / 'checkpoints' / 'prematch_g_02500000.pt')
    monkeypatch.setenv('TORCH_HOME', '/somewhere/torch_cache')
    with pytest.raises(SystemExit, match=r'kNN-VC weights missing: .*prematch_g_02500000.pt.*TORCH_HOME=/somewhere'):
        knnvc.mod.load('cpu')
    assert not [c for c in knnvc.torch.calls if c[0] == 'hub.load']   # hubconf never gets to download them


def test_knnvc_convert_builds_the_matching_set_once_per_reference_object(knnvc):
    m, knn = knnvc.mod, knnvc.knn
    m.load('cpu')
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000 * 3).astype(np.float32)
    ref.flags.writeable = False                         # as the harness passes it
    srcs = [rng.standard_normal(n).astype(np.float32) for n in (16000, 24000)]
    for src in srcs:
        wav, sr = m.convert(src, ref, 16000)
        assert isinstance(wav, np.ndarray) and wav.dtype == np.float32 and wav.ndim == 1 and sr == 16000
    ms = [c for c in knn.calls if c[0] == 'get_matching_set']
    assert len(ms) == 1
    _, wavs, writeable, weights, vad = ms[0]
    assert len(wavs) == 1 and np.array_equal(wavs[0], ref) and writeable == [True] and weights is None and vad == 7
    feats = [c for c in knn.calls if c[0] == 'get_features']
    assert [np.array_equal(f[1], s) for f, s in zip(feats, srcs)] == [True, True]
    assert all(f[2:] == (None, 0) for f in feats)      # the source: no VAD
    match = [c for c in knn.calls if c[0] == 'match']
    assert match == [('match', 1, None, 4, -16, None, None)] * 2   # synth set = matching set, k = 4, -16 LUFS
    m.convert(srcs[0], ref.copy(), 16000)                # equal samples, another object: another reference
    assert len([c for c in knn.calls if c[0] == 'get_matching_set']) == 2


def test_knnvc_describe(knnvc):
    m = knnvc.mod
    m.load('cpu', topk=8)
    d = contract(m, 'knnvc')
    assert set(rb.DESCRIBE_KEYS) <= set(d) and d['repo'] == m.REPO and d['commit'] == m.COMMIT and d['sr_out'] == 16000
    assert list(d['checkpoints']) == m._ckpts
    s = d['settings']
    assert (s['prematched'], s['topk'], s['ref_vad_trigger_level'], s['tgt_loudness_db']) == (True, 8, 7, -16)
    assert (s['wavlm_layer'], s['src_vad_trigger_level'], s['target_duration']) == (6, 0, None)
    want = {f: hashlib.sha256((knnvc.repo / f).read_bytes()).hexdigest() for f in m.CODE}
    assert d['code_sha256'] == want
    (knnvc.repo / 'matcher.py').write_text('# edited\n')  # an edited checkout changes conv_cfg, not only COMMIT
    assert m.describe()['code_sha256']['matcher.py'] != want['matcher.py']


# ── S6 FreeVC ────────────────────────────────────────────────

FREEVC_CONFIG = {
    'train': {'segment_size': 8960, 'seed': 1234},
    'data': {'max_wav_value': 32768.0, 'sampling_rate': 16000, 'filter_length': 1280, 'hop_length': 320,
             'win_length': 1280, 'n_mel_channels': 80, 'mel_fmin': 0.0, 'mel_fmax': None},
    'model': {'inter_channels': 192, 'hidden_channels': 192, 'filter_channels': 768, 'n_heads': 2, 'n_layers': 6,
              'kernel_size': 3, 'p_dropout': 0.1, 'resblock': '1', 'resblock_kernel_sizes': [3, 7, 11],
              'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 'upsample_rates': [10, 8, 2, 2],
              'upsample_initial_channel': 512, 'upsample_kernel_sizes': [16, 16, 4, 4], 'n_layers_q': 3,
              'use_spectral_norm': False, 'gin_channels': 256, 'ssl_dim': 1024, 'use_spk': True},
}

FREEVC_FAKES = {
    'utils.py': '''
import json
import numpy as np
import torchvision  # noqa: F401  (FreeVC utils.py imports it for training augmentations)
CALLS = []


class HParams:   # FreeVC / VITS utils.HParams
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if type(v) == dict else v

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


def get_hparams_from_file(config_path):
    return HParams(**json.load(open(config_path)))


def load_checkpoint(checkpoint_path, model, optimizer=None, strict=False):
    CALLS.append(('load_checkpoint', checkpoint_path, model, optimizer, strict, model.device, model.training))


def get_content(cmodel, y):
    import torch
    CALLS.append(('get_content', cmodel, y.a.copy(), y.device))
    return torch.from_numpy(np.zeros((1, 1024, y.a.shape[-1] // 320), np.float32)).to(y.device)
''',
    'models.py': '''
import numpy as np
import utils


class SynthesizerTrn:
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads,
                 n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, ssl_dim, use_spk,
                 **kwargs):
        utils.CALLS.append(('SynthesizerTrn', spec_channels, segment_size, ssl_dim, use_spk, sorted(kwargs)))
        self.device, self.training = None, True

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def infer(self, c, g=None, mel=None, c_lengths=None):
        import torch
        utils.CALLS.append(('infer', c.a.shape, g.tag, g.a.shape, g.device, mel))
        return torch.from_numpy(np.full((1, 1, c.a.shape[2] * 320), 0.5, np.float32)).to(c.device)
''',
    'speaker_encoder/__init__.py': '',
    'speaker_encoder/voice_encoder.py': '''
import numpy as np
import utils


class SpeakerEncoder:
    def __init__(self, weights_fpath, device=None, verbose=True):
        utils.CALLS.append(('SpeakerEncoder', weights_fpath, device))

    def embed_utterance(self, wav, return_partials=False, rate=1.3, min_coverage=0.75):
        utils.CALLS.append(('embed_utterance', wav.copy(), return_partials, rate, min_coverage))
        return np.full(256, 1 / 16, np.float32)
''',
    'wavlm/__init__.py': '''
import utils


class WavLMConfig:
    def __init__(self, cfg=None):
        self.cfg = cfg


class WavLM:
    def __init__(self, cfg):
        utils.CALLS.append(('WavLM', cfg.cfg))
        self.device, self.training = None, True

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict, strict=True):
        utils.CALLS.append(('WavLM.load_state_dict', state_dict, strict, self.device))

    def eval(self):
        self.training = False
        return self
''',
    'checkpoints/freevc.pth': b'generator weights',
    'speaker_encoder/ckpt/pretrained_bak_5805000.pt': b'speaker encoder weights',
    'wavlm/WavLM-Large.pt': b'wavlm weights',
}


@pytest.fixture
def freevc(isolated, monkeypatch, tmp_path):
    root = tmp_path / 'FreeVC'
    write_tree(root, dict(FREEVC_FAKES, **{'configs/freevc.json': json.dumps(FREEVC_CONFIG)}))
    git = Git()
    monkeypatch.setattr(subprocess, 'check_output', git)
    torch = fake_torch()

    def load(f, map_location=None, weights_only=None):   # the WavLM-Large.pt layout
        torch.calls.append(('load', str(f), map_location, weights_only))
        return {'cfg': {'encoder_layers': 24}, 'model': {'weights': 'wavlm'}}
    torch.load = load
    sys.modules['torch'] = torch                              # (isolated)
    mod = importlib.import_module('baselines.freevc')
    return types.SimpleNamespace(mod=mod, root=root, git=git, torch=torch)


def test_freevc_settings(freevc):
    m = freevc.mod
    assert (m.SR, m.TOP_DB, m.REPO) == (16000, 20, 'https://github.com/OlaWod/FreeVC')
    assert (m.SPK_CKPT, m.WAVLM_CKPT) == ('speaker_encoder/ckpt/pretrained_bak_5805000.pt', 'wavlm/WavLM-Large.pt')
    assert inspect.signature(m.load).parameters['ptfile'].default == 'checkpoints/freevc.pth'


def test_freevc_load_refuses_a_changed_checkout_before_importing_anything(freevc):
    freevc.git.changed = ' M models.py\n'
    with pytest.raises(SystemExit, match=f'differs from its commit {COMMIT}:\n M models.py'):
        freevc.mod.load('cpu', root=str(freevc.root))
    git = ['git', '-C', str(freevc.root), '--no-optional-locks']
    assert freevc.git.calls == [git + ['rev-parse', 'HEAD'], git + ['status', '--porcelain', '--untracked-files=no']]
    assert 'utils' not in sys.modules and freevc.mod.M == {}


def test_freevc_load_is_convert_py_model_loading(freevc):
    m, root = freevc.mod, str(freevc.root)
    m.load('cpu', root=root)
    assert sys.path[0] == root and m.M['commit'] == COMMIT and m.M['device'] == 'cpu'
    assert sys.modules['torchvision'].__dict__.keys() <= {'__name__', '__doc__', '__package__', '__loader__',
                                                          '__spec__'}                  # the empty stand-in
    calls = sys.modules['utils'].CALLS
    assert [c[0] for c in calls] == ['SynthesizerTrn', 'load_checkpoint', 'WavLM', 'WavLM.load_state_dict',
                                     'SpeakerEncoder']
    assert calls[0][1:] == (641, 28, 1024, True, ['n_layers_q', 'use_spectral_norm'])   # 1280 // 2 + 1, 8960 // 320
    net_g = m.M['net_g']
    assert calls[1] == ('load_checkpoint', f'{root}/checkpoints/freevc.pth', net_g, None, True, 'cpu', False)
    assert calls[2] == ('WavLM', {'encoder_layers': 24})
    assert calls[3] == ('WavLM.load_state_dict', {'weights': 'wavlm'}, True, 'cpu')
    assert calls[4] == ('SpeakerEncoder', f'{root}/speaker_encoder/ckpt/pretrained_bak_5805000.pt', 'cpu')
    assert ('load', f'{root}/wavlm/WavLM-Large.pt', 'cpu', None) in freevc.torch.calls
    assert not net_g.training and not m.M['cmodel'].training


@pytest.mark.parametrize('change', [{'data': {'sampling_rate': 22050}}, {'model': {'use_spk': False}}])
def test_freevc_load_refuses_another_config(freevc, change):
    cfg = json.loads(json.dumps(FREEVC_CONFIG))
    for k, v in change.items():
        cfg[k].update(v)
    (freevc.root / 'configs' / 'other.json').write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="runs convert.py's use_spk path at 16000 Hz"):
        freevc.mod.load('cpu', root=str(freevc.root), hpfile='configs/other.json')


def test_freevc_convert_trims_and_embeds_each_reference_once(freevc):
    m = freevc.mod
    m.load('cpu', root=str(freevc.root))
    calls = sys.modules['utils'].CALLS
    del calls[:]
    t = np.arange(16000 * 2) / 16000
    speech = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ref = np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(8000, np.float32)])
    ref.flags.writeable = False
    src = np.random.default_rng(1).standard_normal(16000 + 123).astype(np.float32)
    for _ in range(2):
        wav, sr = m.convert(src, ref, 16000)
        assert sr == 16000 and wav.dtype == np.float32 and wav.ndim == 1 and len(wav) == (len(src) // 320) * 320
    emb = [c for c in calls if c[0] == 'embed_utterance']
    assert len(emb) == 1                                       # cached on `ref is M['ref']`
    trimmed = librosa.effects.trim(ref, top_db=20)[0]
    assert np.array_equal(emb[0][1], trimmed) and len(trimmed) < len(ref) - 8000   # both silent ends dropped
    assert emb[0][2:] == (False, 1.3, 0.75)                    # embed_utterance's defaults (rate, min_coverage)
    content = [c for c in calls if c[0] == 'get_content']
    assert len(content) == 2 and content[0][1] is m.M['cmodel'] and content[0][2].shape == (1, len(src))
    assert np.array_equal(content[0][2][0], src) and content[0][3] == 'cpu'
    infer = [c for c in calls if c[0] == 'infer']
    assert infer[0][1:] == ((1, 1024, len(src) // 320), None, (1, 256), 'cpu', None)
    m.convert(src, ref.copy(), 16000)
    assert len([c for c in calls if c[0] == 'embed_utterance']) == 2


def test_freevc_describe(freevc):
    m, root = freevc.mod, str(freevc.root)
    m.load('cpu', root=root)
    d = contract(m, 'freevc')
    assert set(rb.DESCRIBE_KEYS) <= set(d) and d['flags'] == ['VCTK-seen'] and d['commit'] == COMMIT
    assert list(d['checkpoints']) == [f'{root}/checkpoints/freevc.pth',
                                      f'{root}/speaker_encoder/ckpt/pretrained_bak_5805000.pt',
                                      f'{root}/wavlm/WavLM-Large.pt']
    s = d['settings']
    assert d['sr_out'] == 16000 and s['config'] == FREEVC_CONFIG and s['hpfile'] == f'{root}/configs/freevc.json'
    assert (s['ref_trim_top_db'], s['embed_rate'], s['embed_min_coverage'], s['prior_noise_scale'],
            s['strict_load']) == (20, 1.3, 0.75, 1.0, True)
    assert json.dumps(m.describe(), sort_keys=True) == json.dumps(m.describe(), sort_keys=True)


# ── S7 Seed-VC ───────────────────────────────────────────────

SEEDVC_CONFIG = {
    'preprocess_params': {'sr': 22050, 'spect_params': {'n_fft': 1024, 'win_length': 1024, 'hop_length': 256,
                                                        'n_mels': 80, 'fmin': 0, 'fmax': 'None'}},
    'model_params': {'dit_type': 'DiT', 'speech_tokenizer': {'type': 'whisper', 'name': 'openai/whisper-small'},
                     'vocoder': {'type': 'bigvgan', 'name': 'nvidia/bigvgan_v2_22khz_80band_256x'},
                     'style_encoder': {'dim': 192}},
}

SEEDVC_FAKES = {
    'modules/__init__.py': '',
    'modules/audio.py': '''
CALLS = []


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    CALLS.append(('mel_spectrogram', dict(n_fft=n_fft, num_mels=num_mels, sampling_rate=sampling_rate,
                                          hop_size=hop_size, win_size=win_size, fmin=fmin, fmax=fmax, center=center)))
    return y
''',
    'modules/commons.py': '''
from modules.audio import CALLS


class Munch(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    if isinstance(d, list):
        return [recursive_munch(v) for v in d]
    return d


class Part:
    def __init__(self, name):
        self.name = name
        self.estimator = self

    def eval(self):
        CALLS.append((self.name + '.eval',))
        return self

    def to(self, device):
        CALLS.append((self.name + '.to', str(device)))
        return self

    def setup_caches(self, max_batch_size, max_seq_length):
        CALLS.append(('setup_caches', max_batch_size, max_seq_length))


def build_model(args, stage="DiT"):
    CALLS.append(('build_model', args.dit_type, stage))
    return Munch(cfm=Part('cfm'), length_regulator=Part('length_regulator'))


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[], is_distributed=False,
                    load_ema=False):
    CALLS.append(('load_checkpoint', path, optimizer, load_only_params, ignore_modules, is_distributed, load_ema))
    return model, None, None, None
''',
    'modules/bigvgan/__init__.py': '',
    'modules/bigvgan/bigvgan.py': '''
from modules.audio import CALLS


class BigVGAN:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, use_cuda_kernel=False, **kw):
        CALLS.append(('BigVGAN.from_pretrained', pretrained_model_name_or_path, use_cuda_kernel, kw))
        return cls()

    def remove_weight_norm(self):
        CALLS.append(('BigVGAN.remove_weight_norm',))

    def eval(self):
        CALLS.append(('BigVGAN.eval',))
        return self

    def to(self, device):
        CALLS.append(('BigVGAN.to', str(device)))
        return self
''',
    'modules/campplus/__init__.py': '',
    'modules/campplus/DTDNN.py': '''
from modules.audio import CALLS


class CAMPPlus:
    def __init__(self, feat_dim=80, embedding_size=512, growth_rate=32, bn_size=4, init_channels=128,
                 config_str='batchnorm-relu', memory_efficient=True):
        CALLS.append(('CAMPPlus', feat_dim, embedding_size))

    def load_state_dict(self, state_dict, strict=True):
        CALLS.append(('CAMPPlus.load_state_dict', state_dict))

    def eval(self):
        CALLS.append(('CAMPPlus.eval',))
        return self

    def to(self, device):
        CALLS.append(('CAMPPlus.to', str(device)))
        return self
''',
}


@pytest.fixture
def seedvc(isolated, monkeypatch, tmp_path):
    repo, hf = tmp_path / 'seed-vc', tmp_path / 'hf_cache'
    write_tree(repo, SEEDVC_FAKES)
    git = Git()
    monkeypatch.setattr(subprocess, 'check_output', git)
    hub_calls, tf_calls = [], []

    def hf_hub_download(repo_id, filename, **kw):
        hub_calls.append((repo_id, filename, kw))
        p = hf / repo_id / filename
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f'{repo_id}/{filename}'.encode())
        return str(p)
    hub = types.ModuleType('huggingface_hub')
    hub.hf_hub_download = hf_hub_download
    cfg = hf / 'Plachta/Seed-VC' / 'config_dit_mel_seed_uvit_whisper_small_wavenet.yml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps(SEEDVC_CONFIG))            # JSON is YAML; the stand-in yaml reads it with json
    yaml = types.ModuleType('yaml')
    yaml.safe_load = lambda f: json.load(f)

    whisper = []

    class WhisperModel:
        @classmethod
        def from_pretrained(cls, name, torch_dtype=None):
            tf_calls.append(('WhisperModel.from_pretrained', name, torch_dtype))
            whisper.append(cls())
            whisper[-1].decoder, whisper[-1].encoder = 'decoder', 'encoder'
            return whisper[-1]

        def to(self, device):
            tf_calls.append(('WhisperModel.to', str(device)))
            return self

    class AutoFeatureExtractor:
        @classmethod
        def from_pretrained(cls, name):
            tf_calls.append(('AutoFeatureExtractor.from_pretrained', name))
            return cls()
    transformers = types.ModuleType('transformers')
    transformers.WhisperModel, transformers.AutoFeatureExtractor = WhisperModel, AutoFeatureExtractor
    torch = fake_torch()
    sys.modules.update(torch=torch, yaml=yaml, huggingface_hub=hub, transformers=transformers)   # (isolated)
    mod = importlib.import_module('baselines.seedvc')
    monkeypatch.setattr(mod, 'REPO', str(repo))
    monkeypatch.setattr(mod, '_st', {})
    return types.SimpleNamespace(mod=mod, repo=repo, hf=hf, git=git, torch=torch, hub_calls=hub_calls,
                                 tf_calls=tf_calls, whisper=whisper)


def test_seedvc_defaults_are_those_of_the_plan(seedvc):
    m = seedvc.mod
    p = inspect.signature(m.load).parameters
    assert {k: p[k].default for k in ('diffusion_steps', 'length_adjust', 'inference_cfg_rate', 'fp16')} == {
        'diffusion_steps': 25, 'length_adjust': 1.0, 'inference_cfg_rate': 0.7, 'fp16': True}   # 25: the README's
    assert m.SR == 22050 and m.REPO_URL == 'https://github.com/Plachtaa/seed-vc'
    assert m.DIT == ('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth',
                     'config_dit_mel_seed_uvit_whisper_small_wavenet.yml')


def test_seedvc_load_is_inference_py_load_models(seedvc):
    m, hf = seedvc.mod, seedvc.hf
    m.load('cpu')
    calls = sys.modules['modules.audio'].CALLS
    dit = str(hf / 'Plachta/Seed-VC' / 'DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth')
    assert calls[:9] == [
        ('build_model', 'DiT', 'DiT'), ('load_checkpoint', dit, None, True, [], False, False),
        ('cfm.eval',), ('cfm.to', 'cpu'), ('length_regulator.eval',), ('length_regulator.to', 'cpu'),
        ('setup_caches', 1, 8192), ('CAMPPlus', 80, 192),
        ('CAMPPlus.load_state_dict', {'state_dict_of': 'campplus_cn_common.bin'})]
    assert ('BigVGAN.from_pretrained', 'nvidia/bigvgan_v2_22khz_80band_256x', False, {}) in calls
    assert calls.index(('BigVGAN.remove_weight_norm',)) < calls.index(('BigVGAN.eval',)) < calls.index(
        ('BigVGAN.to', 'cpu'))
    assert ('load', str(hf / 'funasr/campplus/campplus_cn_common.bin'), 'cpu', None) in seedvc.torch.calls
    assert seedvc.tf_calls == [('WhisperModel.from_pretrained', 'openai/whisper-small', 'float16'),
                               ('WhisperModel.to', 'cpu'), ('AutoFeatureExtractor.from_pretrained',
                                                            'openai/whisper-small')]
    st = m._st
    assert len(seedvc.whisper) == 1 and not hasattr(seedvc.whisper[0], 'decoder')   # del whisper_model.decoder
    assert st['checkpoints'] == [dit, str(hf / 'funasr/campplus/campplus_cn_common.bin'),
                                 str(hf / 'nvidia/bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt'),
                                 str(hf / 'openai/whisper-small/model.safetensors')]
    assert st['configs'] == [str(hf / 'Plachta/Seed-VC/config_dit_mel_seed_uvit_whisper_small_wavenet.yml'),
                             str(hf / 'nvidia/bigvgan_v2_22khz_80band_256x/config.json'),
                             str(hf / 'openai/whisper-small/config.json'),
                             str(hf / 'openai/whisper-small/preprocessor_config.json')]
    s = st['settings']
    assert (s['model'], s['diffusion_steps'], s['length_adjust'], s['inference_cfg_rate'], s['fp16']) == (
        'seed-uvit-whisper-small-wavenet', 25, 1.0, 0.7, True)
    assert (s['ref_max_s'], s['context_window_s'], s['overlap_frames'], s['f0_condition'], s['auto_f0_adjust'],
            s['semi_tone_shift']) == (25, 30, 16, False, False, 0)
    mel = {'n_fft': 1024, 'win_size': 1024, 'hop_size': 256, 'num_mels': 80, 'sampling_rate': 22050, 'fmin': 0,
           'fmax': None, 'center': False}
    assert s['mel'] == mel
    st['mel_fn']('wave')                                # the arguments bind to modules.audio.mel_spectrogram
    assert calls[-1] == ('mel_spectrogram', mel)


def test_seedvc_settings_can_be_overridden(seedvc):
    seedvc.mod.load('cpu', diffusion_steps=10, fp16=False)
    s = seedvc.mod._st['settings']
    assert (s['diffusion_steps'], s['fp16'], s['inference_cfg_rate']) == (10, False, 0.7)


@pytest.mark.parametrize('changed, commit', [('', COMMIT), (' M modules/commons.py\n', COMMIT + '-dirty')])
def test_seedvc_commit_and_changed_tree(seedvc, changed, commit):
    seedvc.git.changed = changed
    seedvc.mod.load('cpu')
    git = ['git', '-C', str(seedvc.repo)]
    assert seedvc.git.calls == [git + ['rev-parse', 'HEAD'], git + ['status', '--porcelain', '--untracked-files=no']]
    d = contract(seedvc.mod, 'seedvc')
    assert d['commit'] == commit and set(rb.DESCRIBE_KEYS) <= set(d)
    assert (d['repo'], d['repo_dir'], d['sr_out']) == ('https://github.com/Plachtaa/seed-vc', str(seedvc.repo), 22050)
    assert list(d['checkpoints']) == seedvc.mod._st['checkpoints'] and d['configs'] == seedvc.mod._st['configs']


class Stop(Exception):
    pass


@pytest.mark.parametrize('ref_s, used_s', [(30, 25), (180, 25), (25, 25), (3, 3)])
def test_seedvc_uses_the_first_25_s_of_the_reference(seedvc, monkeypatch, ref_s, used_s):
    """convert() up to the first resample: the tensors made from the source and the reference."""
    m, sr = seedvc.mod, 22050
    made = []

    def tensor(a):
        made.append(np.array(a))
        return FakeTensor(a)
    seedvc.torch.tensor = tensor

    def resample(*a, **k):
        raise Stop
    torchaudio = types.ModuleType('torchaudio')
    torchaudio.functional = types.SimpleNamespace(resample=resample)
    sys.modules['torchaudio'] = torchaudio                    # (isolated)
    m._st.update(model=None, semantic_fn=None, vocoder=None, campplus=None, mel_fn=None,
                 device=types.SimpleNamespace(type='cpu'),
                 settings={'diffusion_steps': 25, 'length_adjust': 1.0, 'inference_cfg_rate': 0.7, 'fp16': True})
    rng = np.random.default_rng(0)
    src = rng.standard_normal(sr * 40).astype(np.float32)     # sources are never cut
    ref = rng.standard_normal(sr * ref_s).astype(np.float32)
    ref.flags.writeable = False
    with pytest.raises(Stop):
        m.convert(src, ref, sr)
    assert len(made) == 2 and np.array_equal(made[0], src) and np.array_equal(made[1], ref[:sr * used_s])


def test_seedvc_crossfade_is_upstreams_cos2_fade():
    import importlib.util
    spec = importlib.util.spec_from_file_location('seedvc_crossfade', os.path.join(
        os.path.dirname(os.path.abspath(baselines.__file__)), 'seedvc.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                          # imports only numpy and subprocess at module level
    a, b = np.full(4096, 0.3), np.full(5000, 0.3)
    out = m.crossfade(a, b.copy(), 4096)
    assert np.allclose(out, 0.3)                        # cos^2 + sin^2 = 1: a constant passes unchanged
    a, b = np.ones(4096), np.zeros(5000)
    out = m.crossfade(a, b, 4096)
    assert out is b and out[0] == 1 and abs(out[4095]) < 1e-12 and np.all(np.diff(out[:4096]) <= 0)
    assert np.all(out[4096:] == 0)
    short = m.crossfade(np.ones(4096), np.zeros(100), 4096)   # a last chunk shorter than the overlap
    fade_out = np.cos(np.linspace(0, np.pi / 2, 4096)) ** 2
    assert np.allclose(short, fade_out[:100])


# ── S8 FACodec ───────────────────────────────────────────────

NS3_FAKE = '''
import numpy as np
CALLS = []


def _t(a, device, tag):
    import torch
    t = torch.from_numpy(np.asarray(a)).to(device)
    t.tag = tag
    return t


class FACodecEncoderV2:
    def __init__(self, ngf=32, up_ratios=(2, 4, 5, 5), out_channels=1024):
        CALLS.append(('FACodecEncoderV2', dict(ngf=ngf, up_ratios=up_ratios, out_channels=out_channels)))
        self.device, self.training = None, True

    def load_state_dict(self, state_dict, strict=True):
        CALLS.append(('enc.load_state_dict', state_dict, strict))

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, x):
        CALLS.append(('enc', x.a.copy(), x.device))
        return _t(np.zeros((1, 256, x.a.shape[-1] // 200)), x.device, ('enc_out', x.a.shape[-1]))

    def get_prosody_feature(self, x):
        CALLS.append(('get_prosody_feature', x.a.shape))
        return _t(np.zeros((1, 20, x.a.shape[-1] // 200)), x.device, ('prosody', x.a.shape[-1]))


class FACodecDecoderV2:
    def __init__(self, in_channels=256, upsample_initial_channel=1536, ngf=32, up_ratios=(5, 5, 4, 2), vq_num_q_c=2,
                 vq_num_q_p=1, vq_num_q_r=3, vq_dim=1024, vq_commit_weight=0.005, vq_weight_init=False,
                 vq_full_commit_loss=False, codebook_dim=8, codebook_size_prosody=10, codebook_size_content=10,
                 codebook_size_residual=10, quantizer_dropout=0.0, dropout_type="linear", use_gr_content_f0=False,
                 use_gr_prosody_phone=False, use_gr_residual_f0=False, use_gr_residual_phone=False,
                 use_gr_x_timbre=False, use_random_mask_residual=True, prob_random_mask_residual=0.75):
        args = dict(locals())
        args.pop('self')
        CALLS.append(('FACodecDecoderV2', args))
        self.device, self.training = None, True

    def load_state_dict(self, state_dict, strict=True):
        CALLS.append(('dec.load_state_dict', state_dict, strict))

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, x, prosody_feature, vq=True, get_vq=False, eval_vq=True, speaker_embedding=None,
                 n_quantizers=None, quantized=None):
        n = x.tag[1]
        CALLS.append(('dec', x.tag, prosody_feature.tag, vq, get_vq, eval_vq, speaker_embedding, n_quantizers))
        return ('outs', _t(np.zeros((6, 1, n // 200)), x.device, ('codes', n)), 'loss', 'quantized',
                _t(np.zeros((1, 256)), x.device, ('spk_embs', n)))

    def vq2emb(self, vq, use_residual=True):
        CALLS.append(('vq2emb', vq.tag, use_residual))
        return _t(np.zeros((1, 256, vq.a.shape[-1])), vq.device, ('emb', vq.tag[1]))

    def inference(self, x, speaker_embedding):
        CALLS.append(('inference', x.tag, speaker_embedding.tag))
        return _t(np.full((1, 1, x.tag[1]), 0.25, np.float32), x.device, None)
'''


@pytest.fixture
def facodec(isolated, monkeypatch, tmp_path):
    root, site, ckpt = tmp_path / 'facodec', tmp_path / 'envs' / 'site-packages', tmp_path / 'snapshot'
    write_tree(root, {'Amphion/models/codec/ns3_codec/__init__.py': NS3_FAKE})
    write_tree(site, {'einops/__init__.py': "__version__ = '0.8.2'\n"})
    write_tree(ckpt, {f: f.encode() * 10 for f in ('ns3_facodec_encoder_v2.bin', 'ns3_facodec_decoder_v2.bin')})
    git = Git()
    monkeypatch.setattr(subprocess, 'check_output', git)
    torch = fake_torch()
    sys.modules['torch'] = torch                              # (isolated)
    mod = importlib.import_module('baselines.facodec')
    monkeypatch.setattr(mod, 'EINOPS_SITE', str(site))
    monkeypatch.setattr(mod, 'M', {})
    return types.SimpleNamespace(mod=mod, root=root, site=site, ckpt=ckpt, git=git, torch=torch)


def test_facodec_settings(facodec):
    m = facodec.mod
    assert (m.SR, m.HOP, m.REPO) == (16000, 200, 'https://github.com/open-mmlab/Amphion')
    assert m.ENCODER == {'ngf': 32, 'up_ratios': [2, 4, 5, 5], 'out_channels': 256}   # README: V2 takes V1's
    assert m.DECODER == {'in_channels': 256, 'upsample_initial_channel': 1024, 'ngf': 32, 'up_ratios': [5, 5, 4, 2],
                         'vq_num_q_c': 2, 'vq_num_q_p': 1, 'vq_num_q_r': 3, 'vq_dim': 256, 'codebook_dim': 8,
                         'codebook_size_prosody': 10, 'codebook_size_content': 10, 'codebook_size_residual': 10,
                         'use_gr_x_timbre': True, 'use_gr_residual_f0': True, 'use_gr_residual_phone': True}
    assert m.HF_REPO == 'amphion/naturalspeech3_facodec' and m.HF_REVISION.startswith('314afc3')
    assert m.CKPT_DIR.endswith(f'snapshots/{m.HF_REVISION}')
    assert m.CKPTS == ('ns3_facodec_encoder_v2.bin', 'ns3_facodec_decoder_v2.bin')


def test_facodec_load_refuses_a_changed_checkout(facodec):
    facodec.git.changed = ' M models/codec/ns3_codec/facodec.py\n'
    with pytest.raises(SystemExit, match=f'Amphion differs from its commit {COMMIT}'):
        facodec.mod.load('cpu', root=str(facodec.root), ckpt_dir=str(facodec.ckpt))
    git = ['git', '-C', str(facodec.root / 'Amphion'), '--no-optional-locks']
    assert facodec.git.calls == [git + ['rev-parse', 'HEAD'], git + ['status', '--porcelain', '--untracked-files=no']]
    assert 'Amphion' not in sys.modules and facodec.mod.M == {}


def test_facodec_load_builds_the_readme_models_with_strict_cpu_loads(facodec):
    m = facodec.mod
    m.load('cpu', root=str(facodec.root), ckpt_dir=str(facodec.ckpt))
    assert sys.path[0] == str(facodec.root) and sys.path[-1] == str(facodec.site)
    calls = sys.modules['Amphion.models.codec.ns3_codec'].CALLS
    assert calls[0] == ('FACodecEncoderV2', m.ENCODER)
    dec_args = calls[1][1]
    assert calls[1][0] == 'FACodecDecoderV2' and {k: dec_args[k] for k in m.DECODER} == m.DECODER
    paths = [str(facodec.ckpt / f) for f in m.CKPTS]
    assert [c for c in facodec.torch.calls if c[0] == 'load'] == [('load', p, 'cpu', True) for p in paths]
    assert calls[2:] == [('enc.load_state_dict', {'state_dict_of': m.CKPTS[0]}, True),
                         ('dec.load_state_dict', {'state_dict_of': m.CKPTS[1]}, True)]
    assert not m.M['enc'].training and not m.M['dec'].training and m.M['enc'].device == m.M['dec'].device == 'cpu'
    assert m.M['commit'] == COMMIT and m.M['einops'] == f'0.8.2 {facodec.site}/einops'


def test_facodec_describe(facodec):
    m = facodec.mod
    m.load('cpu', root=str(facodec.root), ckpt_dir=str(facodec.ckpt))
    d = contract(m, 'facodec')
    assert set(rb.DESCRIBE_KEYS) <= set(d) and d['commit'] == COMMIT and d['sr_out'] == 16000
    assert d['weights'] == f'amphion/naturalspeech3_facodec@{m.HF_REVISION}'
    assert list(d['checkpoints']) == [str(facodec.ckpt / f) for f in m.CKPTS]
    s = d['settings']
    assert s['encoder'] == m.ENCODER and s['decoder'] == m.DECODER and s['eval'] is True and s['eval_vq'] is False
    assert s['strict_load'] is True and '200 - len(x) % 200' in s['pad']


@pytest.mark.parametrize('n_src', [16000, 16001, 16199, 16200, 33333])
def test_facodec_pads_to_a_multiple_of_200_and_takes_only_the_timbre_of_the_reference(facodec, n_src):
    m = facodec.mod
    m.load('cpu', root=str(facodec.root), ckpt_dir=str(facodec.ckpt))
    calls = sys.modules['Amphion.models.codec.ns3_codec'].CALLS
    del calls[:]
    rng = np.random.default_rng(n_src)
    ref = rng.standard_normal(16000 * 3 + 7).astype(np.float32)
    ref.flags.writeable = False
    src = rng.standard_normal(n_src).astype(np.float32)
    pad = lambda n: n + 200 - n % 200                   # 1 to 200 zeros, a full hop when n % 200 == 0
    for _ in range(2):
        wav, sr = m.convert(src, ref, 16000)
        assert sr == 16000 and isinstance(wav, np.ndarray) and wav.ndim == 1 and len(wav) == pad(n_src)
    enc = [c for c in calls if c[0] == 'enc']
    assert [c[1].shape for c in enc] == [(1, 1, pad(len(ref))), (1, 1, pad(n_src)), (1, 1, pad(n_src))]
    for c, x in zip(enc, (ref, src)):
        assert np.array_equal(c[1][0, 0, :len(x)], x) and not c[1][0, 0, len(x):].any() and c[2] == 'cpu'
    dec = [c for c in calls if c[0] == 'dec']
    assert all(c[3:] == (True, False, False, None, None) for c in dec)   # vq=True, get_vq, eval_vq=False
    assert [c[1:3] for c in dec] == [(('enc_out', pad(len(ref))), ('prosody', pad(len(ref))))] + [
        (('enc_out', pad(n_src)), ('prosody', pad(n_src)))] * 2       # the reference once: cached per object
    assert [c for c in calls if c[0] in ('vq2emb', 'inference')] == [
        ('vq2emb', ('codes', pad(n_src)), False), ('inference', ('emb', pad(n_src)), ('spk_embs', pad(len(ref))))] * 2
