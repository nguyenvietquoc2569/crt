"""
Baseline adapters for run_baselines.py (ANALYSIS_PLAN_v1 §2.4): knnvc = S5 kNN-VC, freevc = S6 FreeVC, seedvc = S7
Seed-VC v1 small, facodec = S8 FACodec. One module per system, baselines/<name>.py, <name> in SYSTEMS (the manifest's
system name). run_baselines imports a module only when its system has work, one system per interpreter, so a module
may keep its model in module globals and put its repository on sys.path. A module defines:
  SR                     int: the rate (Hz) at which the harness loads the source and the reference.
  load(device, **cfg)    builds the model once. device: 'cuda' or 'cpu'; cfg: the system's entry of run_baselines
                         --cfg, keyword overrides of the module's defaults (the fixed settings of §2.4).
  convert(src, ref, sr)  -> (wav, sr_out). src: the source utterance; ref: the pair's reference R[budget] as
                         run_conversions.py gets it (make_pairs.load_reference: each utterance for its keep_s seconds,
                         concatenated, cut to exactly budget_s); both 1-D float32 numpy at sr == SR. ref is read-only
                         (copy it for code that writes in place) and the same object for every pair of one target
                         reference, so reference features may be cached on `ref is last_ref`. random, numpy and torch
                         are seeded from the pair_id before each call. wav: 1-D float numpy on the host at sr_out ==
                         describe()['sr_out']. No target audio other than ref may be used; a model limit that uses
                         less than all of ref goes in settings.
  describe()             -> dict, called after load(): repo (code URL), commit (of the code actually imported),
                         checkpoints (paths of every weight file loaded; the harness adds their sha256), sr_out (int),
                         settings (every inference setting in force), and anything else to record. It enters the rows'
                         conv_cfg_sha, so it must be JSON-serialisable and fixed for a fixed install.
"""
SYSTEMS = ('knnvc', 'freevc', 'seedvc', 'facodec')
