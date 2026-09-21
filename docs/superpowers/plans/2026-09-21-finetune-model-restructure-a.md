# Finetune restructure, sub-project A: model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MeSAEFeatureHead`, `MeSAEFinetune` and `PerChannelHeadAttn` with one composable `FeatureHead` (feature front-end, spatial filter, time pooling, optional branches, readout) plus a `StampExtractor` for the backbone-dependent part, built from numeric head-config keys and wrapped by a small `FinetuneModel` so `train_finetune.py` keeps working with two small edits. Checkpoints store the head config and load without shape inference.

**Architecture:** New classes go in the finetune section of `model/MeSAE/MeSAE_modules.py` (below the ADR 0016 pieces). `MeSAE.py` keeps only `FinetuneModel` (frozen backbone + extractor + head, same call signature as before: `forward(x, coords, time_idx, valid_channels, pad_mask) -> (logits, None, None)`) and `build_finetune`. `model/factory.py` gains a `channel_idx` argument and a checkpoint loader. The composition lives inside `FeatureHead` (slots chosen by config); `build_finetune` only resolves and validates the config, finds the real channels and wraps the backbone. Task 1 is additive; Task 2 switches over and deletes the old code; Task 3 sweeps docs and stale references.

**Tech Stack:** PyTorch (`eeg_fm` env), existing `viz/`, `model/factory.py`, `train_finetune.py`.

**Spec:** `docs/superpowers/specs/2026-09-21-finetune-restructure-design.md`, sub-project A (read it fully first). The extractor output shape is **`[B, N', C_valid, S, 2]`** (patch-major, as the head math uses). Old code for comparison: git tag `pre-head-cleanup` (`model/MeSAE/MeSAE.py` there holds `MeSAEFeatureHead`; the ADR 0016 pieces already live in `MeSAE_modules.py`).

## Global Constraints

- **No experiment is running; the GPU is free.** Smoke runs may use it briefly (Task 2 Step 8); everything else is CPU (`CUDA_VISIBLE_DEVICES=''`).
- **Python env:** `/home/mamechin/anaconda3/envs/eeg_fm/bin/python` for every command (never `base`).
- **Line endings:** `model/MeSAE/MeSAE.py` and `config/config.json` are CRLF (check with `grep -c $'\r' <file>` equals `wc -l`); edit in place with a byte-level script or Edit, never rewrite the whole file, and `git diff --stat` must show only the intended lines. `MeSAE_modules.py`, `model/factory.py` (check!), `viz/__init__.py` (check!), `train_finetune.py` (LF), docs and new files: preserve whatever each file already uses.
- **No old-run compatibility** (user decision). Old checkpoints and old finetune configs (`output/*/artifacts/config.json`, `config/phase2/*.json`) stop loading; the tag `pre-head-cleanup` preserves the old code. `config/phase2/*.json` stay stale until sub-project D deletes them.
- **`MeSAEPretrain` and all pretrain code stay untouched.** `overlap_add_patches`, `StampBank`, `TSAEncoder` etc. are reused as they are.
- Removing `freeze_backbone` from `train_finetune.py` belongs to sub-project C. In A the backbone is always frozen by `FinetuneModel`, and the head-config resolver silently ignores the two legacy training keys `freeze_backbone` and `backbone_lr_mult` if they appear under `model_params.MeSAE.finetune`.
- No test suite exists (CLAUDE.md): validation is the equivalence script and smoke checks below, not pytest files.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
  Do not push (the controller pushes). Never commit anything under `output/` or `.superpowers/`.

## Head config (the contract every task uses)

Keys under `model_params.MeSAE.finetune` (defaults in brackets):
- `feature` [`"stamp_power"`]: `stamp_power` (per-stamp log power), `stamp_band` (template mu/beta energy of the stamp code, the old `stamp_bandpow`), `raw_band` (mu/beta power of the raw signal, computed per patch), `raw_signal` (signed time samples, ~20 Hz average pooling, the ERP baseline).
- `spatial_k` [8]: signed spatial filters over the real channels.
- `time_pool` [`"learned"`]: `flat` | `learned` | `window` | `none`. `learned` uses `time_rank` [2] (rank R, weights per feature dimension, softmax over patches); `window` uses `window` [null] = `[lo, hi]` seconds (patches fully inside, then flat mean); `none` keeps the patch axis as features.
- `phase_advance` [false], `evoked_rank` [0]: optional branches, `stamp_power` only.
- `dropout` [0.5].

Derived at build time (never in the user config): `num_classes`, `num_patches`, `num_channels` (= number of real channels), `num_stamps`, `patch_len`, `patch_stride`, `sample_freq`.

Validity (each violation raises `ValueError` naming the keys):
- `feature` must be one of the four; `time_pool` one of the four.
- `raw_signal` requires `time_pool == "none"` (softmax-weighting signed samples is meaningless; the flat mean of a signed waveform is a scalar).
- `phase_advance` and `evoked_rank > 0` require `feature == "stamp_power"`.
- `time_pool == "window"` requires `window`; `learned` requires `time_rank >= 1`; `evoked_rank > 0` is incompatible with `window` (it needs the full patch axis).
- `num_patches` is required when `feature == "raw_signal"`, `time_pool` is `learned` or `none`, or `evoked_rank > 0`.
- Unknown keys raise; `freeze_backbone` and `backbone_lr_mult` are dropped before the check.

Feature width entering the readout, `feature_dim(cfg)`: with `K = spatial_k`, `N = num_patches`, `F` = `num_stamps` for `stamp_power` and `len(BANDS)` (2) otherwise: `raw_signal`: `K * (T // pool)` with `T = (N - 1) * patch_stride + patch_len`, `pool = max(1, round(sample_freq / 20))`; all others: `(N if time_pool == "none" else 1) * K * F`, plus `2 * K * F` for each active branch.

Per-feature pipeline (the spatial filter always acts at the earliest linear stage):
- `stamp_power`: `a, b` (extractor amplitudes) -> spatial mix over channels -> `power = a^2 + b^2` `[B, N, K, S]` -> time pool -> `log(. + 1e-12)`; branches read `a, b`.
- `stamp_band`: same mix -> per-patch `a^2 @ E_D + b^2 @ E_H` `[B, N, K, 2]` -> time pool -> log.
- `raw_band`: raw patches `[B, C, N, L]` -> spatial mix over channels -> per-patch `|rfft|^2` (frequency resolution `sample_freq / patch_len`, 4 Hz) -> sum over the mu (8-13 Hz) and beta (13-30 Hz) bins `[B, K, N, 2]` -> time pool -> log.
- `raw_signal`: raw patches -> overlap-add -> spatial mix -> `avg_pool1d(pool)` -> flatten (no log).
- Feature order when branches are on: pooled features first, then phase advance, then evoked (each `[B, K, .]` joined on the last axis before flattening; with `time_pool == "none"` each part is flattened and joined).

---

## Task 1: extractor, feature head and the config resolver (additive)

**Files:**
- Modify: `model/MeSAE/MeSAE_modules.py` (append to the finetune section, after `phase_advance` / `_selfcheck_head_modules`)
- Create (not committed): `.superpowers/sdd/2026-09-21-finetune-model-restructure-a/head_equiv.py`

**Interfaces:**
- Produces (used by Task 2):
  - `BANDS`, `FEATURES`, `resolve_head_config(ft_params, **derived) -> dict`, `feature_dim(cfg) -> int`
  - `StampExtractor(backbone, channel_idx)`: `nn.Module` registering `backbone` and buffers `keep` (alive routed + shared stamp indices) and `channel_idx`; `forward(x, coords, time_idx=None, valid_channels=None) -> amp [B, N', C_valid, S, 2]` float32; method `band_tables(sample_freq) -> (E_D, E_H)` each `[S, len(BANDS)]`.
  - `NoTimePool()` (identity over the patch axis).
  - `FeatureHead(cfg)`: attributes `spatial` (`nn.Linear(num_channels, spatial_k, bias=False)`), `time` (`FlatTimePool`, `LearnedTimePool(R, F, N)` or `NoTimePool`; parameters only when learned: `time.p`, `time.q`), `evoked` (`EvokedBranch(R, F, N)` or `None`), `cls` (`Sequential(BatchNorm1d(n_feat), Dropout(dropout), Linear(n_feat, num_classes))`), buffers `E_D`, `E_H` (zeros `[S, 2]`, only for `stamp_band`). `forward(inp) -> logits`: `inp` is the extractor's `amp [B, N', C, S, 2]` for `stamp_*` features and the patched raw signal `[B, C, N', L]` (padded channels zeroed) for `raw_*` features.

- [ ] **Step 1: Read first.** Read the finetune section of `MeSAE_modules.py`, `MeSAEFeatureHead` in `model/MeSAE/MeSAE.py` (`__init__` and `forward`), and `StampBank.dense_amp` / `_template_tables`. The code below ports that math; if the current code differs from what this plan shows (for example the window rule or the fp32 block), the current code wins and you say so in the report.

- [ ] **Step 2: Append this code** to the finetune section of `MeSAE_modules.py` (it uses `torch`, `nn`, `overlap_add_patches`, `spatial_mix`, `FlatTimePool`, `LearnedTimePool`, `EvokedBranch`, `phase_advance` already defined in that file). `LearnedTimePool(rank, num_stamps, num_patches)` and `EvokedBranch` take the feature width as their second argument, so they work for any `F`:

```python
BANDS = ((8.0, 13.0), (13.0, 30.0))   # mu, beta
FEATURES = ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal')
_HEAD_DEFAULTS = dict(feature='stamp_power', spatial_k=8, time_pool='learned', time_rank=2,
                      window=None, phase_advance=False, evoked_rank=0, dropout=0.5)
_LEGACY_TRAINING_KEYS = ('freeze_backbone', 'backbone_lr_mult')   # removed in sub-project C


def resolve_head_config(ft_params, **derived):
    """Head config = defaults + user keys + derived shapes, validated (see the plan's contract)."""
    user = {k: v for k, v in ft_params.items() if k not in _LEGACY_TRAINING_KEYS}
    unknown = set(user) - set(_HEAD_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown head keys {sorted(unknown)}; valid: {sorted(_HEAD_DEFAULTS)}")
    cfg = {**_HEAD_DEFAULTS, **user, **derived}
    f, tp = cfg['feature'], cfg['time_pool']
    if f not in FEATURES:
        raise ValueError(f"feature must be one of {FEATURES}, got {f!r}")
    if tp not in ('flat', 'learned', 'window', 'none'):
        raise ValueError(f"time_pool must be flat|learned|window|none, got {tp!r}")
    if f == 'raw_signal' and tp != 'none':
        raise ValueError("feature='raw_signal' requires time_pool='none'")
    if (cfg['phase_advance'] or cfg['evoked_rank']) and f != 'stamp_power':
        raise ValueError("phase_advance/evoked_rank require feature='stamp_power'")
    if tp == 'window' and not cfg['window']:
        raise ValueError("time_pool='window' requires window=[lo, hi]")
    if tp == 'learned' and int(cfg['time_rank']) < 1:
        raise ValueError("time_pool='learned' requires time_rank >= 1")
    if cfg['evoked_rank'] and tp == 'window':
        raise ValueError("evoked_rank needs the full patch axis, not time_pool='window'")
    if (f == 'raw_signal' or tp in ('learned', 'none') or cfg['evoked_rank']) and cfg.get('num_patches') is None:
        raise ValueError("this configuration needs num_patches (trial length in patches)")
    return cfg


def feature_dim(cfg):
    """Width of the feature vector entering the readout."""
    K, N, f = cfg['spatial_k'], cfg.get('num_patches'), cfg['feature']
    if f == 'raw_signal':
        pool = max(1, round(cfg['sample_freq'] / 20))
        return K * (((N - 1) * cfg['patch_stride'] + cfg['patch_len']) // pool)
    F_ = cfg['num_stamps'] if f == 'stamp_power' else len(BANDS)
    width = (N if cfg['time_pool'] == 'none' else 1) * K * F_
    return width + 2 * K * F_ * (bool(cfg['phase_advance']) + bool(cfg['evoked_rank']))


class StampExtractor(nn.Module):
    """Everything that needs the frozen backbone: stamp code (a, b) per patch, channel and alive
    stamp, scaled by patch RMS. Output [B, N', C_valid, S, 2] float32, padded channels dropped."""
    def __init__(self, backbone, channel_idx):
        super().__init__()
        self.backbone = backbone
        st = backbone.stamps
        alive = (st.fire_ema >= st.dead_threshold).nonzero().flatten()
        self.register_buffer('keep', torch.cat([alive, torch.arange(st.n_routed, st.n_stamps, device=alive.device)]))
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))

    @torch.no_grad()
    def forward(self, x, coords, time_idx=None, valid_channels=None):
        B, C, N, L = x.shape
        vmask = (valid_channels if valid_channels is not None
                 else x.new_ones(B, C, dtype=torch.bool)).float()
        z, _ = self.backbone.stage_features(x, coords, time_idx=time_idx)                 # [B, C, N, D]
        zg = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
        xg = x.permute(0, 2, 1, 3).reshape(B * N, C, L)
        amp = self.backbone.stamps.dense_amp(zg, rms=xg.float().pow(2).mean(-1, keepdim=True).sqrt())
        amp = amp[:, :, self.keep].reshape(B, N, C, -1, 2).float() * vmask[:, None, :, None, None]
        return amp[:, :, self.channel_idx]                                                # [B, N, Cv, S, 2]

    def band_tables(self, sample_freq):
        """Per-stamp template band energies (E_D, E_H), each [S, len(BANDS)], for feature='stamp_band'."""
        with torch.no_grad():
            D_tab, H_tab = (t[self.keep].float() for t in self.backbone.stamps._template_tables())
            fr = torch.fft.rfftfreq(D_tab.shape[-1], 1.0 / sample_freq)
            sel = [((fr >= lo) & (fr < hi)).to(D_tab.device) for lo, hi in BANDS]
            spec = lambda T: torch.stack([torch.fft.rfft(T, dim=-1).abs().pow(2)[:, m].sum(-1) for m in sel], -1)
            return spec(D_tab), spec(H_tab)


class NoTimePool(nn.Module):
    """Keep the patch axis: the features stay [B, N', K, F] and are flattened by the head."""
    def forward(self, power):
        return power


def _window_patches(cfg, N, device):
    """Bool mask [N] of patches fully inside cfg['window'] seconds (same rule as before)."""
    starts = torch.arange(N, device=device) * cfg['patch_stride']
    lo, hi = (w * cfg['sample_freq'] for w in cfg['window'])
    keep = (starts >= lo) & (starts + cfg['patch_len'] <= hi)
    assert keep.any(), f"window {cfg['window']} s keeps no patch"
    return keep


class FeatureHead(nn.Module):
    """Composable finetune head, no backbone inside (ADR 0016): feature front-end -> spatial filter
    -> time pooling -> optional branches -> BatchNorm/Dropout/Linear readout. See the plan's
    'Per-feature pipeline' for the exact math of each feature."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        f, tp = cfg['feature'], cfg['time_pool']
        N = cfg.get('num_patches')
        F_ = cfg['num_stamps'] if f == 'stamp_power' else len(BANDS)
        self.spatial = nn.Linear(cfg['num_channels'], cfg['spatial_k'], bias=False)
        if f == 'raw_signal':
            self.pool_k = max(1, round(cfg['sample_freq'] / 20))
        if tp == 'learned':
            self.time = LearnedTimePool(int(cfg['time_rank']), F_, N)
        else:
            self.time = NoTimePool() if tp == 'none' else FlatTimePool()
        self.evoked = EvokedBranch(int(cfg['evoked_rank']), F_, N) if cfg['evoked_rank'] else None
        if f == 'stamp_band':
            self.register_buffer('E_D', torch.zeros(cfg['num_stamps'], len(BANDS)))
            self.register_buffer('E_H', torch.zeros(cfg['num_stamps'], len(BANDS)))
        n_feat = feature_dim(cfg)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        self.cls = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(cfg['dropout']),
                                 nn.Linear(n_feat, cfg['num_classes']))

    def forward(self, inp):
        cfg, f = self.cfg, self.cfg['feature']
        # Feature math in fp32: under autocast, squared amplitudes overflow in fp16 and the 1e-12
        # epsilon rounds to 0 (NaN loss). The readout stays outside, as before.
        with torch.autocast(device_type=inp.device.type, enabled=False):
            inp, extras = inp.float(), []
            if f == 'raw_signal':                                              # [B, C, N', L]
                sig = spatial_mix(self.spatial, overlap_add_patches(inp, cfg['patch_stride']), 1)
                feat = torch.nn.functional.avg_pool1d(sig, self.pool_k, self.pool_k)   # [B, K, T']
            elif f == 'raw_band':                                              # [B, C, N', L]
                x = inp[:, :, _window_patches(cfg, inp.shape[2], inp.device)] if cfg['time_pool'] == 'window' else inp
                sp = torch.fft.rfft(spatial_mix(self.spatial, x, 1), dim=-1).abs().pow(2)   # [B, K, N, bins]
                fr = torch.fft.rfftfreq(x.shape[-1], 1.0 / cfg['sample_freq']).to(sp.device)
                p = torch.stack([sp[..., (fr >= lo) & (fr < hi)].sum(-1) for lo, hi in BANDS], -1)  # [B, K, N, 2]
                feat = torch.log(self.time(p.permute(0, 2, 1, 3)) + 1e-12)     # [B, K, 2] or [B, N, K, 2]
            else:                                                              # stamp_*: [B, N', C, S, 2]
                amp = inp[:, _window_patches(cfg, inp.shape[1], inp.device)] if cfg['time_pool'] == 'window' else inp
                a, b = spatial_mix(self.spatial, amp[..., 0], 2), spatial_mix(self.spatial, amp[..., 1], 2)
                if f == 'stamp_power':
                    p = a.pow(2) + b.pow(2)                                    # [B, N, K, S]
                else:
                    p = torch.einsum('bnks,sq->bnkq', a.pow(2), self.E_D) \
                        + torch.einsum('bnks,sq->bnkq', b.pow(2), self.E_H)    # [B, N, K, 2]
                feat = torch.log(self.time(p) + 1e-12)                         # [B, K, F] or [B, N, K, F]
                if cfg['phase_advance']:
                    extras.append(phase_advance(a, b))                         # ADR 0014 C3
                if self.evoked is not None:
                    extras.append(self.evoked(a, b))                           # ADR 0014 C4
            if not extras:
                feat = feat.flatten(1)
            elif feat.dim() == 3:
                feat = torch.cat([feat] + extras, dim=-1).flatten(1)
            else:
                feat = torch.cat([feat.flatten(1)] + [e.flatten(1) for e in extras], dim=1)
        return self.cls(feat)
```

- [ ] **Step 3: Write the equivalence script** `.superpowers/sdd/2026-09-21-finetune-model-restructure-a/head_equiv.py` (uncommitted). It compares the old class (from the tag) against `StampExtractor + FeatureHead`, plus new-behaviour checks:

```python
"""Old MeSAEFeatureHead (tag pre-head-cleanup) vs new StampExtractor + FeatureHead. CPU only."""
import os, subprocess, sys, torch
sys.path.insert(0, '.')
from viz import load_config
from model.factory import build_pretrain_from_config
import model.MeSAE.MeSAE_modules as M
sub = subprocess.check_output(['git', 'show', 'pre-head-cleanup:model/MeSAE/MeSAE.py']).decode()
open('model/MeSAE/_old_mesae_tmp.py', 'w', newline='').write(sub)
from model.MeSAE._old_mesae_tmp import MeSAEFeatureHead as Old
try:
    cfgp = load_config('output/pretrain/mesae_v10_small_uw01/artifacts/config.json')
    bb = build_pretrain_from_config(cfgp)
    bb.load_state_dict(torch.load('output/pretrain/mesae_v10_small_uw01/checkpoint/last.pth', map_location='cpu')['model_state_dict'])
    bb.eval()
    C, N, L, NC = 64, 39, 50, 4
    idx = torch.arange(0, 64, 3)[:22]                       # 22 "real" channels
    vc = torch.zeros(2, C, dtype=torch.bool); vc[:, idx] = True
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, C, N, L, generator=g) * vc[:, :, None, None]
    coords = torch.randn(2, C, 3, generator=g)
    derived = dict(num_classes=NC, num_patches=N, num_channels=len(idx), patch_len=L, patch_stride=25, sample_freq=200)
    P = dict(pool_channel='spatial:8')
    # (name, old kwargs, new head keys): every old case that maps exactly onto a new feature
    CASES = [
        ('raw_signal',    dict(input='raw', task='erp', pool_time='trial', **P), dict(feature='raw_signal', time_pool='none')),
        ('band flat',     dict(input='stamp_bandpow', pool_time='trial', **P), dict(feature='stamp_band', time_pool='flat')),
        ('band window',   dict(input='stamp_bandpow', pool_time='window:1.0-4.0', **P), dict(feature='stamp_band', time_pool='window', window=[1.0, 4.0])),
        ('power flat',    dict(input='stamp_induced', pool_time='trial', **P), dict(time_pool='flat')),
        ('power learned', dict(input='stamp_induced', pool_time='learned:2', **P), dict(time_pool='learned', time_rank=2)),
        ('power window',  dict(input='stamp_induced', pool_time='window:1.0-4.0', **P), dict(time_pool='window', window=[1.0, 4.0])),
        ('learned+adv',   dict(input='stamp_induced', pool_time='learned:2', include_advance=True, **P), dict(time_pool='learned', phase_advance=True)),
        ('learned+evk',   dict(input='stamp_induced', pool_time='learned:2', evoked_rank=2, **P), dict(time_pool='learned', evoked_rank=2)),
        ('learned+both',  dict(input='stamp_induced', pool_time='learned:2', include_advance=True, evoked_rank=2, **P), dict(time_pool='learned', phase_advance=True, evoked_rank=2)),
    ]
    def to_new_state(old_sd):
        new = {}
        for k, v in old_sd.items():
            if k.startswith('head.'):
                k = k[5:]
                new[k] = v[:, idx] if k == 'spatial.weight' else v
            elif k in ('E_D', 'E_H'):
                new[k] = v
        return new
    def stamp_input(head_cfg_feature, ext):
        return ext(x, coords, valid_channels=vc) if head_cfg_feature.startswith('stamp') else x[:, idx] * vc[:, idx].float()[:, :, None, None]
    for name, okw, nkw in CASES:
        torch.manual_seed(0)
        old = Old(bb, C, NC, num_patches=N, dropout=0.5, sample_freq=200, **okw).eval()
        cfg = M.resolve_head_config(nkw, **derived, num_stamps=(len(old.keep) if hasattr(old, 'keep') else 0))
        head = M.FeatureHead(cfg).eval()
        ext = M.StampExtractor(bb, idx).eval() if cfg['feature'].startswith('stamp') else None
        if cfg['feature'] == 'stamp_band':
            head.E_D.copy_(ext.band_tables(200)[0]); head.E_H.copy_(ext.band_tables(200)[1])
        missing, unexpected = head.load_state_dict(to_new_state(old.state_dict()), strict=False)
        assert not unexpected and not missing, (name, missing, unexpected)
        if ext is not None:
            assert torch.equal(ext.keep, old.keep), name
        old.zero_grad(); head.zero_grad()
        lo = old(x, coords, valid_channels=vc)[0]
        ln = head(stamp_input(cfg['feature'], ext))
        assert torch.allclose(lo, ln, atol=1e-6), (name, (lo - ln).abs().max().item())
        lo.sum().backward(); ln.sum().backward()
        for pn, p in head.named_parameters():
            po = dict(old.named_parameters())['head.' + pn]
            go = po.grad[:, idx] if pn == 'spatial.weight' else po.grad
            assert torch.allclose(go, p.grad, atol=1e-6), (name, pn)
        print('OK', name)

    # new behaviour 1: raw_band per-patch estimator against an explicit reference
    cfg = M.resolve_head_config(dict(feature='raw_band', time_pool='flat'), **derived, num_stamps=0)
    head = M.FeatureHead(cfg).eval(); head.cls = torch.nn.Identity()
    xr = x[:, idx] * vc[:, idx].float()[:, :, None, None]
    xm = torch.einsum('kc,bcnl->bknl', head.spatial.weight, xr)
    sp = torch.fft.rfft(xm, dim=-1).abs().pow(2); fr = torch.fft.rfftfreq(L, 1 / 200)
    band = torch.stack([sp[..., (fr >= lo) & (fr < hi)].sum(-1) for lo, hi in M.BANDS], -1)   # [B, K, N, 2]
    assert torch.allclose(head(xr), torch.log(band.mean(2) + 1e-12).flatten(1), atol=1e-5)
    print('OK raw_band reference')

    # new behaviour 2: feature_dim matches the real feature width for valid combinations
    S = 25
    amp = torch.randn(2, N, len(idx), S, 2)
    for nkw in [dict(feature='stamp_power', time_pool='none'), dict(feature='stamp_band', time_pool='learned'),
                dict(feature='stamp_band', time_pool='none'), dict(feature='stamp_power', time_pool='flat', phase_advance=True, evoked_rank=2),
                dict(feature='stamp_power', time_pool='none', phase_advance=True), dict(feature='stamp_power', time_pool='window', window=[1.0, 4.0])]:
        cfg = M.resolve_head_config(nkw, **derived, num_stamps=S)
        h = M.FeatureHead(cfg).eval(); h.cls = torch.nn.Identity()
        assert h(amp).shape[1] == M.feature_dim(cfg), nkw
    for nkw in [dict(feature='raw_band', time_pool='learned'), dict(feature='raw_band', time_pool='none'),
                dict(feature='raw_band', time_pool='window', window=[1.0, 4.0]), dict(feature='raw_signal', time_pool='none')]:
        cfg = M.resolve_head_config(nkw, **derived, num_stamps=0)
        h = M.FeatureHead(cfg).eval(); h.cls = torch.nn.Identity()
        assert h(xr).shape[1] == M.feature_dim(cfg), nkw
    print('OK feature_dim')

    # new behaviour 3: invalid combinations raise, legacy training keys are ignored
    for bad in [dict(feature='raw_signal'), dict(feature='raw_band', phase_advance=True), dict(feature='stamp_band', evoked_rank=2),
                dict(foo=1), dict(time_pool='window'), dict(time_pool='window', window=[1, 2], evoked_rank=2),
                dict(feature='nope'), dict(time_pool='learned', time_rank=0)]:
        try:
            M.resolve_head_config(bad, **derived, num_stamps=S)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {bad}")
    try:
        M.resolve_head_config(dict(time_pool='learned'), num_classes=4, num_channels=22, num_stamps=S, patch_len=L, patch_stride=25, sample_freq=200)
    except ValueError:
        pass
    else:
        raise AssertionError("learned without num_patches must raise")
    M.resolve_head_config(dict(freeze_backbone=True, backbone_lr_mult=0.1), **derived, num_stamps=S)
    print('OK resolve_head_config')
finally:
    if os.path.exists('model/MeSAE/_old_mesae_tmp.py'):
        os.remove('model/MeSAE/_old_mesae_tmp.py')
```
If the old class's constructor signature or keyword arguments differ from the above (it took `input, task, pool_channel, pool_time, z_proj, dropout, sample_freq, freeze_backbone, num_patches, include_advance, evoked_rank`), adapt the call, not the comparison. `model/MeSAE/_old_mesae_tmp.py` is a temporary import target and must never be committed.

- [ ] **Step 4: Run it.** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python .superpowers/sdd/2026-09-21-finetune-model-restructure-a/head_equiv.py`
Expected: nine `OK <case>` lines plus `OK raw_band reference`, `OK feature_dim`, `OK resolve_head_config`. A mismatch in the nine old cases is a porting bug: fix the new classes, not the test.

- [ ] **Step 5: Confirm scope and commit.** `git status --short` shows only `M model/MeSAE/MeSAE_modules.py` (the temp file is gone). Commit: `feat: FeatureHead and StampExtractor with a composable, validated numeric head config`.

---

## Task 2: switch over (`FinetuneModel`, factory, train_finetune, viz) and delete the old code

**Files:**
- Modify: `model/MeSAE/MeSAE.py` (CRLF): delete `MeSAEFeatureHead` and `MeSAEFinetune`; add `FinetuneModel`; rewrite `build_finetune`; drop `PerChannelHeadAttn` from the import.
- Modify: `model/MeSAE/MeSAE_modules.py`: delete class `PerChannelHeadAttn`; fix the comment near line 1001 that names `MeSAEFinetune`.
- Modify: `model/factory.py`, `viz/__init__.py`, `train_finetune.py`, `model/base_checker.py`, `model/MeSAE/plugin.py`, `config/config.json` (CRLF).

**Interfaces:**
- Consumes: everything from Task 1.
- Produces:
  - `FinetuneModel(backbone, head_cfg, channel_idx)` in `MeSAE.py`: attributes `backbone` (frozen), `extractor` (or `None` for `raw_*` features), `head`, `head_cfg`, buffer `channel_idx`; `forward(x, coords, time_idx=None, valid_channels=None, pad_mask=None) -> (logits, None, None)`; `train(mode)` keeps the backbone in eval; `head_checkpoint(backbone_checkpoint) -> dict`; `FinetuneModel.from_checkpoint(backbone, ckpt) -> FinetuneModel`.
  - `build_finetune(backbone, num_channels, num_classes, channel_idx=None, num_patches=None, sample_freq=200, **ft_params) -> FinetuneModel` (`channel_idx=None` means all channels).
  - `model.factory.build_finetune_from_config(config, num_classes, mode='finetune', num_patches=None, channel_idx=None)` and `model.factory.load_finetune_checkpoint(config, path, device) -> FinetuneModel`.
  - Checkpoint dict: `{"model_state_dict": head.state_dict(), "head_config": <cfg + "channel_idx" list + "keep" list or None>, "backbone_checkpoint": <path>}`.

- [ ] **Step 1: `FinetuneModel` and `build_finetune`** in `MeSAE.py`, replacing `MeSAEFeatureHead`, `MeSAEFinetune` and the old `build_finetune` (import `StampExtractor`, `FeatureHead`, `resolve_head_config` from `.MeSAE_modules`; drop `PerChannelHeadAttn`):

```python
class FinetuneModel(nn.Module):
    """Frozen MeSAE backbone + one FeatureHead (MeSAE_modules finetune section). Call signature
    matches the old finetune classes so train_finetune.py is unchanged: forward -> (logits, None, None)."""
    def __init__(self, backbone, head_cfg, channel_idx):
        super().__init__()
        self.backbone = backbone
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer('channel_idx', torch.as_tensor(channel_idx, dtype=torch.long))
        self.head_cfg = head_cfg
        stamp = head_cfg['feature'].startswith('stamp')
        self.extractor = StampExtractor(backbone, channel_idx) if stamp else None
        if stamp:
            assert head_cfg['num_stamps'] == len(self.extractor.keep), "num_stamps must equal the alive stamp count"
        self.head = FeatureHead(head_cfg)
        if head_cfg['feature'] == 'stamp_band':
            E_D, E_H = self.extractor.band_tables(head_cfg['sample_freq'])
            self.head.E_D.copy_(E_D); self.head.E_H.copy_(E_H)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()   # frozen: no dropout noise, stable top-k
        return self

    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C = x.shape[:2]
        vm = valid_channels if valid_channels is not None else x.new_ones(B, C, dtype=torch.bool)
        with torch.no_grad():
            if self.extractor is not None:
                inp = self.extractor(x, coords, time_idx, vm)
            else:
                inp = x[:, self.channel_idx] * vm[:, self.channel_idx].float()[:, :, None, None]
        return self.head(inp), None, None

    def head_checkpoint(self, backbone_checkpoint):
        cfg = dict(self.head_cfg, channel_idx=self.channel_idx.tolist(),
                   keep=self.extractor.keep.tolist() if self.extractor is not None else None)
        return {'model_state_dict': self.head.state_dict(), 'head_config': cfg,
                'backbone_checkpoint': backbone_checkpoint}

    @classmethod
    def from_checkpoint(cls, backbone, ckpt):
        cfg = dict(ckpt['head_config'])
        channel_idx, keep = cfg.pop('channel_idx'), cfg.pop('keep')
        model = cls(backbone, cfg, channel_idx)
        if keep is not None and model.extractor.keep.tolist() != keep:
            raise ValueError("backbone's alive stamps differ from the checkpoint's head_config['keep']")
        model.head.load_state_dict(ckpt['model_state_dict'])
        return model


def build_finetune(backbone, num_channels, num_classes, channel_idx=None, num_patches=None,
                   sample_freq=200, **ft_params):
    """finetune_cls entry: builds the frozen backbone + FeatureHead from the numeric head config."""
    channel_idx = list(range(num_channels)) if channel_idx is None else list(channel_idx)
    num_stamps = 0
    if ft_params.get('feature', 'stamp_power').startswith('stamp'):
        st = backbone.stamps
        num_stamps = int((st.fire_ema >= st.dead_threshold).sum()) + (st.n_stamps - st.n_routed)
    cfg = resolve_head_config(ft_params, num_classes=num_classes, num_patches=num_patches,
                              num_channels=len(channel_idx), num_stamps=num_stamps,
                              patch_len=backbone.patch_len, patch_stride=backbone.patch_stride,
                              sample_freq=float(sample_freq))
    return FinetuneModel(backbone, cfg, channel_idx)
```
`st.n_stamps - st.n_routed` is the shared-stamp count (the old code used `torch.arange(st.n_routed, st.n_stamps)`).

- [ ] **Step 2: `model/factory.py`.** `build_finetune_from_config` gains `channel_idx=None` and passes it as `channel_idx=channel_idx` to `plugin.finetune_cls`. Add:

```python
def load_finetune_checkpoint(config, path, device):
    """Rebuild a finetune model from a head checkpoint: backbone from ckpt['backbone_checkpoint'],
    head from ckpt['head_config'] (no shape inference)."""
    from model.MeSAE.MeSAE import FinetuneModel
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    backbone = build_pretrain_from_config(config, mode='finetune')
    backbone.load_state_dict(torch.load(ckpt['backbone_checkpoint'], map_location='cpu')['model_state_dict'])
    return FinetuneModel.from_checkpoint(backbone, ckpt).to(device).eval()
```
(Use the same `mode` argument `build_finetune_from_config` passes to `build_pretrain_from_config`.)

- [ ] **Step 3: `viz/__init__.py` `load_model`.** In the `mode == 'finetune'` branch replace the whole shape-inference block (from `sd = {}` to `model.eval(); return model`) with `from model.factory import load_finetune_checkpoint; return load_finetune_checkpoint(config, checkpoint, device)`; a missing checkpoint file raises (no random-weights fallback for finetune). Update the docstring sentence about state-dict prefixes.

- [ ] **Step 4: `train_finetune.py` (two edits only).**
  1. At the `build_finetune_from_config` call in `run_training_loop` (about line 495) compute the real-channel indices and pass them:
     ```python
     vcs = {tuple(v.tolist()) for v in train_base.all_valid_channels}
     assert len(vcs) == 1, "finetune assumes one real-channel set per dataset"
     channel_idx = torch.nonzero(train_base.all_valid_channels[0]).flatten().tolist()
     model = build_finetune_from_config(config, num_classes, mode='finetune', num_patches=n_patches, channel_idx=channel_idx)
     ```
  2. The best-checkpoint save (about line 554) becomes `torch.save(model.head_checkpoint(train_params['pretrained_checkpoint']), os.path.join(checkpoint_dir, 'best_finetune.pth'))`.
  Nothing else in the file changes in this sub-project (`freeze_backbone`, `backbone_lr_mult`, `_recon_mse`, the viz call, the split modes all wait for sub-project C).

- [ ] **Step 5: dead viz code.** In `model/base_checker.py` delete `render_finetune_attn` and, in `check_finetune`, the `plot_attn_topo` parameter and its `try:` block that calls it (the reconstruction, topomap and PSD snapshot stays; it only needs `model.backbone`). In `model/MeSAE/plugin.py` drop `MeSAEFinetune` from the import, delete the `render_finetune_attn` override, and fix any docstring that names `MeSAEFinetune`. Check `git grep -n "render_finetune_attn\|plot_attn_topo"` afterwards: no caller may still pass `plot_attn_topo` to `check_finetune` (`train_finetune.py`, `check_model.py`).

- [ ] **Step 6: delete `PerChannelHeadAttn`** from `MeSAE_modules.py` and update the comment at about line 1001 that mentions `MeSAEFinetune.encode_post_stamp_expert`. Keep `encode_post_stamp_expert` and every other `MeSAEPretrain` method.

- [ ] **Step 7: `config/config.json` (CRLF).** In `model_params.MeSAE.finetune` replace the old keys with
  `{"feature": "stamp_power", "spatial_k": 8, "time_pool": "learned", "time_rank": 2, "phase_advance": false, "evoked_rank": 0, "dropout": 0.5, "freeze_backbone": true}`
  keeping the file's existing formatting and CRLF; `git diff --stat` shows a handful of lines.

- [ ] **Step 8: verify.**
  - `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -c "import train_finetune, check_model, viz, model.factory"` imports cleanly.
  - Extend `head_equiv.py` with a second check per case in `CASES`: build the new model through the real entry point, `fm = build_finetune(bb, C, NC, channel_idx=idx.tolist(), num_patches=N, sample_freq=200, **nkw)`, load the same name-mapped state with `fm.head.load_state_dict(to_new_state(old.state_dict()))`, put `fm` in `eval()`, and assert `fm(x, coords, valid_channels=vc)[0]` equals the old logits at atol 1e-6 (for `stamp_band` cases `FinetuneModel` fills `E_D/E_H` itself, and the mapped state overwrites them with identical values). Rerun the script: nine `OK` lines for the Task 1 check and nine for this one.
  - Checkpoint round trip (add to the script): `torch.save(fm.head_checkpoint('output/pretrain/mesae_v10_small_uw01/checkpoint/last.pth'), tmp)`, then `load_finetune_checkpoint(cfgp, tmp, 'cpu')` (with `cfgp['training_params']['finetune']['pretrained_checkpoint']` set as the factory expects) reproduces the logits exactly, for one `stamp_power`, one `stamp_band` and one `raw_signal` model.
  - Bad-config errors: `build_finetune` with `time_pool='learned'` and `num_patches=None` raises `ValueError`.
  - Base config builds: `build_finetune_from_config(load_config('config/config.json'), 4, 'finetune', num_patches=39, channel_idx=list(range(22)))` succeeds; print `sum(p.numel() for p in model.head.parameters())` and expect 1,508 (the old C1 head's 1,844 with its 8 x 64 spatial weight replaced by 8 x 22).
  - GPU smoke (2 epochs, about 2 minutes): copy `config/config.json` to the scratchpad with `split_mode: "intra_subject_cv"`, `cv_folds: 2`, `epochs: 2`, `dataset_params.finetune.BCICIV2a.subject_to_use: ["8"]`, `model_name: "smoke_a"`; run `python train_finetune.py --config <copy>`; it must finish without error and write `output/smoke_a/finetune/*/best_finetune.pth`. Load that checkpoint with `viz.load_model(cfg, path, device, mode='finetune')` and print the class and head parameter count. Repeat once with `feature: "raw_band"`, `time_pool: "learned"` in the copy to prove a new combination trains. Then delete `output/smoke_a`.

- [ ] **Step 9: line endings and scope.** `MeSAE.py` and `config/config.json` still have every line ending in CR; `git diff --stat` shows deletions for `MeSAE.py` (about 250 removed, about 60 added), small diffs elsewhere; no whole-file rewrites.

- [ ] **Step 10: commit** in two commits: (1) `refactor: FinetuneModel with StampExtractor and FeatureHead replaces MeSAEFeatureHead/MeSAEFinetune; checkpoints store head_config` (MeSAE.py, MeSAE_modules.py PerChannelHeadAttn removal, factory.py, viz/__init__.py, train_finetune.py, config.json); (2) `refactor: remove the dead finetune attention viz path` (base_checker.py, plugin.py).

---

## Task 3: docs and stale-reference sweep

**Files:**
- Modify: `CLAUDE.md`, `CONTEXT.md`, `docs/adr/0016-finetune-head-modules.md`, `docs/agents/*.md` if they name the removed classes.

- [ ] **Step 1: sweep.** Run `git grep -n -E "MeSAEFeatureHead|MeSAEFinetune|PerChannelHeadAttn|head_z|z_chan|z_proj|pool_channel|pool_time|include_advance|use_topo_feature|stamp_induced|stamp_bandpow"` and list every hit outside `docs/adr/0012*`, `docs/adr/0014*`, `docs/adr/0014_attempts.csv`, older plans under `docs/superpowers/plans/` and `config/phase2/` (historical records, left as written). Every other hit is either a stale reference to fix or a false positive; fix or list it in the report.
- [ ] **Step 2: update docs.**
  - `CLAUDE.md`: the Architecture bullets that name `MeSAEFeatureHead`, `MeSAEFinetune` and `PerChannelHeadAttn` (about lines 88 to 90) now describe `FinetuneModel`, `StampExtractor` and `FeatureHead`; the Config section's finetune head keys become the numeric keys (`feature`, `spatial_k`, `time_pool`, `time_rank`, `window`, `phase_advance`, `evoked_rank`, `dropout`), listed once.
  - `CONTEXT.md` line 29: drop the `PerChannelHeadAttn` example.
  - `docs/adr/0016-finetune-head-modules.md` Consequences: the head is one composable `FeatureHead` (feature front-end, spatial filter, time pooling, branches, readout) plus `StampExtractor`, built from numeric keys; band power can now be combined with learned time weights (`raw_band` uses a per-patch estimator); the checkpoint stores `head_config`; "the state-dict names listed above changed in sub-project A (no old-run compatibility)". Update the ADR's Modules list wording only where it names the old `input` values.
- [ ] **Step 3: commit** `docs: describe the finetune model restructure (sub-project A)`.

---

## Self-Review Notes

- **Spec coverage (A):** one `FeatureHead` with the four slots, the extractor, numeric keys with validation, checkpoint with `head_config`, removals (`MeSAEFinetune`, `PerChannelHeadAttn`, `head_z`, `recon`, `z_chan`, concat, `render_finetune_attn`, dead `check_finetune` branch) are Tasks 1 and 2; real-channel-only spatial mix comes from `channel_idx`; `freeze_backbone` is dropped from the model here and from the training script in C.
- **Interfaces:** shapes are consistent: extractor `[B, N', Cv, S, 2]` feeds `FeatureHead` for `stamp_*`; raw features take `[B, Cv, N', L]`; `spatial_mix(..., dim=2)` for stamp amplitudes and `dim=1` for raw signals, as before.
- **Risks handled:** the equivalence script proves logits and gradients equal the old class for nine configurations under name-mapped weights (padded channels were zero, so dropping them changes nothing); the new `raw_band` estimator is checked against an explicit reference; `feature_dim` is checked against the real feature width on ten combinations; the resolver ports the old guards.
- **Known behaviour change:** `raw_band` (the raw control) is now per-patch (4 Hz resolution) instead of whole-trial; numbers will differ from the earlier raw control (accepted: no old-run compatibility).
- **Deliberately not done here:** feature cache (B), split modes and the slimmer loop (C), experiment runner and `probes/` removal (D), head diagnostics viz (viz refactor). `train_finetune.py` still runs the backbone every epoch until B and C land.
