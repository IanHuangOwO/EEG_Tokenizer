# Feature-as-a-list head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the finetune head's single `feature: "<name>"` config key with `features: ["<name>", ...]`, letting a head combine multiple primaries (e.g. `stamp_power` + `raw_band`) and/or granularities (`stamp_power` + `stamp_band`) in one readout, with per-entry `time_pool`/`time_rank`/`window`/`evoked_rank` overrides.

**Architecture:** `resolve_head_config`/`feature_dim` iterate a list instead of validating one string. `FeatureHead` builds one small submodule per list entry (`_FeatureEntry` for `stamp_power`/`stamp_band`/`raw_band`/`raw_signal`, `_PhaseAdvanceEntry`/`_EvokedEntry` for the two branch-only entries) sharing one `spatial` mixing layer, and concatenates every entry's output before the `BatchNorm/Dropout/Linear` readout — the per-feature math itself (`spatial_mix`, `FlatTimePool`, `LearnedTimePool`, `phase_advance`, `EvokedBranch`) is untouched, only how it's wired together changes. `train_finetune.py`'s source layer gains a `CombinedSource` that serves both a `StampSource` and a `RawSource` together when a head's feature list needs both, and `FinetuneModel` (inference-time single-trial path) gets the matching `needs_stamp`/`needs_raw` dispatch so a checkpoint trained with a mixed head still loads and predicts correctly.

**Tech Stack:** PyTorch (`nn.Module`, `nn.ModuleDict`), existing repo conventions (no test framework — real runs are the verification).

**Spec:** `docs/superpowers/specs/2026-09-22-list-feature-head-design.md`

## Global Constraints

- `features` replaces `feature`. Old bare-string `feature: "x"` configs and checkpoints
  keep loading unchanged via a normalization shim (`_normalize_features`, Task 1) — no
  existing file on disk is migrated or rewritten.
- `phase_advance`/`evoked` (now list entries, not booleans) require **exactly one** of
  `stamp_power`/`stamp_band` present in the same list — not zero, not both. A duplicate
  primary (`features: ["stamp_power", "stamp_power"]`) is rejected.
- `spatial_k` stays one shared value/module across every entry in the list (per the spec)
  — never one `spatial` layer per entry.
- `time_pool`/`time_rank`/`window`/`evoked_rank` are per-entry: an entry's effective value
  is `cfg['overrides'].get(entry_name, {}).get(key, cfg[key])` — the top-level key is the
  default, `overrides[entry_name]` wins when present.
- `raw_signal`'s effective `time_pool` must be `'none'`. `evoked`'s effective `time_pool`
  must not be `'window'` (needs the full patch axis — same rule as before, now scoped to
  its own override instead of the old single global `time_pool`).
- No backward-compat translation needed for the *shape* of existing single-feature
  configs at runtime beyond the key-name shim above — `features: ["stamp_power"]` with no
  `overrides` must produce byte-identical `feature_dim`/forward-pass numerics to the old
  `feature: "stamp_power"` path (this is Task 4's regression check).
- No test suite exists in this repo (see `CLAUDE.md`). Every task's verification is either
  a runnable self-check function (`_selfcheck_head_modules`-style, asserted against
  independent math, not run on import) or a real command run against real data/checkpoints
  on disk, with output pasted into the task's report.

---

### Task 1: `features` list in `resolve_head_config`/`feature_dim`

**Files:**
- Modify: `model/MeSAE/MeSAE_modules.py:1423-1479` (the `BANDS`/`FEATURES`/`_HEAD_DEFAULTS`
  constants through `feature_dim`)
- Modify: `model/MeSAE/MeSAE_modules.py:1393-1420` (`_selfcheck_head_modules` — add cases)

**Interfaces:**
- Consumes: nothing new from other tasks (this task is pure config-resolution logic, no
  `nn.Module` changes).
- Produces (read by Task 2/3):
  - `resolve_head_config(ft_params, **derived) -> dict` — same signature, but the
    returned dict always has `cfg['features']` (a `list[str]`, never `cfg['feature']`).
  - `feature_dim(cfg) -> int` — same signature, sums over `cfg['features']`.
  - `_entry_cfg(cfg, name) -> dict` — effective per-entry `time_pool`/`time_rank`/
    `window`/`evoked_rank` plus the passthrough shape keys (`num_patches`, `num_stamps`,
    `sample_freq`, `patch_stride`, `patch_len`) an entry module needs.
  - `needs_stamp(cfg) -> bool`, `needs_raw(cfg) -> bool` — whether any entry in
    `cfg['features']` needs the frozen-backbone stamp extractor / the raw compiled signal.
  - `FEATURES_ALL = ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal',
    'phase_advance', 'evoked')` — every legal list entry name.

- [ ] **Step 1: Replace the constants and `resolve_head_config`/`feature_dim` block**

Current code at `model/MeSAE/MeSAE_modules.py:1423-1479`:

```python
BANDS = ((8.0, 13.0), (13.0, 30.0))   # mu, beta
FEATURES = ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal')
_HEAD_DEFAULTS = dict(feature='stamp_power', spatial_k=8, time_pool='learned', time_rank=2,
                      window=None, phase_advance=False, evoked_rank=0, dropout=0.5)


def make_head_checkpoint(head, head_cfg, channel_idx, keep, backbone_checkpoint):
    """Head-only checkpoint: state, resolved config (plus the real channels and alive stamps it was
    built for) and the frozen backbone it belongs to. Loaded by FinetuneModel.from_checkpoint."""
    return {'model_state_dict': head.state_dict(),
            'head_config': dict(head_cfg, channel_idx=list(channel_idx), keep=None if keep is None else list(keep)),
            'backbone_checkpoint': backbone_checkpoint}


def resolve_head_config(ft_params, **derived):
    """Head config = defaults + user keys + derived shapes, validated (see the plan's contract)."""
    user = dict(ft_params)
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
    if cfg['spatial_k'] not in (None, 0) and not (isinstance(cfg['spatial_k'], int) and cfg['spatial_k'] > 0):
        raise ValueError(f"spatial_k must be a positive int, or null/0 for no spatial mixing (channel concat), got {cfg['spatial_k']!r}")
    return cfg


def spatial_width(cfg):
    """Effective K: spatial_k filters, or every real channel kept separate (spatial_k None/0 = concat, no mixing)."""
    return cfg['spatial_k'] or cfg['num_channels']


def feature_dim(cfg):
    """Width of the feature vector entering the readout."""
    K, N, f = spatial_width(cfg), cfg.get('num_patches'), cfg['feature']
    if f == 'raw_signal':
        pool = max(1, round(cfg['sample_freq'] / 20))
        return K * (((N - 1) * cfg['patch_stride'] + cfg['patch_len']) // pool)
    F_ = cfg['num_stamps'] if f == 'stamp_power' else len(BANDS)
    width = (N if cfg['time_pool'] == 'none' else 1) * K * F_
    return width + 2 * K * F_ * (bool(cfg['phase_advance']) + bool(cfg['evoked_rank']))
```

Replace with:

```python
BANDS = ((8.0, 13.0), (13.0, 30.0))   # mu, beta
FEATURES_ALL = ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal', 'phase_advance', 'evoked')
_PRIMARY_FEATURES = ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal')
_STAMP_ENTRIES = frozenset({'stamp_power', 'stamp_band', 'phase_advance', 'evoked'})
_RAW_ENTRIES = frozenset({'raw_band', 'raw_signal'})
_ENTRY_OVERRIDE_KEYS = ('time_pool', 'time_rank', 'window', 'evoked_rank')
_HEAD_DEFAULTS = dict(features=['stamp_power'], spatial_k=8, time_pool='learned', time_rank=2,
                      window=None, evoked_rank=0, overrides={}, dropout=0.5)


def make_head_checkpoint(head, head_cfg, channel_idx, keep, backbone_checkpoint):
    """Head-only checkpoint: state, resolved config (plus the real channels and alive stamps it was
    built for) and the frozen backbone it belongs to. Loaded by FinetuneModel.from_checkpoint."""
    return {'model_state_dict': head.state_dict(),
            'head_config': dict(head_cfg, channel_idx=list(channel_idx), keep=None if keep is None else list(keep)),
            'backbone_checkpoint': backbone_checkpoint}


def _normalize_features(cfg):
    """In-place: old bare `feature: "<name>"` (single string, pre-list-features configs
    and every existing on-disk checkpoint's baked-in head_config) -> `features: ["<name>"]`.
    Called at both places a head config can enter the system unresolved: resolve_head_config
    (fresh training-time config) and FinetuneModel.__init__ (an already-resolved config
    loaded straight from an old checkpoint's head_config, which never passes through
    resolve_head_config again) -- see FinetuneModel.from_checkpoint in MeSAE.py. Raises if
    both keys are present (ambiguous, never legal)."""
    has_old, has_new = 'feature' in cfg, 'features' in cfg
    if has_old and has_new:
        raise ValueError("head config has both 'feature' (old) and 'features' (new) -- use only 'features'")
    if has_old:
        cfg['features'] = [cfg.pop('feature')]


def resolve_head_config(ft_params, **derived):
    """Head config = defaults + user keys + derived shapes, validated (see the plan's contract)."""
    user = dict(ft_params)
    _normalize_features(user)
    unknown = set(user) - set(_HEAD_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown head keys {sorted(unknown)}; valid: {sorted(_HEAD_DEFAULTS)}")
    cfg = {**_HEAD_DEFAULTS, **user, **derived}
    feats, tp = cfg['features'], cfg['time_pool']
    if not feats:
        raise ValueError("features must be a non-empty list")
    if len(feats) != len(set(feats)):
        raise ValueError(f"features has duplicate entries: {feats}")
    bad = [f for f in feats if f not in FEATURES_ALL]
    if bad:
        raise ValueError(f"features entries must be one of {FEATURES_ALL}, got {bad}")
    primaries = [f for f in feats if f in ('stamp_power', 'stamp_band')]
    for branch in ('phase_advance', 'evoked'):
        if branch in feats and len(primaries) != 1:
            raise ValueError(f"'{branch}' requires exactly one of stamp_power/stamp_band in "
                              f"features, got primaries={primaries}")
    if tp not in ('flat', 'learned', 'window', 'none'):
        raise ValueError(f"time_pool must be flat|learned|window|none, got {tp!r}")
    if cfg['spatial_k'] not in (None, 0) and not (isinstance(cfg['spatial_k'], int) and cfg['spatial_k'] > 0):
        raise ValueError(f"spatial_k must be a positive int, or null/0 for no spatial mixing (channel concat), got {cfg['spatial_k']!r}")
    overrides = cfg.get('overrides') or {}
    bad_ov = set(overrides) - set(feats)
    if bad_ov:
        raise ValueError(f"overrides keys must name an entry present in features, got {sorted(bad_ov)}")
    needs_np = False
    for name in feats:
        e = _entry_cfg(cfg, name)
        etp = e['time_pool']
        if etp not in ('flat', 'learned', 'window', 'none'):
            raise ValueError(f"features['{name}'] effective time_pool must be flat|learned|window|none, got {etp!r}")
        if name == 'raw_signal' and etp != 'none':
            raise ValueError(f"features entry 'raw_signal' requires effective time_pool='none', got {etp!r}")
        if name == 'evoked' and etp == 'window':
            raise ValueError("features entry 'evoked' needs the full patch axis, not time_pool='window'")
        if etp == 'window' and not e['window']:
            raise ValueError(f"features['{name}'] effective time_pool='window' requires a window=[lo, hi]")
        if etp == 'learned' and int(e['time_rank']) < 1:
            raise ValueError(f"features['{name}'] effective time_pool='learned' requires time_rank >= 1")
        if name == 'evoked' and int(e['evoked_rank']) < 1:
            raise ValueError("features entry 'evoked' requires evoked_rank >= 1 (effective)")
        if name == 'raw_signal' or etp in ('learned', 'none') or name == 'evoked':
            needs_np = True
    if needs_np and cfg.get('num_patches') is None:
        raise ValueError("this configuration needs num_patches (trial length in patches)")
    return cfg


def _entry_cfg(cfg, name):
    """Effective per-entry time_pool/time_rank/window/evoked_rank (top-level default,
    overrides[name] wins) plus the shape/derived keys every entry needs -- single source of
    truth for both FeatureHead's submodule construction (Task 2) and feature_dim below."""
    eff = {k: cfg[k] for k in _ENTRY_OVERRIDE_KEYS}
    eff.update((cfg.get('overrides') or {}).get(name, {}))
    for k in ('num_patches', 'num_stamps', 'sample_freq', 'patch_stride', 'patch_len', 'num_channels'):
        eff[k] = cfg.get(k)
    return eff


def needs_stamp(cfg):
    """Whether any entry in cfg['features'] needs the frozen-backbone stamp extractor."""
    return any(f in _STAMP_ENTRIES for f in cfg['features'])


def needs_raw(cfg):
    """Whether any entry in cfg['features'] needs the compiled raw signal."""
    return any(f in _RAW_ENTRIES for f in cfg['features'])


def spatial_width(cfg):
    """Effective K: spatial_k filters, or every real channel kept separate (spatial_k None/0 = concat, no mixing)."""
    return cfg['spatial_k'] or cfg['num_channels']


def _entry_dim(cfg, name):
    """Feature-vector width contributed by one features[] entry."""
    K = spatial_width(cfg)
    e = _entry_cfg(cfg, name)
    if name == 'raw_signal':
        pool = max(1, round(cfg['sample_freq'] / 20))
        return K * (((cfg['num_patches'] - 1) * cfg['patch_stride'] + cfg['patch_len']) // pool)
    if name in ('phase_advance', 'evoked'):
        return 2 * K * cfg['num_stamps']   # z_re/z_im (or a/b) over every alive stamp
    F_ = cfg['num_stamps'] if name == 'stamp_power' else len(BANDS)
    N = cfg.get('num_patches')
    return (N if e['time_pool'] == 'none' else 1) * K * F_


def feature_dim(cfg):
    """Width of the concatenated feature vector entering the readout."""
    return sum(_entry_dim(cfg, name) for name in cfg['features'])
```

- [ ] **Step 2: Add self-check cases to `_selfcheck_head_modules`**

`model/MeSAE/MeSAE_modules.py:1393` currently starts the function with
`def _selfcheck_head_modules():` and a docstring, then `torch.manual_seed(0)`. Insert
these assertions right after the existing `print('head_modules self-check OK')` line is
reached is wrong -- insert them **before** that print line, at the end of the existing
function body (so the function still ends with the print):

```python
    # -- features list resolution (Task 1) --
    base = dict(num_channels=22, num_stamps=25, sample_freq=200.0, patch_len=50, patch_stride=50)
    old_style = resolve_head_config(dict(feature='stamp_power', spatial_k=8, time_pool='learned',
                                          time_rank=2, dropout=0.5), num_patches=16, **base)
    assert old_style['features'] == ['stamp_power'], old_style['features']
    new_style = resolve_head_config(dict(features=['stamp_power'], spatial_k=8, time_pool='learned',
                                          time_rank=2, dropout=0.5), num_patches=16, **base)
    assert feature_dim(old_style) == feature_dim(new_style)
    both_raw_stamp = resolve_head_config(dict(features=['stamp_power', 'raw_band'], spatial_k=8,
                                               time_pool='flat', dropout=0.5), num_patches=16, **base)
    assert needs_stamp(both_raw_stamp) and needs_raw(both_raw_stamp)
    assert feature_dim(both_raw_stamp) == _entry_dim(both_raw_stamp, 'stamp_power') + _entry_dim(both_raw_stamp, 'raw_band')
    with_branch = resolve_head_config(dict(features=['stamp_power', 'phase_advance', 'evoked'],
                                           spatial_k=8, time_pool='learned', time_rank=2,
                                           evoked_rank=2, dropout=0.5), num_patches=16, **base)
    assert needs_stamp(with_branch) and not needs_raw(with_branch)
    try:
        resolve_head_config(dict(features=['phase_advance'], spatial_k=8, dropout=0.5), num_patches=16, **base)
        assert False, "phase_advance with no stamp primary should have raised"
    except ValueError:
        pass
    try:
        resolve_head_config(dict(features=['stamp_power', 'stamp_power'], spatial_k=8, dropout=0.5),
                             num_patches=16, **base)
        assert False, "duplicate feature entry should have raised"
    except ValueError:
        pass
    overridden = resolve_head_config(dict(features=['stamp_power', 'raw_band'], spatial_k=8,
                                          time_pool='learned', time_rank=2,
                                          overrides={'raw_band': {'time_pool': 'flat'}}, dropout=0.5),
                                     num_patches=16, **base)
    assert _entry_cfg(overridden, 'stamp_power')['time_pool'] == 'learned'
    assert _entry_cfg(overridden, 'raw_band')['time_pool'] == 'flat'
```

- [ ] **Step 3: Run the self-check**

```bash
source /home/mamechin/anaconda3/etc/profile.d/conda.sh && conda activate eeg_fm
python -c "from model.MeSAE.MeSAE_modules import _selfcheck_head_modules; _selfcheck_head_modules()"
```
Expected: `head_modules self-check OK`, no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add model/MeSAE/MeSAE_modules.py
git commit -m "feat: features list in resolve_head_config/feature_dim (list-feature-head Task 1)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `FeatureHead` per-entry submodules

**Files:**
- Modify: `model/MeSAE/MeSAE_modules.py:1530-1593` (the whole `FeatureHead` class)

**Interfaces:**
- Consumes: `resolve_head_config`/`feature_dim`/`_entry_cfg`/`_entry_dim`/`needs_stamp`/
  `needs_raw`/`FEATURES_ALL` from Task 1 (already committed). `spatial_mix`,
  `FlatTimePool`, `LearnedTimePool`, `NoTimePool`, `EvokedBranch`, `phase_advance`,
  `_window_patches`, `overlap_add_patches` — all unchanged, already in this file.
- Produces (read by Task 3): `FeatureHead.forward(self, inp)` where `inp` is now a
  `dict` with optional keys `'raw'` ([B, C, N', L] patchified raw signal, present iff
  `needs_raw(cfg)`) and `'stamp'` ([B, N', C, S, 2] stamp amplitudes, present iff
  `needs_stamp(cfg)`) — **not** a single tensor any more. `FeatureHead.cfg['features']`
  is a `list[str]` (was `cfg['feature']`, a single `str`).

- [ ] **Step 1: Replace the `FeatureHead` class**

Current code at `model/MeSAE/MeSAE_modules.py:1530-1593`:

```python
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
        # spatial_k None/0 = no mixing (channel concat, ADR 0016 ablation control): spatial_mix(None, ...)
        # is identity, so each real channel stays its own feature row instead of being pooled to K filters.
        self.spatial = nn.Linear(cfg['num_channels'], cfg['spatial_k'], bias=False) if cfg['spatial_k'] else None
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

Replace with:

```python
class _PrimaryEntry(nn.Module):
    """One stamp_power/stamp_band/raw_band/raw_signal entry's own time pooling (+ stamp_band's
    template band tables). Reads the SHARED spatial-mixed tensor(s) FeatureHead computes once;
    owns no spatial layer itself (spatial_k is one value shared across every entry, per the
    design spec)."""
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        e = _entry_cfg(cfg, name)
        self.e = e
        F_ = cfg['num_stamps'] if name == 'stamp_power' else len(BANDS)
        if name == 'raw_signal':
            self.pool_k = max(1, round(e['sample_freq'] / 20))
        elif e['time_pool'] == 'learned':
            self.time = LearnedTimePool(int(e['time_rank']), F_, e['num_patches'])
        else:
            self.time = NoTimePool() if e['time_pool'] == 'none' else FlatTimePool()
        if name == 'stamp_band':
            self.register_buffer('E_D', torch.zeros(cfg['num_stamps'], len(BANDS)))
            self.register_buffer('E_H', torch.zeros(cfg['num_stamps'], len(BANDS)))

    def forward(self, raw_mixed=None, ab=None):
        """raw_mixed: [B, K, N', L], already spatial-mixed on the channel axis (raw_signal/raw_band
        only). ab: (a, b), each [B, N', K, S], already spatial-mixed (stamp_power/stamp_band only).
        Returns a flattened [B, width] tensor matching this entry's _entry_dim."""
        e = self.e
        if self.name == 'raw_signal':
            sig = overlap_add_patches(raw_mixed, e['patch_stride'])
            return torch.nn.functional.avg_pool1d(sig, self.pool_k, self.pool_k).flatten(1)
        if self.name == 'raw_band':
            x = raw_mixed[:, :, _window_patches(e, raw_mixed.shape[2], raw_mixed.device)] \
                if e['time_pool'] == 'window' else raw_mixed
            sp = torch.fft.rfft(x, dim=-1).abs().pow(2)                        # [B, K, N, bins]
            fr = torch.fft.rfftfreq(x.shape[-1], 1.0 / e['sample_freq']).to(sp.device)
            p = torch.stack([sp[..., (lo <= fr) & (fr < hi)].sum(-1) for lo, hi in BANDS], -1)  # [B, K, N, 2]
            feat = torch.log(self.time(p.permute(0, 2, 1, 3)) + 1e-12)          # [B, K, 2] or [B, N, K, 2]
            return feat.flatten(1)
        a, b = ab
        if e['time_pool'] == 'window':
            m = _window_patches(e, a.shape[1], a.device)
            a, b = a[:, m], b[:, m]
        if self.name == 'stamp_power':
            p = a.pow(2) + b.pow(2)                                             # [B, N, K, S]
        else:
            p = torch.einsum('bnks,sq->bnkq', a.pow(2), self.E_D) \
                + torch.einsum('bnks,sq->bnkq', b.pow(2), self.E_H)             # [B, N, K, 2]
        feat = torch.log(self.time(p) + 1e-12)                                  # [B, K, F] or [B, N, K, F]
        return feat.flatten(1)


class _PhaseAdvanceEntry(nn.Module):
    """phase_advance branch (ADR 0014 C3): reads the same stamp entry's spatial-mixed (a, b),
    windowed by its OWN effective time_pool/window (independent of the paired primary's)."""
    def __init__(self, cfg):
        super().__init__()
        self.e = _entry_cfg(cfg, 'phase_advance')

    def forward(self, ab):
        a, b = ab
        if self.e['time_pool'] == 'window':
            m = _window_patches(self.e, a.shape[1], a.device)
            a, b = a[:, m], b[:, m]
        return phase_advance(a, b).flatten(1)


class _EvokedEntry(nn.Module):
    """evoked branch (ADR 0014 C4): owns its own EvokedBranch (its own learned rank-r time
    filter), reads the same stamp entry's spatial-mixed (a, b). No windowing (validated in
    resolve_head_config: evoked's effective time_pool may not be 'window')."""
    def __init__(self, cfg):
        super().__init__()
        e = _entry_cfg(cfg, 'evoked')
        self.branch = EvokedBranch(int(e['evoked_rank']), cfg['num_stamps'], cfg['num_patches'])

    def forward(self, ab):
        return self.branch(*ab).flatten(1)


class FeatureHead(nn.Module):
    """Composable finetune head, no backbone inside (ADR 0016; list-feature-head plan): one
    _PrimaryEntry/_PhaseAdvanceEntry/_EvokedEntry submodule per cfg['features'] entry, a single
    shared spatial mixing layer, concatenated -> BatchNorm/Dropout/Linear readout."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feats = cfg['features']
        # spatial_k None/0 = no mixing (channel concat, ADR 0016 ablation control): spatial_mix(None, ...)
        # is identity, so each real channel stays its own feature row instead of being pooled to K filters.
        self.spatial = nn.Linear(cfg['num_channels'], cfg['spatial_k'], bias=False) if cfg['spatial_k'] else None
        self.entries = nn.ModuleDict()
        for name in feats:
            if name in ('stamp_power', 'stamp_band', 'raw_band', 'raw_signal'):
                self.entries[name] = _PrimaryEntry(name, cfg)
            elif name == 'phase_advance':
                self.entries[name] = _PhaseAdvanceEntry(cfg)
            elif name == 'evoked':
                self.entries[name] = _EvokedEntry(cfg)
        n_feat = feature_dim(cfg)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        self.cls = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(cfg['dropout']),
                                 nn.Linear(n_feat, cfg['num_classes']))

    def forward(self, inp):
        """inp: {'raw': [B, C, N', L] or absent, 'stamp': [B, N', C, S, 2] or absent} --
        tools.analysis.select_eval_subsets stays unaffected (it never touches FeatureHead)."""
        cfg = self.cfg
        # Feature math in fp32: under autocast, squared amplitudes overflow in fp16 and the 1e-12
        # epsilon rounds to 0 (NaN loss). The readout stays outside, as before.
        with torch.autocast(device_type=next(iter(inp.values())).device.type, enabled=False):
            raw_mixed, ab = None, None
            if 'raw' in inp:
                raw_mixed = spatial_mix(self.spatial, inp['raw'].float(), 1)          # [B, K, N', L]
            if 'stamp' in inp:
                amp = inp['stamp'].float()
                ab = (spatial_mix(self.spatial, amp[..., 0], 2),
                      spatial_mix(self.spatial, amp[..., 1], 2))                      # each [B, N', K, S]
            outs = []
            for name, mod in self.entries.items():
                if name in ('raw_band', 'raw_signal'):
                    outs.append(mod(raw_mixed=raw_mixed))
                else:
                    outs.append(mod(ab=ab))
            feat = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        return self.cls(feat)
```

- [ ] **Step 2: Runnable self-check — append to `_selfcheck_head_modules`**

Add before the final `print('head_modules self-check OK')` line (same function as Task 1
Step 2 — append after those assertions, still before the print):

```python
    # -- FeatureHead multi-entry forward (Task 2) --
    hcfg = resolve_head_config(dict(features=['stamp_power', 'raw_band'], spatial_k=4,
                                    time_pool='flat', dropout=0.0),
                               num_patches=8, num_channels=6, num_classes=3, num_stamps=S,
                               sample_freq=200.0, patch_len=50, patch_stride=50)
    head = FeatureHead(hcfg)
    B_ = 2
    stamp_in = torch.randn(B_, 8, 6, S, 2)
    raw_in = torch.randn(B_, 6, 8, 50)
    out = head({'stamp': stamp_in, 'raw': raw_in})
    assert out.shape == (B_, 3), out.shape
    single = resolve_head_config(dict(feature='stamp_power', spatial_k=4, time_pool='flat',
                                      dropout=0.0), num_patches=8, num_channels=6, num_classes=3,
                                 num_stamps=S, sample_freq=200.0, patch_len=50, patch_stride=50)
    head2 = FeatureHead(single)
    out2 = head2({'stamp': stamp_in})
    assert out2.shape == (B_, 3), out2.shape
```

- [ ] **Step 3: Run the self-check**

```bash
python -c "from model.MeSAE.MeSAE_modules import _selfcheck_head_modules; _selfcheck_head_modules()"
```
Expected: `head_modules self-check OK`, no assertion errors, no shape mismatches.

- [ ] **Step 4: Commit**

```bash
git add model/MeSAE/MeSAE_modules.py
git commit -m "feat: FeatureHead per-entry submodules for features list (list-feature-head Task 2)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: source layer (`train_finetune.py`) + `FinetuneModel` (`MeSAE.py`)

**Files:**
- Modify: `train_finetune.py:115-168` (`StampSource`, `RawSource`, `make_source`,
  `iter_batches`)
- Modify: `model/MeSAE/MeSAE.py:646-708` (`FinetuneModel`, `build_finetune`)

**Interfaces:**
- Consumes: `needs_stamp`/`needs_raw`/`_normalize_features`/`resolve_head_config` from
  Task 1; `FeatureHead` (dict-input `forward`) from Task 2.
- Produces: `make_source(config, ds_name, pool, device)` still returns one object with a
  `.get(idx) -> (x, y)` method, where `x` is now either a plain tensor (single-source
  heads, unchanged from today) or a `dict` (`CombinedSource`, new) — `iter_batches`
  handles both. `FinetuneModel.forward` still returns `(logits, None, None)`.

- [ ] **Step 1: Add `CombinedSource`, update `make_source`, update `iter_batches`**

Current code at `train_finetune.py:115-168`:

```python
class StampSource:
    """Cached stamp amplitudes of the pool (cache_feature.py), in RAM."""
    kind = 'stamp'

    def __init__(self, config, ds_name, pool, device):
        subs = [str(s) for s in pool]
        self.data = CachedStampDataset(get_stamp_cache(config, ds_name, subs, device=device), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx, self.keep = self.data.channel_idx, self.data.keep
        self.num_patches, self.num_stamps = self.data.num_patches, self.data.num_stamps

    def get(self, idx):
        return self.data.amp[idx].float(), self.labels[idx]


class RawSource:
    """Compiled raw trials of the pool, real channels only; patched on the fly."""
    kind = 'raw'

    def __init__(self, config, ds_name, pool):
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds_name: {**config['dataset_params']['finetune'][ds_name],
                                                       'subject_to_use': list(pool)}}
        base = build_dataset_from_config(cfg, mode='finetune').base_dataset
        assert len({tuple(v.tolist()) for v in base.all_valid_channels}) == 1, "one real-channel set per dataset"
        self.channel_idx = torch.nonzero(base.all_valid_channels[0]).flatten().tolist()
        self.x = base.data[:, self.channel_idx].contiguous()                  # [N, C_valid, T]
        self.labels, self.subject_data = base.labels.long(), base.subject_data.long()
        pp = config['preprocess_params']
        self.patch_len = pp.get('patch_length', 100)
        self.patch_stride = pp.get('patch_stride', self.patch_len)
        self.num_patches = num_patches(self.x.shape[-1], self.patch_len, self.patch_stride)
        self.num_stamps, self.keep = 0, None

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return xp, self.labels[idx]


def make_source(config, ds_name, pool, device):
    feature = config['model_params']['MeSAE']['finetune'].get('feature', 'stamp_power')
    return StampSource(config, ds_name, pool, device) if feature.startswith('stamp') else RawSource(config, ds_name, pool)


def iter_batches(source, idx, batch_size, device, shuffle, gen=None):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen)]
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        if shuffle and len(j) < 2:
            continue
        x, y = source.get(j)
        yield x.to(device), y.to(device)
```

Replace with (`StampSource`/`RawSource` bodies unchanged, only `.kind` docstrings and the
new `CombinedSource`/`make_source`/`iter_batches` around them):

```python
class StampSource:
    """Cached stamp amplitudes of the pool (cache_feature.py), in RAM."""
    kind = 'stamp'

    def __init__(self, config, ds_name, pool, device):
        subs = [str(s) for s in pool]
        self.data = CachedStampDataset(get_stamp_cache(config, ds_name, subs, device=device), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx, self.keep = self.data.channel_idx, self.data.keep
        self.num_patches, self.num_stamps = self.data.num_patches, self.data.num_stamps

    def get(self, idx):
        return self.data.amp[idx].float(), self.labels[idx]


class RawSource:
    """Compiled raw trials of the pool, real channels only; patched on the fly."""
    kind = 'raw'

    def __init__(self, config, ds_name, pool):
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds_name: {**config['dataset_params']['finetune'][ds_name],
                                                       'subject_to_use': list(pool)}}
        base = build_dataset_from_config(cfg, mode='finetune').base_dataset
        assert len({tuple(v.tolist()) for v in base.all_valid_channels}) == 1, "one real-channel set per dataset"
        self.channel_idx = torch.nonzero(base.all_valid_channels[0]).flatten().tolist()
        self.x = base.data[:, self.channel_idx].contiguous()                  # [N, C_valid, T]
        self.labels, self.subject_data = base.labels.long(), base.subject_data.long()
        pp = config['preprocess_params']
        self.patch_len = pp.get('patch_length', 100)
        self.patch_stride = pp.get('patch_stride', self.patch_len)
        self.num_patches = num_patches(self.x.shape[-1], self.patch_len, self.patch_stride)
        self.num_stamps, self.keep = 0, None

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return xp, self.labels[idx]


class CombinedSource:
    """Serves a StampSource and a RawSource together, for a head whose features list needs
    both (e.g. features=['stamp_power', 'raw_band']). Exposes the union of attributes either
    single source exposes (num_patches/num_stamps/channel_idx/keep/labels/subject_data) --
    both sources are built from the SAME (ds_name, pool), so their per-trial ordering,
    labels and channel_idx must already agree; asserted once at construction, not re-checked
    per batch."""
    kind = 'combined'

    def __init__(self, stamp_source, raw_source):
        assert stamp_source.channel_idx == raw_source.channel_idx, \
            "StampSource/RawSource channel_idx mismatch -- same dataset/pool should agree"
        assert torch.equal(stamp_source.labels, raw_source.labels), \
            "StampSource/RawSource label order mismatch -- same dataset/pool should agree"
        self.stamp, self.raw = stamp_source, raw_source
        self.labels, self.subject_data = stamp_source.labels, stamp_source.subject_data
        self.channel_idx, self.keep = stamp_source.channel_idx, stamp_source.keep
        self.num_patches, self.num_stamps = stamp_source.num_patches, stamp_source.num_stamps

    def get(self, idx):
        stamp_x, y = self.stamp.get(idx)
        raw_x, _ = self.raw.get(idx)
        return {'stamp': stamp_x, 'raw': raw_x}, y


def make_source(config, ds_name, pool, device):
    ft_cfg = dict(config['model_params']['MeSAE']['finetune'])
    _normalize_features(ft_cfg)
    want_stamp, want_raw = needs_stamp(ft_cfg), needs_raw(ft_cfg)
    if want_stamp and want_raw:
        return CombinedSource(StampSource(config, ds_name, pool, device), RawSource(config, ds_name, pool))
    if want_stamp:
        return StampSource(config, ds_name, pool, device)
    return RawSource(config, ds_name, pool)


def iter_batches(source, idx, batch_size, device, shuffle, gen=None):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen)]
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        if shuffle and len(j) < 2:
            continue
        x, y = source.get(j)
        x = {k: v.to(device) for k, v in x.items()} if isinstance(x, dict) else x.to(device)
        yield x, y.to(device)
```

Add the import `needs_stamp`/`needs_raw`/`_normalize_features` to `train_finetune.py`'s
existing `from model.MeSAE.MeSAE_modules import (...)` line (find it near the top of the
file and extend the tuple — do not add a second import line for the same module).

- [ ] **Step 2: Update `FinetuneModel`/`build_finetune`**

Current code at `model/MeSAE/MeSAE.py:646-708`:

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
        self.backbone.eval()   # frozen backbone never trains, even in .train() mode
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
        return make_head_checkpoint(self.head, self.head_cfg, self.channel_idx.tolist(),
                                    self.extractor.keep.tolist() if self.extractor is not None else None,
                                    backbone_checkpoint)

    @classmethod
    def from_checkpoint(cls, backbone, ckpt):
        cfg = dict(ckpt['head_config'])
        channel_idx, keep = cfg.pop('channel_idx'), cfg.pop('keep')
        model = cls(backbone, cfg, channel_idx)
        if keep is not None and model.extractor.keep.tolist() != keep:
```

(the `from_checkpoint` body continues past what was captured above — its remaining lines
are untouched by this task; only the constructor/`forward` change). Find `class
FinetuneModel` in `model/MeSAE/MeSAE.py` and make these three edits in place (do not
retype the whole class — `train`/`head_checkpoint`/`from_checkpoint`'s bodies are
unchanged):

1. In `__init__`, replace:
```python
        self.head_cfg = head_cfg
        stamp = head_cfg['feature'].startswith('stamp')
        self.extractor = StampExtractor(backbone, channel_idx) if stamp else None
        if stamp:
            assert head_cfg['num_stamps'] == len(self.extractor.keep), "num_stamps must equal the alive stamp count"
        self.head = FeatureHead(head_cfg)
        if head_cfg['feature'] == 'stamp_band':
            E_D, E_H = self.extractor.band_tables(head_cfg['sample_freq'])
            self.head.E_D.copy_(E_D); self.head.E_H.copy_(E_H)
```
with:
```python
        _normalize_features(head_cfg)   # old checkpoints' baked-in head_config: 'feature' -> 'features'
        self.head_cfg = head_cfg
        stamp = needs_stamp(head_cfg)
        self.extractor = StampExtractor(backbone, channel_idx) if stamp else None
        if stamp:
            assert head_cfg['num_stamps'] == len(self.extractor.keep), "num_stamps must equal the alive stamp count"
        self.head = FeatureHead(head_cfg)
        if 'stamp_band' in head_cfg['features']:
            E_D, E_H = self.extractor.band_tables(head_cfg['sample_freq'])
            self.head.entries['stamp_band'].E_D.copy_(E_D)
            self.head.entries['stamp_band'].E_H.copy_(E_H)
```

2. Replace `forward`:
```python
    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C = x.shape[:2]
        vm = valid_channels if valid_channels is not None else x.new_ones(B, C, dtype=torch.bool)
        with torch.no_grad():
            if self.extractor is not None:
                inp = self.extractor(x, coords, time_idx, vm)
            else:
                inp = x[:, self.channel_idx] * vm[:, self.channel_idx].float()[:, :, None, None]
        return self.head(inp), None, None
```
with:
```python
    def forward(self, x, coords, time_idx=None, valid_channels=None, pad_mask=None):
        B, C = x.shape[:2]
        vm = valid_channels if valid_channels is not None else x.new_ones(B, C, dtype=torch.bool)
        inp = {}
        with torch.no_grad():
            if self.extractor is not None:
                inp['stamp'] = self.extractor(x, coords, time_idx, vm)
            if needs_raw(self.head_cfg):
                inp['raw'] = x[:, self.channel_idx] * vm[:, self.channel_idx].float()[:, :, None, None]
        return self.head(inp), None, None
```

(`x` here is already `[B, C, N', L]` patchified raw signal per this method's existing
callers — the `else` branch in the old code covered the "raw-only" case by reusing the
same `x`; the new code covers "raw-only" AND "raw-and-stamp-together" the same way, since
`inp['raw']` is set whenever `needs_raw` is true regardless of whether `inp['stamp']` is
also set.)

3. `head_checkpoint`, `from_checkpoint`: no changes — `head_cfg` already contains
   `'features'` (normalized in `__init__` above) by the time `make_head_checkpoint` saves
   it, so every NEW checkpoint is saved in the new list form; every OLD checkpoint keeps
   loading via the `_normalize_features` call at the top of `__init__` (step 1 above).

- [ ] **Step 3: Add the imports `FinetuneModel`/`train_finetune.py` need**

`model/MeSAE/MeSAE.py`: find its existing
`from model.MeSAE.MeSAE_modules import (...)` import and add `needs_stamp, needs_raw,
_normalize_features` to that tuple (do not add a new import line).

- [ ] **Step 4: Real smoke run — single-feature regression**

Pick any existing baseline checkpoint's own overlay config (they already exist on disk
after this session's rename, e.g. `output/baseline/BNCI2014001_intra_learned/`) and run a
short (2-epoch) `train_finetune.py` using a COPY of its config with `epochs: 2`, once
BEFORE this task's changes (checkout the pre-task commit in a scratch worktree, or just
trust Task 1/2's self-checks plus re-run this same config against the CURRENT HEAD after
all of Task 3's edits) — compare the printed per-epoch `[Val] loss/bal_acc` lines. Exact
command (adjust dataset/config path to whatever exists under `output/baseline/` on your
disk):

```bash
source /home/mamechin/anaconda3/etc/profile.d/conda.sh && conda activate eeg_fm
python - <<'EOF'
import json
cfg = json.load(open('output/baseline/BNCI2014001_intra_learned/artifacts/config.json'))
cfg['training_params']['finetune']['epochs'] = 2
cfg['training_params']['finetune']['model_name'] = 'scratch/features_list_regression_check'
json.dump(cfg, open('/tmp/features_list_regression.json', 'w'))
EOF
python train_finetune.py --config /tmp/features_list_regression.json
```
Expected: runs to completion, 2 epochs of `[Val] loss: ... bal_acc: ...` lines, no crash.
This config's own `model_params.MeSAE.finetune` already has `feature: "stamp_power"` (old
key, baked into that checkpoint's artifact) — this run is exercising the
`_normalize_features` shim through `resolve_head_config` end to end.

- [ ] **Step 5: Real smoke run — genuine multi-entry head**

```bash
python - <<'EOF'
import json
cfg = json.load(open('output/baseline/BNCI2014001_intra_learned/artifacts/config.json'))
cfg['model_params']['MeSAE']['finetune'] = {
    'features': ['stamp_power', 'raw_band'], 'spatial_k': 8, 'time_pool': 'learned',
    'time_rank': 2, 'dropout': 0.5}
cfg['training_params']['finetune']['epochs'] = 2
cfg['training_params']['finetune']['model_name'] = 'scratch/features_list_multi_smoke'
json.dump(cfg, open('/tmp/features_list_multi.json', 'w'))
EOF
python train_finetune.py --config /tmp/features_list_multi.json
```
Expected: runs to completion, `[<tag>] train=... eval=... head_params=...` line shows a
head_params count reflecting the WIDER combined feature vector, 2 epochs of loss/bal_acc
lines, no NaN, loss decreasing or at least finite and stable (2 epochs is too short to
demand convergence — the bar is "doesn't crash, doesn't NaN").

Clean up the scratch output afterward:
```bash
rm -rf output/scratch /tmp/features_list_regression.json /tmp/features_list_multi.json
```

- [ ] **Step 6: Commit**

```bash
git add train_finetune.py model/MeSAE/MeSAE.py
git commit -m "feat: CombinedSource + FinetuneModel dict-input for features list (list-feature-head Task 3)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: config migration, docs, final verification

**Files:**
- Modify: `config/config.json` (`model_params.MeSAE.finetune`)
- Modify: `CLAUDE.md` (the `model_params.MeSAE.finetune` bullet in the Config section)
- Modify: `docs/superpowers/specs/2026-09-22-list-feature-head-design.md` (status line)

**Interfaces:**
- Consumes: everything from Tasks 1-3 (already committed).
- Produces: nothing further downstream — this is the plan's final task.

- [ ] **Step 1: Update `config/config.json`**

Current (`model_params.MeSAE.finetune`):
```json
{
  "feature": "stamp_power",
  "spatial_k": 8,
  "time_pool": "learned",
  "time_rank": 2,
  "phase_advance": false,
  "evoked_rank": 0,
  "dropout": 0.5
}
```
Replace with:
```json
{
  "features": ["stamp_power"],
  "spatial_k": 8,
  "time_pool": "learned",
  "time_rank": 2,
  "evoked_rank": 0,
  "dropout": 0.5
}
```
(`phase_advance` as a bare boolean is gone — it is now a `features` list entry name, not
a flag; this baseline config doesn't use it, so it's simply dropped, not translated.)

- [ ] **Step 2: Update `CLAUDE.md`**

Find this line in the `### Config (\`config/config.json\`)` section:
```
- `model_params.MeSAE.pretrain`: the one architecture block — `patch_len`, `embed_dim`, `enc_depth`, `pool_after_blocks` (also the tokenizer-phase block set), `moe_ffn`, `stamp_bank`, `loss`. `model_params.MeSAE.finetune`: head keys (numeric, validated at build; the checkpoint stores the resolved `head_config`) — `feature` (`stamp_power`/`stamp_band`/`raw_band`/`raw_signal`), `spatial_k`, `time_pool` (`flat`/`learned`/`window`/`none`), `time_rank`, `window` (`[lo, hi]` s), `phase_advance`, `evoked_rank`, `dropout`; defaults in `docs/adr/0016`
```
Replace the `model_params.MeSAE.finetune` sentence (keep the `model_params.MeSAE.pretrain`
sentence before it unchanged) with:
```
`model_params.MeSAE.finetune`: head keys (numeric, validated at build; the checkpoint stores the resolved `head_config`) — `features` (list, one or more of `stamp_power`/`stamp_band`/`raw_band`/`raw_signal`/`phase_advance`/`evoked`; `phase_advance`/`evoked` need exactly one of `stamp_power`/`stamp_band` in the same list — see `docs/superpowers/specs/2026-09-22-list-feature-head-design.md`), `spatial_k` (one shared value across every entry), `time_pool` (`flat`/`learned`/`window`/`none`, default for every entry), `time_rank`, `window` (`[lo, hi]` s), `evoked_rank`, `overrides` (`{entry_name: {time_pool/time_rank/window/evoked_rank}}`, per-entry override of the defaults above), `dropout`; old bare `feature: "<name>"` configs/checkpoints still load (normalized to a one-element `features` list); defaults in `docs/adr/0016`
```

- [ ] **Step 3: Update the spec's status line**

`docs/superpowers/specs/2026-09-22-list-feature-head-design.md:3` currently reads:
```
Date: 2026-09-22. Status: proposal, not scheduled. Not part of the finetune restructure
sub-projects A-D; a candidate follow-up once the restart baseline and D are done.
```
Replace with:
```
Date: 2026-09-22. Status: implemented, see docs/superpowers/plans/2026-09-22-feature-list-head.md.
```

- [ ] **Step 4: Repo-wide reference sweep**

```bash
grep -rn "cfg\['feature'\]\|\.get('feature'\|'feature':" --include='*.py' . | grep -v "^\./tools\|test"
```
Expected: no remaining LIVE code reads `cfg['feature']` (singular) except inside
`_normalize_features` itself (which is supposed to read/pop it) — every other hit should
be a comment/docstring mention, not a functional dependency. If a real hit turns up
outside `_normalize_features`, fix it before proceeding (this step exists specifically to
catch a call site this plan's author missed).

- [ ] **Step 5: Final real run — the existing baseline config end to end**

```bash
source /home/mamechin/anaconda3/etc/profile.d/conda.sh && conda activate eeg_fm
python train_finetune.py --config config/config.json
```
(Use whatever `dataset_params.finetune`/`epochs` `config/config.json` currently has
configured for a quick real run — do not change those fields, only confirm the run
starts, trains at least one epoch without error, and writes
`output/<model_name>/artifacts/group_eval.json`. If `config/config.json`'s current
dataset/epochs make a full run slow, temporarily drop `epochs` to 1 for this
verification only, then restore it — do not leave that edit committed.)

Expected: completes without error, `head_params=...` line printed matches
`feature_dim(cfg)` for `features: ["stamp_power"]` (same width as before this whole
plan, since `feature_dim`'s Task 1 self-check already proved old-vs-new equivalence).

- [ ] **Step 6: Commit**

```bash
git add config/config.json CLAUDE.md docs/superpowers/specs/2026-09-22-list-feature-head-design.md
git commit -m "docs: migrate config.json/CLAUDE.md to features list, mark spec implemented (list-feature-head Task 4)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```
