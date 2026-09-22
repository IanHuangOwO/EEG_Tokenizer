# Experiment C — C1 (learned time weights) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace C0's flat (uniform) time pooling in `MeSAEFeatureHead`'s `stamp_induced`
arm with a learned, low-rank time weighting (`w[s,n] = Σ_r p_r[s]·q_r[n]`, softmax-normalized
over the patch axis), then run the wiring check ADR 0014 build-order step 8 calls for: C1
must beat C0.

**Architecture:** Extend `MeSAEFeatureHead` again (same pattern as C0's `stamp_induced` arm
on top of `stamp_bandpow`) rather than adding a new class — add a new `pool_time` value,
`"learned:R"` (R = rank, 1 or 2 per ADR), valid only for `input="stamp_induced"`. At R's
`time_p`/`time_q` factors, the pre-softmax logits are near zero at init (small random init),
so the learned weighting starts **equivalent to C0's flat mean** and only diverges from it as
training moves the parameters — this makes the "must beat C0" comparison a true ablation of
one factor (learned vs. flat time weights), not a confound with a different starting point.

Implementing this requires knowing the patch count `N` at model-construction time (to size
the `time_q` factor as an `nn.Parameter`, which must exist before `train_finetune.py` builds
the optimizer's param groups — a parameter built lazily on first `forward()` would be
invisible to the optimizer). `N` is dataset-dependent (BNCI2014001's trial length is not the
same as `preprocess_params.window_length`, which is a pretrain-only concept), so it must be
computed from the actual dataset and threaded through `build_finetune_from_config` →
`build_finetune` → `MeSAEFeatureHead.__init__`, none of which currently take it.

**Also required by this plan:** the previous C0/C0-follow-up work never produced a clean,
directly-comparable baseline number for "C0 done right" (5-fold CV, `dropout: 0.5` —
combining the correct protocol with the regularization fix a later review recommended). The
only numbers on record are 5-fold/`dropout 0` (0.456, known to overfit) and 3-fold/`dropout
0.5` (0.468, fold-count-confounded). Task 2 below produces the missing 5-fold/`dropout 0.5`
`stamp_induced` baseline **before** running C1, so "C1 must beat C0" is measured against a
real, uncounfounded number instead of either of the flawed ones on record.

**Tech Stack:** PyTorch (`nn.Parameter`, `torch.einsum`, `torch.softmax`), the existing
`train_finetune.py` / `probes/ft_summary.py` pipeline, `IO/preprocessing.py`'s `num_patches`
helper (already exists, unused by this file so far).

**Spec:** `docs/adr/0014-finetune-head-test-plan.md` — "Experiment C" (the `Head` pseudocode
block, specifically step `2a. induced: log sum_n w[s,n] |u|^2 time weights, low-rank`),
"Ablation ladder" (`C1 | learned time weights | C0 — tests "when"`), "Protocol" (5-fold CV,
optimizer settings), and build-order step 7's now-recorded C0/follow-up results (read this
in full — it has the exact numbers referenced above and the reasoning for why they don't
form a clean baseline).

## Global Constraints

- Backbone stays frozen (ADR 0012) — do not touch `freeze_backbone`.
- Backbone checkpoint: `output/pretrain/mesae_v10_small/checkpoint/last.pth` (same as every prior
  C-experiment measurement; conclusions about the tokenizer itself need the full-data run,
  not in scope here).
- Optimizer/protocol (ADR 0014 §Protocol, carried forward from C0): `learning_rate: 0.01`,
  `min_learning_rate: 0.001`, `epochs: 100`, `warmup_epochs: 2`, `split_mode:
  "intra_subject_cv"`, `cv_folds: 5` (NOT 3 — the 3-fold cut used for one C0 follow-up run
  was an explicit, one-off speed tradeoff, not a new standard; every run in this plan uses 5).
- Head `dropout: 0.5` for every `stamp_induced` run in this plan (both the new baseline and
  C1) — carried forward from the C0 follow-up finding that `dropout: 0` on a 200-feature head
  overfits badly (train accuracy hit ~0.985–1.000 on 230 trials/fold).
- Reporting metric: `balanced_acc`, mean over each subject's last 10 epochs (`probes/
  ft_summary.py`'s "tail" row), never best-val-epoch.
- No project test suite exists (CLAUDE.md) — validation is empirical: a shape/build smoke
  check for Task 1 (no training run needed to prove the plumbing works), then the real
  training runs for Task 2.
- Do not touch `model_params.MeSAE.pretrain.stamp_bank` (must stay `60/4/12/6`, matching the
  `mesae_v10_small` checkpoint — a prior session's mistake here cost a full rerun).
- Do not touch `dataset_params.finetune` (already correctly points at BNCI2014001).
- `input="raw"`, `"recon"`, `"stamp_bandpow"`, and `"z_chan"`'s existing behavior must be
  completely unaffected by this plan — `num_patches` is optional/unused for those arms, and
  the new `pool_time="learned:R"` branch must raise clearly if used with any input other than
  `"stamp_induced"`.

---

## Task 1: Learned low-rank time weights (`pool_time="learned:R"`) on `MeSAEFeatureHead`

**Files:**
- Modify: `model/factory.py:57-83` (`build_finetune_from_config`) — accept and forward an
  optional `num_patches` kwarg.
- Modify: `train_finetune.py` — imports near the top (`from IO.preprocessing import
  slice_patches` becomes `slice_patches, num_patches`), and `run_training_loop` (currently
  `train_finetune.py:462` onward) — compute the trial's patch count from the actual dataset
  and pass it into `build_finetune_from_config`.
- Modify: `model/MeSAE/MeSAE.py` — `MeSAEFeatureHead.__init__` and `.forward` (the
  `stamp_induced` arm specifically), and `build_finetune`'s `allowed` kwargs tuple.

**Interfaces:**
- Consumes: `IO.preprocessing.num_patches(total_T: int, patch_len: int, patch_stride:
  Optional[int] = None) -> int` (existing, `IO/preprocessing.py:164-169` — already the exact
  formula `slice_patches` uses internally, so this can't drift out of sync with the real
  patch count).
- Consumes: a `FinetuneDataset`/`Subset`-wrapped dataset's `__getitem__` returning `(x [C,T],
  coords, label, valid_channels, valid_length)` (existing, `IO/dataset.py:473-479`) — `x.shape[-1]`
  is the trial length `T`.
- Produces: `build_finetune_from_config(config, num_classes, mode='finetune', num_patches=None)`
  — the new parameter is optional (defaults to `None`, so every other caller/input arm is
  unaffected) and is merged into the kwargs passed to `plugin.finetune_cls`.
- Produces: `MeSAEFeatureHead(..., num_patches=None)` — required (raises if `None`) only when
  `pool_time` is `"learned:R"`; ignored otherwise.
- Produces: `pool_time="learned:R"` (R a positive int, e.g. `"learned:2"`) as a new valid
  value alongside the existing `"trial"` and `"window:lo-hi"` forms.

- [ ] **Step 1: Thread `num_patches` through `build_finetune_from_config`**

In `model/factory.py`, change the function signature and body (currently
`model/factory.py:57-83`):

```python
def build_finetune_from_config(config, num_classes, mode='finetune', num_patches=None):
    """
    Builds the pretrained backbone (via build_pretrain_from_config, same config section),
    loads its checkpoint, and wraps it in the classification head (dispatched via
    MODEL_REGISTRY the same way build_pretrain_from_config dispatches its backbone).
    num_classes is dataset-dependent (label set size) so it can't be read from config —
    pass it in. num_patches is likewise dataset-dependent (trial length varies by dataset,
    independent of preprocess_params.window_length, which is a pretrain-only concept) --
    only required by heads whose parameter shapes depend on the patch axis length (e.g.
    MeSAEFeatureHead's pool_time="learned:R"); every other head ignores it.
    """
    train_params = config['training_params'][mode]
    model_type   = train_params.get('model_type', 'MeSAE')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    plugin = MODEL_REGISTRY[model_type]

    backbone = build_pretrain_from_config(config, mode=mode)
    ckpt_path = train_params['pretrained_checkpoint']
    state = torch.load(ckpt_path, map_location='cpu')
    # load_state_dict restores the checkpoint's phase flags (MeSAE _restore_phase)
    backbone.load_state_dict(state['model_state_dict'])

    canonical_channels = resolve_canonical_channels(config['preprocess_params']['canonical_channels'])
    ft_params = config['model_params'][model_type].get('finetune', {})
    # The whole finetune block is passed through; the plugin picks what its head takes.
    return plugin.finetune_cls(
        backbone, len(canonical_channels), num_classes,
        sample_freq=config['preprocess_params']['sample_freq'], num_patches=num_patches,
        **ft_params,
    )
```

Check the actual current file before editing (line numbers may have drifted) — the
functional change is: add `num_patches=None` to the signature, pass `num_patches=num_patches`
into the `plugin.finetune_cls(...)` call, and add the one sentence to the docstring.

- [ ] **Step 2: Compute the real patch count in `run_training_loop` and pass it through**

In `train_finetune.py`, find the import line for `slice_patches` (near the top of the file,
`from IO.preprocessing import slice_patches`) and change it to:

```python
from IO.preprocessing import slice_patches, num_patches
```

In `run_training_loop` (currently starting at `train_finetune.py:462`), after
`patch_stride` is computed (`train_finetune.py:470`: `patch_stride =
config.get('preprocess_params', {}).get('patch_stride', patch_len)`) and before the call to
`build_finetune_from_config` (`train_finetune.py:493`: `model =
build_finetune_from_config(config, num_classes, mode='finetune')`), add:

```python
    trial_T = train_dataset[0][0].shape[-1]
    n_patches = num_patches(trial_T, patch_len, patch_stride)
```

Then change the `build_finetune_from_config` call (`train_finetune.py:493`) to:

```python
    model = build_finetune_from_config(config, num_classes, mode='finetune', num_patches=n_patches)
```

`train_dataset[0]` works whether `train_dataset` is a `FinetuneDataset`-wrapping-`EEGDataset`
directly or a `torch.utils.data.Subset` of one — both support `__getitem__(0)` returning the
same `(x, coords, label, valid_channels, valid_length)` tuple (`Subset.__getitem__` just
indexes into its own index list first). Read the actual current file to confirm this call
site's exact current line numbers and surrounding context before editing (they may have
drifted from prior sessions' edits).

- [ ] **Step 3: Add `num_patches` to `build_finetune`'s allowed kwargs**

In `model/MeSAE/MeSAE.py`, find `build_finetune` (search for `def build_finetune`). Its
non-`head_z` branch currently reads:

```python
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq', 'freeze_backbone')
    return MeSAEFeatureHead(backbone, num_channels, num_classes, input=input,
                            **{k: v for k, v in kw.items() if k in allowed})
```

Add `'num_patches'` to that tuple:

```python
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq',
               'freeze_backbone', 'num_patches')
```

(The `head_z` branch's own `allowed` tuple, a few lines above, is unrelated and must NOT be
touched — `MeSAEFinetune` never takes `num_patches`.)

- [ ] **Step 4: Add the `num_patches` parameter and `pool_time="learned:R"` parsing to
  `MeSAEFeatureHead.__init__`**

Read the actual current `MeSAEFeatureHead.__init__` in full before editing (search for
`class MeSAEFeatureHead` in `model/MeSAE/MeSAE.py`) — it currently looks like this (as of
this plan's writing):

```python
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, input='recon',
                 task='mi', pool_channel='concat', pool_time='trial', z_proj=8,
                 dropout=0.1, sample_freq=200, freeze_backbone=True):
        super().__init__()
        if task != 'mi':
            raise NotImplementedError(f"task={task!r}: only 'mi' is built (ADR 0014 build order)")
        if not freeze_backbone:
            raise NotImplementedError("MeSAEFeatureHead assumes a frozen backbone (ADR 0012)")
        if input not in ('raw', 'recon', 'stamp_bandpow', 'stamp_induced', 'z_chan'):
            raise ValueError(f"unknown input {input!r}")
        self.backbone, self.input, self.fs = backbone, input, float(sample_freq)
        for p in backbone.parameters():
            p.requires_grad_(False)

        C = num_channels
        if input in ('stamp_bandpow', 'stamp_induced'):
            st = backbone.stamps
            alive = (st.fire_ema >= st.dead_threshold).nonzero().flatten()
            keep = torch.cat([alive, torch.arange(st.n_routed, st.n_stamps, device=alive.device)])
            self.register_buffer('keep', keep)
        K = C if pool_channel == 'concat' else int(pool_channel.split(':')[1])
        self.window = None if pool_time == 'trial' else \
            tuple(float(v) for v in pool_time.split(':')[1].split('-'))
        head = {}
        if pool_channel != 'concat':
            head['spatial'] = nn.Linear(C, K, bias=False)
        if input == 'z_chan':
            head['z_proj'] = nn.Linear(backbone.head_dim, z_proj)
            n_feat = K * z_proj
        elif input == 'stamp_induced':
            n_feat = K * len(keep)
        else:
            n_feat = K * len(self.BANDS)
        # BatchNorm stands in for the probe's StandardScaler: log-powers are far from unit scale.
        head['cls'] = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(dropout), nn.Linear(n_feat, num_classes))
        self.head = nn.ModuleDict(head)

        if input == 'stamp_bandpow':
            with torch.no_grad():
                D_tab, H_tab = (t[keep].float() for t in st._template_tables())
                fr = torch.fft.rfftfreq(D_tab.shape[-1], 1.0 / self.fs)
                sel = [((fr >= lo) & (fr < hi)).to(D_tab.device) for lo, hi in self.BANDS]
                spec = lambda T: torch.stack([torch.fft.rfft(T, dim=-1).abs().pow(2)[:, m].sum(-1) for m in sel], -1)
                self.register_buffer('E_D', spec(D_tab))                   # [S, bands]
                self.register_buffer('E_H', spec(H_tab))
```

Make these changes:

1. Add `num_patches=None` to the constructor signature (after `sample_freq=200,`):

```python
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, input='recon',
                 task='mi', pool_channel='concat', pool_time='trial', z_proj=8,
                 dropout=0.1, sample_freq=200, freeze_backbone=True, num_patches=None):
```

2. Replace the `self.window = None if pool_time == 'trial' else ...` line with parsing that
   also recognizes `"learned:R"`:

```python
        self.time_rank = None
        if pool_time == 'trial':
            self.window = None
        elif pool_time.startswith('learned:'):
            if input != 'stamp_induced':
                raise NotImplementedError(
                    f"pool_time='learned:R' is only implemented for input='stamp_induced' "
                    f"(ADR 0014 experiment C1), got input={input!r}")
            if num_patches is None:
                raise ValueError("pool_time='learned:R' requires num_patches (pass it through "
                                  "build_finetune_from_config -- see model/factory.py)")
            self.window = None
            self.time_rank = int(pool_time.split(':')[1])
        else:
            self.window = tuple(float(v) for v in pool_time.split(':')[1].split('-'))
```

3. Right after the `head['cls'] = nn.Sequential(...)` line and `self.head =
   nn.ModuleDict(head)` line — actually, `time_p`/`time_q` must be added to the `head` dict
   BEFORE `self.head = nn.ModuleDict(head)` is called (a `ModuleDict` is built once from the
   dict passed to it; parameters added to the dict afterward would not register). So insert
   this block between the `if pool_channel != 'concat': head['spatial'] = ...` line and the
   `if input == 'z_chan': ... else: n_feat = ...` block:

```python
        if self.time_rank is not None:
            R = self.time_rank
            S = len(keep)
            # Small random init keeps pre-softmax logits near zero, so the learned weighting
            # starts equivalent to C0's flat mean (uniform softmax) and only diverges from it
            # as training proceeds -- makes "does learned beat flat" a clean ablation instead
            # of a different starting point (ADR 0014 build-order step 8).
            head['time_p'] = nn.Parameter(torch.randn(R, S) * 0.02)
            head['time_q'] = nn.Parameter(torch.randn(R, num_patches) * 0.02)
```

   (`keep` is already in scope at this point per the ordering already established for C0 —
   confirm this by reading the actual current file, since this plan's Step 4 assumes that
   ordering fix from the C0 plan is still in place.)

4. Leave the `n_feat` calculation, `head['cls']`, `self.head = nn.ModuleDict(head)`, and the
   `stamp_bandpow`-only `E_D`/`E_H` block completely unchanged — `n_feat` is still `K *
   len(keep)` regardless of which time-pooling mode is used, since the output feature shape
   `[B, K, S]` doesn't change, only how it's computed from `[B, N, K, S]`.

- [ ] **Step 5: Add the learned-weight math to `forward`**

Read the actual current `forward` method's `stamp_induced` branch (search for `elif
self.input == 'stamp_induced':` inside `def forward`). It currently reads:

```python
            elif self.input == 'stamp_induced':
                # C0 (ADR 0014 experiment C): spatial filter (step 1) + induced branch with
                # flat time weights (step 2a, w[s,n] = 1/N) -- log mean power per (filter,
                # stamp), no band collapse. The pre-log per-stamp powers span the same
                # information stamp_bandpow's band-summed powers do (via E_D/E_H), but this
                # readout is on log-power, which is NOT a strict superset of stamp_bandpow's
                # readout -- log doesn't distribute over the band sum, so this is not
                # guaranteed to reproduce stamp_bandpow spatial:8's 0.522 exactly (ADR 0014).
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                feat = torch.log((a.pow(2) + b.pow(2)).mean(1) + 1e-12)                      # [B, K, S]
```

Change it to branch on `self.time_rank`:

```python
            elif self.input == 'stamp_induced':
                # C0 (ADR 0014 experiment C): spatial filter (step 1) + induced branch with
                # flat time weights (step 2a, w[s,n] = 1/N) -- log mean power per (filter,
                # stamp), no band collapse. The pre-log per-stamp powers span the same
                # information stamp_bandpow's band-summed powers do (via E_D/E_H), but this
                # readout is on log-power, which is NOT a strict superset of stamp_bandpow's
                # readout -- log doesn't distribute over the band sum, so this is not
                # guaranteed to reproduce stamp_bandpow spatial:8's 0.522 exactly (ADR 0014).
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                power = a.pow(2) + b.pow(2)                                                   # [B, N', K, S]
                if self.time_rank is not None:
                    # C1: learned low-rank time weights (step 2a, ADR 0014 build-order 8),
                    # softmax-normalized over the patch axis per stamp -- w[s,n] sums to 1
                    # over n for each stamp, so this is a weighted mean, directly comparable
                    # to C0's uniform mean (w[s,n] = 1/N) rather than an unbounded rescaling.
                    logits_time = torch.einsum('rs,rn->sn', self.head['time_p'], self.head['time_q'])  # [S, N']
                    w = torch.softmax(logits_time, dim=-1)                                    # [S, N']
                    pooled = torch.einsum('sn,bnks->bks', w, power)                            # [B, K, S]
                else:
                    pooled = power.mean(1)                                                    # [B, K, S]
                feat = torch.log(pooled + 1e-12)                                              # [B, K, S]
```

Note: `w`'s time axis (`n`) must match `power`'s `N'` axis exactly. Since `pool_time="learned:R"`
forces `self.window = None` (Step 4.2), `self._patch_keep` returns `None` (see
`_patch_keep`'s existing `if self.window is None: return None`), so `pk` is `None` and `amp`
(and therefore `power`) is never sliced down from the full patch count — `N' == num_patches`
exactly, matching `time_q`'s registered width. This is why `pool_time="learned:R"` and any
`window:lo-hi` sub-selection are mutually exclusive in this implementation (Step 4.2 already
encodes this by making `self.window` `None` in the learned case).

- [ ] **Step 6: Update the class docstring**

Add a short note to `MeSAEFeatureHead`'s class docstring's `pool_time:` line (currently
`pool_time: trial | window:lo-hi (seconds from trial start; patches fully inside)`):

```python
    pool_time:    trial | window:lo-hi (seconds from trial start; patches fully inside) |
                  learned:R (ADR 0014 C1, stamp_induced only -- low-rank softmax-weighted
                  time pooling, R = rank; requires num_patches)
```

- [ ] **Step 7: Shape/build smoke check (no training run needed)**

No test suite exists in this repo (CLAUDE.md). Verify the plumbing and math with a shape
check, extending the pattern C0 used:

```bash
python -c "
import json, torch
from model.factory import build_pretrain_from_config
from model.MeSAE.MeSAE import build_finetune

with open('config/config.json') as f:
    config = json.load(f)
backbone = build_pretrain_from_config(config, mode='finetune')
backbone.eval()
backbone.stamps.fire_ema.fill_(1.0)  # force a non-empty alive-stamp set on a fresh backbone
num_channels = 64  # matches canonical_channels='10-10' in this config; adjust if it differs
n_patches = 30      # arbitrary for this shape check -- must match what forward() is fed below
model = build_finetune(backbone, num_channels, num_classes=4, input='stamp_induced',
                        pool_channel='spatial:8', pool_time='learned:2', task='mi',
                        sample_freq=config['preprocess_params']['sample_freq'],
                        num_patches=n_patches, dropout=0.5)
B, C, N, L = 2, num_channels, n_patches, config['preprocess_params']['patch_length']
x = torch.randn(B, C, N, L)
coords = torch.randn(B, C, 3)
time_idx = torch.arange(N).unsqueeze(0).expand(B, N)
valid_channels = torch.ones(B, C, dtype=torch.bool)
logits, _, _ = model(x, coords, time_idx=time_idx, valid_channels=valid_channels)
print('logits shape:', tuple(logits.shape))
assert logits.shape == (B, 4)
print('time_p shape:', tuple(model.head['time_p'].shape))
print('time_q shape:', tuple(model.head['time_q'].shape))
assert model.head['time_q'].shape[-1] == n_patches
print('learned-mode params:', sum(p.numel() for p in model.head.parameters()))

# Confirm flat mode (pool_time='trial') still works unchanged -- regression check.
model2 = build_finetune(backbone, num_channels, num_classes=4, input='stamp_induced',
                         pool_channel='spatial:8', pool_time='trial', task='mi',
                         sample_freq=config['preprocess_params']['sample_freq'],
                         num_patches=None, dropout=0.5)
logits2, _, _ = model2(x, coords, time_idx=time_idx, valid_channels=valid_channels)
assert logits2.shape == (B, 4)
assert 'time_p' not in model2.head
print('flat mode (regression check): OK')

# Confirm the other four input arms still build fine with num_patches=None.
for inp in ('raw', 'recon', 'stamp_bandpow', 'z_chan'):
    m = build_finetune(backbone, num_channels, num_classes=4, input=inp,
                        sample_freq=config['preprocess_params']['sample_freq'],
                        num_patches=None)
    lg, _, _ = m(x, coords, time_idx=time_idx, valid_channels=valid_channels)
    assert lg.shape == (B, 4), inp
    print(f'{inp}: OK')
print('ALL OK')
"
```

Expected: no exception, `logits shape: (2, 4)`, `time_q shape: (2, 30)` (matching `n_patches`),
a small `learned-mode params` count (R=2, S~25-ish depending on the checkpoint's alive-stamp
count, N=30 in this synthetic check → roughly `2*(S+30)` extra parameters, a few hundred at
most), the flat-mode regression check passes with no `time_p` in its head, and all four other
input arms still build and run. `ALL OK` prints last.

- [ ] **Step 8: Commit**

```bash
git add model/factory.py train_finetune.py model/MeSAE/MeSAE.py
git commit -m "feat: add pool_time='learned:R' (low-rank time weights) to MeSAEFeatureHead (ADR 0014 C1)"
```

---

## Task 2: Establish the missing C0 baseline, then run C1, then update the ADR

**Files:**
- Modify: `config/config.json` (`model_params.MeSAE.finetune`, `training_params.finetune`) —
  edited twice in sequence (once per run).
- Modify: `docs/adr/0014-finetune-head-test-plan.md` (build-order step 8, and a short
  addendum to step 7 recording the missing baseline).

**Interfaces:**
- Consumes: `pool_time="learned:R"` and the `num_patches` plumbing from Task 1.
- Consumes: `split_mode="intra_subject_cv"`, `cv_folds` (existing, from the C0 plan's Task 1).

- [ ] **Step 1: Run the missing baseline — `stamp_induced`, flat time weights, 5-fold CV,
  `dropout: 0.5`**

Edit `config/config.json`'s `model_params.MeSAE.finetune` to:

```json
      "finetune": {
        "input": "stamp_induced",
        "task": "mi",
        "pool_channel": "spatial:8",
        "pool_time": "trial",
        "dropout": 0.5,
        "freeze_backbone": true
      }
```

And `training_params.finetune` to (note `cv_folds: 5`, not 3):

```json
    "finetune": {
      "model_type": "MeSAE",
      "model_name": "mesae_finetune_c0_dropout05_5fold",
      "pretrained_checkpoint": "output/pretrain/mesae_v10_small/checkpoint/last.pth",
      "learning_rate": 0.01,
      "min_learning_rate": 0.001,
      "backbone_lr_mult": 0.0,
      "epochs": 100,
      "warmup_epochs": 2,
      "batch_size": 16,
      "weight_decay": 0.01,
      "device": "cuda",
      "split_mode": "intra_subject_cv",
      "cv_folds": 5
    }
```

Run: `python train_finetune.py --config config/config.json`. This is a real, long-running
GPU job (9 subjects x 5 folds x 100 epochs -- budget similar order of magnitude to the
original C0 run, ~3-4 hours). Let it run to completion. Then read the result:
`python probes/ft_summary.py output/archive/experiment_c/mesae_finetune_c0_dropout05_5fold/artifacts/train_<timestamp>.log`.
Record the tail-mean `balanced_acc` -- this is the number C1 (Step 2 below) must beat.

- [ ] **Step 2: Run C1 — `stamp_induced`, learned time weights (rank 2), 5-fold CV,
  `dropout: 0.5`**

Edit `config/config.json`'s `model_params.MeSAE.finetune`, changing only `pool_time`:

```json
      "finetune": {
        "input": "stamp_induced",
        "task": "mi",
        "pool_channel": "spatial:8",
        "pool_time": "learned:2",
        "dropout": 0.5,
        "freeze_backbone": true
      }
```

And `training_params.finetune`, changing only `model_name`:

```json
      "model_name": "mesae_finetune_c1_learned2",
```

(everything else identical to Step 1's `training_params.finetune`). Run the same way, wait
for completion, then read the result the same way:
`python probes/ft_summary.py output/archive/experiment_c/mesae_finetune_c1_learned2/artifacts/train_<timestamp>.log`.

**Rank choice:** use `R=2` (not 1) for this run -- the ADR's ablation ladder names R=1-2 as
the range to try, and R=2 gives the learned weighting slightly more expressiveness (e.g. one
factor could learn "attend to the post-cue window", a second could learn a
frequency-dependent time profile) while staying tiny in parameter count. If time allows a
second run at `R=1` for comparison, that's a reasonable optional addition, but not required
by this plan -- note it as a possible follow-up in the ADR update (Step 3) rather than running
it, to keep this task's own scope to what the build-order step actually asks for (one ablation
factor per run).

- [ ] **Step 3: Update the ADR**

Read `docs/adr/0014-finetune-head-test-plan.md`'s build-order step 8 (currently just
`8. **C1 — learned time weights.** First test of "when"; the stamp analysis and the raw
beta lateralization both point at 0.5-2.5 s.`) and step 7 (already extensively documented
from the C0 work). Add:

1. A short addendum to step 7 noting the missing baseline is now filled in: report Step 1's
   observed tail-mean `balanced_acc` (5-fold, `dropout 0.5`) alongside the existing 5-fold/
   `dropout 0` (0.456) and 3-fold/`dropout 0.5` (0.468) numbers, so a future reader has all
   three data points together with their exact settings labeled.
2. Extend step 8 with C1's result: the observed tail-mean `balanced_acc`, whether it beats
   Step 1's new baseline (the actual acceptance criterion -- "C1 must beat C0"), the
   per-subject breakdown (same style as step 7's inline numbers), and the learned head
   parameter count (`sum(p.numel() for p in model.head.parameters())`, comparable to the
   ADR's "~1k parameters" target mentioned in the Head section's "Size control is mandatory"
   paragraph).
3. If C1 beats the baseline: say so plainly, and note this is now consistent with continuing
   to C2 (per-stamp features) as the next build-order step. If it does NOT beat the baseline:
   say so plainly and do not speculate about why beyond what the numbers directly show (e.g.
   don't invent a story about the softmax normalization or R choice without evidence) --
   record it as an open question for whoever scopes the next step, same restraint the C0
   follow-up wrote-up used.
4. Do not mark build-order step 8 as fully "done" even if C1 beats the baseline -- record the
   result, leave the "Next, in order" framing intact per how steps 7 and earlier were handled.

Match the ADR's existing prose style (short bolded lead phrases, inline numbers, no new
markdown tables for a section this size).

- [ ] **Step 4: Commits**

Two commits for the config changes (one before each run, matching the pattern from the C0
follow-up plan) and one for the ADR update at the end, after both runs' results are known.
Check `git log -5` for this repo's exact commit-message and attribution-trailer format before
committing.

## Self-Review Notes

- **Spec coverage:** ADR build-order step 8 (C1) is Task 2. The plumbing Task 1 needs
  (threading `num_patches` through three call sites) is not itself an ADR requirement but is
  a hard prerequisite for building the `nn.Parameter`s C1's design requires before the
  optimizer is built -- documented in the Architecture section above.
- **Placeholder scan:** no TBD/TODO; every step shows the literal diff, code to insert, or
  command to run.
- **Type consistency:** `pool_time="learned:R"` (Task 1, Steps 4-5) is what Task 2's config
  (Step 2) sets; `num_patches` flows `run_training_loop` → `build_finetune_from_config` →
  `build_finetune` → `MeSAEFeatureHead.__init__` with the same parameter name and `None`
  default at every hop, so any caller that doesn't need it is unaffected. `head['time_p']`/
  `head['time_q']` (Task 1 Step 4.3) are read by name in Task 1 Step 5's forward-pass code --
  names match exactly.
- **Ordering dependency carried over from the C0 plan:** Task 1 Step 4's insertion point for
  `time_p`/`time_q` assumes `keep` is already computed earlier in `__init__` (the ordering
  fix from the C0 plan's pre-flight scan). This plan's Step 4 explicitly calls out
  "confirm this by reading the actual current file" rather than assuming it silently, since a
  regression here would produce a confusing `NameError` deep in construction.
