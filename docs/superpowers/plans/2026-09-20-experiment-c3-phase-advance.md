# Experiment C — C3 (phase advance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the phase-advance branch (`2c` in ADR 0014's Head pseudocode) to
`MeSAEFeatureHead`'s `stamp_induced` arm, on top of C1's best-so-far configuration
(spatial filter + per-stamp induced power + learned rank-2 time weights), then run the
wiring check ADR 0014 build-order step 10 calls for: **C3 must beat C1** (0.536,
5-fold CV, `dropout 0.5`, `pool_time="learned:2"` — the current best head).

**Architecture:** Extend `MeSAEFeatureHead` again (same pattern as C0's `stamp_induced`
arm and C1's `pool_time="learned:R"`) rather than adding a new class. Add a new boolean
constructor flag, `include_advance` (valid only for `input="stamp_induced"`), that — when
true — computes a second feature block from the *same* `a, b` tensors the induced branch
already computes (no new backbone call), and concatenates it to the induced-power
features before the classifier.

The phase-advance term is `z[k,s] = Σ_n u[n+1,k,s]·conj(u[n,k,s])`, where
`u[n,k,s] = a[n,k,s] + i·b[n,k,s]` is the complex code the induced branch already has as
`a, b` (post spatial-filter, shape `[B, N', K, S]` each). Expanded into real arithmetic
(no complex dtype needed):

```
z_re[k,s] = Σ_n ( a[n+1,k,s]·a[n,k,s] + b[n+1,k,s]·b[n,k,s] )   # [B, K, S]
z_im[k,s] = Σ_n ( b[n+1,k,s]·a[n,k,s] − a[n+1,k,s]·b[n,k,s] )   # [B, K, S]
```

`z_re`/`z_im` are fed directly (no log — they can be negative) into the same shared
`BatchNorm1d → Dropout → Linear` classifier as the induced-power features, concatenated
along the flattened feature axis. `z`'s magnitude captures "rhythm steadiness" (how
consistent the phase advance is patch-to-patch) and its angle captures "sub-bin
frequency" (where the rhythm sits inside the stamp's ~4 Hz template bin) — real+imag is
the numerically simplest representation of both at once; no magnitude/angle conversion
is needed since the linear classifier can learn any function of `(z_re, z_im)` a
magnitude/angle pair could express.

**This is exactly the feature-budget regime ADR 0014 warns about.** With `include_advance`
on, feature width becomes `K · S · 3` (induced: `K·S`, advance: `2·K·S`) — at this run's
shape (`K=8`, `S=25`) that's **600 features**, precisely the number the ADR's "Size
control is mandatory" paragraph names as the regime where `z_chan` hit "train 0.99 /
val 0.41." No group penalty or extra size control is implemented here — C0 and C1 didn't
implement one either (both relied on `dropout: 0.5` alone), and this plan follows the
same precedent rather than introducing a new regularization mechanism as a surprise
scope addition. **This is a real, foreseen risk, not an oversight** — if C3 overfits
badly, that is itself the finding (same as C0's first run), to be reported honestly, not
preemptively engineered around.

**Tech Stack:** PyTorch (elementwise ops on existing tensors, no new ops needed), the
existing `train_finetune.py` / `probes/ft_summary.py` pipeline (unmodified — the
subject-level grouping fix already lives there from the C1 plan).

**Spec:** `docs/adr/0014-finetune-head-test-plan.md` — the `Head` pseudocode block
(`2c. advance`), the "Ablation ladder" table (`C3 | phase advance (2c) | C2`, and note
this plan treats "must beat C2" as "must beat C1" per the already-recorded finding that
C2 needed no separate run — see build-order step 9), the "Size control is mandatory"
paragraph, and the Protocol section's acceptance-comparison methodology (added during
the C1 plan: tail-mean comparison, n=9 paired t-test reported but never a pass/fail
gate).

## Global Constraints

- Backbone stays frozen (ADR 0012) — do not touch `freeze_backbone`.
- Backbone checkpoint: `output/pretrain/mesae_v10_small_uw01/checkpoint/last.pth` (same as every
  prior C-experiment measurement).
- Optimizer/protocol (ADR 0014 §Protocol): `learning_rate: 0.01`, `min_learning_rate:
  0.001`, `epochs: 100`, `warmup_epochs: 2`, `split_mode: "intra_subject_cv"`,
  `cv_folds: 5`.
- Head `dropout: 0.5` — unchanged from C1, so the C3 run tests exactly one new factor
  (the advance branch) against C1's exact configuration otherwise. Do NOT raise dropout
  to compensate for the larger feature width in this run — that would confound two
  changes in one comparison; if C3 overfits, a regularization-matched follow-up is a
  separate, later step (same pattern as C0 → C0 follow-up (b)/(c)).
- `pool_time="learned:2"` stays on — C3 builds on C1, not on flat C0. This run's config
  is C1's config plus `include_advance: true`, nothing else different.
- Reporting metric: `balanced_acc`, mean over each subject's last 10 epochs
  (`probes/ft_summary.py`'s "tail" row, which already groups by subject before any
  significance test — use it as-is, don't recompute by hand).
- Acceptance comparison: tail-mean point difference against C1 (0.536), paired t-test
  across the 9 subjects (folds averaged within-subject first — this is what
  `probes/ft_summary.py` now does automatically), reported honestly regardless of
  significance (per the Protocol section's acceptance-comparison methodology — never
  gate advancement on p-value at n=9).
- No project test suite exists (CLAUDE.md) — validation is empirical: a shape/build
  smoke check for Task 1 (no training run needed), then the real training run for
  Task 2.
- Do not touch `model_params.MeSAE.pretrain.stamp_bank` (must stay `60/4/12/6`).
- Do not touch `dataset_params.finetune` (already correctly points at BCICIV2a).
- `input="raw"`, `"recon"`, `"stamp_bandpow"`, `"z_chan"`, and `stamp_induced` with
  `include_advance` unset/false must be completely unaffected by this plan — this is an
  additive, opt-in flag, not a rewrite of the existing forward path.

---

## Task 1: `include_advance` phase-advance branch on `MeSAEFeatureHead`

**Files:**
- Modify: `model/MeSAE/MeSAE.py` — `MeSAEFeatureHead.__init__` (new flag, feature-width
  accounting) and `.forward` (the `stamp_induced` branch), and `build_finetune`'s
  `allowed` kwargs tuple.

**Interfaces:**
- Produces: `MeSAEFeatureHead(..., include_advance=False)` — new optional constructor
  kwarg, valid only when `input="stamp_induced"` (raise `NotImplementedError` otherwise,
  matching the existing pattern for `pool_time="learned:R"`'s input restriction).
- Produces: `build_finetune(..., include_advance=False)` dispatches it through
  `build_finetune`'s `allowed` tuple to `MeSAEFeatureHead`, same as every other kwarg.
- Consumes: the existing `a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)`
  tensors already computed inside `forward`'s `stamp_induced` branch (`model/MeSAE/MeSAE.py`,
  search for `elif self.input == 'stamp_induced':` inside `def forward`) — no new
  backbone computation, no new buffers beyond what C0/C1 already registered.

- [ ] **Step 1: Read the actual current `MeSAEFeatureHead` class in full**

Read `model/MeSAE/MeSAE.py`'s `MeSAEFeatureHead` class (search `class MeSAEFeatureHead`)
top to bottom before editing — confirm the current `__init__` and `forward` structure
matches what this plan assumes (it was last touched by the C1 plan; line numbers below
are anchors, not guarantees — always match against the live file). In particular locate:
the `if self.time_rank is not None:` block in `__init__` (where `head['time']` gets
registered before `self.head = nn.ModuleDict(head)`), the `elif input == 'stamp_induced':
n_feat = K * len(keep)` line, and the `stamp_induced` branch inside `forward` (the block
computing `a, b`, `power`, and `feat`).

- [ ] **Step 2: Add `include_advance` to the constructor**

Add `include_advance=False` to `MeSAEFeatureHead.__init__`'s signature (alongside
`num_patches=None`):

```python
    def __init__(self, backbone: MeSAEPretrain, num_channels, num_classes, input='recon',
                 task='mi', pool_channel='concat', pool_time='trial', z_proj=8,
                 dropout=0.1, sample_freq=200, freeze_backbone=True, num_patches=None,
                 include_advance=False):
```

Immediately after the existing `input not in (...)` validation (the
`if input not in ('raw', 'recon', 'stamp_bandpow', 'stamp_induced', 'z_chan'): raise
ValueError(...)` line), add:

```python
        if include_advance and input != 'stamp_induced':
            raise NotImplementedError(
                f"include_advance is only implemented for input='stamp_induced' "
                f"(ADR 0014 experiment C3), got input={input!r}")
        self.include_advance = include_advance
```

- [ ] **Step 3: Widen `n_feat` when the advance branch is on**

Find the `n_feat` computation (currently `elif input == 'stamp_induced': n_feat = K *
len(keep)`). Change to:

```python
        elif input == 'stamp_induced':
            n_feat = K * len(keep) * (3 if include_advance else 1)
```

`3x` because: 1 induced-power feature per `(k, s)` plus 2 advance features (`z_re`,
`z_im`) per `(k, s)` when `include_advance` is on — matches the ADR's own "K·S·3" size
warning exactly (see this plan's Architecture section — this is intentional, not a
miscount).

- [ ] **Step 4: Compute and concatenate the advance features in `forward`**

Find the `stamp_induced` branch inside `forward` (inside the `with
torch.autocast(device_type=x.device.type, enabled=False):` block). It currently ends
with:

```python
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                power = a.pow(2) + b.pow(2)                                                   # [B, N', K, S]
                if self.time_rank is not None:
                    logits_time = torch.einsum('rs,rn->sn', self.head['time']['p'], self.head['time']['q'])  # [S, N']
                    w = torch.softmax(logits_time, dim=-1)                                    # [S, N']
                    pooled = torch.einsum('sn,bnks->bks', w, power)                            # [B, K, S]
                else:
                    pooled = power.mean(1)                                                    # [B, K, S]
                feat = torch.log(pooled + 1e-12)                                              # [B, K, S]
```

Change the final line and add the advance computation after it:

```python
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                power = a.pow(2) + b.pow(2)                                                   # [B, N', K, S]
                if self.time_rank is not None:
                    logits_time = torch.einsum('rs,rn->sn', self.head['time']['p'], self.head['time']['q'])  # [S, N']
                    w = torch.softmax(logits_time, dim=-1)                                    # [S, N']
                    pooled = torch.einsum('sn,bnks->bks', w, power)                            # [B, K, S]
                else:
                    pooled = power.mean(1)                                                    # [B, K, S]
                feat = torch.log(pooled + 1e-12)                                              # [B, K, S]
                if self.include_advance:
                    # C3 (ADR 0014 build-order step 10): phase-advance branch (2c),
                    # z[k,s] = sum_n u[n+1,k,s] * conj(u[n,k,s]), u = a + i*b -- the same
                    # a, b this branch already computed, no new backbone call. Real/imag
                    # expansion (no complex dtype): z_re captures rhythm steadiness
                    # (magnitude-like), z_im captures sub-bin frequency (phase-like);
                    # feeding both raw (no log -- they can be negative) lets the linear
                    # classifier learn any function of magnitude+angle without an explicit
                    # atan2. Not log-power-scaled like `feat`, but the shared BatchNorm1d
                    # ahead of the classifier normalizes per-feature scale regardless.
                    a_next, a_prev = a[:, 1:], a[:, :-1]                                       # [B, N'-1, K, S]
                    b_next, b_prev = b[:, 1:], b[:, :-1]
                    z_re = (a_next * a_prev + b_next * b_prev).sum(1)                          # [B, K, S]
                    z_im = (b_next * a_prev - a_next * b_prev).sum(1)                          # [B, K, S]
                    feat = torch.cat([feat, z_re, z_im], dim=-1)                                # [B, K, 3*S]
```

Note the `dim=-1` concat: `feat`, `z_re`, `z_im` are all `[B, K, S]` at this point: `cat`
along the last (stamp) axis gives `[B, K, 3*S]`, which flattens to `K*S*3` — matching
Step 3's `n_feat`. (An alternative would be `dim=1` giving `[B, 3*K, S]`, also `3*K*S`
elements after flattening — either is numerically fine since the classifier's `Linear`
doesn't care about feature order, but `dim=-1` keeps all three blocks for a given filter
`k` contiguous in the flattened vector, which is the more natural grouping. Use `dim=-1`
as specified.)

- [ ] **Step 5: Add `include_advance` to `build_finetune`'s allowed kwargs**

Find `build_finetune` (search `def build_finetune`). Its non-`head_z` branch currently
reads (as of the C1 plan):

```python
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq',
               'freeze_backbone', 'num_patches')
```

Add `'include_advance'`:

```python
    allowed = ('task', 'pool_channel', 'pool_time', 'z_proj', 'dropout', 'sample_freq',
               'freeze_backbone', 'num_patches', 'include_advance')
```

(Leave the `head_z` branch's own `allowed` tuple untouched — `MeSAEFinetune` never takes
`include_advance`.)

- [ ] **Step 6: Update the class docstring**

Add a line to `MeSAEFeatureHead`'s class docstring (find the existing `pool_time:` line)
documenting the new flag, e.g. immediately after the `pool_time:` line:

```python
    include_advance: bool, stamp_induced only (ADR 0014 C3) -- concatenates the phase-
                  advance branch (2c) to the induced-power features, tripling feature
                  width (K*S -> K*S*3). No size control beyond dropout is implemented;
                  this is the exact K*S*3 regime ADR 0014's "Size control is mandatory"
                  paragraph warns about, by design -- see the ADR for the reasoning.
```

- [ ] **Step 7: Shape/build smoke check (no training run needed)**

No test suite exists in this repo (CLAUDE.md). Verify the plumbing and math with a shape
check, extending the pattern C0/C1 used:

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
n_patches = 30      # arbitrary for this shape check

model = build_finetune(backbone, num_channels, num_classes=4, input='stamp_induced',
                        pool_channel='spatial:8', pool_time='learned:2', task='mi',
                        sample_freq=config['preprocess_params']['sample_freq'],
                        num_patches=n_patches, dropout=0.5, include_advance=True)
B, C, N, L = 2, num_channels, n_patches, config['preprocess_params']['patch_length']
x = torch.randn(B, C, N, L)
coords = torch.randn(B, C, 3)
time_idx = torch.arange(N).unsqueeze(0).expand(B, N)
valid_channels = torch.ones(B, C, dtype=torch.bool)
logits, _, _ = model(x, coords, time_idx=time_idx, valid_channels=valid_channels)
print('logits shape:', tuple(logits.shape))
assert logits.shape == (B, 4)
n_feat_expected = 8 * model.keep.numel() * 3
print('n_feat expected:', n_feat_expected, 'cls Linear in_features:', model.head['cls'][2].in_features)
assert model.head['cls'][2].in_features == n_feat_expected

# Gradient check: advance features must actually receive gradient (not detached/dead).
loss = logits.sum()
loss.backward()
assert model.head['cls'][2].weight.grad is not None and model.head['cls'][2].weight.grad.abs().sum() > 0
print('gradient reaches classifier: OK')

# Regression check: include_advance=False (C1 config unchanged) still builds and matches n_feat.
model2 = build_finetune(backbone, num_channels, num_classes=4, input='stamp_induced',
                         pool_channel='spatial:8', pool_time='learned:2', task='mi',
                         sample_freq=config['preprocess_params']['sample_freq'],
                         num_patches=n_patches, dropout=0.5, include_advance=False)
logits2, _, _ = model2(x, coords, time_idx=time_idx, valid_channels=valid_channels)
assert logits2.shape == (B, 4)
assert model2.head['cls'][2].in_features == 8 * model2.keep.numel()
print('include_advance=False regression: OK')

# Confirm include_advance raises for non-stamp_induced input.
try:
    build_finetune(backbone, num_channels, num_classes=4, input='stamp_bandpow',
                    pool_channel='spatial:8', task='mi',
                    sample_freq=config['preprocess_params']['sample_freq'],
                    include_advance=True)
    raise SystemExit('expected NotImplementedError, got none')
except NotImplementedError:
    print('include_advance input guard: OK')

print('ALL OK')
"
```

Expected: no exception, `logits shape: (2, 4)`, `n_feat_expected` matches the classifier's
actual `in_features`, gradient reaches the classifier weight (proves `z_re`/`z_im` are
wired into the computation graph, not accidentally detached), the `include_advance=False`
regression check passes with the original `K*S` width, and the input guard raises as
expected. `ALL OK` prints last.

- [ ] **Step 8: Commit**

```bash
git add model/MeSAE/MeSAE.py
git commit -m "feat: add include_advance phase-advance branch to MeSAEFeatureHead (ADR 0014 C3)"
```

---

## Task 2: Wire config, run C3, update the ADR

**Files:**
- Modify: `config/config.json` (`model_params.MeSAE.finetune`, `training_params.finetune`)
- Modify: `docs/adr/0014-finetune-head-test-plan.md` (build-order step 10)

**Interfaces:**
- Consumes: `include_advance=True` from Task 1, on top of C1's exact
  `pool_time="learned:2"` configuration.

- [ ] **Step 1: Point the finetune config at C3**

Edit `config/config.json`'s `model_params.MeSAE.finetune` — this should currently be C1's
config (`input: "stamp_induced"`, `pool_channel: "spatial:8"`, `pool_time: "learned:2"`,
`dropout: 0.5`, `freeze_backbone: true`, `task: "mi"` — confirm against the live file
before editing, since it may have drifted). Add exactly one field:

```json
      "finetune": {
        "input": "stamp_induced",
        "task": "mi",
        "pool_channel": "spatial:8",
        "pool_time": "learned:2",
        "dropout": 0.5,
        "freeze_backbone": true,
        "include_advance": true
      }
```

Edit `training_params.finetune.model_name` to `"mesae_finetune_c3_advance"`. Confirm
every other field (`pretrained_checkpoint`, `learning_rate`, `min_learning_rate`,
`backbone_lr_mult`, `epochs`, `warmup_epochs`, `batch_size`, `weight_decay`, `device`,
`split_mode`, `cv_folds`) already matches C1's run (5-fold, `dropout` unrelated to this
block) — do not change any of them.

- [ ] **Step 2: Run C3**

```bash
python train_finetune.py --config config/config.json
```

Real, long-running GPU job — 9 subjects × 5 folds × 100 epochs, budget similar order of
magnitude to prior 5-fold `stamp_induced` runs (~3-4 hours; the advance branch adds
negligible compute — it's a cheap elementwise reduction over already-computed tensors,
not a new backbone pass). Let it run to completion — do not treat a long runtime as a
failure. Once done, find the log (`output/experiment_c/mesae_finetune_c3_advance/artifacts/
train_<timestamp>.log`) and run `python probes/ft_summary.py <that log path>` — this
script already groups by subject before its paired t-test (fixed during the C1 plan), so
its printed `p=`/`wins` figures are directly usable, no hand-recomputation needed this
time.

Also extract final-epoch train `balanced_acc` across all 45 fold×subject runs (same
overfitting sanity check as every prior C-experiment run — `grep "\[Train\]" <log> |
tail -45 | grep -oE "acc: [0-9.]+"` or a short one-off script) — given this run's larger
600-feature width, this number matters more than usual for interpreting whatever tail-mean
comes out.

**If it fails or crashes** for an environment/config reason, diagnose and report — don't
silently work around it by changing epochs/folds/subjects/dropout. If you hit a genuinely
new code bug in `train_finetune.py` or `model/MeSAE/MeSAE.py` and the fix is narrow and
mechanically justified, you may fix it (own commit, documented as a deviation) — but do
not let a fix change this run's measured configuration.

- [ ] **Step 3: Update the ADR**

Read `docs/adr/0014-finetune-head-test-plan.md`'s build-order step 10 (currently just
`10. **C3 — phase advance (2c).**`) and step 9's now-extensive C2 entry (added by this
same plan's earlier prep work — read it for the exact prose convention: bolded lead
phrases, inline numbers, explicit significance caveats, no new markdown tables).

Extend step 10 with:

1. The observed tail-mean `balanced_acc`, whether it beats C1's 0.536 (the actual
   acceptance criterion — "C3 must beat C1", since C2 needed no separate run), the
   per-subject breakdown, and the `probes/ft_summary.py`-computed n=9 paired-t-test
   figures (mean diff, t, p) — report honestly regardless of significance, per the
   Protocol section's acceptance-comparison methodology; do not gate the writeup on
   whether p clears any threshold.
2. The final-epoch train-accuracy check from Step 2, compared explicitly against C1's
   0.843 (already on record in step 8) and the original unregularized C0 run's ~0.985 —
   state plainly whether the 600-feature width overfit worse than C1's 200-feature
   width despite identical `dropout: 0.5`, since that's the concrete risk this plan's
   Architecture section named upfront.
3. A plain statement of what the result implies for `2c`'s premise (does sub-bin
   frequency / rhythm-steadiness information actually help MI decoding on top of
   induced power + learned time weights, or not) — report what the numbers show, don't
   speculate beyond them.
4. Do not mark build-order step 10 as fully "done" — record the result, keep it under
   "Next, in order" per how steps 7-9 were handled.

- [ ] **Step 4: Commits**

Two commits: one for the config change (before running), one for the ADR update (after
the run's results are known). Check `git log -5` for this repo's exact commit-message and
attribution-trailer format before committing.

## Self-Review Notes

- **Spec coverage:** ADR build-order step 10 (C3) is Task 2. The `include_advance`
  plumbing Task 1 needs is a direct, additive extension of the exact pattern C0
  (`stamp_induced` arm) and C1 (`pool_time="learned:R"`) already established on this
  same class — no new architectural pattern introduced.
- **Placeholder scan:** no TBD/TODO; every step shows the literal diff, code to insert,
  or command to run.
- **Type consistency:** `include_advance=True` (Task 1) is exactly what Task 2's config
  sets; `n_feat = K * len(keep) * (3 if include_advance else 1)` (Task 1 Step 3) is
  consumed by the `head['cls']` `Linear` layer sized from it, and Task 1 Step 7's smoke
  check explicitly asserts this width matches at runtime — not just described, verified.
- **Known risk stated upfront, not discovered mid-review:** the K·S·3 = 600 feature
  count this plan produces is the exact regime ADR 0014 names as dangerous (z_chan's
  overfitting precedent). This plan does not attempt to engineer around it (no group
  penalty added) — consistent with C0/C1's precedent of relying on `dropout: 0.5` alone
  and reporting the outcome honestly. A reviewer should not treat "no size control
  beyond dropout" as an oversight; it's a stated, deliberate scope boundary, matching
  this document's own established pattern of running first and diagnosing honestly
  rather than pre-engineering for a failure mode that may not materialize.
