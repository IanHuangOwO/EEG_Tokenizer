# Experiment C — C0 (induced branch, wiring check) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the first rung of ADR 0014 experiment C — a per-stamp, per-channel-filter
"induced power" feature with flat (uniform) time pooling — and prove it reproduces the
existing `stamp_bandpow` spatial:8 result (0.522 balanced acc) under the new 5-fold-CV
protocol, before any of C1–C5 add real learned structure on top.

**Architecture:** Reuse the existing `MeSAEFeatureHead` (model/MeSAE/MeSAE.py) rather than
adding a new head class — it already has the exact building blocks C0 needs (frozen
backbone, `StampBank.dense_amp`, a shared signed `Linear(C, K)` channel mixer via `_mix`,
alive-stamp bookkeeping). C0 only needs a new `input='stamp_induced'` arm that skips the
existing `stamp_bandpow` arm's band-energy collapse (`E_D`/`E_H`) and instead keeps one
log-power feature per (filter, stamp) pair. Separately, `train_finetune.py` gets a new
`split_mode='intra_subject_cv'` (5-fold `StratifiedKFold` per subject) because ADR 0014's
protocol section requires this for every experiment-C number, replacing experiment B's
80/20 split.

**Tech Stack:** PyTorch, scikit-learn (`StratifiedKFold`, already a dependency via
`sklearn.model_selection` used elsewhere in the repo), the existing `train_finetune.py` /
`probes/ft_summary.py` pipeline.

**Spec:** `docs/adr/0014-finetune-head-test-plan.md` — sections "Experiment C — one head for
every EEG paradigm", "Protocol", "Implementation notes", and build-order steps 7 and the
"C uses 5-fold CV per subject" note under "Method fix".

## Global Constraints

- Backbone stays frozen (ADR 0012); C0 must not touch `freeze_backbone`.
- Backbone checkpoint for every C measurement is `output/pretrain/mesae_v10_small_uw01/checkpoint/last.pth`
  (ADR 0014 §Protocol: "for plumbing"; conclusions about the tokenizer itself need the
  full-data run, not in scope here).
- Optimizer for every trained head (ADR 0014 §Protocol): `learning_rate: 0.01`,
  `min_learning_rate: 0.001`, `epochs: 100`, `warmup_epochs: 2`, head `dropout: 0`. The repo's
  finetune defaults (lr 1e-3, 50 epochs, dropout 0.3) under-train a linear head — do not use
  them for this experiment.
- Reporting metric is `balanced_acc`, mean over each subject's last 10 epochs (`probes/ft_summary.py`'s
  "tail" row) — never best-val-epoch, which is optimistic on small val sets.
- C0's acceptance check is architectural, not statistical: it must land at
  `stamp_bandpow` spatial:8's 0.522 (ADR 0014 Experiment B table), not merely "close under a
  paired t-test". A clear miss (not noise-sized) means a wiring bug, per the ADR's own framing
  of this step ("A miss is a wiring bug, not a design result").
- No project test suite exists (per CLAUDE.md) — validation is empirical: run
  `train_finetune.py`, then `probes/ft_summary.py` on the resulting log, and read the printed
  `tail` mean.
- Do not touch the currently-uncommitted, unrelated `mesae_v10_all_share` pretrain config
  changes already sitting in `config/config.json` / `config/analysis.json` — those are a
  separate in-flight pretrain experiment (`stamp_bank` shrunk to `n_routed_stamps: 32,
  n_shared_stamps: 0, stamp_top_k: 32`). C0 reads `model_params.MeSAE.finetune` and
  `training_params.finetune` only.

---

## Task 1: `intra_subject_cv` split mode in `train_finetune.py`

**Files:**
- Modify: `train_finetune.py:280-354` (add a CV split helper and a CV runner next to
  `_intra_subject_split` / `_run_intra_subject`)
- Modify: `train_finetune.py:583-590` (dispatch the new split_mode in `main()`)

**Interfaces:**
- Consumes: `run_training_loop(config, train_dataset, val_dataset, checkpoint_dir, vis_dir, artifact_dir, logger, patch_len, fold_tag="")` (existing, unchanged, `train_finetune.py:390`) —
  returns `{'epoch', 'train', 'val', 'last_val'}` or `None`.
- Consumes: `build_dataset_from_config(config, transform=None, mode='finetune')` (existing,
  `IO/dataset.py`).
- Produces: `_run_intra_subject_cv(config, dataset_params, base_output_dir, artifact_dir, logger, patch_len, split_ratio, n_folds=5)` — no return value, writes
  `artifact_dir/intra_subject_cv_summary.json` and logs per-fold results, same shape as
  `_run_intra_subject`'s summary.
- Produces: fold tags of the form `f"{ds_name}_f{k}_S{s}"` (fold index before subject id) so
  `probes/ft_summary.py`'s `int(t.split('_S')[-1])` sort key still parses correctly and its
  flat mean over all fold×subject entries equals the mean of each subject's 5-fold mean
  (every subject contributes exactly `n_folds` entries of equal weight).

- [ ] **Step 1: Add the CV split helper**

Insert right after `_intra_subject_split` (`train_finetune.py:294`):

```python
def _intra_subject_cv_splits(dataset, subject, n_folds=5, seed=42):
    """One subject's own trials, stratified into n_folds train/val Subset pairs
    (sklearn StratifiedKFold — same balance guarantee _intra_subject_split's
    per-class split gives, but partitioning instead of one fixed holdout)."""
    from sklearn.model_selection import StratifiedKFold
    base = dataset.base_dataset
    subjects = base.subject_data.numpy()
    labels = base.labels.numpy()
    idx = np.flatnonzero(subjects == subject)
    y = labels[idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_pos, val_pos in skf.split(idx, y):
        train_idx = sorted(idx[train_pos].tolist())
        val_idx = sorted(idx[val_pos].tolist())
        folds.append((Subset(dataset, train_idx), Subset(dataset, val_idx)))
    return folds
```

- [ ] **Step 2: Add the CV runner**

Insert right after `_run_intra_subject` (`train_finetune.py:354`), modeled on
`_run_intra_subject`'s subject loop and `_run_loso`'s fold aggregation:

```python
def _run_intra_subject_cv(config, dataset_params, base_output_dir, artifact_dir, logger,
                           patch_len, n_folds=5):
    """5-fold CV per subject (ADR 0014 experiment C protocol) — replaces experiment B's
    one 80/20 split so head numbers are comparable to probe_v10.py's 5-fold LDA and the
    +/-0.05 gaps B left non-significant at n=9 shrink."""
    ds_name, ds_args = next(iter(dataset_params.items()))
    full = build_dataset_from_config(copy.deepcopy(config), transform=None, mode='finetune')
    subjects = sorted(set(full.base_dataset.subject_data.numpy().tolist()))
    logger.info(f"INTRA-subject CV: {len(subjects)} subjects x {n_folds} folds ({ds_name})")

    per_subject_tail = {}
    for s in subjects:
        folds = _intra_subject_cv_splits(full, s, n_folds=n_folds)
        fold_metrics = []
        for k, (tr, va) in enumerate(folds):
            fold_tag = f"{ds_name}_f{k}_S{s}"
            logger.info(f"===== intra-subject-cv {fold_tag}: train={len(tr)} val={len(va)} =====")
            ck = os.path.join(base_output_dir, "finetune", f"fold{k}_subj_{s}")
            vz = os.path.join(base_output_dir, "visualization", f"fold{k}_subj_{s}")
            os.makedirs(ck, exist_ok=True); os.makedirs(vz, exist_ok=True)
            best = run_training_loop(config, tr, va, ck, vz, artifact_dir, logger,
                                      patch_len, fold_tag=fold_tag)
            fold_metrics.append(best)
        per_subject_tail[f"{ds_name}_S{s}"] = fold_metrics

    logger.info("===== INTRA-subject-CV Summary (mean over folds' best-val-acc epoch) =====")
    metric_keys = ['acc', 'f1', 'f1_weighted', 'balanced_acc', 'kappa']
    per_metric = {k: [] for k in metric_keys}
    summary = {'subjects': {}, 'aggregate': {}}
    for tag, folds in per_subject_tail.items():
        valid = [f for f in folds if f is not None]
        summary['subjects'][tag] = folds
        if not valid:
            logger.info(f"  {tag}: no valid epoch"); continue
        means = {k: statistics.mean(f['val'][k] for f in valid) for k in metric_keys}
        logger.info(f"  {tag}: " + " | ".join(f"{k}={v:.4f}" for k in metric_keys)
                    + f" ({len(valid)}/{len(folds)} folds)")
        for k in metric_keys:
            per_metric[k].append(means[k])

    for k in metric_keys:
        vals = per_metric[k]
        if vals:
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary['aggregate'][k] = {'mean': mean, 'std': std}
            logger.info(f"  MEAN {k}: {mean:.4f} +/- {std:.4f}")
    path = os.path.join(artifact_dir, 'intra_subject_cv_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"intra-subject-cv summary written to {path}")
```

- [ ] **Step 3: Wire the new split_mode into `main()`**

In `train_finetune.py`, `main()` currently has (`train_finetune.py:583-590`):

```python
    if split_mode == 'intra_subject':
        _run_intra_subject(config, dataset_params, base_output_dir, artifact_dir, logger,
                           patch_len, split_ratio)
        return
```

Add directly below it:

```python
    if split_mode == 'intra_subject_cv':
        n_folds = train_params.get('cv_folds', 5)
        _run_intra_subject_cv(config, dataset_params, base_output_dir, artifact_dir, logger,
                              patch_len, n_folds)
        return
```

And extend the `ValueError` a few lines down (`train_finetune.py:594-597`) to mention it:

```python
        raise ValueError(
            f"Unknown split_mode: {split_mode!r} "
            "(expected 'inter_subject', 'intra_subject', 'intra_subject_cv' or 'loso')")
```

- [ ] **Step 4: Smoke-check the split helper in isolation**

No test suite exists in this repo (CLAUDE.md), so verify with a short interactive check
instead of a pytest file. From the repo root:

```bash
python -c "
import json, copy
from IO.dataset import build_dataset_from_config
from train_finetune import _intra_subject_cv_splits

with open('config/config.json') as f:
    config = json.load(f)
ds_params = config['dataset_params']['finetune']
full = build_dataset_from_config(copy.deepcopy(config), transform=None, mode='finetune')
subjects = sorted(set(full.base_dataset.subject_data.numpy().tolist()))
s = subjects[0]
folds = _intra_subject_cv_splits(full, s, n_folds=5)
assert len(folds) == 5
sizes = [(len(tr), len(va)) for tr, va in folds]
print('subject', s, 'fold sizes (train, val):', sizes)
total_val = sum(va for _, va in sizes)
n_trials_this_subject = int((full.base_dataset.subject_data.numpy() == s).sum())
assert total_val == n_trials_this_subject, (total_val, n_trials_this_subject)
print('OK: every trial appears in exactly one val fold')
"
```

Expected: 5 (train, val) pairs of roughly equal val size, and the OK line prints (every
trial covered exactly once across the 5 val folds, the defining property of k-fold CV).

- [ ] **Step 5: Commit**

```bash
git add train_finetune.py
git commit -m "feat: add intra_subject_cv split mode (ADR 0014 experiment C protocol)"
```

---

## Task 2: `stamp_induced` input arm on `MeSAEFeatureHead` (C0)

**Files:**
- Modify: `model/MeSAE/MeSAE.py:706-845` (`MeSAEFeatureHead` class)

**Interfaces:**
- Consumes: `StampBank.dense_amp(z, rms=None) -> amp [G, C, n_stamps, 2]` (existing,
  `model/MeSAE/MeSAE_modules.py:995`).
- Consumes: `StampBank.fire_ema`, `StampBank.dead_threshold`, `StampBank.n_routed`,
  `StampBank.n_stamps` (existing buffers/attrs, `model/MeSAE/MeSAE_modules.py:619-689`).
- Consumes: `self._mix(t, dim)` (existing, `model/MeSAE/MeSAE.py:794-798`) — signed
  `Linear(C, K)` over axis `dim`, identity when `pool_channel == 'concat'`.
- Produces: `MeSAEFeatureHead(backbone, num_channels, num_classes, input='stamp_induced', ...)`
  — same constructor/forward contract as the other four `input` values, returns
  `(logits, None, None)`.

- [ ] **Step 1: Extend the allowed-input check**

`model/MeSAE/MeSAE.py:737` currently:

```python
        if input not in ('raw', 'recon', 'stamp_bandpow', 'z_chan'):
            raise ValueError(f"unknown input {input!r}")
```

Change to:

```python
        if input not in ('raw', 'recon', 'stamp_bandpow', 'stamp_induced', 'z_chan'):
            raise ValueError(f"unknown input {input!r}")
```

- [ ] **Step 2: Compute `keep` (alive-stamp indices) before `n_feat`, and split it out of
  the `stamp_bandpow`-only block**

In the current file, `model/MeSAE/MeSAE.py:743-757` (the `C = num_channels` ... `self.head =
nn.ModuleDict(head)` block, which computes `n_feat` and therefore needs `keep` already
available) runs *before* `model/MeSAE/MeSAE.py:759-770` (the `if input == 'stamp_bandpow':`
block that currently computes `keep` alongside `E_D`/`E_H`). Ordering must change: `keep`
has to exist before `n_feat` is computed, and `stamp_induced` needs `keep` but not
`E_D`/`E_H`. Do both moves as one edit:

1. Insert this block immediately after `C = num_channels` (before the `K = C if
   pool_channel == 'concat' ...` line, i.e. right at the top of `__init__`'s body where `C`
   is first assigned):

```python
        if input in ('stamp_bandpow', 'stamp_induced'):
            st = backbone.stamps
            alive = (st.fire_ema >= st.dead_threshold).nonzero().flatten()
            keep = torch.cat([alive, torch.arange(st.n_routed, st.n_stamps, device=alive.device)])
            self.register_buffer('keep', keep)
```

2. Replace the existing `if input == 'stamp_bandpow':` block (`model/MeSAE/MeSAE.py:759-770`)
   — which currently redefines `st`, `alive`, and `keep` — with just the `E_D`/`E_H` part,
   reusing the `keep` buffer registered in step 1 above:

```python
        if input == 'stamp_bandpow':
            with torch.no_grad():
                D_tab, H_tab = (t[keep].float() for t in st._template_tables())
                fr = torch.fft.rfftfreq(D_tab.shape[-1], 1.0 / self.fs)
                sel = [((fr >= lo) & (fr < hi)).to(D_tab.device) for lo, hi in self.BANDS]
                spec = lambda T: torch.stack([torch.fft.rfft(T, dim=-1).abs().pow(2)[:, m].sum(-1) for m in sel], -1)
                self.register_buffer('E_D', spec(D_tab))                   # [S, bands]
                self.register_buffer('E_H', spec(H_tab))
```

`st` here is the same `backbone.stamps` bound in step 1's block — since both blocks now run
inside the same `if input in (...)` / `if input == 'stamp_bandpow':` pair in sequence, `st`
from step 1 is still in scope in step 2 (both are plain local variables in the same
`__init__` call, not separate closures). If your edit tool applies these as two disjoint
`Edit` calls and `st` ends up out of scope, add `st = backbone.stamps` as the first line of
step 2's block — either form is correct, prefer reusing step 1's `st` when the diff tool
lets you keep it in one contiguous block.

- [ ] **Step 3: Give `stamp_induced` its feature width**

`model/MeSAE/MeSAE.py:750-754` currently:

```python
        if input == 'z_chan':
            head['z_proj'] = nn.Linear(backbone.head_dim, z_proj)
            n_feat = K * z_proj
        else:
            n_feat = K * len(self.BANDS)
```

Change to:

```python
        if input == 'z_chan':
            head['z_proj'] = nn.Linear(backbone.head_dim, z_proj)
            n_feat = K * z_proj
        elif input == 'stamp_induced':
            n_feat = K * len(keep)
        else:
            n_feat = K * len(self.BANDS)
```

`keep` is in scope here because Step 2 registers `self.keep` (and the plain local `keep`,
in the same `__init__` call) before this block runs, per the reordering in Step 2.

- [ ] **Step 4: Share the amp computation between `stamp_bandpow` and `stamp_induced`**

`model/MeSAE/MeSAE.py:816` currently:

```python
                if self.input == 'stamp_bandpow':
```

Change to:

```python
                if self.input in ('stamp_bandpow', 'stamp_induced'):
```

(the six lines under it, `model/MeSAE/MeSAE.py:817-822`, are unchanged — they already
compute the shared `amp [B, N, C, S, 2]` tensor sliced to `self.keep`.)

- [ ] **Step 5: Add the induced-branch feature math**

`model/MeSAE/MeSAE.py:833-837` currently:

```python
            elif self.input == 'stamp_bandpow':
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                pw = torch.einsum('bnks,sq->bkq', a.pow(2), self.E_D) \
                    + torch.einsum('bnks,sq->bkq', b.pow(2), self.E_H)
                feat = torch.log(pw / a.shape[1] + 1e-12)                                    # [B, K, bands]
```

Add a `stamp_induced` branch right after it (before the `else:` that handles `z_chan`):

```python
            elif self.input == 'stamp_induced':
                # C0 (ADR 0014 experiment C): spatial filter (step 1) + induced branch with
                # flat time weights (step 2a, w[s,n] = 1/N) -- log mean power per (filter,
                # stamp), no band collapse. Wiring check: must land near stamp_bandpow
                # spatial:8's 0.522, since summing this over bands via E_D/E_H would give
                # back exactly the stamp_bandpow feature.
                a, b = self._mix(amp[..., 0], 2), self._mix(amp[..., 1], 2)                  # [B, N', K, S]
                feat = torch.log((a.pow(2) + b.pow(2)).mean(1) + 1e-12)                      # [B, K, S]
```

- [ ] **Step 6: Update the class docstring**

`model/MeSAE/MeSAE.py:706-720` lists the four `input` values. Add a line for the fifth:

```python
                  stamp_induced  ADR 0014 experiment C, step C0: per-(channel-filter, stamp)
                                 log power, flat (uniform) time weights -- same amp tensor as
                                 stamp_bandpow, kept per-stamp instead of band-collapsed
```

- [ ] **Step 7: Confirm the module still imports and builds**

No test suite exists (CLAUDE.md) — this is a shape/wiring check, not a training run:

```bash
python -c "
import json, torch
from model.factory import build_pretrain_from_config
from model.MeSAE.MeSAE import build_finetune

with open('config/config.json') as f:
    config = json.load(f)
backbone = build_pretrain_from_config(config, mode='finetune')
backbone.eval()
num_channels = 22  # BCICIV2a channel count; adjust if canonical_channels differs
model = build_finetune(backbone, num_channels, num_classes=4, input='stamp_induced',
                        pool_channel='spatial:8', task='mi',
                        sample_freq=config['preprocess_params']['sample_freq'])
B, C, N, L = 2, num_channels, 3, config['preprocess_params']['patch_length']
x = torch.randn(B, C, N, L)
coords = torch.randn(B, C, 3)
time_idx = torch.arange(N).unsqueeze(0).expand(B, N)
valid_channels = torch.ones(B, C, dtype=torch.bool)
logits, _, _ = model(x, coords, time_idx=time_idx, valid_channels=valid_channels)
print('logits shape:', tuple(logits.shape))
assert logits.shape == (B, 4)
print('n_feat width:', model.head['cls'][0].num_features)
print('OK')
"
```

Expected: no exception, `logits shape: (2, 4)`, and an `n_feat` printed as
`8 * len(alive_routed_stamps + n_shared)` for whatever checkpoint
`training_params.finetune.pretrained_checkpoint` currently points at. `OK` prints last.

- [ ] **Step 8: Commit**

```bash
git add model/MeSAE/MeSAE.py
git commit -m "feat: add stamp_induced input arm to MeSAEFeatureHead (ADR 0014 experiment C0)"
```

---

## Task 3: Wire config and run the C0 wiring check

**Files:**
- Modify: `config/config.json` (`model_params.MeSAE.finetune`, `training_params.finetune`)

**Interfaces:**
- Consumes: `input='stamp_induced'` and the `intra_subject_cv` split mode from Tasks 1–2.
- Produces: a training log under `output/<model_name>/artifacts/train_*.log`, readable by
  `probes/ft_summary.py`.

- [ ] **Step 1: Point the finetune config at C0**

Edit `config/config.json`'s `model_params.MeSAE.finetune` block (currently
`{"hidden": 100, "freeze_backbone": true, "dropout": 0.3}`, `config/config.json:217-220`) to:

```json
      "finetune": {
        "input": "stamp_induced",
        "task": "mi",
        "pool_channel": "spatial:8",
        "pool_time": "trial",
        "dropout": 0,
        "freeze_backbone": true
      }
```

- [ ] **Step 2: Point `training_params.finetune` at the C0 protocol**

Edit `config/config.json`'s `training_params.finetune` block
(`config/config.json:239-249`) to:

```json
    "finetune": {
      "model_type": "MeSAE",
      "model_name": "mesae_finetune_c0",
      "pretrained_checkpoint": "output/pretrain/mesae_v10_small_uw01/checkpoint/last.pth",
      "learning_rate": 0.01,
      "min_learning_rate": 0.001,
      "backbone_lr_mult": 0.0,
      "epochs": 100,
      "warmup_epochs": 2,
      "batch_size": 16,
      "weight_decay": 0.01,
      "device": "cuda",
      "split_mode": "intra_subject_cv",
      "cv_folds": 5,
      "train_val_split": 0.8
    }
```

(`backbone_lr_mult: 0.0` is inert here since `freeze_backbone: true` already stops the
backbone's `requires_grad`, but setting it to 0 keeps the optimizer's param-group list from
ever picking up backbone params if that flag is ever flipped without updating this value too.)

Confirm `dataset_params.finetune` already points at BCICIV2a with `subject_to_use: ["all"]`
(ADR 0014 §Protocol) — read the existing block and adjust only if it targets a different
dataset; do not change channel/subject selection otherwise.

- [ ] **Step 3: Run the wiring check**

```bash
python train_finetune.py --config config/config.json
```

This trains 9 subjects x 5 folds = 45 per-subject-fold models (frozen backbone, ~1-2k
parameter head, 100 epochs each) and writes
`output/experiment_c/mesae_finetune_c0/artifacts/train_<timestamp>.log`.

- [ ] **Step 4: Read the headline number**

```bash
python probes/ft_summary.py output/experiment_c/mesae_finetune_c0/artifacts/train_<timestamp>.log
```

Expected: the printed `tail` mean (last-10-epoch `balanced_acc`, averaged over all 45
fold x subject entries — equivalent to the mean of each subject's 5-fold mean, since every
subject contributes exactly 5 entries) lands close to 0.522, the `stamp_bandpow` spatial:8
number from ADR 0014's experiment B table. Record the exact number and the per-subject
breakdown in the ADR's build-order table (step 7) once observed — do not silently accept a
large miss as "probably fine": ADR 0014 is explicit that a clear miss here means a wiring
bug, not a result to report.

- [ ] **Step 5: Commit the config change (only after Step 4 confirms the wiring check)**

```bash
git add config/config.json
git commit -m "config: point finetune at C0 (stamp_induced, intra_subject_cv)"
```

## Self-Review Notes

- **Spec coverage:** Build-order step 7 ("C0 — induced branch only, flat time weights, must
  reproduce stamp_bandpow spatial:8") is Tasks 2–3. The Protocol section's "C uses 5-fold CV
  per subject" is Task 1. Steps C1–C5, the regime tests, and stamp attribution (build-order
  steps 8–14) are deliberately out of scope for this plan — each depends on C0's result and
  gets its own plan once C0's wiring check passes, per the ADR's own "one factor per run"
  discipline.
- **Placeholder scan:** no TBD/TODO; every step shows the literal diff or command.
- **Type consistency:** `MeSAEFeatureHead(..., input='stamp_induced', ...)` in Task 2 matches
  the `build_finetune` dispatcher's existing `allowed` kwarg tuple
  (`model/MeSAE/MeSAE.py:854`, unchanged by this plan — `task`, `pool_channel`, `pool_time`,
  `dropout`, `sample_freq`, `freeze_backbone` already cover everything C0's config in Task 3
  sets). `fold_tag` format (`f"{ds_name}_f{k}_S{s}"`) is used consistently in Task 1's Step 2
  and matched against `probes/ft_summary.py`'s existing (unmodified) parsing in Task 3 Step 4.
