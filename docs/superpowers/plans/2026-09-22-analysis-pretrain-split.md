# analysis_pretrain.py / analysis_finetune.py split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `check_model.py` to `analysis_pretrain.py` (pretrain-only), move its
finetune-only logic verbatim into a new `analysis_finetune.py`, and move `viz/__init__.py`'s
non-plotting orchestration helpers into a new `analysis/` package that both scripts import
from.

**Architecture:** Pure relocation, no behavior change. `git mv` for whole-file moves keeps
history; `Edit` for the parts of `check_model.py` that split across two destination files.
Each task ends with a real-checkpoint smoke run compared against a baseline captured before
any file moves.

**Tech Stack:** Python, PyTorch, `eeg_fm` conda env (`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`
— has `mne`; do not use `base`, see CLAUDE.md).

**Spec:** `docs/superpowers/specs/2026-09-22-analysis-pretrain-split-design.md`

## Global Constraints

- No test suite exists (CLAUDE.md) — validation is smoke runs, never pytest files.
- Run everything in `eeg_fm` (`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`), not `base`.
- This is a pure relocation — no function's internal logic changes, only where it lives and
  what it's called. Any smoke-run output difference (besides file mtimes) is a bug in the
  move, not something to "fix" by changing behavior.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
- `probes/` is explicitly out of scope — do not touch it.
- Scratch/throwaway overlay configs go in the scratchpad directory, never in `config/`.

---

## Task 1: `analysis/` package — move `viz/__init__.py`'s orchestration helpers

**Files:**
- Create: `analysis/__init__.py` (via `git mv` from `viz/__init__.py`)
- Modify: `viz/__init__.py` (new, minimal content)
- Modify: `train_pretrain.py:21`
- Modify: `docs/agents/adding-a-model.md:72`

**Interfaces:**
- Produces: `analysis.load_config`, `analysis.resolve_output_dir`, `analysis.select_subject_dataset`,
  `analysis.filter_config_to_subject`, `analysis.load_model`, `analysis.pick_trial`,
  `analysis.setup_mne_info`, `analysis._deep_merge` — same signatures as today's
  `viz.<name>`, only the import path changes. Task 2 and Task 3 both consume
  `load_model`, `select_subject_dataset`, `filter_config_to_subject`, `pick_trial`,
  `resolve_output_dir`, `_deep_merge` from `analysis`.

- [ ] **Step 1: Move the file, keeping history**

```bash
git mv viz/__init__.py analysis/__init__.py
```

- [ ] **Step 2: Update the moved file's docstring**

In `analysis/__init__.py`, replace:

```python
"""
Shared infrastructure for the viz package.

Every viz module imports from here instead of duplicating
dataset filtering, model loading, and trial selection logic.
"""
```

with:

```python
"""
Shared orchestration helpers for the analysis scripts (analysis_pretrain.py,
analysis_finetune.py): config loading/merging, model loading, dataset/trial/subject
selection, output-dir resolution. Nothing here renders a plot — see viz/ for that.
"""
```

- [ ] **Step 3: Create the new, minimal `viz/__init__.py`**

```python
"""
viz/ — plotting only (extract.py, panels.py, topomap.py, timeseries.py, codebook.py,
iclabel.py). Config/model/dataset orchestration helpers live in analysis/, not here.
"""
```

- [ ] **Step 4: Fix the one live cross-file import**

In `train_pretrain.py`, line 21, replace:
```python
from viz import pick_trial
```
with:
```python
from analysis import pick_trial
```

- [ ] **Step 5: Fix the doc reference to the moved function**

In `docs/agents/adding-a-model.md`, line 72, replace:
```
def enable_spatial(self):
    """Turns on cross-channel mixing. Called at construction-time load (viz/__init__.py
    load_model), at the start of train_tokenizer.py (BaseTrainer.on_tokenizer_start), and
```
with:
```
def enable_spatial(self):
    """Turns on cross-channel mixing. Called at construction-time load (analysis/__init__.py
    load_model), at the start of train_tokenizer.py (BaseTrainer.on_tokenizer_start), and
```

- [ ] **Step 6: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import analysis, train_pretrain; print('ok')"
```
Expected: `ok`, no `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 7: Confirm nothing else broke**

```bash
git grep -n "from viz import\|^import viz$"
```
Expected: exactly one hit, in `check_model.py` (its `from viz import (...)` block — that
file isn't touched until Task 2). `train_pretrain.py` must show none — if it still does,
Step 4 above didn't take.

- [ ] **Step 8: Commit**

```bash
git add analysis viz/__init__.py train_pretrain.py docs/agents/adding-a-model.md
git commit -m "refactor: move viz/__init__.py orchestration helpers into new analysis/ package

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 2: `check_model.py` → `analysis_pretrain.py` (strip finetune logic)

**Files:**
- Create: `analysis_pretrain.py` (via `git mv` from `check_model.py`)
- Test/baseline artifacts (scratchpad, not committed): pretrain and finetune baseline
  listings, captured from the still-intact `check_model.py` before it's moved.

**Interfaces:**
- Produces: `analysis_pretrain.run(config, output_dir, model, dataset, trial_idx, mode='pretrain', ...)`
  — same signature as today's `check_model.run`, minus the `mode='finetune'` branch (that
  branch, and the standalone helpers it needs, move to Task 3's `analysis_finetune.py`
  instead — copy their exact source from this task's Step 2 diff, do not re-derive it).

- [ ] **Step 1: Capture BOTH baselines before touching any file**

Pretrain baseline (uses the checkpoint `config/analysis.json` already points at):
```bash
cd /media/mamechin/PortableSSD/iansaididontcare/CNElab/cnelab_model_trainer/EEG_Tokenizer
/home/mamechin/anaconda3/envs/eeg_fm/bin/python check_model.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0 \
  > /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/pretrain_baseline.log 2>&1
find output/mesae_v10_all_share/analysis -type f | sort \
  > /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/pretrain_baseline_files.txt
```

Finetune baseline (write the overlay first — the run's own `artifacts/config.json` has a
stale pre-rename `dataset_params.finetune` entry, so this overlay replaces it with the
current dataset location):
```bash
cat > /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/finetune_smoke_overlay.json <<'EOF'
{
  "checkpoint": "output/baseline/BNCI2014001_intra_raw_band/finetune/run_1_fold0/head.pth",
  "mode": "finetune",
  "dataset_params": {
    "finetune": {
      "BNCI2014001": {"dataset_path": "datas/finetune/BNCI2014001", "subject_to_use": ["1"], "channels_to_use": ["all"]}
    }
  },
  "check": {"plot_recon": true, "plot_topo_psd": true, "cmap": "YlOrRd"}
}
EOF
/home/mamechin/anaconda3/envs/eeg_fm/bin/python check_model.py \
  --config /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/finetune_smoke_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001 \
  > /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/finetune_baseline.log 2>&1
find output/baseline/BNCI2014001_intra_raw_band/analysis -type f | sort \
  > /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/finetune_baseline_files.txt
```
Check both `.log` files end with `[check] done: ...` lines and exit 0 (`echo $?`). Keep all
four scratchpad files — Task 3 reuses the finetune ones.

- [ ] **Step 2: Move the file, keeping history**

```bash
git mv check_model.py analysis_pretrain.py
```

- [ ] **Step 3: Update the module docstring**

In `analysis_pretrain.py`, replace the top docstring:
```python
"""
Post-training checker: per-subject topo/PSD/attention snapshot (MeSAE, resolved
from the model instance) via BaseEpochChecker.check_pretrain/check_finetune
(model/base_checker.py).

Config resolution: config/analysis.json (or --config) is a small overlay — checkpoint,
mode, dataset_params.pretrain (one dataset entry, subject_to_use = subjects to visualize;
shared by Tokenizer and Pretrain-stage checkpoints, see CLAUDE.md), check.plot_* toggles.
It's deep-merged onto the full run config, taken from the checkpoint's own
output/<model_name>/artifacts/config.json snapshot unless overlay['base_config'] or
--base-config points elsewhere. Output goes to
output/<model_name>/analysis/<dataset_name>/recon/ (separate from training's own
output/<model_name>/visualization/).
"""
```
with:
```python
"""
Post-training checker for the PRETRAIN stage only: per-subject topo/PSD/attention
snapshot (MeSAE, resolved from the model instance) via BaseEpochChecker.check_pretrain
(model/base_checker.py), plus cross-dataset codebook/vocab diagnostics
(model/base_codebook_checker.py). Finetune-stage analysis lives in analysis_finetune.py.

Config resolution: config/analysis.json (or --config) is a small overlay — checkpoint,
dataset_params.pretrain (one dataset entry, subject_to_use = subjects to visualize;
shared by Tokenizer and Pretrain-stage checkpoints, see CLAUDE.md), check.plot_* toggles.
It's deep-merged onto the full run config, taken from the checkpoint's own
output/<model_name>/artifacts/config.json snapshot unless overlay['base_config'] or
--base-config points elsewhere. Output goes to
output/<model_name>/analysis/<dataset_name>/recon/ (separate from training's own
output/<model_name>/visualization/).
"""
```

- [ ] **Step 4: Switch the `viz` import to `analysis`**

Replace:
```python
    from viz import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
```
with:
```python
    from analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
```

- [ ] **Step 5: Drop the finetune branch from `run()`**

Replace:
```python
def run(config, output_dir, model, dataset, trial_idx, mode='pretrain', subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, plot_attn_topo=True,
        tag=''):
    model_type = 'MeSAE'  # only registered model (MeFSQ removed, docs/adr/0013)
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    if mode == 'finetune':
        return checker.check_finetune(
            config, output_dir, model, dataset, trial_idx,
            subject_id=subject_id, epoch=epoch, cmap=cmap,
            plot_recon=plot_recon, plot_topo_psd=plot_topo_psd,
            trainer=trainer, tag=tag,
        )
    return checker.check_pretrain(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
        trainer=trainer,
    )
```
with:
```python
def run(config, output_dir, model, dataset, trial_idx, subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, plot_attn_topo=True):
    model_type = 'MeSAE'  # only registered model (MeFSQ removed, docs/adr/0013)
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    return checker.check_pretrain(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd, plot_attn_topo=plot_attn_topo,
        trainer=trainer,
    )
```
(`mode`/`tag` dropped — `run()` is pretrain-only now, `tag` was only ever passed by the
finetune per-class snapshot loop this file no longer has.)

- [ ] **Step 6: Remove the three finetune-only helper functions**

Delete `_load_target_names`, `_safe_name`, and `_predict_all` entirely from
`analysis_pretrain.py` (their exact source, to paste into `analysis_finetune.py`, is given
in Task 3 Step 1 below — copy it from there, not from git history).

- [ ] **Step 7: Drop `--mode` from the CLI and hardcode `mode`/`data_mode`**

Replace:
```python
    parser.add_argument('--mode',        default=None, choices=['pretrain', 'finetune'])
    parser.add_argument('--analysis',    default=None, choices=['snapshot', 'codebook', 'both'],
```
with:
```python
    parser.add_argument('--analysis',    default=None, choices=['snapshot', 'codebook', 'both'],
```

Replace:
```python
    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = args.mode or overlay.get('mode', 'pretrain')
    # dataset_params only has 'pretrain'/'finetune' — the Tokenizer stage shares the
    # Pretrain stage's dataset entries (same raw data, no masking), see CLAUDE.md.
    data_mode  = 'finetune' if mode == 'finetune' else 'pretrain'
```
with:
```python
    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = 'pretrain'
    # dataset_params only has 'pretrain'/'finetune' — the Tokenizer stage shares the
    # Pretrain stage's dataset entries (same raw data, no masking), see CLAUDE.md.
    data_mode  = 'pretrain'
```

- [ ] **Step 8: Delete the finetune CLI branch**

In the `if analysis in ('snapshot', 'both'):` block, delete the entire
`if mode == 'finetune': ... ` sub-branch (from `if mode == 'finetune':` through the line
right before `elif cfg.get('training_params', {}).get('visualize_params', {})...`), and
change the following `elif` to `if` since it's now the first condition in the chain:

Replace:
```python
    if analysis in ('snapshot', 'both'):
        if mode == 'finetune':
            # Per-target correct/wrong snapshot pairs, not a single per-subject trial pick:
            # for each of the num_classes targets, find one trial the model got right and one
            # it got wrong, and render the full 3-panel snapshot (recon_signal, topo_psd_filter,
            # attn_topo) for each — num_classes * 2 * 3 files total, searched across every
            # subject in dataset_params.finetune[dataset_name].subject_to_use (not one subject
            # at a time), since a single subject isn't guaranteed to contain both a correct and
            # a wrong example of every class.
            dataset_name = args.dataset or next(iter(ds_params))
            ds_cfg       = ds_params[dataset_name]

            filtered = copy.deepcopy(cfg)
            filtered['dataset_params'][data_mode] = {dataset_name: ds_cfg}
            # subject_to_use=["all"] needs the same resolution train_finetune.py's
            # build_subject_split_datasets does — build_dataset_from_config takes it literally
            # and fails ("Subject all not found") since EEGDataset expects real subject ids.
            from train_finetune import _resolve_all_subjects, _resolve_requested_subjects
            all_subjects = _resolve_all_subjects(ds_cfg['dataset_path'])
            filtered['dataset_params'][data_mode][dataset_name]['subject_to_use'] = \
                _resolve_requested_subjects(ds_cfg, all_subjects)
            ds = build_dataset_from_config(filtered, mode=data_mode)

            patch_len = filtered.get('preprocess_params', {}).get('patch_length', 100)
            preds, labels = _predict_all(mdl, ds, patch_len, device)
            num_classes  = int(labels.max()) + 1
            target_names = _load_target_names(ds_cfg['dataset_path'], num_classes)

            out = resolve_output_dir(filtered, 'analysis', dataset_name, mode=mode)
            for cls_idx in range(num_classes):
                name = target_names[cls_idx]
                safe = _safe_name(name)
                cls_mask = labels == cls_idx
                correct_idxs = (cls_mask & (preds == cls_idx)).nonzero()[0]
                wrong_idxs   = (cls_mask & (preds != cls_idx)).nonzero()[0]
                for status, idxs in (('correct', correct_idxs), ('wrong', wrong_idxs)):
                    if len(idxs) == 0:
                        print(f"[check] target{cls_idx}_{safe}: no {status} example found in {dataset_name}, skipping")
                        continue
                    t_idx = int(idxs[0])
                    subject_id = int(ds.base_dataset.subject_data[t_idx].item())
                    tag = f'_target{cls_idx}_{safe}_{status}'
                    metrics = run(
                        filtered, out, mdl, ds, t_idx, mode=mode, subject_id=subject_id, cmap=cmap,
                        plot_recon=check_cfg.get('plot_recon', True),
                        plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                        plot_attn_topo=check_cfg.get('plot_attn_topo', True),
                        tag=tag,
                    )
                    metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[check] done: target={cls_idx}({name}) status={status} subject={subject_id} "
                          f"trial_idx={t_idx}  |  {metrics_str}")

        elif cfg.get('training_params', {}).get('visualize_params', {}).get(data_mode, {}).get('targets'):
```
with:
```python
    if analysis in ('snapshot', 'both'):
        if cfg.get('training_params', {}).get('visualize_params', {}).get(data_mode, {}).get('targets'):
```

- [ ] **Step 9: Fix the two remaining `run(...)` call sites to match the new signature**

Both remaining `run(...)` calls in `analysis_pretrain.py` (in the `targets` branch and the
final `else` branch) already pass `mode=mode`. Remove that keyword from both — e.g.:
```python
                metrics = run(
                    cfg, out, mdl, ds, t_idx, mode=mode, subject_id=subject_id, cmap=cmap,
```
becomes
```python
                metrics = run(
                    cfg, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap,
```
(same edit in the other call site, a few lines below in the final `else` branch). Every
other `resolve_output_dir(..., mode=mode)` / `select_subject_dataset(..., mode=data_mode)`
call in the file keeps `mode=mode`/`mode=data_mode` as-is — those functions still take a
`mode` parameter, only `run()`'s own signature dropped it.

- [ ] **Step 10: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import analysis_pretrain; print('ok')"
```
Expected: `ok`.

- [ ] **Step 11: Smoke — rerun the pretrain baseline through the new script**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
find output/mesae_v10_all_share/analysis -type f | sort > /tmp/analysis_pretrain_after.txt
diff /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/pretrain_baseline_files.txt /tmp/analysis_pretrain_after.txt
```
Expected: `diff` prints nothing (identical file lists). Also compare the printed
`recon_mse=...` value in this run's stdout against the baseline log — must match to the
printed decimals.

- [ ] **Step 12: `git diff --stat` sanity check**

```bash
git diff --stat check_model.py analysis_pretrain.py 2>/dev/null; git diff --stat --cached
```
Expected: a rename shown, plus a moderate-size diff (helper functions removed, a few lines
changed) — not a full rewrite.

- [ ] **Step 13: Commit**

```bash
git add analysis_pretrain.py
git rm --cached check_model.py 2>/dev/null  # no-op if git mv already staged the rename
git commit -m "refactor: split check_model.py into analysis_pretrain.py, drop finetune branch

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 3: New `analysis_finetune.py` (verbatim move)

**Files:**
- Create: `analysis_finetune.py`

**Interfaces:**
- Consumes: `analysis.load_config`, `analysis.resolve_output_dir`,
  `analysis.select_subject_dataset` (unused by the finetune branch, but importable),
  `analysis.filter_config_to_subject` (same), `analysis.load_model`, `analysis._deep_merge`
  — from Task 1.
- Produces: `analysis_finetune.run(config, output_dir, model, dataset, trial_idx, subject_id=None, epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, tag='')`.

- [ ] **Step 1: Write `analysis_finetune.py`**

```python
"""
Post-training checker for the FINETUNE stage only: per-target correct/wrong snapshot
pairs via BaseEpochChecker.check_finetune (model/base_checker.py). Pretrain-stage
analysis (snapshot + codebook) lives in analysis_pretrain.py.

Config resolution: --config is a small overlay — checkpoint (a finetune head.pth),
dataset_params.finetune (one dataset entry), check.plot_* toggles. It's deep-merged onto
the head checkpoint's own run config, taken via --base-config (finetune runs write a
timestamped artifacts/config_<timestamp>.json, not a fixed name, so this cannot be
auto-derived from the checkpoint path the way analysis_pretrain.py's --base-config default
can — always pass --base-config explicitly). Output goes to
output/<model_name>/analysis/<dataset_name>/recon/.
"""

import os
import json

from model.factory import MODEL_REGISTRY


def _load_target_names(dataset_path, num_classes):
    """data_metadata.targets["<idx>"].label, e.g. BNCI2014001's {"0": {"label": "Left hand"}, ...}
    — falls back to "class<idx>" for any index missing from metadata (or if metadata has no
    targets section at all, e.g. a dataset that hasn't been annotated with class names)."""
    try:
        with open(os.path.join(dataset_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        targets = meta.get('data_metadata', {}).get('targets', {})
    except Exception:
        targets = {}
    return [targets.get(str(i), {}).get('label', f'class{i}') for i in range(num_classes)]


def _safe_name(s):
    """Filename-safe version of a target label, e.g. 'Left hand' -> 'Left_hand'."""
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_') or 'unnamed'


def _predict_all(model, dataset, patch_len, device):
    """Runs the finetune model over every trial in `dataset` (in index order, no shuffle)
    and returns (preds, labels) numpy arrays aligned to dataset indices — used to find one
    correctly- and one incorrectly-classified trial per target class.

    One trial at a time, no DataLoader/collate: `FinetuneCollate`, which used to batch and
    pad variable-length trials for this loop, no longer exists in train_finetune.py — its
    whole batching pipeline moved to a cached-feature/`source` model (ADR 0016) that has no
    equivalent for raw per-trial patches. `FinetuneModel.forward` itself is unchanged
    (model/MeSAE/MeSAE.py's docstring: "matches the old finetune classes"), and
    model/base_checker.py's own `check_finetune` already patchifies one trial the same way
    (`_patchify`, same reshape) with no collate/padding needed — this mirrors that exact
    pattern instead of reimplementing the vanished collate function."""
    import torch
    import numpy as np

    was_training = model.training
    model.eval()
    preds, labels = [], []
    try:
        with torch.no_grad():
            for i in range(len(dataset)):
                x_raw, coords, label, valid_channels, _valid_length = dataset[i]
                C, T = x_raw.shape
                P = T // patch_len
                x_patches = x_raw[:, :P * patch_len].reshape(C, P, patch_len).unsqueeze(0).to(device)
                time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0).to(device)
                c_in = coords.unsqueeze(0).to(device)
                vc_in = valid_channels.unsqueeze(0).to(device)
                logits = model(x_patches, c_in, time_idx=time_idx, valid_channels=vc_in)[0]
                preds.append(int(logits.argmax(dim=-1).item()))
                labels.append(int(label))
    finally:
        model.train(was_training)
    return np.array(preds), np.array(labels)


def run(config, output_dir, model, dataset, trial_idx, subject_id=None,
        epoch=None, cmap='YlOrRd', plot_recon=True, plot_topo_psd=True, tag=''):
    model_type = 'MeSAE'  # only registered model (MeFSQ removed, docs/adr/0013)
    plugin  = MODEL_REGISTRY[model_type]
    checker = plugin.checker_cls()
    trainer = plugin.trainer_cls()
    return checker.check_finetune(
        config, output_dir, model, dataset, trial_idx,
        subject_id=subject_id, epoch=epoch, cmap=cmap,
        plot_recon=plot_recon, plot_topo_psd=plot_topo_psd,
        trainer=trainer, tag=tag,
    )


if __name__ == '__main__':
    import argparse
    import copy
    import torch
    from IO.dataset import build_dataset_from_config
    from analysis import _deep_merge, load_model, resolve_output_dir

    parser = argparse.ArgumentParser(description='Post-training EEG finetune checker (MeSAE)')
    parser.add_argument('--config',      required=True)
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        overlay = json.load(f)

    checkpoint = args.checkpoint or overlay.get('checkpoint', '')
    mode       = 'finetune'
    data_mode  = 'finetune'

    base_path = args.base_config or overlay.pop('base_config', None)
    if not base_path:
        raise ValueError(
            "analysis_finetune.py needs --base-config (or overlay['base_config']): a "
            "finetune run's artifacts/config_<timestamp>.json has no fixed name to guess.")
    with open(base_path, 'r') as f:
        base = json.load(f)
    cfg = _deep_merge(base, overlay)
    for m, dsp in overlay.get('dataset_params', {}).items():
        cfg['dataset_params'][m] = dsp

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl    = load_model(cfg, checkpoint, device, mode=mode)

    check_cfg = cfg.get('check', {})
    cmap = args.recon_cmap or check_cfg.get('cmap', 'YlOrRd')
    ds_params = cfg['dataset_params'][data_mode]

    # Per-target correct/wrong snapshot pairs, not a single per-subject trial pick:
    # for each of the num_classes targets, find one trial the model got right and one
    # it got wrong, and render the full 3-panel snapshot (recon_signal, topo_psd_filter,
    # attn_topo) for each — num_classes * 2 * 3 files total, searched across every
    # subject in dataset_params.finetune[dataset_name].subject_to_use (not one subject
    # at a time), since a single subject isn't guaranteed to contain both a correct and
    # a wrong example of every class.
    dataset_name = args.dataset or next(iter(ds_params))
    ds_cfg       = ds_params[dataset_name]

    filtered = copy.deepcopy(cfg)
    filtered['dataset_params'][data_mode] = {dataset_name: ds_cfg}
    # subject_to_use=["all"] needs the same resolution train_finetune.py's own dataset
    # builder does (same call shape as its own subject_to_use resolution, see
    # train_finetune.py's build_dataset_from_config caller) — build_dataset_from_config
    # takes it literally and fails ("Subject all not found") since EEGDataset expects
    # real subject ids.
    from train_finetune import _resolve_all_subjects, resolve_subjects
    all_subjects = _resolve_all_subjects(ds_cfg['dataset_path'])
    filtered['dataset_params'][data_mode][dataset_name]['subject_to_use'] = \
        resolve_subjects(ds_cfg['subject_to_use'], all_subjects)
    ds = build_dataset_from_config(filtered, mode=data_mode)

    patch_len = filtered.get('preprocess_params', {}).get('patch_length', 100)
    preds, labels = _predict_all(mdl, ds, patch_len, device)
    num_classes  = int(labels.max()) + 1
    target_names = _load_target_names(ds_cfg['dataset_path'], num_classes)

    out = resolve_output_dir(filtered, 'analysis', dataset_name, mode=mode)
    for cls_idx in range(num_classes):
        name = target_names[cls_idx]
        safe = _safe_name(name)
        cls_mask = labels == cls_idx
        correct_idxs = (cls_mask & (preds == cls_idx)).nonzero()[0]
        wrong_idxs   = (cls_mask & (preds != cls_idx)).nonzero()[0]
        for status, idxs in (('correct', correct_idxs), ('wrong', wrong_idxs)):
            if len(idxs) == 0:
                print(f"[check] target{cls_idx}_{safe}: no {status} example found in {dataset_name}, skipping")
                continue
            t_idx = int(idxs[0])
            subject_id = int(ds.base_dataset.subject_data[t_idx].item())
            tag = f'_target{cls_idx}_{safe}_{status}'
            metrics = run(
                filtered, out, mdl, ds, t_idx, subject_id=subject_id, cmap=cmap,
                plot_recon=check_cfg.get('plot_recon', True),
                plot_topo_psd=check_cfg.get('plot_topo_psd', True),
                tag=tag,
            )
            metrics_str = '  '.join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[check] done: target={cls_idx}({name}) status={status} subject={subject_id} "
                  f"trial_idx={t_idx}  |  {metrics_str}")
```

- [ ] **Step 2: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import analysis_finetune; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Smoke — run the finetune path through the new script**

**No pre-existing finetune baseline exists to diff against.** Task 2 discovered (and
verified: `train_finetune.py` now defines `resolve_subjects(entry, pool)`, not
`_resolve_requested_subjects`) that `check_model.py --mode finetune` was already broken on
this branch before this refactor started — an earlier, unrelated `train_finetune.py`
rename left it uncallable, so no working finetune baseline was ever captured (no
`finetune_baseline_files.txt` exists in the scratchpad; `finetune_smoke_overlay.json`
does, from before the baseline attempt failed, and is still reusable). This step is
therefore the first successful run of this code path on this branch, not a
regression-identity check — verify it works, not that it matches history:

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py \
  --config /tmp/claude-1000/-media-mamechin-PortableSSD-iansaididontcare-CNElab-cnelab-model-trainer-EEG-Tokenizer/fa469676-ae07-4623-a89e-52c7daef5798/scratchpad/finetune_smoke_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001 ; echo "exit: $?"
find output/baseline/BNCI2014001_intra_raw_band/analysis -type f | sort > /tmp/analysis_finetune_after.txt
cat /tmp/analysis_finetune_after.txt
```
Expected: exit 0; the script prints one `[check] done: target=...` line per correct/wrong
example found (2 classes for BNCI2014001, so up to 4 lines — fewer if an example is
missing for some class/status, which the script itself reports and continues past, not a
failure); `/tmp/analysis_finetune_after.txt` lists PNG files under
`output/baseline/BNCI2014001_intra_raw_band/analysis/BNCI2014001/recon/` named per the
`_target{cls}_{name}_{status}` tag pattern in the code. If it does NOT exit 0, read the
traceback: if it's the same `resolve_subjects`/`_resolve_requested_subjects` mismatch,
your Step 1 file has the bug (compare against this plan's current text, not memory or
check_model.py's old source); any other failure, report BLOCKED with the traceback.

- [ ] **Step 4: `git status` sanity check**

```bash
git status --short analysis_finetune.py
git diff --stat check_model.py -- analysis_finetune.py 2>/dev/null || true
```
Confirm this is a genuinely new, untracked file (not a rename — `check_model.py` is already
gone as of Task 2), and its size is in the same ballpark as the removed finetune code
(roughly 90-110 lines).

- [ ] **Step 5: Commit**

```bash
git add analysis_finetune.py
git commit -m "feat: add analysis_finetune.py, moved verbatim from check_model.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 4: Doc and config fixups

**Files:**
- Modify: `CLAUDE.md`
- Modify: `config/analysis.json`

**Interfaces:** none (documentation/config only, no code).

- [ ] **Step 1: Update CLAUDE.md's Commands section**

Replace:
```
# Post-training checker (checkpoint -> topo/PSD/attn snapshot per subject;
# base config auto-derived from the checkpoint's output/<model>/artifacts/config.json,
# config/analysis.json is a small overlay; viz/extract.py, panels.py, timeseries.py,
# topomap.py are shared primitives it and model/base_checker.py both call — not run directly)
python check_model.py --config config/analysis.json --checkpoint <path>
```
with:
```
# Post-training checker, PRETRAIN stage (checkpoint -> topo/PSD/attn snapshot per subject,
# plus cross-dataset codebook/vocab diagnostics; base config auto-derived from the
# checkpoint's output/<model>/artifacts/config.json, config/analysis.json is a small
# overlay; viz/extract.py, panels.py, timeseries.py, topomap.py are shared primitives it
# and model/base_checker.py both call — not run directly)
python analysis_pretrain.py --config config/analysis.json --checkpoint <path>

# Post-training checker, FINETUNE stage (checkpoint -> per-class correct/wrong snapshot
# pairs; --base-config is required, a finetune run's artifacts/config_<timestamp>.json
# has no fixed name to auto-derive)
python analysis_finetune.py --config <overlay.json> --base-config <path/to/artifacts/config.json> --checkpoint <head.pth>
```

- [ ] **Step 2: Update the `fft_resolution` mention**

Replace:
```
`fft_resolution` (Hz/bin for check_model.py's diagnostic PSD panels —
```
with:
```
`fft_resolution` (Hz/bin for analysis_pretrain.py's/analysis_finetune.py's diagnostic PSD panels —
```

- [ ] **Step 3: Drop the redundant `mode` key from `config/analysis.json`**

Replace:
```json
{
  "checkpoint": "output/pretrain/mesae_v10_all_share/checkpoint/last.pth",
  "mode": "pretrain",
  "dataset_params": {
```
with:
```json
{
  "checkpoint": "output/pretrain/mesae_v10_all_share/checkpoint/last.pth",
  "dataset_params": {
```

- [ ] **Step 4: Validate the JSON and do a final end-to-end smoke**

```bash
python3 -c "import json; json.load(open('config/analysis.json')); print('valid json')"
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
```
Expected: `valid json`, then the same `[check] done: ...` output as Task 2 Step 11.

- [ ] **Step 5: Final repo-wide check**

```bash
git grep -n "check_model\.py" -- '*.py' '*.md' '*.json' | grep -v docs/adr | grep -v docs/superpowers/plans
```
Expected: no output (every live reference updated; historical ADR/plan docs are
intentionally left alone per the spec's "Out of scope" section).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md config/analysis.json
git commit -m "docs: update CLAUDE.md and config/analysis.json for the analysis_pretrain/analysis_finetune split

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```
