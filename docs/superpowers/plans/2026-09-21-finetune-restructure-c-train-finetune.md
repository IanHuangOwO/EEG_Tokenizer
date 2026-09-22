# Finetune restructure, sub-project C: train_finetune.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slim `train_finetune.py` (795 lines) to one head-only training script with two split modes (`intra_subject`, `inter_subject`), training the `FeatureHead` directly on the stamp-amplitude cache (or the patched raw signal for `raw_*` features), one output format (`group_eval.json`) and an environment stamp in the saved config.

**Architecture:** The frozen backbone is no longer part of training. A *source* object serves batches: `StampSource` (the cache from `cache_feature.py`) or `RawSource` (the compiled dataset, real channels only). `make_runs` turns the `split` config block into runs (train indices, evaluation subjects), the training loop optimises only `FeatureHead`, and one pass over the evaluation trials per epoch yields the per-subject balanced accuracy for the tail-mean. Task 1 adds the data side next to the old code (additive, so the old runners stay available for comparison); Task 2 replaces the file; Task 3 does docs and sweeps.

**Tech Stack:** PyTorch (`eeg_fm` env), scikit-learn (`StratifiedKFold`, metrics), `cache_feature.py` (sub-project B), `model/MeSAE/MeSAE_modules.py` (`FeatureHead`, `StampExtractor`, `resolve_head_config`, sub-project A).

**Spec:** `docs/superpowers/specs/2026-09-21-finetune-restructure-design.md`, sub-project C (read it first). Refinements made here (written back in Task 3): no `DataLoader` and no autocast/GradScaler (the head is tiny and all inputs are already in RAM, so batches are indexed directly and the head trains in fp32); one evaluation pass per epoch serves both the validation metrics and the per-subject tail history (no per-subject loaders); the model trained is the bare `FeatureHead`, not `FinetuneModel`; one dataset per run.

## Global Constraints

- **Python env:** `/home/mamechin/anaconda3/envs/eeg_fm/bin/python` for every command (never `base`). The GPU is free (smoke runs in Task 2 use it); Task 1 checks are CPU.
- **Line endings:** `train_finetune.py` is LF and may be rewritten whole in Task 2. `model/MeSAE/MeSAE.py` and `config/config.json` are CRLF: edit in place with a byte-level script (`grep -c $'\r' <file>` must equal `wc -l` before and after; small `git diff --stat`). `MeSAE_modules.py`, `cache_feature.py`, docs and scripts are LF.
- **No old-run compatibility** (user decision). `config/phase2/*.json` and every old finetune config stop working (unknown `split_mode`, removed keys); sub-project D deletes `config/phase2/`. The old code stays available at git tag `pre-head-cleanup` (before sub-project A) and, for the runners this sub-project replaces, at the commit before Task 2 (record its hash in the Task 2 report).
- **Do not touch** pretraining code, `IO/`, `viz/`, `cache_feature.py` (except through its public API), `MeSAE_modules.py` beyond Task 2 Step 2.
- **Layering:** `train_finetune.py` builds datasets and runs training (glue, as today); it imports `MeSAE_modules` directly (MeSAE is the only model; the plugin registry is still used for the plotter).
- No test suite exists (CLAUDE.md): validation is the check scripts and smoke runs below, not pytest files.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
  Do not push (the controller pushes). Never commit anything under `output/` or `.superpowers/`.

## Config contract (what the script reads)

Mode names (user decision): `intra_subject` = each subject's own trials are split (the old `intra_subject_cv`); `inter_subject` = subjects are split into training and evaluation groups (k-fold over subjects, LOSO, or explicit lists).

`training_params.finetune`: `model_name` (output dir `output/<model_name>`), `model_type` (default `MeSAE`), `pretrained_checkpoint`, `learning_rate`, `min_learning_rate`, `weight_decay`, `epochs`, `warmup_epochs`, `batch_size`, `device` (default cuda if available), `seed` (default 42), and the `split` block:

- `"split": {"mode": "intra_subject", "n_folds": k, "seed": 42}`: for every subject of the pool, stratified k-fold over that subject's own trials (`sklearn.StratifiedKFold(shuffle=True, random_state=seed)`, the same partition as the old `intra_subject_cv`). Run names `<subject>_fold<i>`; train subjects `[subject]`; one evaluation group `heldout` holding that subject.
- `"split": {"mode": "inter_subject", ...}` with **exactly one** of
  - `n_folds: k` (2 <= k <= number of pool subjects): the pool is shuffled with `random.Random(seed)` (`seed` default 42) and dealt into k groups (`subs[i::k]`, the same partition as the old `subject_kfold`); run `fold<i>` evaluates group `i` (group name `heldout`) and trains on the rest of the pool (or on `train_subjects` minus the fold if given). `k` equal to the number of subjects is LOSO;
  - `eval_subjects`: a list (group `heldout`) or a dict of named lists (for example `{"seen": [...], "unseen": [...]}`): one run named `main`; trains on `train_subjects` if given, else on the pool minus all evaluation subjects.
  - optional `train_subjects` (any entry form below); training and evaluation subjects must be disjoint; empty groups or an empty training set raise `ValueError`.
- A subject entry (`subject_to_use`, `train_subjects`, `eval_subjects` groups) is an explicit list, `"all"` / `["all"]`, or `{"random": n, "seed": s}`; every listed subject must exist in the pool (a missing one raises, unlike the old silent drop). `train_subjects` and `eval_subjects` resolve against the pool (`dataset_params.finetune.<dataset>.subject_to_use`).
- Removed keys: `split_mode`, `cv_folds`, `train_val_split`, `subject_kfold`, `subject_kfold_seed`, `subject_group_runs`, `backbone_lr_mult`, and `model_params.MeSAE.finetune.freeze_backbone`. Exactly one dataset per run (`dataset_params.finetune` with one entry).

Output (`output/<model_name>/`): `artifacts/config.json` and `artifacts/config_<timestamp>.json` (the config plus an `env` block: `git_commit`, `git_dirty`, `python`, `torch`, `cuda`, `mne`), `artifacts/train_<timestamp>.log`, `artifacts/group_eval.json`, `finetune/run_<name>/head.pth` (the head checkpoint: `{"model_state_dict", "head_config", "backbone_checkpoint"}`, loadable by `viz.load_model` / `model.factory.load_finetune_checkpoint`), `visualization/run_<name>/training_dashboard.png`.

`group_eval.json` schema (unchanged from ADR 0014's Protocol):
```json
{"<run_name>": {"train_subjects": ["3", "7"], "epochs": 50, "tail_epochs": 10,
  "groups": {"<group>": {"subjects": {"<subject_id>": {"tail": 0.61, "last": 0.60, "n_trials": 88}},
                         "n_subjects": 1, "mean_tail": 0.61, "mean_last": 0.60}}}}
```
`tail` = mean over the last `min(10, epochs)` epochs of that subject's balanced accuracy on the model as it stood; `last` = the final epoch.

---

## Task 1: data side, added next to the old code (additive)

**Files:**
- Modify: `train_finetune.py` (add imports at the top and new functions; **no old function is changed or removed in this task**)
- Create (not committed): `.superpowers/sdd/2026-09-21-finetune-restructure-c/split_equiv.py`

**Interfaces (produced, used by Task 2):**
- `resolve_subjects(entry, pool) -> list`: `entry` per the contract; result elements are pool elements (so int when the pool is int).
- `make_runs(split, pool, subject_data, labels) -> list[dict]`: `subject_data`, `labels` numpy arrays over the source's trials; each run `{"name": str, "train": np.ndarray (sorted trial indices), "train_subjects": [str], "eval": {group: {subject_str: np.ndarray (sorted trial indices)}}}`.
- `StampSource(config, ds_name, pool, device)`, `RawSource(config, ds_name, pool)`, `make_source(config, ds_name, pool, device)`: attributes `kind` (`"stamp"`/`"raw"`), `labels` (`LongTensor`), `subject_data` (`LongTensor`), `channel_idx` (list), `keep` (list or `None`), `num_patches`, `num_stamps` (0 for raw); method `get(idx: LongTensor) -> (inp, labels)` where `inp` is fp32 `[B, N', C_valid, S, 2]` (stamp) or `[B, C_valid, N', L]` (raw).
- `iter_batches(source, idx, batch_size, device, shuffle, gen=None)`: generator of `(inp, labels)` on `device`; when `shuffle` a batch of one trial is skipped (BatchNorm needs >= 2).

- [ ] **Step 1: Read first.** Read `train_finetune.py` fully (the splitters `_intra_subject_cv_splits`, `_subject_group_runs`, `_loso_folds`, `_resolve_all_subjects`, `_resolve_requested_subjects`, `run_training_loop`, `main`), `cache_feature.py` (`get_stamp_cache`, `CachedStampDataset`), and `IO/dataset.py` (`build_dataset_from_config`, `EEGDataset.all_valid_channels/subject_data/labels`). If a name or signature below differs from the real code, the real code wins; say so in the report.

- [ ] **Step 2: Add imports** (keep the existing ones): `from sklearn.model_selection import StratifiedKFold`, `import torch.nn.functional as F`, `from cache_feature import CachedStampDataset, get_stamp_cache`, `from model.MeSAE.MeSAE_modules import FeatureHead, StampExtractor, resolve_head_config`, `from model.factory import load_backbone` (extend the existing `from model.factory import ...`).

- [ ] **Step 3: Append these functions** to `train_finetune.py` (above `main`; they use `_resolve_all_subjects` which already exists):

```python
def resolve_subjects(entry, pool):
    """Subject entry -> list of pool subjects: 'all' | ['all'] | explicit list | {'random': n, 'seed': s}."""
    if entry in ('all', ['all']):
        return list(pool)
    if isinstance(entry, dict):
        return sorted(random.Random(entry.get('seed', 42)).sample(list(pool), entry['random']))
    by_str = {str(s): s for s in pool}
    missing = [s for s in entry if str(s) not in by_str]
    if missing:
        raise ValueError(f"subjects {missing} are not in the pool {sorted(by_str, key=str)}")
    return [by_str[str(s)] for s in entry]


def _trials_of(subject_data, subjects):
    return np.flatnonzero(np.isin(subject_data, [int(s) for s in subjects]))


def make_runs(split, pool, subject_data, labels):
    """split block -> runs [{name, train, train_subjects, eval}] (see the plan's contract)."""
    mode, seed = split.get('mode'), split.get('seed', 42)
    if mode == 'intra_subject':
        k = int(split['n_folds'])
        runs = []
        for s in pool:
            idx = np.flatnonzero(subject_data == int(s))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for i, (tr, va) in enumerate(skf.split(idx, labels[idx])):
                runs.append(dict(name=f'{s}_fold{i}', train=np.sort(idx[tr]), train_subjects=[str(s)],
                                 eval={'heldout': {str(s): np.sort(idx[va])}}))
        return runs
    if mode != 'inter_subject':
        raise ValueError(f"split.mode must be 'intra_subject' or 'inter_subject', got {mode!r}")
    if ('n_folds' in split) == ('eval_subjects' in split):
        raise ValueError("inter_subject needs exactly one of n_folds / eval_subjects")
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    if 'n_folds' in split:
        k = int(split['n_folds'])
        if not 2 <= k <= len(pool):
            raise ValueError(f"n_folds must be in [2, {len(pool)}], got {k}")
        subs = list(pool)
        random.Random(seed).shuffle(subs)
        evals = [(f'fold{i}', {'heldout': subs[i::k]}) for i in range(k)]
    else:
        ev = split['eval_subjects']
        ev = {'heldout': ev} if not isinstance(ev, dict) or 'random' in ev else ev
        evals = [('main', {g: resolve_subjects(v, pool) for g, v in ev.items()})]
    runs = []
    for name, groups in evals:
        ev_subs = {s for v in groups.values() for s in v}
        train_subs = [s for s in (train_pool if train_pool is not None else pool) if s not in ev_subs]
        if not train_subs or any(not v for v in groups.values()):
            raise ValueError(f"run {name}: empty training set or evaluation group")
        if 'eval_subjects' in split and train_pool is not None and set(train_pool) & ev_subs:   # n_folds: the fold is subtracted instead
            raise ValueError(f"run {name}: train_subjects and evaluation subjects overlap: {sorted(set(train_pool) & ev_subs)}")
        runs.append(dict(name=name, train=_trials_of(subject_data, train_subs), train_subjects=[str(s) for s in train_subs],
                         eval={g: {str(s): _trials_of(subject_data, [s]) for s in v} for g, v in groups.items()}))
    return runs


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

- [ ] **Step 4: Write the check script** `.superpowers/sdd/2026-09-21-finetune-restructure-c/split_equiv.py` (uncommitted; CPU; `PYTHONPATH=.`). It must:
  1. Import the old file from the tag `pre-head-cleanup` as a temporary module (`git show pre-head-cleanup:train_finetune.py > _old_train_finetune_tmp.py` at the repo root; delete it in a `finally`; never commit it) to get `_intra_subject_cv_splits`, `_subject_group_runs`, `_loso_folds`, `_resolve_all_subjects`.
  2. Build the BNCI2014001 pool dataset through `RawSource` (all 9 subjects) and compare **intra_subject**: for `n_folds=5, seed=42` and every subject, the new `make_runs` train/eval index arrays equal `sorted(...)` from the old `_intra_subject_cv_splits(full_dataset, subject, n_folds=5)` `Subset.indices` (build `full_dataset` with `build_dataset_from_config(config, mode='finetune')` and compare positions; both sides enumerate subjects in the same order, so global trial indices agree). Assert equal for all 9 x 5 folds and that run names are `<s>_fold<i>` in the same order.
  3. Compare **inter_subject n_folds** with the old `_subject_group_runs({'subject_kfold': k, 'subject_kfold_seed': 42}, dataset_params)` for `k in (3, 9)`: for every fold the sets of held-out subjects and train subjects are identical (old ids may be int or str: compare via `str`). For `k == 9` also check each subject is held out exactly once and the train set is the other eight (LOSO), matching `_loso_folds`.
  4. Check **eval_subjects**: `{"seen": ["1"], "unseen": ["8", "9"]}` with `train_subjects: ["2", "3", "4"]` gives one run `main` whose train trials are exactly the trials of subjects 2, 3, 4 and whose groups map to the right subjects; a plain list gives group `heldout`.
  5. Check **errors**: each of these raises `ValueError`: unknown mode; both `n_folds` and `eval_subjects`; neither; `n_folds` 1 and `n_folds` 10 on 9 subjects; overlapping `train_subjects` and `eval_subjects` (but with `n_folds` and `train_subjects: ["1","2","3","4"]` the fold's subjects are simply removed from the training list: check that instead); an eval subject not in the pool; an empty eval group. Check `resolve_subjects` for `"all"`, `["all"]`, an explicit list of strings against an int pool, `{"random": 3, "seed": 1}` (deterministic, sorted, 3 distinct pool members).
  6. Check the sources: `RawSource` on BNCI2014001 subjects `['8', '9']` gives `get(idx)` shapes `[B, 22, 39, 50]`, `channel_idx` of length 22, `num_patches == 39`; `StampSource` (cache exists from sub-project B) gives `[B, 39, 22, 25, 2]` fp32, `keep` of length 25, and for the same trial the head applied to the stamp input is finite; `iter_batches` with `shuffle=True` and a seeded `torch.Generator` yields every index exactly once (except a trailing single trial) and is reproducible; without `shuffle` it yields indices in order.
  Print one `OK <check>` line per check.

- [ ] **Step 5: Run it.** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python .superpowers/sdd/2026-09-21-finetune-restructure-c/split_equiv.py`. Every check must print `OK`. A mismatch with the old splitters is a bug in `make_runs` (fix the new code, not the check). Also confirm `python -c "import train_finetune"` still works (the old runners are untouched) and `git status --short` shows only `M train_finetune.py`.

- [ ] **Step 6: Commit** `feat: subject resolver, split runs and stamp/raw sources for the finetune restructure` (with the two attribution lines).

---

## Task 2: replace the training script, config and head checkpoint helper

**Files:**
- Modify: `train_finetune.py` (rewrite whole, LF)
- Modify: `model/MeSAE/MeSAE_modules.py` (add `make_head_checkpoint`; drop the legacy-key tuple)
- Modify: `model/MeSAE/MeSAE.py` (CRLF; `FinetuneModel.head_checkpoint` calls the shared helper)
- Modify: `config/config.json` (CRLF)
- Create (not committed): `.superpowers/sdd/2026-09-21-finetune-restructure-c/smoke.py` (optional helper) and smoke configs in the scratchpad

**Interfaces:**
- Consumes: everything from Task 1 and sub-projects A and B.
- Produces:
  - `make_head_checkpoint(head, head_cfg, channel_idx, keep, backbone_checkpoint) -> dict` in `MeSAE_modules.py` (the format in the contract); `FinetuneModel.head_checkpoint(bb_path)` returns `make_head_checkpoint(self.head, self.head_cfg, self.channel_idx.tolist(), self.extractor.keep.tolist() if self.extractor is not None else None, bb_path)`.
  - The new `train_finetune.py` CLI: `python train_finetune.py --config <config.json>`.

- [ ] **Step 1: Record the pre-rewrite commit** (`git rev-parse HEAD`) in the Task 2 report; the old file lives there.

- [ ] **Step 2: `make_head_checkpoint`.** In `MeSAE_modules.py` (LF, finetune section) add
```python
def make_head_checkpoint(head, head_cfg, channel_idx, keep, backbone_checkpoint):
    """Head-only checkpoint: state, resolved config (plus the real channels and alive stamps it was
    built for) and the frozen backbone it belongs to. Loaded by FinetuneModel.from_checkpoint."""
    return {'model_state_dict': head.state_dict(),
            'head_config': dict(head_cfg, channel_idx=list(channel_idx), keep=None if keep is None else list(keep)),
            'backbone_checkpoint': backbone_checkpoint}
```
and delete `_LEGACY_TRAINING_KEYS` and its use in `resolve_head_config` (the config no longer carries `freeze_backbone` / `backbone_lr_mult`; unknown keys now raise for them too). In `MeSAE.py` (CRLF, in place) replace the body of `FinetuneModel.head_checkpoint` by the call above (import `make_head_checkpoint` from `.MeSAE_modules`). Re-run the sub-project A equivalence script (`.superpowers/sdd/2026-09-21-finetune-model-restructure-a/head_equiv.py`, CPU): the round-trip cases must still print `OK` (its resolver checks include `freeze_backbone=True`: update that one assertion, which now must raise `ValueError`, and say so in the report).

- [ ] **Step 3: Rewrite `train_finetune.py`** as follows (keep `setup_logger` verbatim from the old file; keep `_resolve_all_subjects` verbatim; everything else in the old file goes, except the Task 1 functions which stay as written). Structure and the new code:

```python
"""Finetune a FeatureHead on a frozen MeSAE backbone (ADR 0016; finetune restructure, sub-project C).

The backbone never trains: stamp features come from the amplitude cache (cache_feature.py), raw
features from the compiled dataset; only the head is optimised (fp32, batches indexed from RAM).
training_params.finetune.split has two modes, intra_subject and inter_subject (see the plan /
CLAUDE.md). Every run writes artifacts/group_eval.json: per-subject tail (mean of the last 10
epochs) and last-epoch balanced accuracy."""
import argparse, copy, json, logging, os, random, shutil, subprocess, statistics, sys, warnings
from datetime import datetime      # only if setup_logger needs it; keep whatever setup_logger uses

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

from IO.dataset import build_dataset_from_config
from IO.preprocessing import num_patches, slice_patches
from cache_feature import CachedStampDataset, get_stamp_cache
from model.factory import MODEL_REGISTRY, load_backbone
from model.MeSAE.MeSAE_modules import (FeatureHead, StampExtractor, make_head_checkpoint,
                                       resolve_head_config)

# setup_logger(output_dir) -> (logger, timestamp)            [verbatim from the old file]
# _resolve_all_subjects(data_root)                            [verbatim from the old file]
# resolve_subjects, _trials_of, make_runs, StampSource, RawSource, make_source, iter_batches  [Task 1]


def env_stamp():
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    try:
        import mne
        mne_version = mne.__version__
    except ImportError:
        mne_version = None
    return {'git_commit': sh('git', 'rev-parse', 'HEAD'), 'git_dirty': bool(sh('git', 'status', '--porcelain')),
            'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'mne': mne_version}


def _metrics(labels, preds, loss):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'loss': float(loss), 'acc': float((labels == preds).mean()),
                'f1': f1_score(labels, preds, average='macro', zero_division=0),
                'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
                'balanced_acc': balanced_accuracy_score(labels, preds), 'kappa': cohen_kappa_score(labels, preds)}


@torch.no_grad()
def _predict(head, source, idx, batch_size, device):
    """logits [n, C] (numpy) and mean loss over the trials idx, in the given (sorted) order."""
    head.eval()
    logits, ys = [], []
    for x, y in iter_batches(source, idx, batch_size, device, shuffle=False):
        logits.append(head(x).float().cpu()); ys.append(y.cpu())
    logits, ys = torch.cat(logits), torch.cat(ys)
    return logits.numpy(), F.cross_entropy(logits, ys).item()


def build_head_factory(config, source, num_classes):
    """(resolved head config, function returning a freshly initialised FeatureHead)."""
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    cfg = resolve_head_config(
        config['model_params']['MeSAE']['finetune'], num_classes=num_classes, num_patches=source.num_patches,
        num_channels=len(source.channel_idx), num_stamps=source.num_stamps, patch_len=patch_len,
        patch_stride=pp.get('patch_stride', patch_len), sample_freq=float(pp['sample_freq']))
    tables = None
    if cfg['feature'] == 'stamp_band':   # the template spectra need the backbone, once
        tables = StampExtractor(load_backbone(config), [0]).band_tables(cfg['sample_freq'])

    def new_head():
        head = FeatureHead(cfg)
        if tables is not None:
            head.E_D.copy_(tables[0]); head.E_H.copy_(tables[1])
        return head
    return cfg, new_head


def run_one(config, run, source, head_cfg, new_head, tag, out_dir, logger, device):
    """Train one head on run['train'], evaluate on the union of run['eval'] every epoch."""
    tp = config['training_params']['finetune']
    E, bs, seed, warm = tp['epochs'], tp['batch_size'], tp.get('seed', 42), tp['warmup_epochs']
    torch.manual_seed(seed)
    head = new_head().to(device)
    opt = optim.AdamW(head.parameters(), lr=tp['learning_rate'], weight_decay=tp['weight_decay'])
    sched = optim.lr_scheduler.SequentialLR(
        opt, schedulers=[optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
                         optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, E - warm), eta_min=tp['min_learning_rate'])],
        milestones=[warm])
    ev_idx = np.unique(np.concatenate([i for g in run['eval'].values() for i in g.values()]))
    pos = {(g, s): np.searchsorted(ev_idx, i) for g, subs in run['eval'].items() for s, i in subs.items()}
    y_all = source.labels.numpy()
    y_ev = y_all[ev_idx]
    gen = torch.Generator().manual_seed(seed)
    plotter = MODEL_REGISTRY[tp.get('model_type', 'MeSAE')].plotter_cls(output_dir=out_dir['vis'])
    tail_start, hist = max(0, E - 10), {}
    logger.info(f"[{tag}] train={len(run['train'])} eval={len(ev_idx)} head_params={sum(p.numel() for p in head.parameters())}")
    for epoch in range(1, E + 1):
        head.train()
        losses, preds, ys = [], [], []
        for x, y in iter_batches(source, run['train'], bs, device, shuffle=True, gen=gen):
            opt.zero_grad()
            logits = head(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item()); preds.append(logits.argmax(1).cpu()); ys.append(y.cpu())
        sched.step()
        train_metrics = _metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy(), float(np.mean(losses)))
        logits, val_loss = _predict(head, source, ev_idx, bs, device)
        val_pred = logits.argmax(1)
        val_metrics = _metrics(y_ev, val_pred, val_loss)
        if epoch > tail_start:                       # per-subject balanced accuracy, from the same pass
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for key, p in pos.items():
                    hist.setdefault(key, []).append(balanced_accuracy_score(y_ev[p], val_pred[p]))
        logger.info(f"--- [{tag}] Epoch {epoch}/{E} Summary ---")
        for name, m in (('Train', train_metrics), ('Val  ', val_metrics)):
            logger.info(f"  [{name}] loss: {m['loss']:.4f} | acc: {m['acc']:.4f} | f1: {m['f1']:.4f} | f1_w: {m['f1_weighted']:.4f}"
                        f" | bal_acc: {m['balanced_acc']:.4f} | kappa: {m['kappa']:.4f}")
        logger.info("-" * 40)
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
    plotter.plot_finetune(freeze_backbone=True)       # once, at the end (the 'Backbone Recon MSE' panel stays empty)
    torch.save(make_head_checkpoint(head, head_cfg, source.channel_idx, source.keep, tp['pretrained_checkpoint']),
               os.path.join(out_dir['ckpt'], 'head.pth'))
    out = {}
    for g, subs in run['eval'].items():
        sd = {s: {'tail': float(np.mean(hist[(g, s)])), 'last': float(hist[(g, s)][-1]), 'n_trials': int(len(i))}
              for s, i in subs.items()}
        out[g] = {'subjects': sd, 'n_subjects': len(sd),
                  'mean_tail': float(np.mean([v['tail'] for v in sd.values()])),
                  'mean_last': float(np.mean([v['last'] for v in sd.values()]))}
    return out
```

`main()`:
```python
def main():
    ap = argparse.ArgumentParser(description='Finetune a FeatureHead on a frozen MeSAE backbone')
    ap.add_argument('--config', default='config/config.json')
    args = ap.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    tp = config['training_params']['finetune']
    base = f"output/{tp.get('model_name', 'default_finetune_run')}"
    artifact_dir = os.path.join(base, 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    logger, timestamp = setup_logger(artifact_dir)
    snapshot = dict(config, env=env_stamp())
    for name in ('config.json', f'config_{timestamp}.json'):
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(snapshot, f, indent=2)
    dp = config['dataset_params']['finetune']
    if len(dp) != 1:
        raise ValueError("finetune runs one dataset at a time: dataset_params.finetune must have exactly one entry")
    ds_name, ds_args = next(iter(dp.items()))
    pool = resolve_subjects(ds_args['subject_to_use'], _resolve_all_subjects(ds_args['dataset_path']))
    device = tp.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    source = make_source(config, ds_name, pool, device)
    labels = source.labels.numpy()
    num_classes = int(labels.max()) + 1
    assert set(labels.tolist()) == set(range(num_classes)), f"labels must be contiguous 0..{num_classes - 1}"
    head_cfg, new_head = build_head_factory(config, source, num_classes)
    logger.info(f"dataset={ds_name} pool={len(pool)} subjects, {len(labels)} trials, classes={num_classes}, head={head_cfg}")
    runs = make_runs(tp['split'], pool, source.subject_data.numpy(), labels)
    result = {}
    for run in runs:
        tag = f"{ds_name}_{run['name']}"
        dirs = {'ckpt': os.path.join(base, 'finetune', f"run_{run['name']}"), 'vis': os.path.join(base, 'visualization', f"run_{run['name']}")}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        groups = run_one(config, run, source, head_cfg, new_head, tag, dirs, logger, device)
        for g, d in groups.items():
            logger.info(f"  [{run['name']}] group {g}: n={d['n_subjects']} mean_tail={d['mean_tail']:.4f} mean_last={d['mean_last']:.4f}")
        result[run['name']] = {'train_subjects': run['train_subjects'], 'epochs': tp['epochs'],
                               'tail_epochs': min(10, tp['epochs']), 'groups': groups}
        with open(os.path.join(artifact_dir, 'group_eval.json'), 'w') as f:      # rewritten per run: partial results survive
            json.dump(result, f, indent=2)
    logger.info("Finetuning complete.")


if __name__ == '__main__':
    main()
```
Sanity: the file ends near 450 lines. Delete every other old function (`FinetuneCollate`, `_unpack_batch`, `_classification_metrics`, `_finalize_epoch`, `_recon_mse`, `train_one_epoch`, `validate_one_epoch`, `_underlying_base_dataset`, `_resolve_requested_subjects`, `build_subject_split_datasets`, `_intra_subject*`, `_loso*`, `run_training_loop`, `_run_loso`, `_subject_group_runs`, `_run_subject_groups`) and every import that becomes unused (`tqdm`, `DataLoader`, `Subset`, `viz.pick_trial`, `build_finetune_from_config`).

- [ ] **Step 4: `config/config.json` (CRLF).** In `training_params.finetune` replace `split_mode` and `cv_folds` by `"split": {"mode": "intra_subject", "n_folds": 5}` and delete `backbone_lr_mult` (and `train_val_split` if present); in `model_params.MeSAE.finetune` delete `freeze_backbone`. Keep formatting and CRLF; `git diff --stat` shows a handful of lines. Confirm `json.load` works and no other finetune key was touched.

- [ ] **Step 5: static checks.** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -c "import train_finetune, check_model, viz, model.factory, cache_feature"` clean; `wc -l train_finetune.py` about 450 (report the real number); rerun Task 1's `split_equiv.py` **against the new file** for the checks that do not need the old runners (its old-code comparisons keep working because they import the tag, not the working tree): all `OK`; rerun the sub-project A equivalence script (25 `OK`, with the one changed assertion).

- [ ] **Step 6: smoke runs on the GPU** (scratchpad configs copied from `config/config.json`, `epochs: 3`, `warmup_epochs: 1`, `dataset_params.finetune` restricted to BNCI2014001 subjects, `model_name` `smoke_c_<name>`; delete `output/smoke_c_*` afterwards; the first stamp run builds or reuses the BNCI2014001 cache under `output/pretrain/mesae_v10_small/feature_cache/`):
  1. **intra_subject**, subjects `["8","9"]`, `n_folds: 2`, default head (`stamp_power`, `learned`): four runs (`8_fold0`, `8_fold1`, `9_fold0`, `9_fold1`); `group_eval.json` has them with group `heldout` and one subject each; `finetune/run_8_fold0/head.pth` and `visualization/run_8_fold0/training_dashboard.png` exist.
  2. **inter_subject LOSO**, subjects `["7","8","9"]`, `n_folds: 3`: three runs `fold0..fold2`, each with train subjects the other two and one held-out subject; every subject appears exactly once as held out across the runs.
  3. **inter_subject eval_subjects**, pool `["1","2","3","4","8","9"]`, `train_subjects: ["2","3","4"]`, `eval_subjects: {"seen": ["1"], "unseen": ["8","9"]}`, with `feature: "raw_band"`, `time_pool: "learned"` (the `RawSource` path and a combination the old class could not build): one run `main`, groups `seen` (1 subject) and `unseen` (2 subjects), `n_trials` filled.
  4. **Other features** (intra_subject, subject `["8"]`, `n_folds: 2`): `stamp_band` with `time_pool: "flat"`; `raw_signal` with `time_pool: "none"`; `stamp_power` with `phase_advance: true`; `stamp_power` with `time_pool: "none"`. Each finishes and writes a valid `group_eval.json`.
  5. **Checkpoint round trip:** for run 1's `head.pth`, `viz.load_model(cfg, path, device, mode='finetune')` returns a `FinetuneModel`, and `model.head(cached_amp)` for a few cached trials equals the logits of the trained head reloaded from the file (max abs diff 0).
  6. **Errors:** a config with the old `split_mode` key and no `split` block raises a `KeyError` or `ValueError` that names `split` (say what you saw; a clear message is preferred: if the raw `KeyError: 'split'` is all that appears, wrap it: `raise ValueError("training_params.finetune.split is required (mode: intra_subject | inter_subject)")`).
  7. **Sanity against the old result:** subject 8, `intra_subject`, `n_folds: 5`, default head, `epochs: 100` (`warmup_epochs`, learning rates as in `config/config.json`). The old C1 run on this subject (5-fold, same head family, 100 epochs) had tail-mean balanced accuracy 0.667 (per-fold 0.608, 0.738, 0.735, 0.610, 0.648). Report the new per-fold tails and mean and the seconds per epoch; a mean below 0.45 is a red flag to investigate before committing (report what you find); values within roughly +-0.1 of 0.667 are expected (fp32 training, cached fp16 amplitudes, real-channel-only spatial filter and a different RNG stream change the numbers a little).

- [ ] **Step 7: commit** in two commits: (1) `refactor: head-only train_finetune.py with intra_subject and inter_subject splits, group_eval output and env stamp` (`train_finetune.py`, `MeSAE_modules.py`, `MeSAE.py`, `config/config.json`); no `output/` files.

---

## Task 3: docs and stale-reference sweep

**Files:** `CLAUDE.md`, `docs/superpowers/specs/2026-09-21-finetune-restructure-design.md`, `docs/adr/0016-finetune-head-modules.md`, `docs/agents/*.md` and any doc or script found by the sweep.

- [ ] **Step 1: sweep.** `git grep -n -E "split_mode|cv_folds|train_val_split|subject_kfold|subject_group_runs|loso_summary|intra_subject|inter_subject|backbone_lr_mult|freeze_backbone|best_finetune|run_training_loop|FinetuneCollate"` and list every hit outside `docs/adr/0012*`, `docs/adr/0014*`, `docs/adr/0014_attempts.csv`, older plans under `docs/superpowers/plans/`, `config/phase2/` and `probes/` (historical records or deleted in sub-project D). Fix every other hit or list it in the report (for example `CLAUDE.md` Commands/Config text, `model/`/`viz/` docstrings that name removed modes).
- [ ] **Step 2: docs.**
  - `CLAUDE.md`: the Commands entry for `train_finetune.py` says it trains a head on the frozen backbone (stamp cache or raw signal) with a `split` block (`intra_subject` / `inter_subject`); the Config section's `training_params.finetune` list drops `split_mode`/`cv_folds`/`freeze_backbone` and names `split` (with its keys), `seed`, `batch_size`, `epochs`, learning-rate fields; the Outputs section lists `finetune/run_<name>/head.pth` and `artifacts/group_eval.json`.
  - Spec sub-project C: record the refinements listed in this plan's header (no `DataLoader`, fp32, one evaluation pass, bare head model, one dataset per run, missing subjects raise).
  - ADR 0016 Consequences: one sentence that the finetune script trains the head directly on the cache.
- [ ] **Step 3: commit** `docs: describe the restructured train_finetune.py (sub-project C)`; preserve each file's line endings, small diffs.

---

## Self-Review Notes

- **Spec coverage (C):** nested `split` block; `intra_subject` and `inter_subject` with `n_folds` (LOSO when `k` equals the number of subjects), `eval_subjects`, `train_subjects`; subject-entry helper (`all`, list, `random`); training on the cache / raw source with only head parameters optimised; removal of `freeze_backbone`, `backbone_lr_mult`, `recon_mse`, best-validation checkpoints and the generic viz; `group_eval.json` for both modes; environment stamp; expected size about 450 lines.
- **Interfaces:** `make_runs` output feeds `run_one` unchanged; sources expose `labels`, `subject_data`, `channel_idx`, `keep`, `num_patches`, `num_stamps`; the head checkpoint format is the one `FinetuneModel.from_checkpoint` already reads.
- **Behaviour vs the old script:** fold composition is checked against the old splitters (Task 1); the training numerics change deliberately (fp32, no DataLoader workers, real-channel spatial filter), so Task 2 Step 6.7 compares with the old C1 subject-8 result as a sanity check, not an equality test.
- **Known limits:** the whole pool is held in RAM (PhysionetMI stamp cache 6.6 GiB, raw about 8 GiB); a batch of one trial is skipped in training (BatchNorm); one dataset per run; `config/phase2/*.json` do not run any more (replaced by experiment specs in sub-project D).
- **Deliberately not done here:** the experiment runner and `results.py` (D), finetune viz and head diagnostics (viz refactor), a plugin hook for other models' heads.
