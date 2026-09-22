# Finetune restructure, sub-project B: feature cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the frozen backbone's stamp amplitudes once per (backbone checkpoint, dataset, preprocessing) and store them next to the backbone, so a stamp-feature head can later be trained on the cache without rerunning the backbone every epoch (sub-project C consumes it).

**Architecture:** One new top-level module `cache_feature.py` (repo root, beside `cache_dataset.py` and `train_finetune.py`, because it is glue between the data pipeline and the model and `model/` must not depend on `IO/`) with `get_stamp_cache(...)` (build missing per-subject files, return the folder), `CachedStampDataset` (in-RAM dataset over those files) and a small CLI. `model/` changes by one builder only: `load_backbone` in `model/factory.py`. The cache reuses `StampExtractor` from sub-project A, so it stores exactly what the head consumes: fp16 `amp [n_trials, N', C_valid, S, 2]`. Training itself is not changed here.

**Tech Stack:** PyTorch (`eeg_fm` env), numpy `.npz`, existing `IO/dataset.py` (`build_dataset_from_config`), `IO/preprocessing.py` (`slice_patches`, `cache_suffix`), `model/factory.py`, `model/MeSAE/MeSAE_modules.py` (`StampExtractor`, read only).

**Spec:** `docs/superpowers/specs/2026-09-21-finetune-restructure-design.md`, sub-project B (read it first). Two refinements to the spec are made here and written back in Task 2: the cache is validated **per subject** (a stored fingerprint of the compiled data file) instead of hashing all data files into the folder key, so adding subjects never invalidates the others; and the folder key also covers what determines the electrode coordinates (whether `mne` is installed and its version, `metadata.json`, `config/montages.json`), because the backbone's spatial embedding depends on them.

## Global Constraints

- **Python env:** `/home/mamechin/anaconda3/envs/eeg_fm/bin/python` for every command (never `base`: it has no `mne`, which changes the coordinates and therefore the cache).
- **The GPU is free.** Task 2's verification uses it briefly (BNCI2014001, two subjects); nothing else is running.
- **Layering:** `model/` does not build datasets and `IO/` does not import `model/`. `model/` already imports two small helpers from `IO/` (`slice_patches` in `plugin.py`, `resolve_canonical_channels` in `factory.py`); those stay as they are. Code that builds datasets and runs the backbone over them lives at the repo root (like `train_finetune.py`). `model/MeSAE/MeSAE_modules.py`, `model/MeSAE/MeSAE.py` and `IO/` are **not touched**; the only edit under `model/` is `load_backbone` in `factory.py`.
- **Line endings:** `cache_feature.py` is a new LF file. `model/factory.py` is LF (verify with `grep -c $'\r'` before editing and keep it LF); `model/MeSAE/MeSAE.py` and `config/config.json` are CRLF and are not touched. `CLAUDE.md` and the spec keep their own endings.
- **No change to training, splits or any head math.** `train_finetune.py` is not touched.
- **Cache location:** `<backbone run folder>/feature_cache/<dataset>/<key>/<subject>.npz`, where the run folder is the parent of the checkpoint's `checkpoint/` directory (`output/pretrain/mesae_v10_small/` today). `output/` is git-ignored and the cache is regenerable, so nothing is committed from it.
- **fp16 storage:** the builder asserts every stored value is finite and that the maximum absolute value is below 6e4 (fp16 max is 65504); a violation raises with the subject id.
- No test suite exists (CLAUDE.md): validation is the acceptance script in Task 2, not pytest files.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
  Do not push (the controller pushes). Never commit anything under `output/` or `.superpowers/`.

## Interfaces (the contract Task 1 builds and sub-project C consumes)

- `model.factory.load_backbone(config, checkpoint_path=None, mode='finetune') -> nn.Module`: `build_pretrain_from_config(config, mode)` plus `load_state_dict` from `checkpoint_path` (default `config['training_params'][mode]['pretrained_checkpoint']`). Used by `build_finetune_from_config`, `load_finetune_checkpoint` and the cache.
- `cache_feature.py` (repo root; CLI `python cache_feature.py --config config/config.json [--batch-size 64]` builds the cache for every subject of every dataset in `dataset_params.finetune`, resolving `"all"` from `metadata.json`, and prints the folders):
  - `get_stamp_cache(config, dataset_name, subjects, device=None, batch_size=64) -> str`: builds any missing or stale per-subject files and returns the folder. `subjects` are subject-id strings as in `subject_to_use`; the dataset entry is `config['dataset_params']['finetune'][dataset_name]`.
  - `cache_key(config, dataset_name, keep, checkpoint_path) -> str` (12 hex chars).
  - `CachedStampDataset(folder, subjects)`: `torch.utils.data.Dataset` holding everything in RAM. Attributes: `amp` (fp16 `[n, N', C_valid, S, 2]`), `labels` (`LongTensor [n]`), `subject_data` (`LongTensor [n]`, `int(subject_id)` per trial, the same convention as `EEGDataset.subject_data`), `valid_length` (`LongTensor [n]`), `channel_idx` (list), `keep` (list), `num_patches`, `num_channels`, `num_stamps`. `__getitem__(i) -> (amp[i], labels[i], valid_length[i])`. All subjects must share `channel_idx` and `keep` (asserted).
- Per-subject file `<subject>.npz` keys: `amp` fp16, `labels` int64 `[n]`, `valid_length` int64 `[n]`, `channel_idx` int64, `keep` int64, `meta` (a 0-d string array holding JSON `{"data": [size, mtime_ns] of the compiled data file}`).

---

## Task 1: `load_backbone` and `cache_feature.py`

**Files:**
- Modify: `model/factory.py` (add `load_backbone`; use it in `build_finetune_from_config` and `load_finetune_checkpoint`)
- Create: `cache_feature.py` (repo root)

- [ ] **Step 1: Read first.** Read `cache_dataset.py` (style of a root pipeline script and its CLI), `model/factory.py` (`build_finetune_from_config`, `load_finetune_checkpoint`), `StampExtractor` in `MeSAE_modules.py`, `IO/dataset.py` (`build_dataset_from_config`, `FinetuneDataset`, the `_load_task` cache path `dataset_path/cache/{subject}_{cache_suffix}.npz`), `IO/preprocessing.py` (`slice_patches`, `cache_suffix`), and `IO/loader.py` (`get_standard_coords`). Check the line endings of `factory.py`. If any name, signature or path below differs from the real code, the real code wins; say so in the report.

- [ ] **Step 2: `load_backbone`.** In `model/factory.py` add

```python
def load_backbone(config, checkpoint_path=None, mode='finetune'):
    """Frozen-backbone loader shared by the finetune builders and the feature cache."""
    backbone = build_pretrain_from_config(config, mode=mode)
    path = checkpoint_path or config['training_params'][mode]['pretrained_checkpoint']
    # load_state_dict restores the checkpoint's phase flags (MeSAE _restore_phase)
    backbone.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'])
    return backbone
```
and replace the duplicated build-and-load lines in `build_finetune_from_config` (`backbone = build_pretrain_from_config(...)` through `backbone.load_state_dict(...)`) and in `load_finetune_checkpoint` (use `load_backbone(config, ckpt['backbone_checkpoint'])`) with calls to it. Behaviour must not change.

- [ ] **Step 3: `cache_feature.py`** (new, repo root, LF):

```python
"""Stamp-amplitude cache for the frozen backbone (finetune restructure, sub-project B).
Pipeline stage between the compiled data (cache_dataset.py) and the finetune head, hence a root script:
    python cache_feature.py --config config/config.json

The backbone never changes during finetuning, so its stamp amplitudes are computed once per
(checkpoint, dataset, preprocessing) and stored next to the backbone:
    <backbone run folder>/feature_cache/<dataset>/<key>/<subject>.npz
Only stamp features use it (StampExtractor output); raw features never touch the backbone."""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from IO.dataset import build_dataset_from_config
from IO.loader import get_standard_coords
from IO.preprocessing import cache_suffix, slice_patches
from model.factory import load_backbone
from model.MeSAE.MeSAE_modules import StampExtractor


def _fingerprint(path):
    """[size, mtime_ns] of a file, or None if missing."""
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    return [st.st_size, st.st_mtime_ns]


def _mne_version():
    try:
        import mne
        return mne.__version__ if get_standard_coords('Cz') is not None else 'installed-but-unusable'
    except ImportError:
        return 'none'   # coordinates fall back to the flat metadata polar values


def cache_key(config, dataset_name, keep, checkpoint_path):
    """Folder key: everything that changes the amplitudes of a given subject file."""
    ds_args = config['dataset_params']['finetune'][dataset_name]
    parts = dict(
        ckpt=[os.path.basename(checkpoint_path), *_fingerprint(checkpoint_path)],
        keep=[int(k) for k in keep],
        preprocess=config.get('preprocess_params', {}),
        dataset={k: v for k, v in ds_args.items() if k != 'subject_to_use'},
        metadata=_fingerprint(os.path.join(ds_args['dataset_path'], 'metadata.json')),
        montages=_fingerprint(os.path.join('config', 'montages.json')),
        mne=_mne_version(),
    )
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _data_path(config, dataset_name, subject):
    pp = config['preprocess_params']
    suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                          pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))
    return os.path.join(config['dataset_params']['finetune'][dataset_name]['dataset_path'],
                        'cache', f"{subject}_{suffix}.npz")


def _is_current(path, data_fp):
    """True if the stored subject file exists and was built from the current compiled data file."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path) as z:
            return json.loads(str(z['meta']))['data'] == data_fp
    except Exception:
        return False


@torch.no_grad()
def _build_subject(config, dataset_name, subject, backbone, device, batch_size, path):
    sub_cfg = json.loads(json.dumps(config))
    sub_cfg['dataset_params']['finetune'] = {
        dataset_name: {**config['dataset_params']['finetune'][dataset_name], 'subject_to_use': [subject]}}
    base = build_dataset_from_config(sub_cfg, mode='finetune').base_dataset
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    patch_stride = pp.get('patch_stride', patch_len)
    valid = base.all_valid_channels[0]
    channel_idx = torch.nonzero(valid).flatten().tolist()
    coords, vlen = base.all_coords[0], int(base.all_valid_length[0])
    extractor = StampExtractor(backbone, channel_idx).to(device).eval()
    out = []
    for i in range(0, len(base.data), batch_size):
        x = base.data[i:i + batch_size]                                   # [b, C, T]
        xp, _ = slice_patches(x, patch_len, patch_stride)                 # [b, C, N', L]
        b, P = xp.shape[0], xp.shape[2]
        amp = extractor(xp.to(device), coords.unsqueeze(0).expand(b, -1, -1).to(device),
                        torch.arange(P, device=device).unsqueeze(0).expand(b, P),
                        valid.unsqueeze(0).expand(b, -1).to(device))      # [b, N', Cv, S, 2] fp32
        out.append(amp.cpu())
    amp = torch.cat(out)
    if not torch.isfinite(amp).all() or amp.abs().max() >= 6e4:
        raise ValueError(f"subject {subject}: stamp amplitudes are not finite or exceed the fp16 range "
                         f"(max abs {amp.abs().max().item():.3g})")
    data_fp = _fingerprint(_data_path(config, dataset_name, subject))
    np.savez(path, amp=amp.half().numpy(), labels=base.labels.numpy().astype(np.int64),
             valid_length=np.full(len(amp), vlen, dtype=np.int64), channel_idx=np.asarray(channel_idx, dtype=np.int64),
             keep=extractor.keep.cpu().numpy().astype(np.int64), meta=np.array(json.dumps({'data': data_fp})))
    return amp.shape


def get_stamp_cache(config, dataset_name, subjects, device=None, batch_size=64):
    """Build any missing or stale per-subject file and return the cache folder."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = config['training_params']['finetune']['pretrained_checkpoint']
    backbone = load_backbone(config).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    keep = StampExtractor(backbone, [0]).keep.cpu().tolist()   # alive stamps do not depend on the channels
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    folder = os.path.join(run_dir, 'feature_cache', dataset_name, cache_key(config, dataset_name, keep, ckpt))
    os.makedirs(folder, exist_ok=True)
    for sub in subjects:
        sub = str(sub)
        path = os.path.join(folder, f'{sub}.npz')
        if _is_current(path, _fingerprint(_data_path(config, dataset_name, sub))):
            continue
        shape = _build_subject(config, dataset_name, sub, backbone, device, batch_size, path)
        print(f"  [feature_cache] built {dataset_name} subject {sub}: amp {tuple(shape)} -> {path}")
    return folder


class CachedStampDataset(Dataset):
    """Stamp amplitudes of the given subjects, in RAM. See the plan's Interfaces section."""
    def __init__(self, folder, subjects):
        parts = []
        for s in subjects:
            with np.load(os.path.join(folder, f'{s}.npz')) as z:
                parts.append({k: z[k] for k in ('amp', 'labels', 'valid_length', 'channel_idx', 'keep')})
        for p in parts[1:]:
            assert np.array_equal(p['channel_idx'], parts[0]['channel_idx']), "subjects have different real-channel sets"
            assert np.array_equal(p['keep'], parts[0]['keep']), "subjects were cached with different alive stamps"
        self.amp = torch.from_numpy(np.concatenate([p['amp'] for p in parts]))
        self.labels = torch.from_numpy(np.concatenate([p['labels'] for p in parts])).long()
        self.valid_length = torch.from_numpy(np.concatenate([p['valid_length'] for p in parts])).long()
        self.subject_data = torch.cat([torch.full((len(p['labels']),), int(s), dtype=torch.long)
                                       for s, p in zip(subjects, parts)])
        self.channel_idx = parts[0]['channel_idx'].tolist()
        self.keep = parts[0]['keep'].tolist()
        self.num_patches, self.num_channels, self.num_stamps = self.amp.shape[1], self.amp.shape[2], self.amp.shape[3]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.amp[i], self.labels[i], self.valid_length[i]


def _subjects(ds_args):
    subs = ds_args['subject_to_use']
    if subs in (['all'], 'all'):
        with open(os.path.join(ds_args['dataset_path'], 'metadata.json'), encoding='utf-8') as f:
            ids = list(json.load(f)['data_structure'].keys())
        try:
            return sorted(ids, key=int)
        except ValueError:
            return sorted(ids)
    return [str(s) for s in subs]


def main():
    ap = argparse.ArgumentParser(description="Build the stamp-amplitude cache for every finetune dataset in a config.")
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = json.load(f)
    for name, ds_args in config['dataset_params']['finetune'].items():
        folder = get_stamp_cache(config, name, _subjects(ds_args), batch_size=args.batch_size)
        print(f"{name}: cache ready in {folder}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Import and layering check.** `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "import cache_feature, model.factory, viz, train_finetune; print('ok')"` prints `ok`; `git grep -n "build_dataset_from_config\|EEGDataset" -- model/` prints nothing (`model/` builds no datasets). Rerun the sub-project A equivalence script `.superpowers/sdd/2026-09-21-finetune-model-restructure-a/head_equiv.py` (`CUDA_VISIBLE_DEVICES='' PYTHONPATH=.`): 25 `OK` lines, so the `load_backbone` refactor changed nothing.

- [ ] **Step 5: Commit** in two commits: (1) `refactor: extract load_backbone in the model factory` (`model/factory.py`); (2) `feat: cache_feature.py, stamp-amplitude cache next to the backbone` (`cache_feature.py`). `git status --short` shows nothing else.

---

## Task 2: acceptance verification and docs

**Files:**
- Create (not committed): `.superpowers/sdd/2026-09-21-finetune-restructure-b/cache_accept.py`
- Modify: `docs/superpowers/specs/2026-09-21-finetune-restructure-design.md` (sub-project B), `CLAUDE.md` (Commands and Outputs sections)

- [ ] **Step 1: Write the acceptance script** (GPU allowed, `eeg_fm` python, run from the repo root with `PYTHONPATH=.`):

```python
"""Acceptance for sub-project B: feature cache matches the on-the-fly extractor, is reused, and invalidates correctly."""
import copy, os, time, torch, numpy as np
from viz import load_config
import cache_feature as FC
from model.MeSAE.MeSAE import build_finetune
from model.factory import load_backbone
from IO.dataset import build_dataset_from_config
from IO.preprocessing import slice_patches

cfg = load_config('config/config.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ds_name, subs = 'BNCI2014001', ['8', '9']
cfg['dataset_params']['finetune'] = {ds_name: {**cfg['dataset_params']['finetune'][ds_name], 'subject_to_use': subs}}

# 1. build, then reuse
t = time.time(); folder = FC.get_stamp_cache(cfg, ds_name, subs, device=dev); t_build = time.time() - t
files = [os.path.join(folder, f'{s}.npz') for s in subs]
m0 = [os.stat(f).st_mtime_ns for f in files]
t = time.time(); folder2 = FC.get_stamp_cache(cfg, ds_name, subs, device=dev); t_reuse = time.time() - t
assert folder2 == folder and [os.stat(f).st_mtime_ns for f in files] == m0, "second call must not rebuild"
print(f"OK build {t_build:.1f}s, reuse {t_reuse:.1f}s (no rebuild), folder {folder}")

# 2. dataset shape and conventions
ds = FC.CachedStampDataset(folder, subs)
n = len(ds)
assert ds.amp.dtype == torch.float16 and ds.amp.shape[0] == n and ds.amp.shape[1:] == (39, len(ds.channel_idx), ds.num_stamps, 2), ds.amp.shape
assert set(ds.subject_data.tolist()) == {8, 9} and len(ds.labels) == n and ds.labels.dtype == torch.long
assert torch.isfinite(ds.amp.float()).all()
print(f"OK dataset amp {tuple(ds.amp.shape)}, {ds.amp.numel() * 2 / 2**20:.1f} MiB, max abs {ds.amp.float().abs().max():.3g}")

# 3. cached amplitudes and logits match the on-the-fly extractor (fp32, no autocast)
bb = load_backbone(cfg).to(dev).eval()
fm = build_finetune(bb, 64, 4, channel_idx=ds.channel_idx, num_patches=ds.num_patches, sample_freq=200, dropout=0.5).to(dev).eval()
assert fm.extractor.keep.tolist() == ds.keep
sub_cfg = copy.deepcopy(cfg); sub_cfg['dataset_params']['finetune'][ds_name]['subject_to_use'] = ['8']
base = build_dataset_from_config(sub_cfg, mode='finetune').base_dataset
idx = list(range(0, 32, 2))
x = base.data[idx]; xp, _ = slice_patches(x, cfg['preprocess_params']['patch_length'], cfg['preprocess_params'].get('patch_stride'))
b, P = xp.shape[0], xp.shape[2]
coords = base.all_coords[0].unsqueeze(0).expand(b, -1, -1).to(dev)
vc = base.all_valid_channels[0].unsqueeze(0).expand(b, -1).to(dev)
tix = torch.arange(P, device=dev).unsqueeze(0).expand(b, P)
with torch.no_grad():
    fresh = fm.extractor(xp.to(dev), coords, tix, vc).cpu()
    cached = ds.amp[idx].float()
    err = (fresh - cached).abs().max().item(); scale = fresh.abs().max().item()
    assert err <= 1e-3 * scale, (err, scale)
    lf = fm.head(fresh.to(dev)); lc = fm.head(cached.to(dev))
    assert torch.allclose(lf, lc, atol=1e-2) and (lf.argmax(1) == lc.argmax(1)).all(), (lf - lc).abs().max().item()
    lm = fm(xp.to(dev), coords, time_idx=tix, valid_channels=vc)[0]
    assert torch.allclose(lm, lf, atol=1e-6)
print(f"OK cached amp rel err {err / scale:.2e}, logits max diff {(lf - lc).abs().max().item():.2e}")

# 4. invalidation: key covers preprocessing/dataset/checkpoint settings, per-subject file covers the data file
k0 = FC.cache_key(cfg, ds_name, ds.keep, cfg['training_params']['finetune']['pretrained_checkpoint'])
for path, val in [(('preprocess_params', 'patch_stride'), 30), (('preprocess_params', 'normalization_type'), 'robust')]:
    c2 = copy.deepcopy(cfg); c2[path[0]][path[1]] = val
    assert FC.cache_key(c2, ds_name, ds.keep, cfg['training_params']['finetune']['pretrained_checkpoint']) != k0, path
assert FC.cache_key(cfg, ds_name, ds.keep[:-1], cfg['training_params']['finetune']['pretrained_checkpoint']) != k0
assert not FC._is_current(files[0], [1, 2]) and FC._is_current(files[0], FC._fingerprint(FC._data_path(cfg, ds_name, '8')))
assert not FC._is_current(os.path.join(folder, 'nope.npz'), None)
print("OK invalidation (key changes with preprocessing, keep; per-subject file checks the data fingerprint)")

# 5. cache tied to the subject list: a third subject is added without touching the first two
subs3 = subs + ['7']
cfg3 = copy.deepcopy(cfg); cfg3['dataset_params']['finetune'][ds_name]['subject_to_use'] = subs3
folder3 = FC.get_stamp_cache(cfg3, ds_name, subs3, device=dev)
assert folder3 == folder and [os.stat(f).st_mtime_ns for f in files] == m0 and os.path.exists(os.path.join(folder, '7.npz'))
print("OK adding a subject reuses the folder and leaves existing files untouched")
```
Remove the temporary subject-7 file afterwards if you like; the cache is regenerable.

- [ ] **Step 2: Run it.** `PYTHONPATH=. /home/mamechin/anaconda3/envs/eeg_fm/bin/python .superpowers/sdd/2026-09-21-finetune-restructure-b/cache_accept.py`. Expected: five `OK` lines. Failures are bugs in `cache_feature.py`: fix it, not the script (unless the script hits a real API mismatch such as the dataset's config layout, then adapt it and report). Record in the report: build time, reuse time, cache size per subject, and the printed relative error.

- [ ] **Step 3: Timing note for the big datasets.** Time the build of one BETA_4s subject (`get_stamp_cache` with `dataset_name='BETA_4s'`, `subjects=['19']`, batch size 64) and report seconds per trial and MiB per trial; extrapolate to EEGMMIdb (39,569 trials) and BETA_4s (8,800 trials) in the report. Delete the test folders' extra subjects afterwards only if disk is a concern.

- [ ] **Step 4: Docs.**
  - Spec sub-project B: the module is the repo-root `cache_feature.py` (a pipeline stage like `cache_dataset.py`: it builds datasets and runs the backbone over them, which `model/` does not do), not `model/MeSAE/feature_cache.py`; replace the cache-key sentence and the "Uniform channels" bullet with: the folder key hashes the checkpoint file identity (name, size, mtime), the `keep` stamp set, the whole `preprocess_params` block, the dataset entry (minus `subject_to_use`), `metadata.json` and `config/montages.json` fingerprints and the `mne` version (electrode coordinates depend on it); each subject file additionally stores the fingerprint of the compiled data file it was built from and is rebuilt when it changes, so adding subjects never invalidates existing files; `CachedStampDataset` asserts all subjects share `channel_idx` and `keep`. Over-invalidation (for example a changed masking setting) only costs a rebuild.
  - `CLAUDE.md`: Commands section, add `python cache_feature.py --config config/config.json` next to the `cache_dataset.py` lines (build the stamp-amplitude cache of the finetune datasets; the runner will do this automatically); Outputs section, add one line `<backbone run folder>/feature_cache/<dataset>/<key>/<subject>.npz` = regenerable cache built by `cache_feature.py`, safe to delete.
- [ ] **Step 5: Commit** `docs: feature cache key and per-subject validation (sub-project B)` (spec and `CLAUDE.md`). Preserve each file's line endings; `git diff --stat` small.

---

## Self-Review Notes

- **Spec coverage (B):** `cache_feature.py` at the repo root (name consistent with `cache_dataset.py`), cache next to the backbone, fp16 real-channel-only storage, uniform-channel assertion, fp16 range check, key over checkpoint/preprocessing/data, acceptance against the on-the-fly extractor (Task 2 step 3: amplitudes within 1e-3 of the max, logits within 1e-2 and identical argmax, plus exact agreement between `FinetuneModel` and `extractor + head`), reuse and invalidation checks.
- **Interfaces:** `get_stamp_cache` and `CachedStampDataset` attributes are exactly what sub-project C's split code and training loop need (`labels`, `subject_data`, `amp`, `channel_idx`, `keep`, `num_patches`, `num_stamps`); `load_backbone` deduplicates the factory code used by A and B.
- **Known limits:** the cache is computed in fp32 while the old training path ran the backbone under fp16 autocast, so cached features are slightly more accurate than what the previous runs saw; fp16 storage costs about 1e-3 relative error; the whole cache of a run is loaded into RAM (EEGMMIdb about 10 GB).
- **Deliberately not done here:** using the cache in training and dropping `freeze_backbone`/`recon_mse` (sub-project C), the experiment runner that builds the cache before a set (D).
