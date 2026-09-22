# /tools panel framework (Sub-project A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `analysis/` and `viz/` under a new `tools/` package, add a filename-convention
`panel_*.py` contract (calc via `tools.analysis`, render via `tools.viz`), wire `--panel` into
`analysis_pretrain.py`/`analysis_finetune.py` additively (old `--analysis` path untouched), and
prove it with `panel_profile.py`, replacing today's standalone `profile_model.py`.

**Architecture:** Task 1 is a pure mechanical relocation (like the earlier
`analysis_pretrain`/`analysis_finetune` split's Task 1). Task 2 builds the new framework and its
first panel, self-contained, touching nothing that exists today. Task 3 adds an opt-in `--panel`
branch to each CLI script's `__main__`, short-circuiting before the existing code when used, so
the default (no `--panel`) path is provably unchanged. Task 4 retires `profile_model.py`.

**Tech Stack:** Python, PyTorch, `eeg_fm` conda env (`/home/mamechin/anaconda3/envs/eeg_fm/bin/python`).

**Spec:** `docs/superpowers/specs/2026-09-22-tools-panels-framework-design.md`

## Global Constraints

- No test suite exists (CLAUDE.md) — validation is smoke runs, never pytest files.
- Pure relocation in Task 1 — no logic changes, only import paths and file locations.
- With no `--panel` given, `analysis_pretrain.py`/`analysis_finetune.py` must behave
  byte-identically to before this plan (Task 3's own verification step proves this).
- `probes/` stays untouched — out of scope for this whole effort, per standing ruling.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`

---

## Task 1: Relocate `analysis/` and `viz/` under `tools/`

**Files:**
- Create: `tools/analysis/__init__.py` (via `git mv` from `analysis/__init__.py`)
- Create: `tools/viz/` (via `git mv` from `viz/`, whole directory)
- Modify: `model/MeSAE/plugin.py` (4 import lines: 17, 19, 21, 560-561)
- Modify: `model/base_checker.py` (4 import lines: 21-24)
- Modify: `model/base_codebook_checker.py` (1 import line: 17)
- Modify: `analysis_pretrain.py` (1 import line: 97)
- Modify: `analysis_finetune.py` (1 import line: 95)
- Modify: `train_pretrain.py` (1 import line: 21)
- Modify: `tools/viz/extract.py` (1 internal import: `from viz.iclabel import` — now
  `from tools.viz.iclabel import`)
- Modify: `tools/viz/panels.py` (2 internal imports: `from viz.topomap import`,
  `from viz.iclabel import` — now `from tools.viz.topomap import`, `from tools.viz.iclabel import`)
- Modify: `CLAUDE.md` (lines 27, 127: `viz/` path mentions)
- Modify: `docs/agents/adding-a-model.md` (lines 72, 162, 165, 169: `analysis/`/`viz/` path
  mentions)

**Interfaces:**
- Produces: `tools.analysis` (same 8 names as before: `_deep_merge`, `load_config`,
  `resolve_output_dir`, `select_subject_dataset`, `filter_config_to_subject`, `load_model`,
  `pick_trial`, `setup_mne_info`), `tools.viz.extract`, `tools.viz.panels`, `tools.viz.codebook`,
  `tools.viz.timeseries`, `tools.viz.topomap`, `tools.viz.iclabel` — same public names as
  today's `viz.*`, just under `tools.viz.*`. Task 2 imports from `tools.analysis`.

- [ ] **Step 1: Create the `tools/` package marker**

```bash
mkdir -p tools
touch tools/__init__.py
```

- [ ] **Step 2: Move both packages, keeping history**

```bash
git mv analysis/__init__.py tools/analysis/__init__.py
git mv viz tools/viz
```
(`git mv analysis/__init__.py tools/analysis/__init__.py` creates `tools/analysis/`
automatically since `analysis/` only ever contained the one `__init__.py` file — confirm
with `ls analysis/` returning nothing/gone and `git status` showing no leftover empty
`analysis/` directory tracked.)

- [ ] **Step 3: Fix `model/MeSAE/plugin.py`'s imports**

Read the file first. Lines 17, 19, 21 each start with `from viz.` — change to `from
tools.viz.` (three edits, e.g. `from viz.extract import (...)` → `from tools.viz.extract
import (...)`, same for the `viz.panels` and `viz.codebook` lines). Lines 560-561 (inside a
function body): `from viz.codebook import plot_stamp_identity_consistency` → `from
tools.viz.codebook import plot_stamp_identity_consistency`, and `from viz.iclabel import
ICLABEL_CLASSES` → `from tools.viz.iclabel import ICLABEL_CLASSES`.

- [ ] **Step 4: Fix `model/base_checker.py`'s imports**

Read the file first. Replace:
```python
from viz.extract import PsdResult, SpectraResult
from viz.topomap import project_coords_2d
from viz.panels import plot_topo_psd_filter, plot_attn_topo as render_attn_topo, plot_stamp_panel
from viz.timeseries import visualize_reconstruction
```
with:
```python
from tools.viz.extract import PsdResult, SpectraResult
from tools.viz.topomap import project_coords_2d
from tools.viz.panels import plot_topo_psd_filter, plot_attn_topo as render_attn_topo, plot_stamp_panel
from tools.viz.timeseries import visualize_reconstruction
```

- [ ] **Step 5: Fix `model/base_codebook_checker.py`'s import**

Read the file first. Its `from viz.codebook import (...)` block (starting line 17) — change
`from viz.codebook import` to `from tools.viz.codebook import`, same parenthesized name
list.

- [ ] **Step 6: Fix `analysis_pretrain.py`'s import**

Replace:
```python
    from analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
```
with:
```python
    from tools.analysis import (
        _deep_merge, load_model,
        select_subject_dataset, filter_config_to_subject, pick_trial, resolve_output_dir,
    )
```

- [ ] **Step 7: Fix `analysis_finetune.py`'s import**

Replace:
```python
    from analysis import _deep_merge, load_model, resolve_output_dir
```
with:
```python
    from tools.analysis import _deep_merge, load_model, resolve_output_dir
```

- [ ] **Step 8: Fix `train_pretrain.py`'s import**

Replace:
```python
from analysis import pick_trial
```
with:
```python
from tools.analysis import pick_trial
```

- [ ] **Step 9: Fix `tools/viz/extract.py`'s internal import**

Read the file first (it's now at `tools/viz/extract.py` after Step 2's move). Replace:
```python
    from viz.iclabel import stamp_iclabel_probs
```
with:
```python
    from tools.viz.iclabel import stamp_iclabel_probs
```

- [ ] **Step 10: Fix `tools/viz/panels.py`'s internal imports**

Read the file first. Replace:
```python
from viz.topomap import draw_topomap, build_triangulation
```
with:
```python
from tools.viz.topomap import draw_topomap, build_triangulation
```
and replace:
```python
        from viz.iclabel import ICLABEL_CLASSES
```
with:
```python
        from tools.viz.iclabel import ICLABEL_CLASSES
```

- [ ] **Step 11: Update `CLAUDE.md`'s two `viz/` path mentions**

Line 27: replace `viz/extract.py, panels.py, timeseries.py, topomap.py are shared
primitives it` with `tools/viz/extract.py, panels.py, timeseries.py, topomap.py are shared
primitives it`.

Line 127 (the long `visualize_params` line): replace the substring `` `viz/timeseries.py`'s
`_canonical_bands` `` with `` `tools/viz/timeseries.py`'s `_canonical_bands` ``.

- [ ] **Step 12: Update `docs/agents/adding-a-model.md`'s four path mentions**

Line 72: replace `Called at construction-time load (analysis/__init__.py` with `Called at
construction-time load (tools/analysis/__init__.py`.

Line 162: replace `` """Returns viz/extract.py's PsdResult(psd_ch_x, norms, affinity,
importance) — `` with `` """Returns tools/viz/extract.py's PsdResult(psd_ch_x, norms,
affinity, importance) — ``.

Line 165: replace `norm), or write a MeXXX-specific one in viz/extract.py returning the
same` with `norm), or write a MeXXX-specific one in tools/viz/extract.py returning the
same`.

Line 169: replace `` """Returns viz/extract.py's SpectraResult(psd [Q, C, F], freqs [F],
importance [Q])."""`` with `` """Returns tools/viz/extract.py's SpectraResult(psd [Q, C, F],
freqs [F], importance [Q])."""``.

- [ ] **Step 13: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import tools.analysis, tools.viz.extract, tools.viz.panels, tools.viz.codebook, tools.viz.timeseries, tools.viz.topomap, tools.viz.iclabel
import analysis_pretrain, analysis_finetune, train_pretrain
import model.base_checker, model.base_codebook_checker, model.MeSAE.plugin
print('ok')
"
```
Expected: `ok`, no `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 14: Confirm no stale import paths remain**

```bash
git grep -n "^from viz\|^import viz\b\| from viz\.\|from analysis import" -- '*.py'
```
Expected: no output (every `viz.`/`analysis` import now reads `tools.viz.`/`tools.analysis`).

- [ ] **Step 15: `git diff --stat` sanity check**

```bash
git diff --stat --cached
git status --short
```
Expected: two renames shown (`analysis/__init__.py` → `tools/analysis/__init__.py`, and
each file under `viz/` → `tools/viz/`), plus small edits in the 8 Python files and 2 doc
files listed above — no unrelated files touched.

- [ ] **Step 16: Commit**

```bash
git add tools analysis_pretrain.py analysis_finetune.py train_pretrain.py \
  model/MeSAE/plugin.py model/base_checker.py model/base_codebook_checker.py \
  CLAUDE.md docs/agents/adding-a-model.md
git commit -m "refactor: relocate analysis/ and viz/ under tools/

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 2: Panel framework + `panel_profile.py`

**Files:**
- Create: `tools/panels/__init__.py`
- Create: `tools/analysis/profile.py`
- Create: `tools/panels/panel_profile.py`

**Interfaces:**
- Consumes: nothing from Task 1 beyond `tools/` existing as an importable package.
- Produces: `tools.panels.PanelContext` (dataclass: `config: dict`, `output_dir: str`,
  `device`, `args`, `model: Optional = None`, `dataset: Optional = None`, `checkpoint:
  Optional[str] = None`), `tools.panels.discover_panel_names() -> list[str]`,
  `tools.panels.load_panel(name: str) -> module`, `tools.panels.run_panels(names:
  list[str], stage: str, ctx: PanelContext) -> None`, `tools.panels.any_needs(names:
  list[str], attr: str) -> bool`. Task 3 imports all five of these from `tools.panels`.
  `tools.analysis.profile.run_profile(config: dict, device, train_mode: bool = False) ->
  ProfileResult` (dataclass: `model_type: str`, `batch: int`, `channels: int`, `patches:
  int`, `patch_len: int`, `children: list[tuple[str, object]]`, `param_map: dict[str,
  int]`, `time_stats: dict[str, float]`, `loss_ms: float`, `total_ms: float`). Task 2's own
  `panel_profile.py` consumes this; no other task calls it directly.

- [ ] **Step 1: Write `tools/panels/__init__.py`**

```python
"""Panel discovery/dispatch shared by analysis_pretrain.py and analysis_finetune.py.

A panel is a module at tools/panels/panel_<name>.py exposing:
  STAGES: frozenset[str]     -- subset of {'pretrain', 'finetune'}: which analysis_*.py
                                 script(s) may run it
  NEEDS_CHECKPOINT: bool     -- whether the CLI must resolve a model before calling it
  NEEDS_DATASET: bool        -- whether the CLI must resolve a dataset before calling it
  def run(ctx) -> None       -- does the work; panels are terminal actions, not
                                 composable functions

Discovery is filename convention only -- no registry, no decorator, no base class:
adding a panel is adding a file, removing one is deleting a file."""
import glob
import importlib
import os
from dataclasses import dataclass
from typing import Optional

PANELS_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PanelContext:
    """Built once per analysis_*.py CLI invocation, passed to every selected panel's
    run(). A panel with NEEDS_CHECKPOINT=NEEDS_DATASET=False simply never reads
    model/dataset/checkpoint."""
    config: dict
    output_dir: str
    device: object            # torch.device -- kept untyped here so this module doesn't
                               # need to import torch just to define the dataclass
    args: object               # argparse.Namespace, for panel-specific CLI flags
    model: Optional[object] = None
    dataset: Optional[object] = None
    checkpoint: Optional[str] = None


def discover_panel_names():
    """Sorted panel names (each tools/panels/panel_<name>.py filename, minus the
    'panel_' prefix and '.py' suffix)."""
    paths = glob.glob(os.path.join(PANELS_DIR, 'panel_*.py'))
    return sorted(os.path.basename(p)[len('panel_'):-len('.py')] for p in paths)


def load_panel(name):
    """Import tools.panels.panel_<name>, return the module."""
    return importlib.import_module(f'tools.panels.panel_{name}')


def run_panels(names, stage, ctx):
    """Loads and runs each named panel in order, checked against `stage`
    ('pretrain'/'finetune'). Raises ValueError naming the panel if it doesn't declare
    support for this stage -- a clear error beats a silent no-op."""
    for name in names:
        mod = load_panel(name)
        if stage not in mod.STAGES:
            raise ValueError(f"panel {name!r} does not support stage {stage!r} "
                              f"(STAGES={sorted(mod.STAGES)})")
        mod.run(ctx)


def any_needs(names, attr):
    """True if any named panel declares `attr` (NEEDS_CHECKPOINT/NEEDS_DATASET) True."""
    return any(getattr(load_panel(name), attr) for name in names)
```

- [ ] **Step 2: Write `tools/analysis/profile.py`**

```python
"""Model profiling: parameter counts and per-component forward-pass timing, from a fresh
untrained model (config/config.json's architecture) -- no checkpoint, no dataset. Moved
from the old standalone profile_model.py; see tools/panels/panel_profile.py for the CLI
entry point and the printed report."""
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn

from model.factory import build_pretrain_from_config

logging.getLogger("fvcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ProfilerHooks:
    def __init__(self):
        self.timings = defaultdict(list)
        self.starts = {}
        self.device = None

    def _sync(self):
        if self.device and self.device.type == 'cuda':
            torch.cuda.synchronize()

    def register(self, model, device):
        self.device = device
        hooks = []
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleList):
                for sub_module in module:
                    h1 = sub_module.register_forward_pre_hook(self._make_pre_hook(name))
                    h2 = sub_module.register_forward_hook(self._make_hook(name))
                    hooks.extend([h1, h2])
            else:
                h1 = module.register_forward_pre_hook(self._make_pre_hook(name))
                h2 = module.register_forward_hook(self._make_hook(name))
                hooks.extend([h1, h2])
        return hooks

    def _make_pre_hook(self, name):
        def hook(module, input):
            self._sync()
            self.starts[name] = time.perf_counter()
        return hook

    def _make_hook(self, name):
        def hook(module, input, output):
            self._sync()
            end = time.perf_counter()
            if name in self.starts:
                duration = (end - self.starts[name]) * 1000  # ms
                self.timings[name].append(duration)
        return hook

    def get_summary(self, n_iters, model):
        stats = []
        for name, module in model.named_children():
            times = self.timings.get(name, [])
            if not times:
                continue
            if isinstance(module, nn.ModuleList):
                total_ms = sum(times) / n_iters
            else:
                total_ms = sum(times) / len(times)
            stats.append((name, total_ms))
        return stats


@dataclass
class ProfileResult:
    model_type: str
    batch: int
    channels: int
    patches: int
    patch_len: int
    children: list      # [(name, module)] in named_children() order -- also print order
    param_map: dict      # name -> trainable param count
    time_stats: dict     # name -> avg forward-pass ms
    loss_ms: float
    total_ms: float


def run_profile(config, device, train_mode=False):
    """Builds a fresh (untrained) model from config, times its forward pass and get_loss
    component-by-component via forward hooks. No checkpoint, no dataset -- dummy input
    shaped from preprocess_params. Returns a ProfileResult; printing the report is the
    caller's (panel_profile.py's) job, not this function's."""
    model_type = config['training_params']['pretrain'].get('model_type', 'MeSAE')
    preprocess = config['preprocess_params']

    B, C = 16, 64
    L = preprocess.get('patch_length', 25)
    N = 800 // L  # 4 seconds @ 200Hz = 800 samples

    model = build_pretrain_from_config(config).to(device)
    model.train() if train_mode else model.eval()

    x = torch.randn(B, C, N, L).to(device)
    coords = torch.randn(B, C, 3).to(device)
    time_idx = torch.zeros(B, N, dtype=torch.long).to(device)

    with torch.no_grad():
        model(x, coords, time_idx)

    children = list(model.named_children())
    param_map = {name: count_parameters(module) for name, module in children}

    profiler = ProfilerHooks()
    profiler.register(model, device)

    with torch.no_grad():
        for _ in range(5):
            model(x, coords, time_idx)
    profiler.timings.clear()

    n_iters = 20
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_total = time.perf_counter()

    loss_times = []
    bool_masked_pos = torch.zeros(B, C, N, dtype=torch.bool).to(device)
    with torch.no_grad():
        for _ in range(n_iters):
            out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            # get_loss's signature differs by model type (MeSAE inserts aux_loss before
            # bool_masked_pos, MeFSQ doesn't have aux_loss at all) -- passing bool_masked_pos
            # positionally silently mis-binds it into MeSAE's aux_loss slot.
            loss_kwargs = dict(bool_masked_pos=bool_masked_pos)
            if hasattr(out, 'aux_loss'):
                loss_kwargs['aux_loss'] = out.aux_loss
            model.get_loss(x, out.recon, **loss_kwargs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss_times.append((t2 - t1) * 1000)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_avg_ms = ((time.perf_counter() - start_total) / n_iters) * 1000
    time_stats = dict(profiler.get_summary(n_iters, model))
    avg_loss_ms = sum(loss_times) / n_iters

    return ProfileResult(
        model_type=model_type, batch=B, channels=C, patches=N, patch_len=L,
        children=children, param_map=param_map, time_stats=time_stats,
        loss_ms=avg_loss_ms, total_ms=total_avg_ms,
    )
```

- [ ] **Step 3: Write `tools/panels/panel_profile.py`**

```python
"""profile panel: parameter counts and per-component forward-pass timing for a fresh
(untrained) model built from config/config.json -- no checkpoint, no dataset. Replaces
the old standalone profile_model.py; run via
`python analysis_pretrain.py --panel profile [--train]` or the same from
analysis_finetune.py -- both stages support it, see STAGES below."""
from tools.analysis.profile import run_profile

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    train_mode = getattr(ctx.args, 'train', False)
    mode_str = 'TRAIN (eigh skipped)' if train_mode else 'EVAL (eigh active)'
    print(f"Profiling on device: {ctx.device}  |  Mode: {mode_str}")

    result = run_profile(ctx.config, ctx.device, train_mode=train_mode)

    print(f"\nModel: {result.model_type}")
    print(f"Input: Batch={result.batch}, Channels={result.channels}, "
          f"Patches={result.patches}, Samples={result.patch_len}")
    print("-" * 60)
    print("Detected Components:")
    for name, _ in result.children:
        p = result.param_map.get(name, 0)
        print(f"  - {name:<20} : {p / 1e6:>6.2f} M params")
    print("-" * 60)

    print("\nPerformance Summary (Avg of 20 runs):")
    print(f"{'Component':<22} | {'Params (M)':<10} | {'Time (ms)':<10} | {'% Total'}")
    print("-" * 70)
    for name, _ in result.children:
        t_ms = result.time_stats.get(name, 0.0)
        p = result.param_map.get(name, 0)
        print(f"{name:<22} | {p / 1e6:<10.2f} | {t_ms:<10.2f} | "
              f"{(t_ms / result.total_ms) * 100:>6.1f}%")
    print(f"{'Method: get_loss':<22} | {'-':<10} | {result.loss_ms:<10.2f} | "
          f"{(result.loss_ms / result.total_ms) * 100:>6.1f}%")
    print("-" * 70)
    total_params = sum(result.param_map.values())
    print(f"{'Total (Fwd + Loss + Rec)':<22} | {total_params / 1e6:<10.2f} | "
          f"{result.total_ms:<10.2f} | 100.0%")
    print("-" * 70)
```

- [ ] **Step 4: Smoke — imports clean**

```bash
CUDA_VISIBLE_DEVICES='' /home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import tools.panels, tools.analysis.profile, tools.panels.panel_profile
print('ok')
print('discovered:', tools.panels.discover_panel_names())
"
```
Expected: `ok`, then `discovered: ['profile']`.

- [ ] **Step 5: Smoke — the panel runs and matches `profile_model.py`'s output shape**

`profile_model.py` still exists at this point (Task 4 deletes it) — run both and compare:

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python profile_model.py > /tmp/profile_old.txt 2>&1
/home/mamechin/anaconda3/envs/eeg_fm/bin/python -c "
import argparse, torch
from tools.panels import PanelContext
from tools.panels.panel_profile import run
import json
with open('config/config.json') as f:
    cfg = json.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = argparse.Namespace(train=False)
ctx = PanelContext(config=cfg, output_dir='output/tools-profile', device=device, args=args)
run(ctx)
" > /tmp/profile_new.txt 2>&1
diff /tmp/profile_old.txt /tmp/profile_new.txt
```
Expected: `diff` shows only expected differences — the printed component names, param
counts (M), and table structure/headers must match exactly; per-run timing numbers (ms,
%) will differ slightly between runs (real wall-clock measurements, not deterministic) —
that's expected, not a bug. If a component name, param count, or table column is missing
or reordered, that's a real regression — stop and fix before continuing.

- [ ] **Step 6: `git diff --stat` sanity check**

```bash
git status --short tools/panels tools/analysis/profile.py
```
Expected: three new untracked files (`tools/panels/__init__.py`,
`tools/analysis/profile.py`, `tools/panels/panel_profile.py`) — nothing else.

- [ ] **Step 7: Commit**

```bash
git add tools/panels tools/analysis/profile.py
git commit -m "feat: panel framework (tools/panels) and panel_profile, the first panel

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 3: Wire `--panel` into `analysis_pretrain.py` and `analysis_finetune.py`

**Files:**
- Modify: `analysis_pretrain.py`
- Modify: `analysis_finetune.py`

**Interfaces:**
- Consumes: `tools.panels.PanelContext`, `tools.panels.load_panel`, `tools.panels.run_panels`,
  `tools.panels.any_needs` (Task 2).
- Produces: nothing new for later tasks — this is the last task before cleanup.

- [ ] **Step 1: Add `--panel`/`--train` args and the new branch to `analysis_pretrain.py`**

Read the file first. In the `__main__` block, replace:
```python
    parser.add_argument('--subject',     type=int, default=None)
    parser.add_argument('--trial',       type=int, default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        overlay = json.load(f)
```
with:
```python
    parser.add_argument('--subject',     type=int, default=None)
    parser.add_argument('--trial',       type=int, default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    parser.add_argument('--panel',       action='append', default=[],
                         help='Run one or more panels (repeatable) instead of the legacy '
                              '--analysis path. See tools/panels/.')
    parser.add_argument('--train',       action='store_true',
                         help='(panel_profile only) profile in train mode (eigh skipped)')
    args = parser.parse_args()

    if args.panel:
        from tools.panels import PanelContext, run_panels, any_needs, load_panel

        for name in args.panel:
            if 'pretrain' not in load_panel(name).STAGES:
                parser.error(f"panel {name!r} does not support the pretrain stage")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if any_needs(args.panel, 'NEEDS_CHECKPOINT'):
            with open(args.config, 'r') as f:
                overlay = json.load(f)
            checkpoint = args.checkpoint or overlay.get('checkpoint', '')
            base_path = args.base_config or overlay.pop('base_config', None)
            if not base_path:
                model_dir = os.path.dirname(os.path.dirname(checkpoint))
                base_path = os.path.join(model_dir, 'artifacts', 'config.json')
            with open(base_path, 'r') as f:
                base = json.load(f)
            cfg = _deep_merge(base, overlay)
            for m, dsp in overlay.get('dataset_params', {}).items():
                cfg['dataset_params'][m] = dsp
            mdl = load_model(cfg, checkpoint, device, mode='pretrain')
            out_dir = resolve_output_dir(cfg, 'analysis', mode='pretrain')
        else:
            with open('config/config.json', 'r') as f:
                cfg = json.load(f)
            checkpoint, mdl = None, None
            out_dir = 'output/tools-profile'
        if any_needs(args.panel, 'NEEDS_DATASET'):
            raise NotImplementedError(
                "no NEEDS_DATASET=True panel exists yet (sub-project A only ships "
                "panel_profile, which needs neither) -- dataset resolution for panels "
                "is deferred to whichever future panel needs it")

        ctx = PanelContext(config=cfg, output_dir=out_dir, device=device, args=args,
                            model=mdl, dataset=None, checkpoint=checkpoint)
        run_panels(args.panel, 'pretrain', ctx)
        raise SystemExit(0)

    with open(args.config, 'r') as f:
        overlay = json.load(f)
```
(The trailing `with open(args.config, 'r') as f: overlay = json.load(f)` is intentionally
duplicated — it's the first line of the existing, untouched code path, now reached only
when `args.panel` is empty.)

- [ ] **Step 2: Smoke — `analysis_pretrain.py --panel profile` runs**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --panel profile
```
Expected: exit 0; stdout starts with `Profiling on device: ... | Mode: EVAL (eigh active)`,
then a `Detected Components:` block listing each top-level model component with its
parameter count in M, then a `Performance Summary` table with `Component | Params (M) |
Time (ms) | % Total` columns, ending in a `Total (Fwd + Loss + Rec)` row at `100.0%`.

- [ ] **Step 3: Smoke — old `--analysis` path still works, byte-identical**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --config config/analysis.json \
  --checkpoint output/pretrain/mesae_v10_all_share/checkpoint/last.pth \
  --analysis snapshot --dataset BNCI2014001 --subject 1 --trial 0
```
Expected: identical `[check] done: ...` output (same `recon_mse` etc.) as any prior run of
this exact command — this is the same smoke command used to verify the earlier
`check_model.py` → `analysis_pretrain.py` split; if this regresses, Task 1 or this task's
Step 1 broke something.

- [ ] **Step 4: Add `--panel`/`--train` args and the new branch to `analysis_finetune.py`**

Read the file first. Replace:
```python
    parser = argparse.ArgumentParser(description='Post-training EEG finetune checker (MeSAE)')
    parser.add_argument('--config',      required=True)
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        overlay = json.load(f)
```
with:
```python
    parser = argparse.ArgumentParser(description='Post-training EEG finetune checker (MeSAE)')
    parser.add_argument('--config',      default=None)
    parser.add_argument('--base-config', default=None, dest='base_config')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--dataset',     type=str, default=None)
    parser.add_argument('--recon_cmap',  type=str, default=None)
    parser.add_argument('--panel',       action='append', default=[],
                         help='Run one or more panels (repeatable) instead of the legacy '
                              'per-target snapshot path. See tools/panels/.')
    parser.add_argument('--train',       action='store_true',
                         help='(panel_profile only) profile in train mode (eigh skipped)')
    args = parser.parse_args()

    if args.panel:
        from tools.panels import PanelContext, run_panels, any_needs, load_panel

        for name in args.panel:
            if 'finetune' not in load_panel(name).STAGES:
                parser.error(f"panel {name!r} does not support the finetune stage")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if any_needs(args.panel, 'NEEDS_CHECKPOINT'):
            if not args.config:
                parser.error('--config is required unless every selected panel has '
                              'NEEDS_CHECKPOINT=False')
            with open(args.config, 'r') as f:
                overlay = json.load(f)
            checkpoint = args.checkpoint or overlay.get('checkpoint', '')
            base_path = args.base_config or overlay.pop('base_config', None)
            if not base_path:
                raise ValueError(
                    "analysis_finetune.py needs --base-config (or overlay['base_config']): "
                    "a finetune run's artifacts/config_<timestamp>.json has no fixed name "
                    "to guess.")
            with open(base_path, 'r') as f:
                base = json.load(f)
            cfg = _deep_merge(base, overlay)
            for m, dsp in overlay.get('dataset_params', {}).items():
                cfg['dataset_params'][m] = dsp
            mdl = load_model(cfg, checkpoint, device, mode='finetune')
            out_dir = resolve_output_dir(cfg, 'analysis', mode='finetune')
        else:
            with open('config/config.json', 'r') as f:
                cfg = json.load(f)
            checkpoint, mdl = None, None
            out_dir = 'output/tools-profile'
        if any_needs(args.panel, 'NEEDS_DATASET'):
            raise NotImplementedError(
                "no NEEDS_DATASET=True panel exists yet (sub-project A only ships "
                "panel_profile, which needs neither) -- dataset resolution for panels "
                "is deferred to whichever future panel needs it")

        ctx = PanelContext(config=cfg, output_dir=out_dir, device=device, args=args,
                            model=mdl, dataset=None, checkpoint=checkpoint)
        run_panels(args.panel, 'finetune', ctx)
        raise SystemExit(0)

    if not args.config:
        parser.error('--config is required')

    with open(args.config, 'r') as f:
        overlay = json.load(f)
```
(`--config` changes from `required=True` at the argparse level to a manual check after
parsing — it must still be required for the legacy path, just not when every selected
panel is checkpoint-free.)

- [ ] **Step 5: Smoke — `analysis_finetune.py --panel profile` runs**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py --panel profile
```
Expected: same printed table shape as before, exit 0 — proves `--panel` dispatch works
from both entrypoints via the shared `tools.panels` helper, not just `analysis_pretrain.py`.

- [ ] **Step 6: Smoke — old finetune path still works, byte-identical**

```bash
cat > /tmp/finetune_smoke_overlay.json <<'EOF'
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
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_finetune.py \
  --config /tmp/finetune_smoke_overlay.json \
  --base-config output/baseline/BNCI2014001_intra_raw_band/artifacts/config.json \
  --dataset BNCI2014001
```
Expected: exit 0, printed `[check] done: target=...` lines matching the shape from the
earlier `analysis_finetune.py` smoke test (per-class correct/wrong pairs) — same behavior
as before this task, since no `--panel` was given.

- [ ] **Step 7: `git diff --stat` sanity check**

```bash
git diff --stat analysis_pretrain.py analysis_finetune.py
```
Expected: moderate-size diffs (new argparse args + one new `if args.panel:` branch each) —
not full rewrites; every line outside the new branch should be unchanged.

- [ ] **Step 8: Commit**

```bash
git add analysis_pretrain.py analysis_finetune.py
git commit -m "feat: wire --panel into analysis_pretrain.py and analysis_finetune.py

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```

---

## Task 4: Retire `profile_model.py`

**Files:**
- Delete: `profile_model.py`
- Modify: `CLAUDE.md` (Commands section)

**Interfaces:** none (documentation/cleanup only).

- [ ] **Step 1: Delete `profile_model.py`**

```bash
git rm profile_model.py
```

- [ ] **Step 2: Update `CLAUDE.md`'s Commands section**

Read the file first. Replace:
```
# Profile model
python profile_model.py
```
with:
```
# Profile model (parameter counts + per-component forward-pass timing, no checkpoint/dataset needed)
python analysis_pretrain.py --panel profile [--train]
```

- [ ] **Step 3: Smoke — the deleted script's job still works via the panel**

```bash
/home/mamechin/anaconda3/envs/eeg_fm/bin/python analysis_pretrain.py --panel profile --train
python3 -c "import os; assert not os.path.exists('profile_model.py'); print('gone')"
```
Expected: the profile table prints (mode line reads `TRAIN (eigh skipped)`), then `gone`.

- [ ] **Step 4: Final repo-wide check**

```bash
git grep -n "profile_model\.py" -- '*.py' '*.md' '*.json' | grep -v docs/superpowers/specs | grep -v docs/superpowers/plans
```
Expected: no output — every live reference to the old script name is gone (historical
spec/plan docs are exempt, same convention as the earlier `check_model.py` retirement).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "refactor: retire profile_model.py, replaced by analysis_pretrain.py --panel profile

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18"
```
