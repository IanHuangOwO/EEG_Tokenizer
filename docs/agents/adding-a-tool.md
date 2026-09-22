# Adding a tool (panel / analysis function / viz function)

`tools/` is three sibling packages, one concern each:

- **`tools/analysis/`** — calculation. Anything that touches a model, a dataset, or a
  cached artifact and turns it into numbers. When a tool's whole output IS numbers/text/a
  file (not a figure), the analysis function does the printing/saving itself too — see
  `group_summary.py`'s `print_group_summary`, `select_eval_subsets.py`'s
  `select_eval_subsets`. Only produce a figure here if drawing it is inseparable from the
  computation; the normal split is analysis returns data, viz draws it.
- **`tools/viz/`** — pure rendering. Takes already-computed data + an output path, writes
  an image file (matplotlib). No model/dataset access, no numeric derivation beyond
  display formatting (log-scaling, axis ticks). Exception you'll run into:
  `tools/viz/extract.py` is actually computation (pulls model activations, builds PSD/
  topo arrays) despite living under `viz/` — a pre-existing naming mismatch from before
  this convention was written down. Don't copy it: new computation code belongs in
  `tools/analysis/`, even if its output feeds a viz function.
- **`tools/panels/panel_<name>.py`** — the CLI-reachable entrypoint. A thin hub: calls
  into `tools/analysis/` for numbers, `tools/viz/` for figures, and does nothing else.
  See "Panel contract" below.

**Rule of thumb:** if you're about to write a `def _cell(...)` or a `matplotlib.pyplot`
call inside a `panel_*.py` file, stop — that belongs in `tools/viz/`. If you're about to
write a `model(...)` forward pass or a `build_dataset_from_config` call inside a
`panel_*.py` file that already has `NEEDS_CHECKPOINT`/`NEEDS_DATASET` covering it, stop —
that belongs in `tools/analysis/`. A panel with exactly one caller for some helper
function does **not** justify inlining that helper into the panel file — panels stay
thin even when nothing else calls the helper (see this branch's history: two rendering
functions were briefly inlined into their panel files for having a single caller, then
reverted — the win from removing one import doesn't outweigh a consistent hub/helper
split across every panel).

## Panel contract

A panel is a module at `tools/panels/panel_<name>.py` exposing:

```python
STAGES: frozenset[str]     # subset of {'pretrain', 'finetune'} -- which analysis_*.py
                            # script(s) may run it
NEEDS_CHECKPOINT: bool      # whether the CLI must resolve a model before calling run()
NEEDS_DATASET: bool         # whether the CLI must resolve a ctx.bundle before calling
                            # run() -- see the caveat below, this is narrower than it sounds

def run(ctx) -> None        # does the work; panels are terminal actions, not composable
```

Discovery is filename convention only (`tools/panels/__init__.py`'s
`discover_panel_names`/`load_panel`, via `glob.glob('tools/panels/panel_*.py')`) — no
registry, no decorator. Adding a panel is adding a file; removing one is deleting a file.

`ctx` is a `tools.panels.PanelContext`: `config`, `output_dir`, `device`, `args`
(the CLI's `argparse.Namespace`, for panel-specific flags), and optionally `model`,
`dataset`, `checkpoint`, `bundle` (a `tools.analysis.snapshot.SnapshotBundle`), `cmap`.

**`NEEDS_DATASET=True` currently means "needs `ctx.bundle`"** — only the legacy
`--analysis`/finetune-per-target path builds one today (via
`build_pretrain_bundle`/`build_finetune_bundle`, see `tools/analysis/snapshot.py`); a
`NEEDS_DATASET=True` panel picked via `--panel` raises `NotImplementedError` naming
itself (`tools/panels/__init__.py`'s `build_panel_context`). If your panel needs a whole
dataset but not a per-trial bundle (no correct/wrong-trial picking, no reconstruction),
**build it yourself inside `run()`** and declare `NEEDS_CHECKPOINT=NEEDS_DATASET=False`
— see `panel_profile.py` (builds its own fresh model) and
`panel_select_eval_subsets.py` (builds its own dataset via
`build_dataset_from_config`) for the pattern.

## Procedure

1. **Decide what the panel needs**, and set `NEEDS_CHECKPOINT`/`NEEDS_DATASET`
   accordingly (see the caveat above — most new panels that touch a dataset will want
   `False`/`False` plus their own self-contained build inside `run()`, not `True`/`True`).
2. **Write the calculation** in `tools/analysis/<topic>.py` — a new file per topic unless
   this is a natural extension of an existing one (`snapshot.py` for
   per-trial bundles, `profile.py` for model profiling). If the tool's output is
   print/CSV/JSON rather than a figure, write the save/print at the end of this function
   — no viz file needed for non-image output.
3. **Write the rendering** in `tools/viz/<topic>.py`, if the tool produces a figure: a
   pure `plot_<name>(out_path, ...data..., ...) -> None` that saves and closes the
   figure. Reuse an existing viz file if the new plot is a close relative (another
   `stamp_plots.py`-style function); otherwise start a new one.
4. **Write `tools/panels/panel_<name>.py`**: a docstring naming which analysis/viz
   functions it calls and which `ctx` fields it reads, the three contract attributes,
   and a `run(ctx)` that reads inputs from `ctx` (+ `ctx.args` for panel-specific flags),
   calls the analysis function, calls the viz function if there is one, and prints one
   status line (`[panel] -> <path>` or similar — match the existing panels' style).
5. **Wire any panel-specific CLI flags** into whichever of `analysis_pretrain.py`/
   `analysis_finetune.py`'s `argparse` block(s) apply (both, if `STAGES` covers both
   scripts) — one `add_argument` per flag, with a `help=` comment naming which panel it's
   for (see `--train`, `--group-eval`, `--se-datasets`/`--se-run-config`/`--se-out-dir`
   for the pattern). The panel reads them back via `ctx.args.<dest>`.
6. **Never name a new `tools/viz/` or `tools/analysis/` file `panels.py`** (or anything
   else that collides with the `tools/panels/` directory name) — `tools/viz/panels.py`
   was renamed to `tools/viz/stamp_plots.py` this session for exactly this confusion.
7. **Verify for real** — no test suite exists in this repo. Run
   `python analysis_pretrain.py --panel <name> [flags...]` or the `analysis_finetune.py`
   equivalent against a real checkpoint/dataset already on disk, and confirm the output
   file or printed numbers look right. For a panel with no real precedent to compare
   against, sanity-check shapes/ranges by hand once.

## Example: a hypothetical `panel_channel_impedance` (no checkpoint, no dataset)

```python
# tools/analysis/impedance.py
def summarize_impedance(dataset_path) -> dict: ...   # reads metadata.json, returns stats

# tools/panels/panel_channel_impedance.py
"""channel_impedance panel: per-channel impedance summary from a dataset's metadata.json
(tools/analysis/impedance.py's summarize_impedance). No checkpoint/dataset build needed."""
from tools.analysis.impedance import summarize_impedance

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False

def run(ctx):
    path = ctx.args.dataset_path   # wired into analysis_pretrain.py's argparse
    stats = summarize_impedance(path)
    for ch, v in stats.items():
        print(f'  {ch}: {v:.1f} kOhm')
```
