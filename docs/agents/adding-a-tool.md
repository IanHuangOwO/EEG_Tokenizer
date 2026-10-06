# Adding a tool (panel / analysis function / viz function)

`tools/` is four sibling packages, one concern each:

- **`tools/analysis/`**: calculation. Anything that touches a model, a dataset, run outputs or a cached
  artifact and turns it into numbers or text (e.g. `backbone_eval.evaluate` writes its JSON,
  `backbone_report.report` returns markdown, `atom_dist.atom_similarity` returns a matrix).
- **`tools/viz/`**: pure rendering. Takes computed data and an output path, writes an image. No model or
  dataset access. (`tools/viz/extract.py` is computation despite its location, a pre-existing mismatch; new
  computation goes in `tools/analysis/`.)
- **`tools/panels/panel_<name>.py`**: the unit `analysis_pretrain.py` / `analysis_finetune.py` run. A thin
  hub: reads its inputs from `ctx`, calls `tools/analysis/` for numbers and `tools/viz/` for figures, prints
  one line per output file. No `matplotlib` calls and no forward passes in a panel file, even when nothing
  else calls the helper.
- **`tools/misc/`**: pipeline utilities run directly (`sweep`, `run_queue`, corpus / inventory builders),
  not analysis. A one-off analysis script that becomes routine graduates into a panel.

## Panel contract

```python
STAGES = frozenset({'pretrain'})   # or {'finetune'}: which entrypoint may run it
def run(ctx) -> None               # raise on failure; the runner prints it and continues
```

Discovery is by filename (`tools/panels/panel_*.py`), no registry. `ctx` is a
`tools.panels.PanelContext`:

- both stages: `out_dir` (write here), `args` (the entrypoint's argparse namespace), `device`, `cmap`.
- pretrain: `config` (the backbone's `artifacts/config.json` plus any `--config` overlay), `checkpoint`,
  `model` (built from the checkpoint's `build_config` on first access). A panel that needs data builds
  its own dataset inside `run()`, capped with `tools.analysis.cap_subjects` (loading every subject of a
  large dataset once ran out of memory).
- finetune: `groups` ({name: backbone}), `ref` (a group name or None), `head` (the finetune label: runs
  are `output/<backbone>/finetune/<head>/<cell>/`).

`PRESETS` in `tools/panels/__init__.py` lists what each entrypoint runs by default; add a panel there
only if it should run every time.

## Procedure

1. Calculation in `tools/analysis/<topic>.py` (extend an existing topic file when it fits:
   `atom_dist.py` for Q-atom statistics, `snapshot.py` for per-trial bundles).
2. Rendering in `tools/viz/<topic>.py` if there is a figure: `plot_<name>(out_path, data...)` that saves
   and closes. Never name a file `panels.py` (clashes with `tools/panels/`).
3. `tools/panels/panel_<name>.py`: docstring naming what it writes and its options, `STAGES`, `run(ctx)`.
4. Panel options: one `add_argument` in the entrypoint(s) whose stage it runs in, `help=` starting with
   the panel name; read back as `ctx.args.<dest>`.
5. Verify for real: `python analysis_pretrain.py --run <backbone> --panel <name>` (or the finetune
   equivalent) on outputs already on disk, and look at the file it wrote.
