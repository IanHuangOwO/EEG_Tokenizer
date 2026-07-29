# Model-specific training/viz logic moves behind BaseTrainer/BaseEpochChecker/BasePlotter

Adding a model (MeFSQ, MeSAE, and whatever comes next) currently means touching four
separate `if model_type == ...` dispatch sites: `model/factory.py`'s build function,
`train_pretrain.py`'s `_step_loss`/`_epoch_metrics`/stage1→stage2 transition, the merged
`check_epoch_pretrain.py`/`check_epoch_finetune.py` (hasattr dispatch), and `viz/train.py`'s
`Plotter` (two disconnected methods, `plot_all` vs `plot_tokenizer` — the latter is
currently dead code, never called). Each new model requires editing all four scattered
locations, and nothing enforces that they stay in sync (the dead-code dashboard bug is a
symptom of that).

Decided to consolidate all four into three base classes (`model/plugin_base.py`):
`BaseTrainer`, `BaseEpochChecker`, `BasePlotter` (kept as three separate classes, not one
bundle, since they're used at three different call sites/lifecycles). Each model dir gets
one `plugin.py` (e.g. `model/MeFSQ/plugin.py`) implementing all three for that model.
`model/factory.py` gains a `MODEL_REGISTRY` dict mapping `model_type` -> plugin classes,
replacing the if/elif build dispatch too.

Key shape decisions within this:

- **`BaseEpochChecker` uses template method, not full override.** The panel-building flow
  (recon_signal / topo_psd_filter / attn_topo) is written once in the base class; subclasses
  only implement extraction hooks (`extract_psd`, `extract_spectra`). This is a deliberate
  reversal of the *previous* decision (see the merged `check_epoch_*.py` docstrings) to keep
  MeFSQ and MeSAE in one shared script specifically to prevent the two panel formats
  drifting apart. Template method preserves that guarantee (format is literally one method
  body) while still letting each model own its file and dispatch logic explicitly instead of
  via `hasattr` sniffing.
- **`BasePlotter` renders from a declarative panel spec**, not hardcoded per-model plotting
  code. A panel is `{title, ylabel, series: [{key, label, color, style, band?}], twin?}`;
  `pool_pair` (routed/shared metric variants) and `indexed_family` (unbounded `skip_gate_N`
  series) both reduce to "more series in the list" rather than needing their own kind.
  Rejected a richer typed-kind enum (`line`/`pool_pair`/`twin_axis`/`std_band`/
  `indexed_family`) as unnecessary — one generic series-list renderer covers every panel
  currently in `plot_all`/`plot_tokenizer`.
- **`BaseTrainer.compute_loss` takes `**hparams`** sourced directly from
  `config.model_params[model_type].pretrain.loss`, not explicit named kwargs
  (`load_balance_weight`, `aux_weight`, ...). A new model's loss hyperparameter needs no
  change to the shared training loop's call signature.
- **Pretrain/finetune variants are explicit methods** (`plot_pretrain`/`plot_finetune`,
  `check_pretrain`/`check_finetune`), not a `mode='pretrain'|'finetune'` string flag —
  matches "plug and play" call sites picking a method directly.

## Consequences

- `viz/check_epoch_pretrain.py`, `viz/check_epoch_finetune.py`, and `viz/train.py`'s
  `Plotter` are retired; their logic is redistributed into `model/plugin_base.py` (shared
  template/renderer) and each model's `plugin.py` (hooks/specs).
- Adding a third model means: implement one `plugin.py` (Trainer + Checker + Plotter hooks),
  register it in `MODEL_REGISTRY`. No shared file needs editing for the model to work.
- The dead `plot_tokenizer` dashboard bug is fixed as a byproduct — `MODEL_REGISTRY` makes
  "which dashboard renders" an explicit lookup instead of a call site (`train_pretrain.py`)
  that never branched on `model_type` in the first place.
