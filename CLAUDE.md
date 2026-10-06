# CLAUDE.md

Guidance for Claude Code in this repository. Canonical terms: `CONTEXT.md`. Design history: `docs/adr/` (decisions),
`docs/cards/` (pre-registered experiments), `docs/reports/` (results) -- kept locally, not tracked in git.

## Environment

Run everything with the `qtome` conda env: `/home/mamechin/anaconda3/envs/qtome/bin/python` (torch, mne,
moabb). A bare `python` is `base`, which has no `mne`/`moabb`: coordinates silently fall back to flat polar
values from `metadata.json` and MOABB loaders fail. The machine is CPU-bound (20 cores, one 12 GB GPU):
cap threads (`--threads`, `training_params.<mode>.num_threads`). Parallel jobs: 3 for unattended (overnight) queues,
2 while the user is working. Measured 2026-10-02 on tiny pretrains: 3 jobs give ~10% more total throughput than 2
(tokenizer epoch 127 s vs 93 s per job) but each job runs ~35-40% slower; three pretrains use ~8 of 12 GB GPU memory.

## Commands

```bash
# Compile raw datasets into per-subject .npz caches (bandpass + resample baked in; verification runs after;
# --dataset NAME narrows it, --verify-only / --deep for checks). See docs/agents/adding-a-dataset.md.
python cache_dataset.py --config configs/compile.json

# Pretrain: one run, tokenizer phase then masked phase (docs/adr/0013)
python train_pretrain.py --config configs/runs/<backbone>/pretrain.json

# Finetune a head on the frozen backbone (atom-code cache built on first use by cache_feature.py)
python train_finetune.py --config configs/runs/<backbone>/finetune/<head>/<cell>.json

# From-scratch baseline (model/EEGNet/, settings configs/EEGNet/settings.json, Compass-checked) on one cell, same
# splits and group_eval.json as a Qtome cell -> output/EEGNet/finetune/compass/<dataset>_<mode>_seed<k>/
python train_baseline.py --model EEGNet --dataset BNCI2014004 --protocol mi_fewshot

# Any config value can be overridden (dotted path, JSON value, repeatable; artifacts/config.json records it)
python train_finetune.py --config <cfg> --set training_params.finetune.learning_rate=0.003

# Experiments: a sweep file (base config + cases x grid, optional "backbones") -> queue plan; run_queue runs it
# resumably (state in output/queue/<plan>/: .done/.failed/.log per job; plan lines 'job <name> :: <cmd>' / 'wait')
python -m tools.misc.sweep configs/Qtome/sweeps/<sweep>.json > output/queue/<plan>.plan
python -m tools.misc.run_queue output/queue/<plan>.plan --max-parallel 2 --threads 8

# Analysis: one panel mechanism (tools/panels/, presets in tools/panels/__init__.py PRESETS), failures don't stop
# the other panels. Pretrain = one backbone -> output/Qtome/<backbone>/pretrain/analysis/ (standard: backbone_eval,
# attention_range, atom_usage, ridge_probe, atom_vs_raw, atom_templates, atom_duplicates, atom_distribution,
# snapshot, codebook; quick: the first seven; on request: coord_robustness, atom_maps, probe_maps). Judge each
# change by its own mechanism metric, not finetuning (ADR 0020).
python analysis_pretrain.py --run <backbone> [--preset quick] [--panel <name> ...]
python analysis_pretrain.py --panel profile [--train]     # parameter counts + timing, no checkpoint
# Finetune = several backbones under one head label -> output/analysis/<groups>/ (standard: summary, report,
# time_weights; also seed_equivalence, class_snapshots --checkpoint <head.pth>). Run backbone_eval first.
python analysis_finetune.py --group base=<backbone> --group X=<backbone> --ref base [--head frozen_learned]
python -m tools.analysis.summarize_runs 'output/Qtome/<backbone>/finetune/<head>/*' [--ref <backbone>:<head>] [--rank N]
```

No test suite: modules carry runnable self-checks (e.g. `Qtome_modules._selfcheck_head_modules()`), and
validation runs during training. Refactors here are verified by reproducing recorded results bit-identically.

### Long-running jobs: always monitor exit and errors

Past failures: a watcher waiting on a log line that never came sat silent for 7 h and blocked the step
chained behind it; a finished queue went unreported for 40 min.
- Launch so the exit is noticed: an agent uses the tool's background mode (exit notification), not a detached
  `nohup ... &`. If detaching is unavoidable, add a background watcher on the PID.
- Wait on a process exiting (`while kill -0 $PID`) or a file existing, never on a specific log line.
- Every job records its exit code; a chained step checks the previous step's status.
- Report a job's end right away (success, or failure + last error lines). On a status check, list anything
  that died or went quiet.
- Once a queue has finished and its outcome is reported, delete its state (`output/queue/<plan>/`, `.plan`,
  `.log`): results live in each run's own output dir. Keep it only while a job failed and needs a rerun.

## Configs

Configs are grouped by owner: shared ones at the top of `configs/`, model-specific ones in `configs/<Model>/`.
`configs/Qtome/pretrain.template.json` / `configs/Qtome/finetune.template.json` are starting points, never
run directly: copy to `configs/runs/<backbone>/pretrain.json` and `configs/runs/<backbone>/finetune/<head>/<cell>.json`
(a finetune overlay sets `base_config` to its backbone's pretrain file; `load_config` deep-merges them).
`configs/runs/` is gitignored; every run's effective config is saved in its `artifacts/config.json`.
The corpus list is the same for every size (generated by `tools/misc/build_pretrain_corpus.py`); the size is one
setting, `preprocess_params.window_fraction`: tiny 0.05 / small 0.2 / medium 0.5 / large 1.0 (the template's value)
of every subject's windows, nested at one `window_fraction_seed`. Set it in the run config or with `--set`.
Shared: `configs/compile.json` (compile params + dataset list), `configs/montages.json` (canonical and
sub-montages), `configs/finetune_protocols.json` (evaluation splits per protocol, every model). Qtome:
`configs/Qtome/` (templates, `protocol_heads.json` = its head and optimiser per protocol, `sweeps/`). EEGNet:
`configs/EEGNet/settings.json`. See `configs/README.md`.

Key fields (templates show defaults):
- `model_params.Qtome.pretrain`: `patch_len`, `embed_dim`, `enc_depth`, `blocks_per_stage` (2), `skip_mode`
  (`gated` UNet skips / `finest` only the finest skip / `none`), `skip_drop` (per-sample skip drop-path probability; a list = one per skip, finest first), `decoder_blocks`
  (per-channel temporal conv blocks after each upsample, no channel mixing),
  `temporal_bias` (true, the templates' default since 2026-10-06: a learned bias on the signed time lag in every block's temporal attention; code default false),
  `spatial_heads`, `moe_ffn`, `atom_bank`, `loss`, `spatial_embedding` (true: Fourier electrode-coordinate
  embedding + per-block directional relative spatial bias; false: neither -- the spatial ablation).
- `preprocess_params`: `canonical_channels` (a `montages.json` name or a list), `channel_layout` (`native`, the
  templates' default: every EEG channel of the dataset, non-10-10 ones in free slots, > 64 channels reduced to the 64
  sites; `grid`: 10-10 names only -- the code default, so configs without the key reproduce; docs/adr/0023), `coords`
  (`template`, the default: MNE's position for the channel name; `recorded`: the dataset's own positions, aligned --
  tested and rejected; old values `real` / `dataset` still accepted), `window_length`,
  `window_min_real`, `window_fraction`, `patch_length` 50, `patch_stride` 25 (50% overlap), `sample_freq` 200,
  `bandpass_filter`, `normalization_type`, and `mask` (`IO/masking.py`): `masking_strategy` `mixture` (one
  MaskMode per window -- channel_cluster / random_channel / time_block / random_token -- on a shared ratio
  ramp; the default) or `random` (the masking baseline); masks redrawn every masked epoch; `time_run` masks
  runs of >= 3 patches (a lone patch leaks through the overlap); `time_block` `max_blocks` 2 (templates since 2026-10-05; the
  v2 backbones predate it): 1-2 holes splitting the ratio at random cut points, each >= 3 patches; `subsample` removes channels down to a
  `montages.json` sub-montage for part of the dense-cap windows (`prob` 0.1 in the templates since 2026-10-06, the
  user's choice without a test; earlier backbones used 0.2). Draw probabilities of the three masking methods: time
  blocks 0.30, random channels 0.35, channel cluster 0.35.
- `training_params.pretrain`: `model_name`, `output_path`, `epochs`, `tokenizer_epochs`, `freeze_atoms`,
  `warmup_epochs`, `batch_size`, LR fields, `train_val_split`, `seed`.
- `model_params.Qtome.finetune` (the head, validated at build): `features` = a list of `{"type": <entry>,
  <per-entry keys>}`; entries are classes in `Qtome_modules.py`'s `ENTRY_TYPES` (`atom_power`, `atom_band`,
  `signed_ab`, `evoked`, `phase_advance`, `raw_band`, `raw_signal`, `latent_power`, `latent_signed`) -- a new
  head feature is one class + one registry line. Every entry has its own spatial filter. Per-entry keys
  (`spatial_k`, `time_pool`, `time_rank`, `window`, `evoked_rank`, `atom_rank`, `latent_proj`) may also be set
  at the top level as defaults; plus `dropout`. Defaults: `_HEAD_DEFAULTS`.
- `training_params.finetune`: `pretrained_checkpoint`, `protocol` (a `configs/finetune_protocols.json` split + its `configs/Qtome/protocol_heads.json` head:
  mi_loso / mi_fewshot / p300_loso / p300_fewshot, tuned on DEV sets BNCI2015001 / BNCI2014009 only; each sets
  the combined head; applied over the merged config, so a head in the run config is overridden -- `--set` still wins), `split` = `{"type": ...}` from `train_finetune.py`'s `SPLITS`:
  `loso`, `subject_kfold` (`n_folds`), `eval_subjects`, `kfold`, `blocked_kfold`, `fewshot` (`train_fraction`,
  EEG-FM-Compass calibration); all take `sessions` and `seed`, per-subject types also `purge` (P300 overlap);
  unknown keys are rejected. Plus LR fields, `epochs`, `class_weight` (`balanced`), `batch_size`, `seed`.
  `fit`: `sgd` (default) or `closed_form` (`model/Qtome/closed_form.py`, no SGD). Few-shot always uses closed-form with
  branch `structured`: the all-atom head's own factors (spatial filter x Q-atom weights x time course) set in closed
  form, one head for every paradigm, signed half at full time resolution (the protocols set it;
  docs/reports/2026-10-06-structured-fewshot-head.md). Older branches `power` / `signed` / `trca` stay for comparison.
  Loso uses SGD (closed-form loses there). SSVEP DEV sets: Kalunga2016 (not phase-locked), Wang2016_dev (phase-locked,
  in the pretraining corpus as unlabelled windows).

## Who owns what

| Owner | Code | Configs |
|---|---|---|
| shared (every model) | `cache_dataset.py`, `datas/`, `IO/` (incl. `IO/splits.py`: protocol splits, `make_runs`), `train_pretrain.py` (dispatches through `model/factory.py`'s plugin registry), `model/base_*.py`, `model/factory.py`, `tools/misc/` | `compile.json`, `montages.json`, `finetune_protocols.json` |
| Qtome | `model/Qtome/`, `train_finetune.py` (frozen backbone + head), `cache_feature.py` (atom cache), `analysis_pretrain.py`, `analysis_finetune.py`, `tools/analysis/`, `tools/panels/`, `tools/viz/` | `configs/Qtome/` |
| from-scratch baselines | `model/EEGNet/`, `train_baseline.py` | `configs/EEGNet/` |

A new pretrained backbone follows `docs/agents/adding-a-model.md` (plugin); a new from-scratch baseline is
`model/<Name>/<Name>.py` + a line in `train_baseline.py`'s `BASELINES` + `configs/<Name>/settings.json`.

## Architecture

**Qtome**: an EEG tokenizer. A TSA encoder feeds a static Q-atom dictionary (AtomBank: every Q-atom active at
every patch; each reconstructs a patch as `a*D + b*H`, D a template and H its quadrature partner), trained by
masked reconstruction. Routed (top-k) Q-atoms were removed (docs/adr/0022; code on the `routed-stamps`
branch). The default downstream head reads only the Q-atoms: Q-atom power (`atom_power`, induced band power: MI) and the
signed Q-atom gains (`signed_ab`: phase-locked / P300). The stronger head swaps `signed_ab` for the signed encoder
output z (`latent_signed`, PCA): level on loso, better on MI few-shot (BNCI2014004 +3.5)
(docs/reports/2026-10-02-head-ablation.md, docs/reports/2026-09-29-combined-head.md). Plugged in via `model/Qtome/plugin.py` (`model/factory.py`
`MODEL_REGISTRY`, docs/adr/0004).

```
datas/<split>/<Name>/loader.py  compile time only; MOABB datasets use IO/loader.py's MoabbLoader
 └ cache_dataset.py    bandpass + resample each CONTINUOUS recording (BaseSubjectLoader._filter_run),
 │                     then cut epochs (event-anchored [event-1 s, event+4 s) for pretraining; the reported finetune
 │                     sets use EEG-FM-Compass post-event windows via metadata moabb.onset_window/window, or fixed windows); drops
 │                     flat-line dropout windows -> datas/<split>/<Name>/cache/*.npz. BETA_3s/4s ship
 │                     pre-epoched and are filtered per epoch.
 └ IO/dataset.py       EEGDataset maps channels onto the 64 slots (channel_layout native / grid; missing -> zero
 │                     padding, valid_channels marks real ones), normalises per trial; PretrainDataset cuts
 │                     windows -> patches and draws masks; MontageBatchSampler makes one-montage batches
 └ train_pretrain.py   tokenizer phase (every block, temporal attention only, unmasked) -> masked phase
                       (spatial attention + coordinate embedding on, mask curriculum starts)
```

- `Qtome_modules.py`: `SpatialTemporalEmbeddings`, `RelativeSpatialBias`, `TSABlock` (temporal attention ->
  spatial attention -> MoE FFN, LayerScale), `TSAEncoder` (stages of `blocks_per_stage` blocks; patch axis
  pooled by 2 between stages with a centred [1,3,3,1]/8 kernel, linear-interpolation upsample, gated skips:
  8 blocks = 4 stages, 39 -> 20 -> 10 -> 5 patches; padded channels and padded tail patches are masked out of
  attention), `AtomBank`, and the finetune side: `AtomExtractor` (Q-atom codes and optionally z from the
  frozen backbone) and `FeatureHead` (the `ENTRY_TYPES` registry).
- `Qtome.py`: `QtomePretrain` (phases, per-sample masked loss: a sample counts as masked only if every patch
  covering it is masked), `FinetuneModel`.
- `model/EEGNet/EEGNet.py`: the EEGNet baseline (Compass's architecture), trained from scratch by `train_baseline.py`;
  not a plugin (no backbone). Finetune caches store volts: raw-input baselines multiply by 1e6.
- `factory.py`: a pretrain checkpoint stores its `build_config`; `build_from_checkpoint` rebuilds a trained
  backbone from it, never from the editable run config.
- `train_pretrain.py`: subject split by `IO/dataset.py`'s `split_pretrain_subjects` -- person-disjoint per
  cohort (`metadata.json` `data_metadata.cohort`) and independent of dataset order; a subject's near-flat
  channels (std < 0.10 x median, dead electrodes / the recording reference) are treated as padding.

**Sparsity budget, a hard ceiling:** each active Q-atom gives two free scalars per channel, so keep
`2 * n_atoms < patch_len` with margin. Past it, the active slots fit any patch
regardless of the templates and it stops being sparse coding (measured at DOF 56 > 50: recon MSE ~0 on every
dataset, kurtosis 6.7 -> 1.2). docs/adr/0011.

## Outputs

Each model has its own output folder: a run writes to `output/<model_type>/<training_params.<mode>.output_path>`
(default `<model_name>/pretrain` for pretrain; `model_type` defaults to Qtome; `tools/analysis` QTOME_OUTPUT). EEGNet:
`output/EEGNet/finetune/compass/<dataset>_<mode>_seed<k>/`.
Per Qtome backbone: `output/Qtome/<backbone>/pretrain/` (`checkpoint/last.pth` -- prefer it over `best.pth`, which locks
onto an easy epoch of the mask curriculum; `artifacts/config.json`; `visualization/`; `analysis/`;
`feature_cache/` -- regenerable, keyed by checkpoint, build_config, code hash and data fingerprint) and
`output/Qtome/<backbone>/finetune/<head>/<cell>/` (`artifacts/group_eval.json`: per-subject `tail` = mean of the last
10 epochs' balanced accuracy, `last`, kappa). `output/analysis/` holds generated multi-backbone analysis (regenerable),
`output/queue/` queue state, `output/Qtome/archive/<date>_<topic>/` superseded experiments. Once a result is final, write it up
in `docs/reports/` (fixed format and index in `docs/reports/README.md`; copy the figures, tables and
pipeline scripts it cites) and delete the generated `output/analysis/` and queue folders.

## Data

`datas/<split>/<name>/metadata.json`: `data_metadata` (`acquisition.sample_frequency`, 1-indexed `channels`
with labels and, where the dataset ships them, its own positions -- polar `coordinates` with
`polar_equator_radius`, or `xyz`; per-subject `channel_xyz` in `data_structure` -- read only by `coords: recorded`;
`event_onset_seconds`, `moabb` class/kwargs/window, optional `cohort`) and
`data_structure` (per-subject files or `moabb_subject`). `datas/DATASETS.md` lists every dataset; regenerate
with `python -m tools.misc.dataset_inventory`. Read `docs/finetune-caveats.md` before reporting finetune numbers.

## Agent docs

- Issues: GitHub Issues (IanHuangOwO/Qtome) via `gh`; labels `needs-triage` / `needs-info` /
  `ready-for-agent` / `ready-for-human` / `wontfix` -- `docs/agents/issue-tracker.md`, `triage-labels.md`.
- Adding a dataset / model / montage / tool: `docs/agents/adding-a-*.md`. Raw downloads (NEMAR / MOABB, resumable,
  checksummed): `python -m tools.misc.fetch_datasets`. `tools/` = `analysis/`
  (calculation), `viz/` (rendering), `panels/` (the analysis entrypoints' units); `tools/misc/` = pipeline utilities run directly.
- `.reshape(`/`.view(` silently scrambles data when it merges non-adjacent axes (it has hit training data
  three times): check any new one against `docs/agents/reshape-pitfalls.md`.
