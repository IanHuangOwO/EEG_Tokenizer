# Finetune restructure: one feature head, feature cache, two split modes, experiment specs

Date: 2026-09-21. Status: design for review (revised after the structure check). Builds on
ADR 0016 (head modules) and ADR 0014 (test plan and Protocol).

## Problem

The finetune ablation work is spread thin.
- Adding a head option touches the head class flags, the factory, the config and a config
  generator. Numbers are encoded inside strings (`pool_channel: "spatial:8"`, `pool_time:
  "learned:2"`), which makes a sweep over K or R awkward.
- One class (`MeSAEFeatureHead`) serves several heads through `input` and `task` flags whose
  combinations are hard-coded (for example learned time weights only exist for one feature),
  and `viz.load_model` guesses `num_classes` and `num_patches` from tensor shapes.
- The backbone is frozen, yet every epoch of every ablation reruns it. The raw control costs
  0.71x of C1 only because the backbone still runs.
- `train_finetune.py` (795 lines) mixes batching and metrics, dataset and split building, the
  training loop, five split runners and reporting, plus optimistic best-validation checkpoints,
  `recon_mse` logging and freeze options that no reported number uses.
- The analysis scripts in `probes/` overlap (three summary scripts, two dead Experiment A
  probes, a config generator). Run folders and environment notes are assembled by hand, which
  let a missing `mne` in the `base` env go unnoticed.

## Decisions (agreed in the design discussion)

1. **Four sub-projects, in order, each with its own plan and review.** A: model. B: feature
   cache. C: `train_finetune.py`. D: experiments and results tooling. Files keep their names
   except where stated; the only new folder is `experiments/`.
2. **No compatibility or reproducibility promise for old runs** (user decision). The finished
   runs stay in `output/` as the record of why the head was chosen; they are not maintained.
   The last commit with the removed code is tagged `pre-head-cleanup`.
3. **`train_finetune.py` stays one file, with two split modes** and a slimmer loop.
4. **One experiment spec file per ablation set and one runner.** No manifest, exit-status or
   timing tracking: the spec and each run's saved config show what was run. One exception: a
   small environment stamp in each run's saved config.
5. **`probes/` is dissolved into `experiments/`.**
6. **Finetune-run viz is removed from the training loop for now** (training on cached
   features has no raw signal or backbone at hand). Head diagnostics (spatial-filter
   topomaps, time-weight maps) come back with the separate viz refactor.

## Sub-project A: model

Replaces `MeSAEFeatureHead`, `MeSAEFinetune` and `PerChannelHeadAttn`. This absorbs Task 3 of
`docs/superpowers/plans/2026-09-21-head-modules-refactor.md`; Task 2 (the head pieces in
`MeSAE_modules.py`) has landed.

- **Feature extractor** `StampExtractor(backbone)`: `forward(x, coords, time_idx,
  valid_channels) -> amp [B, N', C_valid, S, 2]` (the stamp code (a, b) scaled by patch RMS,
  alive routed and shared stamps only, padded channels dropped). It also exposes the `keep`
  stamp indices and the template spectra tables. It owns everything that needs the backbone.
- **One `FeatureHead` class** (no backbone inside) composed from swappable slots, pipeline
  `feature front-end -> spatial filter -> time pooling -> (optional branches) -> readout`:
  - `feature` (front-end): `stamp_power` (per-stamp log power, K x S features), `stamp_band`
    (template mu/beta energy of the stamp code, the old `stamp_bandpow`), `raw_band` (mu/beta
    power of the raw signal, computed per 50-sample patch so it has the same time axis as the
    stamp features), `raw_signal` (signed time samples averaged to ~20 Hz, the ERP baseline).
    `stamp_*` read the extractor's amplitudes; `raw_*` read the patched raw signal.
  - `spatial_k`: number of signed spatial filters over the real channels; `null`/`0` disables the mixing
    (channel concat, each real channel its own feature row) — an ADR 0016 ablation control, not a candidate.
  - `time_pool`: `flat`, `learned` (weights per feature dimension, rank `time_rank`),
    `window` (`[lo, hi]` seconds, then flat), or `none` (keep the patch axis as features).
  - `phase_advance` and `evoked_rank` branches, on `stamp_power` only.
  - readout `BatchNorm1d -> Dropout -> Linear`, `dropout` configurable.
  The backbone-dependent part is `StampExtractor`; `FinetuneModel` prepares the input for the
  chosen feature. Any combination not excluded by the rules in the plan works, for example
  band power with a spatial filter and learned time weights.
- **Head config, numeric keys** (`model_params.MeSAE.finetune`): `feature`, `spatial_k`,
  `time_pool`, `time_rank`, `window`, `phase_advance`, `evoked_rank`, `dropout`. `task` is no
  longer a model argument. Invalid combinations (for example `raw_signal` with anything but
  `time_pool: none`, branches on a non-`stamp_power` feature) raise a `ValueError` naming the
  keys.
- **Checkpoint format:** `{"model_state_dict": <head only>, "head_config": {resolved keys plus
  num_classes, num_patches, num_channels, num_stamps, keep}, "backbone_checkpoint": <path>}`.
  Loading builds the head from `head_config`; no shape inference. Only the head is saved
  (the backbone is frozen and loaded from `backbone_checkpoint`).
- **Kept unchanged:** every `MeSAEPretrain` method; the ADR 0016 pieces (`spatial_mix`,
  `FlatTimePool`, `LearnedTimePool`, `EvokedBranch`, `phase_advance`) inside
  `MeSAE_modules.py` under the finetune section header. Spatial mix now runs over the dataset's
  real channels only (`spatial_k` filters over `C_valid`).
- **Removed:** `MeSAEFinetune`, `PerChannelHeadAttn`, `head_z`, the `recon` and `z_chan`
  inputs with `z_proj`, channel-concat pooling, `render_finetune_attn` and the dead
  `check_finetune` branches, `freeze_backbone`.
- **Verification:** for each kept head configuration, `StampExtractor + FeatureHead` matches the
  previous `MeSAEFeatureHead` (taken from tag `pre-head-cleanup`) numerically when the new head
  is given the same weights, copied across by name and restricted to the real channels (the old padded channels were zero, so dropping them changes no output) (logits and gradients, atol 1e-6, on a fixed input); `raw_band` uses a new per-patch estimator, so it is checked against an explicit per-patch
  FFT reference instead of the old whole-trial estimator; a checkpoint save/load round trip
  reproduces logits; the base config builds.

## Sub-project B: feature cache

Because the backbone is frozen, the stamp amplitudes of every trial are computed once and the
head is trained on them.

- **`cache_feature.py`** (repo root, a pipeline stage like `cache_dataset.py`: it builds datasets
  and runs the backbone over them, which `model/` does not do):
  `get_stamp_cache(config, dataset_name, subjects) -> path`. Stores per subject `amp` fp16
  `[n_trials, N', C_valid, S, 2]`, `labels`, `valid_length`, under
  `<backbone run folder>/feature_cache/<dataset>/<key>/<subject>.npz`, where the backbone run
  folder is the parent of the checkpoint's `checkpoint/` directory (today
  `output/pretrain/mesae_v10_small/`, later `output/pretrain/mesae_v10_small/`). The cache
  travels with the model it came from. Raw heads do not use the cache (they read the patched
  raw signal from the dataset, as today, and never run the backbone).
- **Cache key:** the folder key hashes the checkpoint file identity (name, size, mtime), the
  `keep` stamp set, the whole `preprocess_params` block, the dataset entry (minus
  `subject_to_use`), the `metadata.json` and `config/montages.json` fingerprints and the `mne`
  version (electrode coordinates depend on it). Each subject file additionally stores the
  fingerprint of the compiled data file it was built from and is rebuilt when it changes, so
  adding subjects never invalidates existing files. Over-invalidation (for example a changed
  masking setting) only costs a rebuild.
- **Uniform channels:** `CachedStampDataset` asserts that all subjects share `channel_idx` and
  `keep`; only the real channels are stored.
- **Size:** about 0.25 MB per 62-channel trial in fp16 (EEGMMIdb about 10 GB, BETA_4s about
  2 GB, BCICIV2a about 0.8 GB). It lives under `output/`, which is already git-ignored, and is
  regenerable, so deleting it is always safe.
- **Acceptance:** for a fixed sample of trials, the head's logits from the cache match
  `StampExtractor` on the fly within 1e-3 (fp16 storage), and the resulting per-subject
  balanced accuracy after a 2-epoch smoke run matches the uncached path within noise.

## Sub-project C: `train_finetune.py`

One file. It keeps the training loop, metrics and collate; it loses everything listed under
Problem. Expected size: about 450 lines.

- **Config keys:** a nested `split` block, separate from the optimiser settings under
  `training_params.finetune`. `subject_to_use` (in `dataset_params.finetune`) is the pool.
  - `"split": {"mode": "intra_subject", "n_folds": k}`: for each subject in the pool, k-fold
    CV over that subject's own trials (seed 42 by default, key `seed`).
  - `"split": {"mode": "inter_subject", ...}` with either `n_folds: k` (seeded partition of
    the pool into k groups; fold i evaluates group i, trains on the rest; `k` = number of
    subjects is LOSO) or `eval_subjects` (a list of subject indices, or a dict of named
    lists such as `{"seen": [...], "unseen": [...]}`; a plain list is one group `heldout`).
    Optional `train_subjects` restricts the training subjects (default: the pool minus the
    fold's or run's evaluation subjects). Training and evaluation subjects must be disjoint;
    exactly one of `n_folds` and `eval_subjects` is given.
  - An unknown mode raises an error listing the two valid ones.
- **Subject list helper** inside `train_finetune.py`: a subject entry may be an explicit list,
  `"all"`, or `{"random": n, "seed": s}`. The old difficulty-proxy selector
  (`probes/select_eval_subsets.py`) is dropped; the chosen lists stay in
  `config/subject_groups/*.json` (seed and statistics included) and are pasted into specs.
- **Training on the cache:** the dataset returns the cached `amp` (stamp heads) or the patched
  raw signal (raw heads); the head is the only trainable module and the optimiser sees only
  head parameters. Removed: `freeze_backbone`, `backbone_lr_mult`, `recon_mse` logging, the
  best-validation checkpoint (only the last head checkpoint plus `head_config` is saved), and
  the generic viz calls.
- **One output format for both modes:** `artifacts/group_eval.json` (schema of ADR 0014's
  Protocol): per run, per group, per subject `tail` (mean of the last 10 epochs) and `last`.
  Within-subject runs are named `<subject>_fold<i>` with the single subject in group
  `heldout`; the results tool pools per subject.
- **Environment stamp:** the run's `artifacts/config.json` gets an `env` block: git commit and
  dirty flag, Python path, torch, mne and CUDA versions.
- **Refinements made in implementation:** no `DataLoader` and no autocast/GradScaler (the head
  is tiny and all inputs are already in RAM, so batches are indexed directly and the head
  trains in fp32); one evaluation pass per epoch serves both the validation metrics and the
  per-subject tail history (no per-subject loaders); the model trained is the bare
  `FeatureHead`, not `FinetuneModel`; one dataset per run; a subject listed but missing from
  the pool raises instead of being dropped silently.
- **Verification:** `intra_subject` reproduces the fold composition of the old
  `intra_subject_cv` (same seed, same train and eval trial indices per subject and fold), and
  `inter_subject` with `n_folds` equal to the number of subjects reproduces the old LOSO fold
  composition; a 2-epoch smoke run of each mode writes a valid `group_eval.json`. These check
  the new split code, not old outputs.

## Sub-project D: experiments and results

A new `experiments/` folder (with a README); `probes/` is deleted.

- **Spec file** `experiments/specs/<set>.json`: dataset, backbone checkpoint, `split` block
  (the C keys), epochs, batch size and learning rate, a `base_head` (the numeric head keys),
  `variants` (name to override dict), and an optional `sweep` block for one-factor-at-a-time
  ablations: `{"spatial_k": [4, 8, 16], "time_rank": [1, 2, 4], "dropout": [0.3, 0.5, 0.7]}`
  yields one variant per value against the base (the ADR 0016 grid). A variant may instead be
  `{"baseline": "psda"}`.
- **`experiments/run.py`:** `python -m experiments.run --spec <file>` [`--only <variant>`]
  [`--resume`]. Builds each variant's full config from `config/config.json` defaults plus the
  spec, builds the feature cache once, then runs each variant as a fresh `train_finetune.py`
  process in the Python that launched it, one after another, into
  `output/<set>/<variant>/`. `--resume` skips variants that already have `group_eval.json`.
- **`experiments/results.py`:** `python -m experiments.results output/<set> --ref <variant>`.
  Merges `group_summary.py`, `ft_summary.py` and `epoch_curve.py`: per-subject tails, paired
  difference against the reference (subject level, point difference first, p reported never
  gated, `descriptive only, n<5` for small groups), seen-versus-unseen where groups exist, the
  convergence check (last 10 epochs against the previous 10, from the logs), and
  `results.csv` in the set folder.
- **`experiments/baselines.py`:** registry of non-model baselines with one entry, `psda`
  (from `probes/phase_probe_beta.py`: SSVEP power-spectral peak picking). It writes the same
  `group_eval.json` format so it appears as a variant. Further baselines (CSP+LDA for MI,
  xDAWN for ERP) are added as functions when needed, not before.
- **Deleted:** all of `probes/` (`probe_v10.py`, `stamp_relevance.py`,
  `select_eval_subsets.py`, `make_phase2_configs.py`, `ft_summary.py`, `group_summary.py`,
  `epoch_curve.py`, `phase_probe_beta.py`, `README.md`) and `config/phase2/*.json`.
- Docs: the Commands section of `CLAUDE.md` names `run.py`, `results.py`, the split modes, the
  feature cache and the `eeg_fm` environment.

## Out of scope

- The Phase 2 Inria runs and the ablation grid runs themselves (they run on the new runner
  once D exists).
- Any change to pretraining, the backbone, the loss, or `MeSAEPretrain` methods.
- The viz refactor (head diagnostics, finetune snapshots), any `IO/` restructuring, and the
  model plugin system.
- New baselines beyond PSDA; a run manifest with timings and exit statuses.

## Risks and mitigations

- **Behaviour drift in A and C** (head math, fold composition): the equivalence checks against
  the tagged old code are each sub-project's acceptance and run before the old code is
  deleted. They guard the new code against bugs, not old outputs.
- **Cache staleness:** the key covers backbone file identity, data cache files and the
  preprocess settings that change patches; a changed input yields a new folder, never a
  silent reuse. Disk use is a few GB per dataset (EEGMMIdb about 10 GB).
- **fp16 storage** could clip large amplitudes: the builder asserts all stored values are
  finite and checks the maximum absolute value against the fp16 range.
- **CRLF files** (`MeSAE.py`, `config/config.json`): edit in place, small diffs.

## Order and gating

A, then B, then C, then D, each a separate plan reviewed before the next starts. No experiment
runs while A to C are edited (the GPU stays idle; the Phase 2 Inria runs wait for D).
