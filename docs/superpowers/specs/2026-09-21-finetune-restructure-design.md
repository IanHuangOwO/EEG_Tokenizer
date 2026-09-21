# Finetune restructure: one head class, two split modes, experiment specs

Date: 2026-09-21. Status: design for review. Builds on ADR 0016 (head modules) and ADR 0014
(test plan and Protocol).

## Problem

The finetune ablation work is spread thin. Adding a head option touches the head class flags,
the factory, the config and a config generator. `train_finetune.py` (795 lines) mixes batching
and metrics, dataset and split building, the training loop, five split runners and reporting.
The analysis scripts in `probes/` overlap (three summary scripts, two dead Experiment A
probes, a config generator). Run folders, config names and environment notes are assembled by
hand per experiment set, which let a missing `mne` in the `base` env go unnoticed.

## Decisions (agreed in the design discussion)

1. **Three sub-projects, in order, each with its own plan.** A: model. B: `train_finetune.py`.
   C: experiments and results tooling. Files keep their names; nothing moves into a new
   package except the new `experiments/` folder.
2. **The finetune model is `MeSAEFeatureHead`.** The original `MeSAEFinetune` head with
   `PerChannelHeadAttn` is removed. The head config keeps today's flat keys.
3. **`train_finetune.py` stays one file with two split modes** (below).
4. **One experiment spec file per ablation set and one runner.** No manifest, exit-status or
   timing tracking: the spec and each run's `config.json` show what was run. One exception:
   a small environment stamp inside each run's config snapshot (kept unless the reviewer
   objects).
5. **`probes/` is dissolved into `experiments/`.**

## Sub-project A: model

Scope: `docs/superpowers/plans/2026-09-21-head-modules-refactor.md` Task 3 (Task 2 has landed). Concretely:
- Delete `MeSAEFinetune`, `PerChannelHeadAttn`, `build_finetune`'s `head_z` branch,
  `render_finetune_attn` and the dead `check_finetune` branches, the `recon` and `z_chan`
  inputs with `z_proj`, and channel-concat pooling. Keep `raw`, `stamp_bandpow`,
  `stamp_induced`, `spatial:K`, `learned:R`, `window:lo-hi`, `include_advance`, `evoked_rank`,
  `task`. Keep every `MeSAEPretrain` method.
- `MeSAEFeatureHead` defaults become `input='stamp_induced'`, `pool_channel='spatial:8'`.
- Head config keys stay flat: `input`, `task`, `pool_channel`, `pool_time`, `include_advance`,
  `evoked_rank`, `dropout`, `freeze_backbone`. A short table of them goes in the class
  docstring; `config/config.json` holds the example.
- Verification: the equivalence script (`.superpowers/sdd/2026-09-21-head-modules-refactor/
  head_equiv.py`) against the class at tag `pre-head-cleanup`, restricted to the kept
  configurations plus the real C1 checkpoint; `viz.load_model` smoke check; base config head
  builds.

## Sub-project B: `train_finetune.py`

The file keeps the training loop, collate, metrics and viz calls. The split logic goes from
five modes to two. Config keys sit under `training_params.finetune`; `subject_to_use` (in
`dataset_params.finetune`) is the subject pool.

- **`split_mode: "within_subject"`** with `n_folds`: for each subject in the pool, k-fold CV
  over that subject's own trials (today's `intra_subject_cv`, seed 42 by default).
- **`split_mode: "cross_subject"`** with either:
  - `n_folds: k`: seeded partition of the pool into k groups; fold i evaluates group i and
    trains on the rest. `k` equal to the number of subjects is LOSO; or
  - `eval_subjects`: a list of subject indices, or a dict of named lists such as
    `{"seen": [...], "unseen": [...]}` (a plain list is one group named `heldout`): one run.
  - `train_subjects` (optional list): restricts the training subjects; default is the pool
    minus the fold's or run's evaluation subjects. Training and evaluation subjects must be
    disjoint.
  - Exactly one of `n_folds` and `eval_subjects` is given.
- **Subject list helper** inside `train_finetune.py`: a subject entry may be an explicit list,
  `"all"`, or `{"random": n, "seed": s}`. The old difficulty-proxy selector
  (`probes/select_eval_subsets.py`) is dropped; the chosen lists stay in
  `config/subject_groups/*.json` (with seed and statistics) and are pasted into specs.
- **One output format for both modes:** `artifacts/group_eval.json` (schema of ADR 0014's
  Protocol / commit 71f952d): per run, per group, per subject `tail` (mean of the last 10
  epochs) and `last`. Within-subject runs are named `<subject>_fold<i>` with the single
  subject in group `heldout`; the results tool pools per subject.
- Removed: `inter_subject`, single-split `intra_subject`, the separate LOSO runner and
  `loso_summary.json`. Old split names in a config raise an error naming the replacement.
  Finished runs on disk are unaffected.
- **Environment stamp:** the run's `artifacts/config.json` gets a small `env` block: git commit
  and dirty flag, Python path, torch, mne and CUDA versions.
- Expected size: about 500 lines. Verification: the new `within_subject` mode reproduces the
  fold composition of the old `intra_subject_cv` (same seeds, same train and eval trial
  indices per subject and fold), and `cross_subject` with `n_folds = number of subjects`
  reproduces the old LOSO fold composition; a two-epoch smoke run of each mode writes a valid
  `group_eval.json`.

## Sub-project C: experiments and results

Everything below lives in a new `experiments/` folder (README included); `probes/` is deleted.

- **Spec file** `experiments/specs/<set>.json`: dataset and task, backbone checkpoint, split
  block (the B keys above), epochs, batch size and learning rate, a `base_head` (flat head
  keys), `variants` (name to override dict), and an optional `sweep` block for one-factor-at-a-
  time ablations: `{"pool_channel": ["spatial:4", "spatial:8", "spatial:16"], "dropout":
  [0.3, 0.5, 0.7]}` yields one variant per value against the base (the ADR 0016 grid). A
  variant may instead be `{"baseline": "psda"}`.
- **`experiments/run.py`:** `python -m experiments.run --spec experiments/specs/<set>.json`
  [`--only <variant>`] [`--resume`]. Builds each variant's full config from
  `config/config.json` defaults plus the spec, runs each as a fresh `train_finetune.py`
  process in the Python that launched it, one after another, into `output/<set>/<variant>/`
  (`artifacts`, `finetune`, `visualization`). `--resume` skips variants that already have
  `group_eval.json`.
- **`experiments/results.py`:** `python -m experiments.results output/<set> --ref <variant>`.
  Merges `probes/group_summary.py`, `ft_summary.py` and `epoch_curve.py`: per-subject tails,
  paired difference against the reference (subject-level, point difference first, p reported
  never gated, `descriptive only, n<5` for small groups), seen-versus-unseen where groups
  exist, the convergence check (last 10 epochs against the previous 10, from the logs), and
  `results.csv` in the set folder.
- **`experiments/baselines.py`:** registry of non-model baselines with one entry, `psda`
  (from `probes/phase_probe_beta.py`, SSVEP power-spectral peak picking on BETA_4s). It writes
  the same `group_eval.json` format so it appears as a variant. CSP+LDA (MI) and xDAWN (ERP)
  would be added as functions when needed, not before.
- **Deleted:** `probes/probe_v10.py`, `stamp_relevance.py`, `select_eval_subsets.py`,
  `make_phase2_configs.py`, `ft_summary.py`, `group_summary.py`, `epoch_curve.py`,
  `phase_probe_beta.py` (absorbed), `probes/README.md`. `config/phase2/*.json` are replaced by
  `experiments/specs/`.
- Docs: the Commands section of `CLAUDE.md` names `run.py`, `results.py`, the two split
  modes and the `eeg_fm` environment.

## Out of scope

- The Phase 2 Inria runs and the ablation grid runs themselves (they resume on the new
  runner once C exists).
- Any change to pretraining, the backbone, the loss, or `MeSAEPretrain` methods.
- Renaming or restructuring `IO/`, `viz/` or the model plugin system.
- New baselines beyond PSDA; a run manifest with timings and exit statuses.

## Risks and mitigations

- **Behaviour drift in B** (fold composition, seeds): the equivalence checks against the old
  runners are part of B's acceptance, run before the old runners are deleted.
- **Old checkpoints and configs:** the head keys and state-dict names are unchanged (ADR 0016
  constraint); the last commit with the removed code is tagged `pre-head-cleanup`.
- **Phase 2 provenance:** the finished runs (`mesae_p2_*`) came from `config/phase2/*.json` and
  the pre-restructure runner. Their `artifacts/config.json` files stay the record. Rerunning
  them through a spec is optional, not required.
- **CRLF files** (`MeSAE.py`, `config/config.json`): edit in place, small diffs.

## Order and gating

A, then B, then C, each a separate plan reviewed before the next starts. No experiment runs
while A or B are edited (the GPU stays idle; the Phase 2 Inria runs wait).
