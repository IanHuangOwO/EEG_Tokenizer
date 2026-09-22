# Phase 2 infrastructure — subject-group evaluation, ERP head support, subset selection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling for the Phase 2 cross-subject experiments (development scale; the
full-scale runs come later once the code is final): (1) a group-holdout / subject-k-fold
runner that trains one model on a set of subjects and reports **per-subject** last-10-epoch
scores for one or more evaluation groups (e.g. pretrained-on "seen" subjects vs. held-out
"unseen" subjects); (2) ERP support in `MeSAEFeatureHead` (relax the `task` guard for the
stamp-code arm, add a raw ERP baseline arm); (3) a reproducible, representative selection of
the evaluation/training subject subsets; (4) a stats script over the per-subject outputs.

**Why (one paragraph):** Phase 1 (plan `2026-09-21-loso-mi-core.md`) runs strict LOSO on the
small MI datasets. Phase 2 needs designs Phase 1's runner cannot express: PhysionetMI has 109
subjects (strict LOSO ≈ 366 GPU-h), and the user wants (a) one model evaluated on BOTH a
"seen" group (subjects whose unlabeled EEG was in the backbone's pretraining) and a small,
representative "unseen" subset, so the same trained head isolates the effect of pretraining
exposure, and (b) grouped k-fold for Inria. Evaluation stays per-subject so C1-vs-control
comparisons can be paired across subjects, matching the ADR's Protocol (n-subject paired
tests, folds/epochs averaged within subject first).

**Spec / context:** `docs/adr/0014-finetune-head-test-plan.md` (Protocol; build-order steps 10,
11 incl. the LOSO MI-core entry once written, 12 incl. C4 and 12a raw control; the ERP block
sketch in Experiment B: "ERP: avg_pool1d to about 20 Hz, then flatten. Signed, no log.").
Dataset survey: `.superpowers/loso-survey.md` (gitignored, local): subjects, trial length,
channel coverage, pretraining overlap per dataset.

## Global Constraints

- Frozen backbone; no change to any model behaviour for existing configs. All existing
  inputs/flags (`raw`, `recon`, `stamp_bandpow`, `z_chan`, `stamp_induced` with
  `pool_time`/`include_advance`/`evoked_rank`) must behave exactly as before for `task="mi"`.
- No test suite exists (CLAUDE.md): validation = smoke runs / small checks, never pytest files.
- **Line endings:** `model/MeSAE/MeSAE.py` is CRLF (991 lines, all CR) — an earlier edit
  converted it to LF and rewrote the whole file in git; preserve CRLF and check
  `git diff --stat` stays small. `train_finetune.py` and `docs/adr/*.md` are LF. New files
  LF. Check with `file`/`grep -c $'\r'` before and after each edit.
- The GPU is shared with a running measured experiment (Phase 1 LOSO). Smoke runs must be
  short (≤ ~12 epochs on a few subjects, `num_workers` low) and must not be left running.
  Never run two measured jobs at once.
- Every commit ends with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18`
  Do not push (the controller pushes). Do not commit anything under `output/`,
  `.superpowers/`. Leave the untracked `docs/adr/0014_attempts.csv` alone.

## Shared schema — `artifacts/group_eval.json` (produced by Task 1, consumed by Task 4)

```json
{
  "<run_name>": {
    "train_subjects": ["3", "7"],
    "epochs": 30,
    "tail_epochs": 10,
    "groups": {
      "<group_name>": {
        "subjects": { "<subject_id>": {"tail": 0.61, "last": 0.60, "n_trials": 88} },
        "n_subjects": 10, "mean_tail": 0.59, "mean_last": 0.58
      }
    }
  }
}
```
Subject ids are strings. `tail` = mean over the LAST 10 epochs of that subject's
balanced_acc on the model as it was at each of those epochs; `last` = final epoch's value.

---

## Task 1: `subject_groups` / `subject_kfold` split modes in `train_finetune.py`

**Files:** Modify `train_finetune.py` only (LF).

**Behaviour to implement**
- New `training_params.finetune.split_mode` value `"subject_groups"`. Exactly ONE dataset in
  `dataset_params.finetune` (assert). Config keys, under `training_params.finetune`:
  - either `subject_group_runs`: list of `{"name": str, "train": [subject ids], "eval":
    {"<group>": [subject ids], ...}}` (ids as in `subject_to_use`: ints or numeric strings),
  - or `subject_kfold`: int N (optional `subject_kfold_seed`, default 42): resolve the
    dataset's subject list with the existing `_resolve_all_subjects` /
    `_resolve_requested_subjects`, shuffle with `random.Random(seed)`, partition into N
    near-equal folds; run i: `train` = all subjects not in fold i, `eval` =
    `{"heldout": fold i}`, `name` = `fold<i>`.
  Exactly one of the two must be given (raise `ValueError` otherwise). Validate that train and
  eval subjects are disjoint and all exist in the dataset.
- For each run: build the train dataset (`subject_to_use` = train list) and ONE combined val
  dataset (`subject_to_use` = union of all eval groups) via `build_dataset_from_config`
  (deep-copy config, set `dataset_params['finetune'][ds]['subject_to_use']`; follow
  `_loso_fold_configs` for the pattern) and call the existing `run_training_loop` with
  `fold_tag = f"{ds}_{run_name}"`, so the existing per-epoch logging/`best_finetune.pth`
  behaviour is unchanged.
- **Per-subject evaluation:** add an optional argument to `run_training_loop`,
  `subject_eval=None` — a dict `{group: {subject_id_str: Dataset}}` (one `Subset`/dataset per
  eval subject built from the combined val dataset's `base_dataset.subject_data`; the
  existing `_intra_subject_split`/`Subset` idioms show how to select one subject's trials).
  When given, in each of the LAST 10 epochs (`epoch > total_epochs - 10`; if `total_epochs`
  < 10 use all epochs) evaluate every subject dataset with the model as it stands (reuse
  `validate_one_epoch`'s logic/metrics; low worker count, e.g. `num_workers=2`, no
  persistent workers; `balanced_acc` may warn when a class is absent — suppress that sklearn
  warning) and accumulate `balanced_acc`. After the loop attach to the returned metrics dict
  `best_metrics['subject_eval'] = {group: {subject: {"tail": mean over the collected epochs,
  "last": value at the final epoch, "n_trials": int}}}`. Default `None` leaves the function's
  behaviour byte-for-byte unchanged for every existing mode (`inter_subject`,
  `intra_subject*`, `loso`).
- The new runner `_run_subject_groups(...)` collects each run's `subject_eval`, writes
  `<artifact_dir>/group_eval.json` in exactly the shared schema above (add `mean_tail` /
  `mean_last` / `n_subjects` per group; `epochs` = `training_params.finetune.epochs`), logs a
  readable per-group summary, and is dispatched from `main()` for `split_mode ==
  "subject_groups"`; extend the "Unknown split_mode" error message to list it.
- Do not change `_run_loso`, `_run_intra_subject*` or `run_training_loop`'s existing
  outputs.

- [ ] **Step 1:** Read `train_finetune.py` (`_run_loso`, `_loso_fold_configs`,
  `_resolve_*`, `run_training_loop`, `validate_one_epoch`, `FinetuneCollate`, `main`).
- [ ] **Step 2:** Implement as above.
- [ ] **Step 3: Smoke (GPU, short).** Throwaway configs in the scratchpad (NOT
  `config/config.json`), BNCI2014004, raw head (`input raw`, `pool_channel spatial:8`,
  `pool_time trial`, `dropout 0`), `epochs 12`, `warmup_epochs 1`, `batch_size 16`:
  (a) `subject_group_runs` = one run `{"name": "smoke", "train": [1,2,3,4], "eval": {"a":
  [5,6], "b": [7]}}`; confirm exit 0, `artifacts/group_eval.json` has groups a and b with
  per-subject `tail`/`last`/`n_trials` and group means, and the run still logs the normal
  `--- [BNCI2014004_smoke] Epoch ...` blocks; (b) `subject_kfold: 3` on 9 subjects
  (`epochs 12`): three runs `fold0..fold2`, each eval group `heldout` with 3 disjoint subjects
  covering all 9 across folds; (c) error cases: both/neither of the two config keys,
  overlapping train/eval subjects → clear exceptions. Also confirm an existing mode still
  starts (e.g. `intra_subject_cv` with `cv_folds 2`, `epochs 1`, `subject_to_use [1]`): the
  refactor must not have changed it. Delete `output/<smoke names>` afterwards.
- [ ] **Step 4:** `git diff --stat` small (tens of lines, not a rewrite). Commit
  `train_finetune.py` only: `feat: subject_groups/subject_kfold split modes with per-subject
  last-10-epoch evaluation`.

---

## Task 2: ERP support in `MeSAEFeatureHead` (`model/MeSAE/MeSAE.py`, CRLF)

**Files:** Modify `model/MeSAE/MeSAE.py`.

**Behaviour**
- `task` currently must be `'mi'` (`NotImplementedError` otherwise). New rule:
  - `input == 'stamp_induced'` (all its flags): `task` ∈ {`'mi'`, `'erp'`, `'ssvep'`} is
    accepted and has NO effect on the computation (the stamp-code branches do not depend on
    task); add a docstring/comment stating that.
  - `input == 'raw'`: `task` ∈ {`'mi'` (unchanged mu/beta log band power), `'erp'` (new)};
    `'ssvep'` raises `NotImplementedError` with a message pointing to the standard
    frequency-power reference (`probes/phase_probe_beta.py` PSDA baseline), since two coarse
    bands cannot separate 40 stimulus frequencies.
  - all other inputs (`recon`, `stamp_bandpow`, `z_chan`) keep requiring `task == 'mi'`.
- **Raw ERP block** (`input='raw'`, `task='erp'`): `sig` (overlap-added signal, `[B,C,T]`) →
  `self._mix(sig, 1)` (signed spatial filter, `[B,K,T]`) → `torch.nn.functional.avg_pool1d`
  with `kernel_size = stride = max(1, round(fs / 20))` (≈ 20 Hz output rate; `fs =
  self.fs`) → `[B, K, T // pool]` → flattened → the shared `BatchNorm1d → Dropout →
  Linear`. Signed, NO log. `pool_time` must be `'trial'` for it (a `window:` form is allowed
  only if it already works for raw; `learned:R` stays invalid for raw as today). `n_feat = K
  * (T // pool)` with `T = (num_patches - 1) * backbone.patch_stride + backbone.patch_len`, so
  `num_patches` is REQUIRED for raw+erp (raise `ValueError` with the same style of message as
  the learned/evoked guards). Concat pooling (`pool_channel='concat'`) uses `K = C`. Padded
  channels are already zeroed in `sig`.
- Nothing else about the class changes; `MeSAEFeatureHead` is still only constructed through
  `build_finetune`, whose `allowed` kwargs already include `task` and `num_patches`.

- [ ] **Step 1:** Read `MeSAEFeatureHead` (`__init__` guards, `n_feat` block, `forward`'s
  `raw/recon` branch, `_band_logpow`, `_mix`) and the ADR Experiment B ERP sketch.
- [ ] **Step 2:** Implement; keep the diff to a few dozen lines; keep CRLF.
- [ ] **Step 3: Smoke (CPU or GPU, no training).** Synthetic backbone as in prior checks
  (`build_pretrain_from_config(config, mode='finetune')`, `backbone.stamps.fire_ema.fill_(1.0)`,
  C=64, `num_patches=39`, input `[2,64,39,50]`): (a) raw+erp+spatial:8 builds, forwards to
  `[2,n_classes]`, `in_features == 8 * (T // pool)` (`T=1000`, `pool=10` → 800), gradient
  reaches the classifier and the spatial filter; (b) raw+mi unchanged (`in_features == 16`);
  (c) stamp_induced + `task='erp'` and `'ssvep'` build and forward exactly like `'mi'` (same
  `in_features`; also with `pool_time='learned:2'`, `include_advance=True`, `evoked_rank=2`);
  (d) raw+ssvep, recon+erp, stamp_bandpow+erp raise `NotImplementedError`; raw+erp without
  `num_patches` raises `ValueError`; (e) confirm every previously valid `task='mi'` build
  still produces identical `in_features` (C1 200, raw 16, C3 600, C4 600).
- [ ] **Step 4:** `git diff --stat` shows a small diff (line endings intact: file still has
  as many CR as lines). Commit `MeSAE.py` only: `feat: ERP task support (stamp_induced
  task-agnostic, raw ERP baseline arm)`.

---

## Task 3: `probes/select_eval_subsets.py` — reproducible subset selection

**Files:** Create `probes/select_eval_subsets.py`; create `config/subject_groups/physionetmi.json`
and `config/subject_groups/beta4s.json` (generated, tracked); add one line to
`probes/README.md` describing the script.

**Behaviour** (seed 42 everywhere; pure CPU; no model needed):
- **Pretraining membership:** read `output/pretrain/mesae_v10_small/artifacts/config.json`
  (`dataset_params.pretrain`) to get, per dataset, the subject ids the backbone saw. If the
  file is unavailable, fail loudly (do not guess).
- **PhysionetMI** (109 subjects, 3-class): `seen` = pretrained subjects; `unseen` = the rest
  (99). For each unseen subject compute a within-subject difficulty proxy: 5-fold stratified
  (seeded) shrinkage-LDA accuracy on log mu/beta band power per canonical channel (use
  `build_dataset_from_config(..., mode='finetune')` on a one-dataset config with
  `subject_to_use` = that subject, the trial signals from `FinetuneDataset`, and the same
  statistic as `MeSAEFeatureHead._band_logpow` / `probes/probe_v10.py`: rfft over the whole
  trial, sum power in 8–13 and 13–30 Hz, log; drop channels that are all zero). Then choose
  **10 eval subjects** by stratifying on the proxy: sort the 99 by proxy, cut into 10
  equal-count bins, take the subject at each bin's median rank; choose **30 train subjects**
  uniformly at random (seeded) from the remaining 89; the 10 seen subjects are the `seen`
  eval group. Record the proxy for ALL subjects, the chosen lists, and representativeness
  statistics comparing the 10 eval subjects to all 99 (mean/std/quantiles of the proxy,
  two-sample KS statistic and p) plus the seen group's proxy for reference. Write
  `config/subject_groups/physionetmi.json` with `{"dataset": "PhysionetMI", "seed": 42, "train":
  [...], "eval": {"seen": [...], "unseen": [...]}, "proxy": {...}, "stats": {...}}` (subject
  ids as strings).
- **BETA_4s** (55 subjects, 40-class SSVEP): `seen` = pretrained subjects among the 55;
  `unseen` = the rest; if `unseen` has ≥ 40 subjects use **10 random eval** + **30 train**
  (the remaining unseen, capped at 30; report if fewer); the `seen` eval group is all seen
  subjects (report the count). No difficulty proxy (band-power LDA is meaningless for 40-class
  SSVEP); state "random, seeded" in the JSON. Write `config/subject_groups/beta4s.json` with
  the same shape.
- CLI: `python probes/select_eval_subsets.py [--datasets eegmmidb beta4s]`; prints the chosen
  sets and the representativeness stats; idempotent (same output on rerun).

- [ ] **Step 1:** Read `probes/probe_v10.py` (feature statistic, LDA usage), `IO/dataset.py`
  (`FinetuneDataset`), `docs/agents/adding-a-dataset.md`.
- [ ] **Step 2:** Implement + run for both datasets; inspect the printed stats. The eval subset
  must not be wildly unrepresentative (report KS p; if p < 0.1, say so — do not silently
  re-draw).
- [ ] **Step 3:** Re-run and confirm identical JSON (determinism); confirm lists are disjoint
  (train ∩ eval groups = ∅, seen ∩ unseen = ∅) and that every eval/train subject exists in the
  dataset and no seen subject is in `train`.
- [ ] **Step 4:** Commit the script, the two JSONs and the README line: `feat: reproducible
  seen/unseen subject subsets for PhysionetMI and BETA_4s`.

---

## Task 4: `probes/group_summary.py` — statistics over `group_eval.json`

**Files:** Create `probes/group_summary.py`; add a line to `probes/README.md`.

**Behaviour:** `python probes/group_summary.py <group_eval.json> [<group_eval.json> ...]`;
each file = one head (name = the file's parent-of-`artifacts` directory name). For every
head print, per run and group, the per-subject `tail` values and `mean_tail ± sd`; for runs
that share a group name across `subject_kfold` folds (`heldout`), pool the subjects across
folds into one group. Then, across heads (first file = reference), paired comparisons over the
SAME subjects within each group (paired t via `scipy.stats.ttest_rel`, mean diff,
`wins`/n, printed as "head − reference"), and, per head, **seen vs unseen** (Welch
`ttest_ind`, unequal n; different subjects, so unpaired) when both groups exist. Follow the
conventions of `probes/ft_summary.py` (subject-level unit, point difference first, p as
information not a gate). No plotting.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Check** with a synthetic pair of JSON files written to the scratchpad (two
  heads, two groups, 8-10 subjects each, one head deliberately better by a known offset, one
  file with two `heldout` folds): assert the printed paired diff equals the known offset within
  tolerance and the pooled-heldout subject count is right; also run it on the real
  `group_eval.json` from Task 1's smoke if it still exists (else skip).
- [ ] **Step 3:** Commit: `feat: group_summary.py -- per-subject group stats and paired head
  comparisons`.

## Self-Review Notes
- **Coverage:** user's Phase 2 design → Task 1 (group-holdout + grouped k-fold), Task 2
  (Inria/ERP: evoked head vs induced needs the guard relaxed; raw ERP baseline), Task 3
  (representative small subsets), Task 4 (stats). The runs themselves (PhysionetMI, BETA_4s,
  Inria; heads C1 / raw / C1+advance / C4 / raw-ERP) are a separate follow-up plan.
- **Interfaces:** `group_eval.json` schema is defined once above and used by Tasks 1 and 4;
  `config/subject_groups/*.json` (Task 3) feeds `subject_group_runs` (Task 1) for the runs.
- **Deliberate limits:** per-subject tail requires evaluation each of the last 10 epochs;
  BETA_3s and Dial are skipped (3 and 2 unseen subjects); no ERP-specific handling of Inria's
  class imbalance beyond `balanced_acc`.
