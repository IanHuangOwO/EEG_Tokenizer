# LOSO on C1 — MI core (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether C1 (the best stamp-code head, within-subject tail-mean 0.536) and
the raw-signal control head transfer to **unseen subjects** (leave-one-subject-out) on the
motor-imagery datasets that are clean for it: BNCI2014001, BNCI2014004, BCICIV1 (5 subjects).
This is ADR 0014 build-order step 11's LOSO regime test, first tier. Every number so far
(C0–C4, raw control) is within-subject; ADR 0014 names cross-subject transfer as the place a
shared pretrained dictionary should beat a per-subject classifier, and it was never tested.

**Scope (deliberate):**
- One LOSO **per dataset** (class label spaces differ; one head cannot span datasets; the
  existing multi-dataset LOSO code would mix label spaces, so every run's
  `dataset_params.finetune` holds exactly one dataset).
- Two heads per dataset, same protocol: **C1** and the **raw control** (interpretation needs
  the control; a bare LOSO accuracy has no reference).
- Out of scope, decided separately with the user (do not build them here): EEGMMIdb
  (two-group seen/unseen design), SSVEP datasets (BETA_3s/BETA_4s/Dial, incl. phase-sensitive
  contrast heads), Inria (ERP). Unusable for supervised LOSO: BCICIV1_Test, Inria_Test,
  GraspAndLift_*, Siena (no labels / no data — see the dataset survey).

**Architecture:** No model changes. One small `train_finetune.py` change so LOSO output is
usable (fold tags compatible with `probes/ft_summary.py`; a last-epoch aggregate next to the
optimistic best-val one), then six config-driven runs, then an ADR entry.

**Tech Stack:** existing `train_finetune.py` (`split_mode: "loso"`), `probes/ft_summary.py`
(groups folds by subject before the paired t-test — in LOSO each subject is one fold).

**Spec:** `docs/adr/0014-finetune-head-test-plan.md` — build-order step 11 (regime tests; the
"best of C0–C4 = C1's config" text), Protocol section (tail-mean, n-subject paired t reported
never gated), step 12a (raw control: the within-subject C1 vs raw comparison is +0.006,
p=0.85).

## Global Constraints

- Frozen backbone `output/pretrain/mesae_v10_small/checkpoint/last.pth` (do not touch
  `model_params.MeSAE.pretrain.stamp_bank`, must stay `60/4/12/6`). Resampling to 200 Hz,
  patch 50 / stride 25, canonical channels 10-10: all from `config/config.json`, unchanged.
- Heads (identical to the within-subject runs they mirror):
  - **C1:** `input stamp_induced`, `task mi`, `pool_channel spatial:8`, `pool_time learned:2`,
    `dropout 0.5`, `freeze_backbone true` (no `evoked_rank`, no `include_advance`).
    `config/config.json` currently holds exactly this (model_name `mesae_finetune_c1_rerun`).
  - **Raw control:** `input raw`, `task mi`, `pool_channel spatial:8`, `pool_time trial`,
    `dropout 0`, `freeze_backbone true` (as step 12a).
- LOSO protocol: `split_mode "loso"`; `learning_rate 0.01`, `min_learning_rate 0.001`,
  `warmup_epochs 2`, `batch_size 16`, `weight_decay 0.01`, `backbone_lr_mult 0.0`,
  `device cuda` unchanged. **`epochs 30`** (not 100): a LOSO fold trains on ~8x more data than
  a within-subject fold, so fewer epochs suffice and 100 would be ~3x the GPU-hours. This is a
  stated deviation from the within-subject protocol; record final-epoch train accuracy so an
  under/over-trained fold is visible. Do not tune epochs per dataset.
- Reporting: **final-epoch / tail-mean** balanced_acc on the held-out subject, NOT the
  best-val epoch (the best epoch is picked on the held-out subject itself, so it is optimistic
  in LOSO). Tail-mean = mean of the last 10 epochs (`probes/ft_summary.py` "tail" row).
  Comparison unit: held-out subject (n = number of subjects). Point difference is the bar;
  p-values reported, never a gate (Protocol section).
- Datasets and subjects (exact; chance in parentheses):
  - `BNCI2014001`: subjects all 9, 4-class MI (0.25). 22 real channels. Fully unseen by the
    backbone's pretraining.
  - `BNCI2014004`: all 9, 2-class MI (0.50). Only 3 real channels (C3/Cz/C4), so a `spatial:8`
    filter on 64 mostly-zero channels is near-degenerate — run as is, and say so in the ADR.
    Fully unseen.
  - `BCICIV1_Train`: subjects **2, 3, 4, 5, 7 only**, 2-class left/right MI (0.50). Subjects
    1 and 6 use a left/foot class pair, so their label 1 means something different; exclude.
    All BCICIV1_Train subjects (1–7) WERE in the backbone's pretraining (unlabeled data), so
    this dataset is "seen": report it in its own row and say so.
- Trial length is fixed within each dataset (2a/2b/BCICIV1: 1000 samples, 39 patches), which
  C1's learned time weights require; `num_patches` is derived from the data by
  `run_training_loop` already (C1 plan).
- No test suite exists (CLAUDE.md): validation is a smoke run for Task 1 and the real runs
  for Task 2.
- Do not commit anything under `output/` or `.superpowers/` (gitignored). Preserve each edited
  file's existing line endings (`file`, `grep -c $'\r'` before/after; `git diff --stat` must
  show only the intended small change). `train_finetune.py` line endings: check first.

---

## Task 1: Make LOSO output usable in `train_finetune.py`

**Files:**
- Modify: `train_finetune.py` — `_run_loso` only (currently ~lines 570–621).

**Interfaces:**
- Produces: LOSO fold tags `"<dataset>_S<subject>"` (subject id after `_S`, nothing after),
  which `probes/ft_summary.py` parses (`int(t.split('_S')[-1])`).
- Produces: `loso_summary.json` gains `aggregate_last` (mean/std over folds of the
  last-epoch val metrics), next to the existing best-val `aggregate`.

- [ ] **Step 1: Read `_run_loso`** and `run_training_loop`'s return value (a dict with keys
  `epoch`, `train`, `val` (best-val epoch) and `last_val` (final epoch), or `None`).

- [ ] **Step 2: Fold tag.** Change

```python
        fold_tag = f"{ds_name}_{subject}"
```
to
```python
        fold_tag = f"{ds_name}_S{subject}"
```
(Checkpoint/visualization directories are named from `fold_tag`, so they become
`fold_<ds>_S<subject>`; fine, no other consumer.)

- [ ] **Step 3: Last-epoch aggregate.** After the existing best-val summary loop and before
`summary_path = ...`, add a second aggregate over `best_metrics['last_val']`, and relabel the
existing header/log so it is clear the best-val numbers are optimistic:

```python
    logger.info("===== LOSO Summary (best-val-acc epoch per fold) -- optimistic: the epoch "
                "is chosen on the held-out subject itself =====")
```
(replace the existing `"===== LOSO Summary (best-val-acc epoch per fold) ====="` line), and
add after the existing aggregate loop:

```python
    logger.info("===== LOSO Summary (final epoch per fold) =====")
    last_metric = {k: [] for k in metric_keys}
    for fold_tag, best_metrics in fold_results.items():
        if best_metrics is None:
            continue
        lv = best_metrics['last_val']
        logger.info(f"  {fold_tag}: " + " | ".join(f"{k}={lv[k]:.4f}" for k in metric_keys))
        for k in metric_keys:
            last_metric[k].append(lv[k])
    summary['aggregate_last'] = {}
    for k in metric_keys:
        vals = last_metric[k]
        if vals:
            mean = statistics.mean(vals)
            std  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary['aggregate_last'][k] = {'mean': mean, 'std': std}
            logger.info(f"  MEAN {k} (final epoch): {mean:.4f} +/- {std:.4f}")
```
`summary['aggregate_last']` must be set before `json.dump(summary, ...)`.

- [ ] **Step 4: Smoke run (GPU, ~minutes).** Build a throwaway config in the scratchpad
(NOT config/config.json): copy `config/config.json`, set `dataset_params.finetune` to
`{"BNCI2014004": {"dataset_path": "datas/BNCI2014004", "subject_to_use": [1, 2, 3], "channels_to_use": ["all"]}}`,
`model_params.MeSAE.finetune` to the raw control head (cheap), `training_params.finetune`:
`split_mode "loso"`, `epochs 2`, `warmup_epochs 1`, `model_name "smoke_loso"`, rest unchanged;
run `python train_finetune.py --config <that file>`. Expect 3 folds with tags
`BNCI2014004_S1..S3`, no crash, `output/smoke_loso/artifacts/loso_summary.json` containing both
`aggregate` and `aggregate_last`; then `python probes/ft_summary.py <its log>` prints 3
columns S1–S3 (tail over only 2 epochs is fine, just confirm parsing). Delete
`output/smoke_loso` afterwards.

- [ ] **Step 5: Commit** `train_finetune.py` only. Message ending with:
```
Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011ZRo7FMWpgbtn1N34keF18
```
e.g. `fix: LOSO fold tags compatible with ft_summary, add final-epoch aggregate`.

---

## Task 2: Run the six LOSO jobs and record the results

**Files:**
- Configs: throwaway per-run copies in the scratchpad (untracked; each run's own
  `output/<name>/artifacts/config.json` snapshot is the record).
- Modify: `docs/adr/0014-finetune-head-test-plan.md` (one new entry under step 11).

- [ ] **Step 1: Six configs** = `config/config.json` (currently C1) with only these changed:
`dataset_params.finetune` (one dataset per config, `channels_to_use ["all"]`,
`subject_to_use` per Global Constraints), `training_params.finetune`: `split_mode "loso"`,
`epochs 30`, `model_name` `mesae_loso_<dataset>_<head>` where `<head>` ∈ {`c1`, `raw`}, and for
the raw runs `model_params.MeSAE.finetune` = raw control head. Datasets: `BNCI2014001`,
`BNCI2014004`, `BCICIV1_Train`. Verify each config against the within-subject run it mirrors
(`output/archive/experiment_c/mesae_finetune_c1_learned2/artifacts/config.json`,
`output/archive/experiment_c/mesae_finetune_raw_control_cv/artifacts/config.json`): only dataset, split_mode,
epochs, model_name (and cv_folds/train_val_split irrelevance) may differ; report the check.

- [ ] **Step 2: Run sequentially** (single GPU; never in parallel):
`python train_finetune.py --config <cfg>` for, in this order, `BNCI2014001 c1`, `BNCI2014001 raw`,
`BNCI2014004 c1`, `BNCI2014004 raw`, `BCICIV1_Train c1`, `BCICIV1_Train raw`. Estimated wall time
(cost model 10.2 ms per sample-epoch for C1, ~0.71x for raw): 2a 2.0 h + 1.4 h, 2b 2.8 h +
2.0 h, BCICIV1 0.5 h + 0.35 h, ~9 GPU-hours total. A crash for an environment/config reason:
diagnose and report; never change what is measured (epochs, heads, subjects). A narrow new
code bug fix is allowed as its own documented commit.

- [ ] **Step 3: Statistics per dataset.** `python probes/ft_summary.py <raw log> <c1 log>`
(raw first, so the printed paired line is C1 − raw at held-out-subject level, n = subjects).
Report per dataset: tail-mean for raw and C1, per-subject tails, C1 − raw mean diff / t / p /
wins; and each head's margin over chance. Also final-epoch train balanced_acc (mean/range)
per run, so under/over-training is visible, and head parameter counts (raw 612, C1 1,844 —
confirm from a saved checkpoint).

- [ ] **Step 4: ADR entry.** Add one new entry under build-order step 11 (regime tests),
titled `**LOSO, MI core (phase 1)**`, in the ADR's prose style (bolded lead phrases, inline
numbers, no new tables; do not renumber; do not edit steps 7–12a): protocol and the stated
deviations (epochs 30, single dataset per run, BCICIV1 subjects 2,3,4,5,7 and why, 2b has 3
real channels so spatial:8 is near-degenerate, BCICIV1 was pretrained on so it is a "seen"
row while 2a/2b are unseen), the per-dataset numbers and paired stats, the train-accuracy
check, and an evidence-only reading: does C1 beat raw on unseen subjects (the ADR's
prediction was: if the tokenizer wins anywhere it is few-shot and cross-subject), how does
seen (BCICIV1) compare to unseen (2a/2b), and what within-subject C1 ≈ raw (step 12a) implies
by contrast. State that "raw" is the simple mu/beta band-power head. Note phase 2+ (EEGMMIdb
seen/unseen groups, SSVEP with phase-sensitive heads, ERP) remain open. Do not mark step 11
done.

- [ ] **Step 5: Commit** the ADR entry (one commit, same trailers).

## Self-Review Notes

- **Spec coverage:** step 11's LOSO (MI core). EEGMMIdb/SSVEP/ERP explicitly deferred.
- **Placeholders:** none; code and commands literal.
- **Consistency:** fold tag `_S<subject>` (Task 1) is exactly what Task 2's `ft_summary.py`
  calls need; `aggregate_last` reads `best_metrics['last_val']`, which `run_training_loop`
  already returns.
- **Known risks (stated, not defects):** 30 epochs is a judgment call (final-epoch train acc
  reported per run); BNCI2014004's 3-channel geometry; LOSO n is only 9 / 9 / 5 subjects, so
  significance is weak by construction (point difference is the bar).
