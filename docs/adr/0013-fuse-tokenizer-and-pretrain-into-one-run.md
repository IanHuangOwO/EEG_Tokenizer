# 0013 — Fuse Tokenizer and Pretrain into one run; remove MeFSQ

Status: Accepted
Date: 2026-09-17
Supersedes: 0005 (script split). Keeps 0003's leakage rule.

## Context

This is the third move of the stage boundary. 0003 merged the two stages into one
script. 0005 split them again because spatial/temporal mixing and masking could only
switch on together. Two things have changed since then:

- 0005's complaint no longer applies to MeSAE. Its tokenizer stage deliberately runs
  temporal-only, and spatial mixing turns on exactly when masking starts.
- 0012 §6 found the masked stage adds nothing measurable over the tokenizer stage. The
  split therefore bought no accuracy. What it did cost: two configs, two architecture
  blocks, a cross-architecture checkpoint loader, and flag state that every loader had
  to restore by hand. The finetune loader restored it wrong: it enabled temporal mixing
  only, so every pretrain-checkpoint finetune ran with spatial attention off.

The merge is for simplicity, not accuracy. 0012 §6 rejected it only on accuracy grounds.

MeFSQ had no runs left in `output/` and is removed.

## Decision

One script, `train_pretrain.py`, and one architecture, `model_params.MeSAE.pretrain`.

| | tokenizer phase (epochs 1..`tokenizer_epochs`) | masked phase |
|---|---|---|
| encoder blocks | only `pool_after_blocks` | all |
| temporal mixing | on | on |
| spatial attention + coord embed | off | on |
| masking | none (`bool_masked_pos=None`) | `preprocess_params.mask` curriculum, counted from the first masked epoch |
| StampBank | trains | `training_params.pretrain.freeze_stamps` |

- **The no-mask length lives in `training_params.pretrain.tokenizer_epochs`, not
  `preprocess_params.mask`.** The boundary switches four things at once, so it is a
  training-schedule event, not a mask property.
- **The tokenizer phase keeps 0003's leakage rule.** Running only the pre-pool blocks
  gives the same shape as the old tokenizer encoder: one block, then a pool, repeated,
  single-channel. With `pool_after_blocks [1,3,5,7]` on `enc_depth 8`, that is 4
  block+pool steps.
- **Newly activated blocks start near-identity.** LayerScale is initialised to 1e-4 and
  the attention `out_proj` layers to zero, so the boundary does not shock the stamps.
- **The phase is persisted.** `MeSAEPretrain.masked_phase` is a buffer, and a
  `load_state_dict` post-hook (`_restore_phase`) re-applies the active blocks and the
  mixing flags. Every loader (finetune, `check_model`, probes) gets the right topology
  from the checkpoint alone, so the manual `enable_*` calls and the
  `on_tokenizer_start` / `on_pretrain_start` hooks are gone. A checkpoint with no
  `masked_phase` predates this ADR and loads fully enabled, which matches the old
  loaders.
- **Checkpoints are `output/<model_name>/checkpoint/{best,last}.pth`.**
  - Best-val tracking resets at the boundary, so `best.pth` is the tokenizer-phase best
    only until the first masked epoch overwrites it.
  - During the mask curriculum, `best.pth` still locks onto the easiest epoch. Use
    `last.pth` (see the comment in `train_pretrain.py`).
- **The optimizer is built once, over all parameters.** Bypassed blocks and frozen
  stamps get `grad=None`, and AdamW skips those. The LR runs on one cosine schedule
  across both phases.

## Removed

- Code and config:
  - `train_tokenizer.py`
  - `TokenizerDataset`
  - `mode='tokenizer'` in `factory.py`, `check_model.py` and `IO/dataset.py`
  - `training_params.tokenizer`, `model_params.MeSAE.tokenizer`, `tokenizer_checkpoint`
  - `model/MeFSQ/` and its registry entry
- Old checkpoints: pre-fusion tokenizer checkpoints (`mesae_tokenizer_v5`, `v9`) used a
  separate architecture block and no longer load through `check_model`.
- Lost ability: the masked phase can no longer be rerun alone, because no tokenizer-phase
  checkpoint survives. A frozen vs. unfrozen A/B therefore needs two full runs.

## First run (`mesae_v10`)

Differences from v9, all at once, so v10 is a new baseline rather than an A/B:

- the fused schedule (15 tokenizer epochs + 35 masked epochs);
- `freeze_stamps: false`;
- `enc_depth` 12 → 8 and `pool_after_blocks` → `[1,3,5,7]`;
- `n_routed_stamps` 120 → 60. v9 had 22 of 120 alive.

What to check:

- **At the end of phase 1:** `mse_patch` and `dead_feature_rate` should be comparable
  to the v9 tokenizer (`mse_patch` ~0.029).
- **After the run:** `pretrain_probe.py` `recon_bandpow` should stay at or above ~0.48.
  Watch the per-stamp PSD panels for dictionary blur caused by the masked objective.

`moe_ffn` is unchanged. In v9 the router load was uniform (entropy 1.385 against a
ln 4 = 1.386 maximum), which the load-balance loss enforces anyway. That gives no
evidence for or against specialisation, so there was no reason to change it in the
same run.
