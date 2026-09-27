# 0021 — Remove the trial-level reconstruction term

Status: Accepted (2026-09-27). The term, its weight and its logging are removed from the code; a config
that still sets `loss.mse_trial_weight` is rejected. Answers ADR 0015's experiment 1.
Date: 2026-09-27

## Context

The pretrain loss had a trial term next to the patch MSE: the patches overlap-added (50% overlap,
linear crossfade) back into the continuous 5 s window and compared with it. It was kept for seam
consistency: neighbouring patches should agree on the samples they share (ADR 0015).

## Test (mechanism card, ADR 0020)

- Hypothesis: without the trial term, neighbouring patches disagree more where they overlap.
- Metric: `backbone_eval` seam disagreement, the squared difference between two neighbouring
  patches' reconstructions on their shared samples / signal power (unmasked, 512 held-out windows).
- Spread: graded vs the ranking-fix reference (near-identical runs) 0.110 vs 0.108, about 2%.

| Run | Seam disagreement | Masked MSE token / channel / time | Ridge probe 004 / 001 / 008 |
|---|---|---|---|
| graded (trial term on) | 0.110 | 0.397 / 0.316 / 0.776 | 71.3 / 37.3 / 70.1 |
| reference (trial term on) | 0.108 | 0.412 / 0.317 / 0.784 | 69.8 / 37.8 / 69.8 |
| `mesae_tiny_notrial_s1` (off) | **0.066** | 0.405 / 0.318 / 0.780 | 69.4 / 38.7 / 69.8 |

Without the term the patches agree **better** (-40%, about 20x the spread); everything else is
within the spread. Logged `mse_trial` rose only 3%.

## Why

The trial term scores the crossfaded average of two overlapping patches against the signal, so two
neighbours can be wrong in opposite directions and still average out: it tolerates disagreement
that cancels (ADR 0015 noted this in principle). Patch MSE alone makes each patch fit its own
samples, so both neighbours converge on the same signal where they overlap.

## Decision

Remove the term. The loss is now patch MSE (masked 1, visible `unmasked_weight`) + per-stamp mp +
MoE load balance. Seam consistency is tracked by `backbone_eval`'s seam disagreement instead.

## Consequences

- One finetune flag for the next recipe freeze (2 seeds): in this single run the BNCI2014004 stamp
  head read 62.9 -> 56.1 (loso, session 0), about 3x the finetune spread between graded and the
  reference. Not a veto under ADR 0020; re-checked at the freeze.
- Old run configs with `mse_trial_weight` cannot be retrained as is (the trainer rejects the key);
  their checkpoints still load (build_config holds no loss settings).
