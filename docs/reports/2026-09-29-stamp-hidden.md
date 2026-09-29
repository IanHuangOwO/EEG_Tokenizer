# Stamp hidden vectors: more than a copy of z?

- **Date:** 2026-09-29
- **Question:** does the StampBank's per-stamp MLP hidden `u_s = GELU(LN(z) W_down_s + b_down_s)` (8 dims per stamp,
  128 per token) carry class information z does not make linearly available?
- **Runs:** `mesae_tiny_p50_s16_s1..3` (judged), `mesae_small_p50_s16_s1` (reported); no training. `u` computed from the
  feature cache's z with each checkpoint's StampBank weights.
- **Protocol:** closed-form ridge probe (loso, primary) and few-shot ridge (secondary) with `u` in place of z, same PCA 8
  pipeline; mean +- SE over the 3 tiny seeds; won when |diff| > 2 x sqrt(SE_u^2 + SE_z^2). Mechanism stats on 20000
  random tokens per dataset. Self-check: the cached amp equals `u w_amp + b_amp` times a per-token rms (relative
  residual 4.6e-4 tiny, 1.5e-3 small).
- **Verdict:** dropped. `u` wins no loso cell and loses two; it is a lossy, low-rank re-expression of z, more so on
  the small corpus. No per-stamp architecture card follows.
- **Cards / ADRs:** [stamp-hidden card](../cards/2026-09-29-stamp-hidden.md); ADR 0020, 0022.

## Results

Loso ridge probe, balanced accuracy (%), tiny seeds:

| Cell | z | u (per seed) | diff | threshold | result |
|---|---|---|---|---|---|
| BNCI2014004 loso | 69.4 | 66.0 (65.1 / 66.6 / 66.3) | -3.4 | 3.8 | tie |
| BNCI2014001 loso | 38.4 | 35.1 (35.9 / 35.3 / 34.2) | -3.3 | 2.1 | u loses |
| BNCI2014008 loso | 69.4 | 66.2 (66.9 / 65.0 / 66.7) | -3.1 | 1.2 | u loses |

Few-shot ridge, tiny seeds (z / u): BNCI2014004 61.9 / 61.5 (log-power 69.4 / 66.6), BNCI2014001 27.5 / 27.0
(29.4 / 29.3), BNCI2014008 57.8 / 56.8 (53.5 / 53.3).

Mechanism, mean over tiny seeds (small backbone in brackets):

| Dataset | free share (random 0.75) | z -> u R^2 | eff. rank u / z | stamp-pair canonical corr |
|---|---|---|---|---|
| BNCI2014004 | 0.62 (0.84) | 0.881 (0.976) | 8.4 / 9.1 (2.4 / 8.5) | 0.64 (0.83) |
| BNCI2014001 | 0.59 (0.82) | 0.875 (0.977) | 9.9 / 11.1 (2.3 / 8.9) | 0.62 (0.82) |
| BNCI2014008 | 0.61 (0.84) | 0.858 (0.984) | 14.8 / 15.9 (1.9 / 9.0) | 0.63 (0.85) |

Small backbone probes (z / u): loso 70.4 / 66.9, 43.8 / 35.7, 70.0 / 65.4; few-shot 64.6 / 64.4, 31.2 / 30.8, 59.4 / 53.9.

## Notes

- Free share: per stamp, share of centred `u_s` variance outside the column space of its 2-dim readout `w_amp_s`.
  Some stamps leave almost all of it unread (max 0.98 tiny, 1.00 small).
- With more pretraining data the hidden collapses further (rank ~2 of 128, stamps sharing one subspace): the
  reconstruction needs only a gain pair per stamp, and nothing else shapes `u`.
- The small backbone's z ridge probe was recomputed here (the overnight file is kept as `ridge_probe_overnight.json`).

## Files

`2026-09-29-stamp-hidden/scripts/`: `run.sh` (driver) and per-backbone logs. Code: `tools/analysis/ridge_probe.py`
(`stamp_hidden`, `stamp_hidden_stats`, `feature='stamp_hidden'`). Raw output: `output/<backbone>/pretrain/analysis/`
(`stamp_hidden_stats.json`, `ridge_probe_stamp_hidden.json`, `fewshot_ridge[_stamp_hidden].json`).
