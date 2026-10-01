# Reports

Curated experiment reports, one file per experiment, named `YYYY-MM-DD-<topic>.md`, with a
same-named folder for its figures, tables and the scripts that produced it. Generated output
(`analysis_finetune.py`, `tools/analysis/*`) still lands in `output/analysis/` and `output/<run>/`;
a report copies what it cites here, so it outlives archived run folders.

Mechanism cards (`docs/cards/`) hold one pre-registered change and its verdict; a report covers a
comparison or pipeline run that may touch several cards, and links them.

## Format

```
# <Title>

- **Date:** YYYY-MM-DD (when the results were produced)
- **Question:** one sentence
- **Runs:** backbone names (output/ or output/archive/) and what differs between them
- **Protocol:** corpus, seeds, evaluation (splits, metrics, test)
- **Verdict:** one or two sentences
- **Cards / ADRs:** links

## Results
Tables, one per question. Numbers with units; pass/fail against the pre-set rule where there is one.

## Notes
Caveats, deviations from the plan, anything a reader needs to trust the numbers.

## Files
What is in the report's folder and where the raw output lives.
```

## Run renames

Reports and cards keep the name a run had when they were written. Renamed 2026-09-29:
`mesae_tiny_notrial_s1` -> `mesae_tiny_p50_s16_s1`, `mesae_small_graded_s1` -> `mesae_small_p50_s16_s1`.
Config values renamed 2026-10-01 (runs keep their names): `channel_layout: real` -> `native`, `coords: dataset` ->
`recorded`; `mesae_tiny_p50_s16_real_s*` are native-layout runs, `*_dcoord_s*` native + recorded positions.

## Index

| Date | Report | Verdict |
|---|---|---|
| 2026-09-26 | [Spatial encoding and masking (base, A, B, AB)](2026-09-26-spatial-and-masking.md) | AB adopted: only combination with a clear key-cell gain |
| 2026-09-28 | [Overnight pipeline: finest skips, tiny to small, trial windows](2026-09-28-overnight.md) | graded skips stay; small corpus helps; Compass windows help MI few-shot stamp heads |
| 2026-09-29 | [Patch length 50 vs 100](2026-09-29-patch-length.md) | trade-off: patch 100 wins MI few-shot, loses P300 loso; patch 50 stays, patch 75 next |
| 2026-09-29 | [Patch length 75, and ranking against Compass](2026-09-29-patch-75.md) | patch 50 stays (P300); all lengths rank 2nd / 2nd / 1st-2nd among Compass frozen linear probes |
| 2026-09-29 | [Test A: latent token pooling vs patch 100](2026-09-29-test-a-token-pooling.md) | pooling recovers ~half the MI few-shot gain, no loso gain, same P300 cost; patch 50 stays |
| 2026-09-29 | [Stamp hidden vectors vs z](2026-09-29-stamp-hidden.md) | u is a lossy low-rank copy of z, loses 2 / 3 loso cells; idea dropped |
| 2026-09-29 | [Source-factorized stamps, K=4 pilot](2026-09-29-source-stamps.md) | not promising: stamp head -9.0 on BNCI2014001 loso, topographies scalp-wide not focal; shelved |
| 2026-09-29 | [Combined head: stamp_power + latent_signed](2026-09-29-combined-head.md) | adopted: wins every loso cell vs both parents (tiny, 3 seeds); overfits few-shot; small 80.2 / 53.1 / 70.0 loso |
| 2026-10-01 | [Real layout with each dataset's own electrode coordinates](2026-10-01-dataset-coordinates.md) | keep grid over dcoord (004 loso +0.8, 001 loso -1.7); the own positions cause the 001 loss -- real layout with template positions wins 004 loso (+1.3), level elsewhere: meets the adopt rule |
