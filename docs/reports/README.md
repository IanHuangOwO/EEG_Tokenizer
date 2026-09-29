# Reports

Curated experiment reports, one file per experiment, named `YYYY-MM-DD-<topic>.md`, with a
same-named folder for its figures, tables and the scripts that produced it. Generated output
(`analysis_finetune.py`, `tools/analysis/*`) still lands in `output/reports/` and `output/<run>/`;
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

## Index

| Date | Report | Verdict |
|---|---|---|
| 2026-09-26 | [Spatial encoding and masking (base, A, B, AB)](2026-09-26-spatial-and-masking.md) | AB adopted: only combination with a clear key-cell gain |
| 2026-09-28 | [Overnight pipeline: finest skips, tiny to small, trial windows](2026-09-28-overnight.md) | graded skips stay; small corpus helps; Compass windows help MI few-shot stamp heads |
| 2026-09-29 | [Patch length 50 vs 100](2026-09-29-patch-length.md) | trade-off: patch 100 wins MI few-shot, loses P300 loso; patch 50 stays, patch 75 next |
| 2026-09-29 | [Patch length 75, and ranking against Compass](2026-09-29-patch-75.md) | patch 50 stays (P300); all lengths rank 2nd / 2nd / 1st-2nd among Compass frozen linear probes |
| 2026-09-29 | [Test A: latent token pooling vs patch 100](2026-09-29-test-a-token-pooling.md) | pooling recovers ~half the MI few-shot gain, no loso gain, same P300 cost; patch 50 stays |
