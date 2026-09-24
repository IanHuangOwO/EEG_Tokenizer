# Finetune evaluation caveats

Known ways our finetune numbers can be optimistic. Audited 2026-09-24. The checks found no
label leak: subjects are disjoint in `inter_subject` (`make_runs` raises on overlap),
normalization is per trial, the head's BatchNorm stats update only in `head.train()`, stamp
features come from the frozen backbone without labels, and the reported score is `tail` or
`last` rather than a best-epoch pick on test. Pretrain-side changes (windowing, recon loss)
don't touch the finetune path.

## 1. P300 intra-subject results leak (BNCI2014008, BNCI2014009)

- **Cause:** P300 trials are 1 s windows ([-0.2, +0.8] s) around flashes that come every
  ~0.25 s, so neighbouring windows share about 0.75 s of the same EEG (1710 of 1725
  consecutive windows overlap, BNCI2014009 subject 1).
- **Effect:** `intra_subject` uses `StratifiedKFold(shuffle=True)` over trials, so almost
  every test window has overlapping windows in the training fold. The head has already seen
  part of the exact test signal.
- **What to do:** treat P300 intra numbers as inflated. Report them only with this caveat, or
  leave them out. LOSO numbers are unaffected, because the whole subject is held out.

## 2. Intra-subject results are optimistic in general

- **Cause:** random trial-level folds put trials recorded seconds apart (same run, same
  impedance, drift and artifacts) on both sides of the split.
- **Effect:** the head can partly learn "this recording" rather than the task.
- **Extent:** checked for motor-imagery windows, which don't overlap: every event gap
  ≥ window length, see the event-spacing check of 2026-09-24. So this is optimism, not a
  data leak.

**Fix for 1 and 2 (planned, not implemented yet):** block-wise intra folds, each fold a
contiguous stretch of the recording, with a purge gap of one window on each side of the test
block so no window overlaps across the split. They'll be built together with the
EEG-FM-Compass protocols (few-shot, one-session LOSO) after the MOABB switch-over. Runs until
then, including the v13 chain, keep the shuffled folds so they stay comparable with v10.

## 3. PhysionetMI is in the v10–v13 pretrain corpus

- **Cause:** the v10–v13 backbones were pretrained with PhysionetMI, without labels. Its
  finetune test subjects' EEG was seen during pretraining.
- **Effect:** not a label leak, but under EEG-FM-Bench / EEG-FM-Compass conventions those
  results are "overlap-sensitive" and shouldn't be reported as clean benchmark numbers.
- **Not affected:** BNCI2014001/004/008/009, BNCI2015001 and Nakanishi2015 aren't in
  pretraining.
- **Future backbones:** PhysionetMI was removed from `configs/pretrain.template.json`
  (2026-09-24), so backbones after v13 are clean on it. Keep every finetune dataset out of
  `dataset_params.pretrain`.
