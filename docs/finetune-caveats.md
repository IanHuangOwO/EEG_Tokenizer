# Finetune evaluation caveats

Read before reporting or comparing finetune numbers. No label leak was found (audit 2026-09-24): subjects are
disjoint across LOSO folds, normalisation is per trial, the head's BatchNorm statistics update only in
training, stamp codes come from the frozen backbone without labels, and the score is `tail` (last 10 epochs),
not a best-epoch pick on the test set.

1. **Data compiled before 2026-09-26 was filtered per epoch.** Bandpass and resample ran on each cut 1-5 s
   epoch, so every epoch carried filter edge transients (BNCI2014008: the samples two overlapping epochs share
   disagreed by 16% median, 33% max). All finetune and backbone numbers from before the recompile were
   measured on that data. Compare only runs made on the same compile.
2. **P300 windows overlap in time** (1 s windows, flashes ~0.25 s apart). A within-subject split without
   `purge` puts overlapping windows on both sides: inflated. The `p300_fewshot` protocol uses `purge: 5`; LOSO
   is unaffected.
3. **Shuffled within-subject folds (`kfold`) are optimistic**: trials seconds apart land on both sides. Use
   `fewshot` (chronological calibration, the Compass convention) or `blocked_kfold`.
4. **Single pretrain seed**: differences within about +-3 balanced-accuracy points are ties; `backbone_report`
   Holm-corrects its paired tests over subjects, which measure subject noise, not seed noise.
5. **Keep finetune datasets out of pretraining.** PhysionetMI was in the v10-v13 corpus (those results are
   overlap-sensitive); it is not in the current corpus. BNCI2014001/004/008/009, BNCI2015001 and
   Nakanishi2015 never were.
6. **DEV sets** (BNCI2015001, BNCI2014009) tuned the frozen protocols: never report them.
7. **MI accuracy may partly read the cue response, not motor imagery.** On the EEG-FM-Compass windows (which
   start at the cue), the z probe for BNCI2014004 / 001 puts most of its weight in the first 0-1 s after the
   cue, with parieto-occipital spatial filters in places (probe maps 2026-09-28 and 2026-09-29), while
   imagery-related ERD typically builds over 0.5-4 s. Part of an MI score -- and of differences between
   backbones, e.g. patch 100's few-shot gain -- may be the visual cue-evoked response. Not tested; the check
   would be the same finetune on a window starting at cue + 0.5 s (accuracy holding = the probe reads imagery).
