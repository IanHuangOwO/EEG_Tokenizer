# Finetune evaluation caveats

Read before reporting or comparing finetune numbers. No label leak was found (audit 2026-09-24): subjects are
disjoint across LOSO folds, normalisation is per trial, the head's BatchNorm statistics update only in
training, Q-atom codes come from the frozen backbone without labels, and the score is `tail` (last 10 epochs),
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
8. **Few-shot heads overfit heavily; read loso first.** Few-shot trains one head per subject on ~21
   calibration trials per class (BNCI2014001: ~86 trials, ~750 head parameters). Seed-1 patch-50 z probe,
   last-10-epoch mean over the 9 subjects: 001 few-shot train 0.955 vs held-out 0.317, 004 few-shot 0.998 vs
   0.670 (001 loso: 0.545 vs 0.456). A smaller head (spatial_k 2) did not help: 001 held-out 0.309, 004
   0.605 (pilot 2026-09-29, output/qtome_tiny_p50_s16_s1/finetune/patch_probe_k2/), so spatial_k stays 8.
   Few-shot differences between backbones partly measure how a head copes with tiny calibration sets.
9. **MI few-shot is limited by feature form, not regularization.** A closed-form ridge with its shrinkage
   picked inside each subject's calibration trials (tools/analysis/ridge_probe.fewshot_ridge) does not beat the
   trained z head: seed-1 patch 50, 001 few-shot z 26.7 / z log-power 27.7 / Q-atom log-power 32.3 vs head 31.7,
   all near chance; 004 Q-atom log-power 72.5 vs head 67.0 is the one gain. Compass's CSP + shrinkage LDA reaches
   60.6 on 001 few-shot: it contrasts variance ACROSS channels (spatial filters before log-power), which
   per-channel features cannot form. A few-shot MI head would need a channel-mixing step before power.
