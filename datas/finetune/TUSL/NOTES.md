# TUSL -- gated, not fetched (2026-09-24)

Temple University EEG Slowing Corpus (von Weltin et al., 2017), EEG-FM-Bench
member. 3-class: seizure / slowing / background.

EEG-FM-Bench Table 1: 21-22ch, 10s windows, 210 / 43 / 37 train/val/test
(small). Split: stratified over ALL data at ~0.8 / 0.1 / 0.1 (not
subject-disjoint), same as TUEV.

Access: same TUH EEG form as TUAB (see ../TUAB/NOTES.md). Rsync path once
approved: data/tuh_eeg/tuh_eeg_slowing/ (check the current version folder).

No scaffolding written yet -- revisit once access is granted.
