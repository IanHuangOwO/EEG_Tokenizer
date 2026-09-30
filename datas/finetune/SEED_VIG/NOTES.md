# SEED-VIG -- raw staged, not compiled (2026-09-30)

SJTU vigilance/fatigue dataset (BCMI Lab), from the gated SEED download. Staged in raw/: Raw_Data/ (23 experiments
of 21 subjects -- subjects 4 and 5 have two -- each a struct EEG.data [1,416,000 x 17] at 200 Hz, EEG.chn
FT7 FT8 T7 T8 TP7 TP8 CP1 CP2 P1 PZ P2 PO3 POZ PO4 O1 OZ O2; EOG struct unused), perclos_labels/ (885 PERCLOS
values per experiment in [0, 1], one per 8 s: 885 x 8 s = 7080 s = the recording), Readme_English.txt.

Not compiled: the target is continuous (PERCLOS regression, Compass reports RMSE) and the pipeline's labels are
int classes. Needs a label decision first: regression support, or thresholds into classes (the SEED-VIG paper's
awake / tired / drowsy at 0.35 / 0.7).
