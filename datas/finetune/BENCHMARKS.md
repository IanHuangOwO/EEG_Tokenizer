# Finetune benchmarks: EEG-FM-Bench and EEG-FM-Compass

Two external benchmarks define the finetune dataset set in `datas/finetune/`.
This note records each benchmark's datasets, their test protocol, which folder
here holds each dataset, and its status. Written 2026-09-24 from the papers
themselves (sources at the bottom).

## Dataset → folder map and status

| Benchmark name | Folder | In Bench | In Compass | Status |
|---|---|---|---|---|
| BCIC-2a / BNCI2014001 | `BNCI2014001` | ✓ | ✓ | compiled |
| BNCI2014004 | `BNCI2014004` | | ✓ | compiled |
| BNCI2015001 | `BNCI2015001` | | ✓ | compiled |
| BNCI2014009 | `BNCI2014009` | | ✓ | compiled |
| BNCI2014008 | `BNCI2014008` | | ✓ | compiled |
| Nakanishi2015 | `Nakanishi2015` | | ✓ | compiled |
| Workload (Zyma 2019) / EEGMat | `EEGMAT` | ✓ | ✓ | compiled |
| PhysioMI | `PhysionetMI` | ✓ | | compiled (moved from pretrain 2026-09-24) |
| Sleep-EDFx | `Sleep_EDFx` | | ✓ | compiled |
| CHB-MIT | `CHB_MIT` | | ✓ | downloading; label design differs from Compass (below) |
| Siena | `Siena` | ✓ | | downloading (moved from pretrain 2026-09-24); loader still gives dummy labels -- needs a seizure-label loader |
| SEED | `SEED` | ✓ | ✓ | gated (BCMI) |
| SEED-V | `SEED_V` | ✓ | | gated (BCMI) |
| SEED-VII | `SEED_VII` | ✓ | | gated (BCMI) |
| SEED-VIG | `SEED_VIG` | | ✓ | gated (BCMI); regression target |
| TUAB | `TUAB` | ✓ | ✓ | gated (TUH) |
| TUEV | `TUEV` | ✓ | | gated (TUH) |
| TUSL | `TUSL` | ✓ | | gated (TUH) |
| Things-EEG-2 | `Things_EEG2` | ✓ | ✓ | skipped (241.5 GB); the two benchmarks use different tasks |
| Mimul-11 | -- | ✓ | | open, not fetched (GigaDB, Jeong et al. 2020) |
| HMC | -- | ✓ | | open, not fetched (PhysioNet `hmc-sleep-staging`) |
| ADFTD | -- | ✓ | | open, not fetched (OpenNeuro ds004504) |

"Workload" in EEG-FM-Bench is Zyma et al. 2019, the PhysioNet mental-arithmetic
set, which is our `EEGMAT`. It is not STEW (`datas/pretrain/STEW`), even though
STEW is also a workload dataset.

## EEG-FM-Bench (Xiong et al., arXiv 2508.17742)

14 datasets, 10 paradigms.

**Preprocessing:**
- High-pass FIR plus a 50/60 Hz notch.
- Resampled to each model's pretraining rate.
- Channels mapped to 10-10 by name.
- Fixed windows (table below). Units in Volts.

**Split:** one train/val/test split per dataset, not k-fold. Subject-independent
unless noted. A greedy multi-label stratified splitter keeps the label mix.

| Dataset | Task | Ch | Win (s) | Train / Val / Test | Split rule |
|---|---|---|---|---|---|
| BCIC-2a | 4-class MI (L/R hand, feet, tongue) | 22 | 4 | 2784 / 1152 / 1152 | subjects 1-5 / 6-7 / 8-9 |
| PhysioMI | 4-class MI (L fist, R fist, both fists, feet) | 64 | 4 | 6210 / 1734 / 1803 | subjects 1-69 / 70-88 / 89-110 |
| Mimul-11 | 3-class upper-limb MI (reach, grasp, twist) | 60 | 5 | 31398 / 5000 / 4949 | stratified ~0.76 / 0.12 / 0.12 |
| SEED | 3-class emotion | 60 | 10 | 22455 / 7875 / 7560 | subject-DEPENDENT: 15 trials split 9:3:3, sessions merged |
| SEED-V | 5-class emotion | 60 | 10 | 3552 / 4638 / 4128 | subject-dependent: trials 1:1:1 |
| SEED-VII | 7-class emotion | 60 | 15 | 15536 / 1942 / 1942 | subjects random 8:1:1 |
| HMC | 5-class sleep (W, N1, N2, N3, REM) | 4 | 30 | 91681 / 22804 / 22440 | subjects random 103:24:24 |
| Siena | binary seizure detection | 29 | 10 | 41631 / 5592 / 3607 | subjects 0-7 / 9-13 / 16-17 |
| Workload (EEGMat) | binary: arithmetic vs rest | 19 | 10 | 1537 / 300 / 297 | stratified ~0.72 / 0.14 / 0.14 |
| TUAB | binary abnormal | 23 | 30 | 247728 / 12315 / 12277 | official train set; official eval set split in half by subject for val/test |
| TUEV | 6-class events | 21 | 5 | 87834 / 12473 / 13046 | stratified over all data ~0.8 / 0.1 / 0.1 |
| TUSL | 3-class (seizure, slowing, background) | 21-22 | 10 | 210 / 43 / 37 | stratified over all data ~0.8 / 0.1 / 0.1 |
| Things-EEG-2 | binary visual target detection | 63 | 5 | 24915 / 8324 / 8331 | subjects 0.6 / 0.2 / 0.2 |
| ADFTD | 3-class (AD, FTD, healthy) | 19 | 10 | 4743 / 1115 / 1155 | stratified ~0.70 / 0.15 / 0.15 |

**Test setups:**
- Three fine-tuning strategies: frozen backbone (head only), full fine-tuning,
  and multi-task (one model fine-tuned on a mix of all tasks).
- Three heads: patch-average-pool MLP, dimension-compression MLP, and
  attention-pool MLP.

**Metrics:**
- Balanced accuracy on every task.
- AUROC and AUC-PR on binary tasks.
- Cohen's kappa and weighted F1 on multi-class tasks.

**Overlap rule:** a model–dataset pair where the dataset was in the model's
pretraining corpus is flagged "overlap-sensitive" and left out of the overall
score.

## EEG-FM-Compass (Liu et al., arXiv 2601.17883)

13 datasets, 9 paradigms. Every dataset is run under two scenarios, each with
both full fine-tuning and linear probing. The main metric is balanced accuracy.
Things-EEG2 uses 2-way accuracy and SEED-VIG uses RMSE.

| Dataset | Subj | Ch | Hz | Trial (s) | Trials | Labels |
|---|---|---|---|---|---|---|
| BNCI2014001 | 9 | 22 | 250 | 4 | 2,592 | L/R hand, feet, tongue |
| BNCI2014004 | 9 | 3 | 250 | 4.5 | 1,400 | L/R hand |
| BNCI2015001 | 12 | 13 | 512 | 5 | 2,400 | R hand, both feet |
| BNCI2014009 | 10 | 16 | 256 | 0.8 | 5,760 | target / non-target |
| BNCI2014008 | 8 | 8 | 256 | 1 | 33,600 | target / non-target |
| CHB-MIT | 23 | 18 | 256 | 4 | 29,840 | interictal / ictal |
| TUAB | 2,383 | 21 | 250 | 10 | 53,604 | normal / abnormal |
| Sleep-EDFx | 78 | 2 | 100 | 30 | 414,961 | W, N1, N2, N3, REM |
| SEED | 15 | 62 | 200 | 1 | 50,910 | pos / neu / neg |
| Nakanishi2015 | 9 | 8 | 256 | 4 | 1,620 | 12 SSVEP freqs 9.25-14.75 Hz |
| EEGMat | 36 | 19 | 500 | 4 | 1,080 | low / high |
| Things-EEG2 | 10 | 63 | 1000 | 1 | 18,540 | 200-image retrieval |
| SEED-VIG | 21 | 17 | 200 | 8 | 18,585 | PERCLOS (regression) |

**Scenario 1, LOSO (cross-subject):**
- Train on the other subjects, test on the held-out one. All trials of the
  training subjects are used, except as listed below.
- MI and P300 sets (BNCI2014001/004, BNCI2015001, BNCI2014009/008): only ONE
  session per subject is used.
- CHB-MIT: ictal segments plus each seizure's 10-min PRE-ictal segment.
- TUAB: first 3 min of each recording.
- Sleep-EDFx and TUAB: 10-fold subject split instead of LOSO.
- Things-EEG2: 3 repetitions per image.

**Scenario 2, within-subject few-shot:**
- Fine-tune on a small labelled part of the target subject, test on the rest
  of that subject.
- MI: 30% of one session (fewer than 30 trials/class).
- BNCI2014009: 10% of one session. BNCI2014008: 5% of one session.
- CHB-MIT: first seizure (ictal plus 10-min pre-ictal) for training, the
  remaining seizures for testing.
- Sleep-EDFx: 10%.
- SEED: one video per class.
- Nakanishi2015: 80%. EEGMat: 60%. SEED-VIG: 10%.
- Things-EEG2: same as LOSO. TUAB: no within-subject run (one label per
  subject).

## Differences from our current protocol

- **Split scheme.**
  - Our `inter` runs are LOSO over all sessions and our `intra` runs are
    per-subject 5-fold.
  - Compass LOSO uses one session per subject for MI/P300.
  - Compass within-subject uses a small fixed calibration fraction, not
    5-fold with 80% train.
  - EEG-FM-Bench uses one fixed train/val/test split per dataset.
  - Numbers are not directly comparable to either paper until the split
    matches.
- **Only a frozen backbone.** We run the frozen backbone with a head only,
  which matches EEG-FM-Bench "frozen-backbone" and Compass "linear probing"
  (ours is not strictly linear). We have no full fine-tuning runs.
- **CHB-MIT labels.** Our loader balances ictal against interictal from
  recordings far from seizures. Compass pairs ictal with the 10-min pre-ictal
  segment. These are different tasks.
- **Siena.** Its loader still assigns dummy labels. EEG-FM-Bench needs binary
  seizure labels from `Seizures-list-PNxx.txt`, 10 s windows, and the subject
  split 0-7 / 9-13 / 16-17.
- **Overlap.**
  - `PhysionetMI` is in the v10–v13 pretrain dataset lists, so any
    PhysionetMI finetune result from those backbones is overlap-sensitive.
    The same was true of Siena before it was moved.
  - Drop both from the pretrain list of future backbones if they should
    count as clean benchmark results.
- **Things-EEG2.** EEG-FM-Bench uses binary target detection, Compass uses
  200-way retrieval.

## Sources

Local copies: `docs/papers/2508.17742_EEG-FM-Bench.pdf`, `docs/papers/2601.17883_EEG-FM-Compass.pdf`.

- EEG-FM-Bench: https://arxiv.org/abs/2508.17742 (Table 1, Appx. A, B.1, B.3, B.4) · code https://github.com/xw1216/EEG-FM-Bench
- EEG-FM-Compass: https://arxiv.org/abs/2601.17883 (Sec. III-A/B, Table IV, Fig. 5)
