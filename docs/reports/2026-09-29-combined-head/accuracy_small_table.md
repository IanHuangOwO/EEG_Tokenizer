
Cross-subject (loso), balanced accuracy (%)

| Dataset | Ours: combined | Ours: z probe | Ours: stamp head | Best FM linear probe | Best FM full fine-tune | Best specialist |
|---|---|---|---|---|---|---|
| BNCI2014004 | 79.1 ± 0.6 | 75.2 ± 0.5 | 74.9 ± 0.4 | 79.40 (MIRepNet) | 78.41 (MIRepNet) | 76.38 (EEGNet) |
| BNCI2014001 | 53.3 ± 0.2 | 49.8 ± 0.2 | 42.3 ± 0.4 | 50.48 (MIRepNet) | 54.21 (MIRepNet) | 46.80 (LMDA) |
| BNCI2014008 | 69.6 ± 0.3 | 68.8 ± 0.2 | 59.1 ± 0.3 | 67.11 (BENDR) | 69.91 (CBraMod) | 72.29 (EEGNet) |

Within-subject few-shot, balanced accuracy (%)

| Dataset | Ours: combined | Ours: z probe | Ours: stamp head | Best FM linear probe | Best FM full fine-tune | Best specialist |
|---|---|---|---|---|---|---|
| BNCI2014004 | 78.0 ± 0.1 | 70.7 ± 1.6 | 76.1 ± 0.6 | 76.82 (MIRepNet) | 81.10 (MIRepNet) | 80.17 (Conformer) |
| BNCI2014001 | 44.0 ± 0.3 | 36.1 ± 0.7 | 47.5 ± 0.3 | 49.82 (Neuro-GPT) | 63.27 (MIRepNet) | 60.62 (CSP+LDA) |
| BNCI2014008 | 59.8 ± 0.5 | 61.5 ± 0.4 | 53.4 ± 0.2 | 61.45 (EEGMamba) | 61.61 (Neuro-GPT) | 70.91 (EEGNet) |
