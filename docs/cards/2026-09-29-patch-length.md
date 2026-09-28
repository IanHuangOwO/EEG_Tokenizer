# Card: patch length 50 vs 100 (tiny corpus, 3 pretrain seeds x 3 finetune seeds)

Written 2026-09-29, before seeds 2-3 trained (ADR 0020: a recipe choice with a downstream trade-off, so
the finetune is the primary measure).

- **Change:** `patch_len` / `patch_length` 100, `patch_stride` 50 (0.5 s patches, 19 per 5 s window)
  vs 50 / 25 (0.25 s, 39 patches). Static dictionary, 16 stamps, everything else as
  `mesae_tiny_notrial_s1`; only the training seed differs between seeds.
  - patch 50: `mesae_tiny_notrial_s1` (seed 1), `mesae_tiny_p50_s16_s2`, `mesae_tiny_p50_s16_s3`
  - patch 100: `mesae_tiny_p100_s16_s1` (seed 1, from the capacity card), `mesae_tiny_p100_s16_s2`, `_s3`
- **Hypothesis (from the seed-1 z probe: 004 +5.0, 001 +0.6, 008 -2.4 for patch 100):** longer patches help
  motor imagery (slow rhythm power: 2 Hz bins, a whole mu/beta cycle per patch) and hurt P300 (BNCI2014008's
  1 s trials become 3 patches; the peak shares a patch with baseline).

## Metrics

1. **Primary -- downstream z probe** (`latent_signed`, pca, spatial_k 8), tail balanced accuracy, 6 cells
   (BNCI2014004 / 001 / 008 x loso / few-shot, Compass protocols), finetune seeds 1-3. Per backbone the mean
   over finetune seeds; per patch length the mean over its 3 backbones, error = SE over the 3 pretrain seeds.
   A cell is won when the difference exceeds 2 x sqrt(SE_50^2 + SE_100^2).
2. **Decision:** patch 100 becomes the default if it wins >= 2 cells and loses none. MI cells won and P300
   cells lost = a real trade-off: patch 75 is the next card. Anything else: patch 50 stays.
3. **Secondary:** closed-form ridge probe on z (`ridge_probe`), mean +- SE over seeds. Masked reconstruction
   (`backbone_eval`) reported, not judged: its test masks are defined in patches, so patch 100 hides twice
   the seconds.

**Deviation (2026-09-29 02:27):** STEW, Weibo2014, Inria_Train and Inria_Test were recompiled mid-run with
the high-pass-only filter (commit 97c45f0; their native band ends at or below 100 Hz, so the old band-pass
only added a no-op edge at Nyquist). `notrial_s1` and all three patch-100 seeds trained on the old caches,
`p50_s16_s2` / `_s3` on the new ones. The difference is confined to 4 of 24 datasets (~19 of 211 h) and to
frequencies at their Nyquist; if patch-50 seeds 2-3 differ systematically from seed 1, check this first.
