# Card: is the coordinate embedding a per-site lookup? (diagnostic, no training)

Written 2026-10-01, before any number was computed.

- **Hypothesis (user):** trained on the fixed 64 10-10 positions only, the backbone memorises each site's coordinate
  as an identity ("this is C3") instead of a smooth map of the head, so a position it has not seen (or a channel at
  another site's position) is handled badly. Motivated by the reverted real layout (ADR 0023 Outcome), which added
  off-grid positions to pretraining and lost BNCI2014001 loso by 10 points.
- **Known so far (backbone_eval, grid s1/s2):** the coordinate embedding's similarity only loosely follows electrode
  closeness (Spearman 0.43 / 0.44) while every block's relative spatial bias follows it closely (~0.9); shuffling the
  coordinates across channels raises channel-mask masked MSE ~2.4x. So coordinates are used heavily; whether as a
  smooth map or a lookup is the open question.
- **Backbones:** `mesae_tiny_p50_s16_s1`, `_s2` (grid). Coordinates are transformed at evaluation only, the data is
  untouched.

## Transforms

- `jitter_{2,5,10}mm`: every channel's position moved by an independent Gaussian offset (sigma per axis = the value /
  sqrt(3), so the mean displacement is about the value), fixed per recording (seeded), then projected back to the
  channel's original distance from the head centre. 5 mm is roughly real cap-placement error; 10 mm is about half the
  10-10 spacing.
- `mirror`: x -> -x (left <-> right). For BNCI2014001 / 004 every channel's mirror site is in the montage, so this is
  a relabelling: C3's data at C4's position and vice versa, Cz unchanged.

## Metrics

1. **Masked MSE** (backbone_eval, same windows and masks as ever): ablation rows for each transform, per test mask.
   Relative rise vs baseline, mean over s1/s2.
2. **Probe robustness (decides):** loso ridge probe on z (the ridge_probe pipeline) for BNCI2014001 and BNCI2014004,
   trained on normal-coordinate features of the training subjects and tested on the held-out subject's features
   under each transform. Drop vs the normal-coordinate test, mean over s1/s2.
3. **Laterality (mirror, BNCI2014004 and BNCI2014001's hand classes):** the fraction of held-out trials whose
   predicted class changes between normal and mirrored coordinates, and the fraction that flips left hand <-> right
   hand.

## Decision

- **Lookup:** jitter_5mm drops either probe by > 3 points, or raises a channel-mask masked MSE by > 5%. Then the
  embedding is brittle at real-cap precision: write an augmentation card (coordinate jitter in pretraining).
- **Smooth:** jitter_5mm drops both probes by < 1 point and raises every masked MSE by < 2%. The lookup hypothesis
  is rejected as the cause of the real-layout loss; the IDW-interpolation explanation (Schirrmeister2017) remains.
- In between: partial; report the jitter curve (2 / 5 / 10 mm) and decide with the user.
- Mirror is reported, not judged: > 50% L <-> R flips on BNCI2014004 means laterality is read from the coordinates,
  not from the channel's slot in the probe.

## Results (2026-10-01): smooth, the lookup hypothesis is rejected

Masked MSE rise vs baseline, mean of s1 / s2 (backbone_eval, 512 held-out windows; the existing rows reproduced to
~1e-9):

| mask | jitter 2 mm | jitter 5 mm | jitter 10 mm | mirror |
|---|---|---|---|---|
| token_runs | +0.1% | +0.4% | +1.4% | +0.0% |
| random_channel | +0.2% | +0.9% | +3.1% | +0.0% |
| channel_cluster | +0.1% | +0.7% | +2.7% | +0.0% |
| time_block | +0.0% | +0.1% | +0.1% | +0.0% |
| motor3_to_bci22 | +0.0% | +0.3% | +1.1% | +0.1% |

Loso ridge probe (trained on normal coordinates), mean of s1 / s2:

| dataset | normal | jitter 2 mm | jitter 5 mm | jitter 10 mm | mirror |
|---|---|---|---|---|---|
| BNCI2014004 | 67.5 | 67.4 | 67.5 | 67.2 | 67.8 |
| BNCI2014001 | 39.2 | 39.2 | 38.9 | 39.0 | 38.9 |

- jitter_5mm: probe drop <= 0.3 points, masked MSE rise <= 0.9%: **smooth** by the pre-set rule. Degradation grows
  gently with the shift (10 mm: +3% on channel masks), no cliff.
- Mirror: masked MSE unchanged (0.0%) and 4-6% of predictions change; the hand error rate (BNCI2014004 32-33%,
  BNCI2014001 18%) is the same as with normal coordinates, so left and right are not read from the coordinates.
  The backbone's output is left-right symmetric in the coordinates: laterality reaches the head only through which
  channel slot a feature sits in. Contrast: shuffling coordinates across channels (which breaks the relative geometry)
  raises channel-mask MSE ~2.4x, so the model uses the relative layout, not absolute positions.
- Consequence for ADR 0023's loss: not a finetune-time position confusion (001's finetune input was identical, and the
  backbone tolerates moved positions). It came from what pretraining on the real-layout corpus learned; the open
  candidates are the IDW-interpolated Schirrmeister2017 sites and the denser 10-05 channels (easier neighbour-copying).
