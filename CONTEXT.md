# EEG Tokenizer

An EEG foundation model: **MeSAE** turns multi-channel EEG into per-patch sparse stamp codes, pretrained by
masked reconstruction, then read by a small head on a frozen backbone. The retired MeFSQ model (discrete
FSQ codes; terms Expert, Code, Codebook, Expert View, pre-/post-VQ feature) is documented in
`docs/adr/0001`, `0002`, `0013` -- only relevant when reading those.

## Data

**Trial**: a labelled segment as the source dataset defines it; event-anchored ones are cut
[event - 1 s, event + 4 s). Finetuning uses whole Trials.
_Avoid_: epoch, segment (for this meaning)

**Window**: the pretrain input unit, 5 s (1000 samples). Each Trial's real content is cut into its own
non-overlapping Windows, never spanning two Trials; a short leftover is dropped or zero-padded, and padding
is excluded from masking and the loss (`IO/preprocessing.py`'s `window_continuous_signal`).

**Patch**: one channel's 50-sample (0.25 s) slice of a Window, stepped by 25 (50% overlap): 39 per Window.
At 200 Hz a patch's FFT grid is 4 Hz (60 Hz lands on a bin, 50 Hz never does).

**Canonical montage / valid channels**: every dataset is mapped onto one channel list (`10-10`, 64);
channels a dataset lacks are zero padding and `valid_channels` marks the real ones. Padded channels are
masked out of attention and the loss; a subject's near-flat channels (dead electrodes, the recording
reference) are treated as padding in pretraining. A **sub-montage** (`configs/montages.json`: motor-3,
p300-8, 10-20, bci-22, ...) is what channel subsampling cuts a dense cap down to.

**Cohort**: datasets recorded from the same people (`data_metadata.cohort`); the pretrain train/val
split keeps a person on one side for the whole cohort.

**Corpus size**: tiny / small / medium / large = 5 / 20 / 50 / 100% of every subject's Windows, nested.

## Model

**Stamp**: a learned unit-norm template `D` (50 samples) plus its derived quadrature partner `H` (Hilbert:
rFFT bins rotated -90 degrees). A stamp contributes `a*D + b*H` to a channel: amplitude `sqrt(a^2+b^2)`,
phase `atan2(b, a)`, the shape unchanged at any phase.

**Stamp code**: the `(a, b)` pair per patch, channel and stamp: `[N', C, S, 2]`. `a^2+b^2` is
phase-invariant power (induced activity, e.g. motor-imagery ERD); signed `(a, b)` keeps phase-locked
content (evoked responses, P300).

**Static dictionary**: every stamp is active at every patch position, shared by all channels, each channel
with its own gains -- so a stamp's per-channel gains form a **mixing column** (an ICA-style topography).
Routed (top-k selected) stamps were removed (docs/adr/0022; `routed-stamps` branch).

**Sparsity budget**: `2 * n_stamps` free scalars per channel must stay well below
`patch_len` (50), or the active slots fit any patch and it stops being sparse coding (docs/adr/0011).

**Residual ordering** (`mp_loss`): matching-pursuit grading -- slots ranked by amplitude, each graded against
the residual the higher ranks leave (detached), so duplicate atoms earn nothing. Only while stamps train.

**Stage**: a group of `blocks_per_stage` (2) encoder blocks at one temporal resolution; the patch axis is
pooled by 2 between stages (centred, no time shift) and upsampled back through gated skips.

**z**: the encoder output before the StampBank, `[N', C, 100]`. The stamp codes on visible input are close
to a fixed projection of each patch onto the templates, so a head on stamp codes sees little of the
encoder's context; `latent_*` head entries read z instead.

**Spatial embedding**: the Fourier electrode-coordinate embedding plus the per-block directional relative
spatial bias, on or off together (`spatial_embedding`).

## Training

**Tokenizer phase**: epochs 1..`tokenizer_epochs`: every block runs with temporal attention only (no
spatial attention, no coordinate embedding), unmasked, so the StampBank learns from single-channel content
(docs/adr/0003, 0013).

**Masked phase**: the rest of the same run: spatial attention and the coordinate embedding on, masked
reconstruction with the mask curriculum counted from here. Stamps stay trainable unless `freeze_stamps`.
The phase is a checkpoint buffer, restored on load.

**Mask mode / mixture**: one mask pattern per Window -- `channel_cluster` (a scalp region), `random_channel`,
`time_block` (runs of patches on every channel), `random_token` -- drawn from a mixture on a shared ratio
ramp, redrawn every masked epoch. `random` (one random_token mode) is the masking baseline. A patch run is
>= 3 patches: with 50% overlap a lone masked patch is visible through its neighbours, and the loss counts a
sample as masked only if every patch covering it is masked.

**Channel subsampling**: removes (not masks) channels of a dense-cap Window down to a sub-montage, to train
sparse-cap robustness.

## Evaluation

**Cell**: one dataset under one protocol, e.g. `BNCI2014001_loso`. **Protocol**: split + head +
hyperparameters, frozen in `configs/finetune_protocols.json` (mi_loso, mi_fewshot, p300_loso, p300_fewshot).
**DEV set**: BNCI2015001 (MI) and BNCI2014009 (P300), used only to tune the protocols, never reported.
Reported: BNCI2014001, BNCI2014004 (3 channels), BNCI2014008 (P300).

**LOSO / fewshot**: leave-one-subject-out; within-subject calibration (per class, the first `train_fraction`
of trials in recording order train). **Purge**: dropping eval trials next to train trials (overlapping P300
windows).

**tail**: a run's score -- balanced accuracy averaged over the last 10 epochs, per subject, then over
subjects. Differences within +-3 points are ties at one pretrain seed.

## Current MeSAE defaults

What `configs/pretrain_tiny.template.json` builds (2.26M parameters):

| module | params | what |
|---|---|---|
| `SpatialTemporalEmbeddings` | 0.51M | patch 50 -> 100, learnable time position embedding, Fourier coordinate MLP |
| `TSAEncoder` | 1.74M | 8 blocks = 4 stages x 2 (39 -> 20 -> 10 -> 5 patches); block = temporal attention -> spatial attention (+ relative spatial bias) -> MoE FFN (4 routed + 1 shared, top-2), LayerScale |
| `StampBank` | 0.01M | 16 stamps (sparsity budget 32 < 50) |

Loss: patch MSE + per-stamp `mp` (weight 1; stamps ranked per patch by strength) + `ffn_lb` (0.01);
visible samples weighted 0.1 in the masked phase. Removed: trial MSE (ADR 0021), STFT (0019), nested (0018). 50 epochs, 10 tokenizer. Masking: mixture (channel_cluster 0.2,
random_channel 0.3, time_block 0.4 max ratios) + channel subsampling.

**Unit**: umbrella term in shared tooling (`model/base_*`, `tools/`) for whatever a model codes per patch --
a Stamp for MeSAE. Plugins: `model/<Name>/plugin.py`, docs/adr/0004.
