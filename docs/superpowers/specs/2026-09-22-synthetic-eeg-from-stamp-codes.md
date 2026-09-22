# Idea to revisit: synthetic EEG from stamp codes

Date: 2026-09-22. Status: not scheduled, parked for later. No code written.

## The idea

`output/baseline/BNCI2014001_intra_c1_learned/analysis/stamp_distributions_subj9.png` plots, for
BNCI2014001 subject 9 (best C1 intra-subject result, tail 0.824), the distribution of `a`, `b`,
amplitude and phase per stamp, pooled over all trials, patches and channels. `a` and `b` have
clean, roughly symmetric, well-behaved marginal distributions. The question raised: could you
sample `a, b` from those marginals and decode them to generate new, plausible EEG for this
subject?

## Why not, as stated

Sampling each stamp's `a, b` independently from its pooled marginal destroys exactly the
structure that makes decoded output look like EEG:

- **Spatial correlation across channels.** `a, b` are per-channel already
  (`[trial, patch, channel, stamp, 2]`); a real trial's channel pattern for one stamp is
  smooth (volume conduction). Independent per-channel draws give spatially incoherent noise
  instead of a topography.
- **Temporal correlation across patches.** The learned time weights (see the time-map
  analysis, `output/experiment_c/mesae_finetune_c1_learned2/analysis/time_weights_subj8.png`)
  show a real, consistent rise-and-fall shape across patches (an ERD-like event spanning
  several consecutive patches). Independent per-patch draws erase that continuity.
- **Cross-stamp correlation.** Stamps likely co-modulate for the same physiological event;
  independent per-stamp draws ignore that.
- **The trial-conditional signal itself.** The marginal pools both classes together, so an
  i.i.d. draw carries no class information — exactly what the classifier reads is gone.

Full joint covariance estimation is not viable either: the per-trial tensor has about
39 (patches) x 22 (channels) x 25 (stamps) x 2 = 42,900 entries, against only 288 trials for
this subject — wildly underdetermined.

## What would actually work, if revisited

Roughly in order of effort:

1. **Bootstrap / interpolate real trials.** Resample or jitter whole `[N', C, S, 2]` tensors
   from real trials instead of independent marginals. Keeps every correlation intact by
   construction. Cheapest option, likely the right first step.
2. **A block-correlated or low-rank model** that at least preserves channel-to-channel and
   stamp-to-stamp covariance within a patch, without needing the full 42,900-dim covariance.
3. **A real generative model** (small VAE/diffusion over the `[N', C, S, 2]` tensor) for
   genuinely novel-but-plausible trials, not just resampling. This is real new work, not an
   add-on to the current pipeline.

## Cheap demo, not yet run

Decode one trial from i.i.d.-sampled marginals and one from a real held-out trial's `a, b`,
compare waveform and PSD side by side. Would make the "sounds like noise" claim above concrete
before investing in option 1-3.

## Why parked

Orthogonal to the finetune-head restructure and the restart baseline currently in progress.
Revisit once those are settled, if there's a reason to want synthetic trials (data
augmentation for the small MI datasets is the obvious motivation).
