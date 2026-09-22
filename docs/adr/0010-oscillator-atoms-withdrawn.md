# Parametric oscillator atoms: added, then withdrawn (MeSAE StampBank)

## Context

A "cross-chunk artifact" was visible in every stamp that captured line noise:
strong amplitude in the middle of a patch, weaker at both edges, repeating every
patch. It looked like a windowing or overlap-add bug. It was neither.

Root cause is FFT-bin quantization. A free-vector atom `D_i` is a length-`patch_len`
vector, so whatever frequency it encodes is bound to the frequency grid that a
`patch_len`-point DFT can represent: `Δf = fs / patch_len`. At `fs=200`,
`patch_len=50` that is `Δf = 4Hz`. 50Hz line noise does not land on that grid
(50/4 = 12.5 bins), so no fixed vector can represent it without the represented
frequency drifting against the real one — producing a beat envelope within each
patch, and a phase discontinuity between consecutive patches, since each patch's
atom is fit independently with no shared notion of absolute time.

Line-noise frequencies were verified per dataset before acting (windowed FFT +
peak sharpness + cross-channel uniformity): BNCI2014001 and Inria_Train carry real
50Hz, EEGMMIdb carries real 60Hz, BETA_4s has no real line noise.

## Decision (later reversed)

A subset of the shared pool was given a different shape source. Instead of a free
`[patch_len]` vector, these atoms evaluated

```
D_i(t) = cos(2*pi*f_i*t + phi_i)
```

with `f_i`, `phi_i` free learnable scalars and `t` the patch's real absolute sample
offset (`time_idx * patch_stride`). A continuous scalar frequency has no grid to
misalign with, so `f_i` could converge on exactly 50.000Hz regardless of
`patch_len`; and because every patch read off the same continuous function of
absolute time, consecutive and overlapping firings were automatically
phase-continuous — no seam.

The `(a, b)` quadrature gain pair stayed exactly as for every other atom (read off
the atom's own hidden bottleneck). Only the *shape source* differed. Scope was
deliberately narrow: transient sources (blinks, muscle bursts) need shape freedom
a pure sinusoid cannot provide, so the routed pool kept the free-vector design.

This worked. A real training run confirmed exact frequency convergence and phase
continuity across patch boundaries.

## Reversal

First withdrawn by config (`n_oscillator_stamps: 0`, commit `b95f640`) on the
grounds that reserving shared slots for a fixed sinusoid presumes line noise is
present, and it is not present in every dataset. Later re-tested properly, with
three escalating interventions, and then deleted on evidence.

### Re-test, and what it showed

Re-opened because 50/60Hz was observed smearing across shared stamps rather than
specializing. Three variants, each on the same slice (`top_k=12`, 50 epochs), each
measuring oscillator gain and its correlation with the patch's real in-band
amplitude:

| intervention | osc gain_rms | gain vs real 50Hz (r) | osc ownership of 50Hz |
|---|---|---|---|
| pinned at `mp_loss` rank 0 | 0.0070 | +0.044 | ~0 |
| + all shared graded in the chain | 0.0070 | +0.044 | ~0 |
| + reconstruction-loss exclusivity | 0.0497 | +0.232 | 0.040 |

For reference, the routed atom that actually owned 50Hz in every one of those runs:
gain 0.30-0.35, r = 0.62-0.63, ownership 0.59-0.66, purity 0.75-0.80.

Exclusivity did wake the oscillator — purity 0.70, so its content genuinely was
50Hz, and its gain finally tracked the real signal. It still lost 16:1 on ownership.

Two reasons, both structural:

1. **Exclusivity is self-limiting.** It protects only what the oscillator has
   already claimed. A small claim protects little, so routed atoms keep the rest.
2. **Line noise is dataset-conditional; shared atoms are unconditional.** 50Hz is
   strong in BCICIV1_Train and Inria_Train, absent in EEGMMIdb, BETA_4s and
   BNCI2014001 (verified model-free: peak sharpness vs a local ring baseline, plus
   cross-channel uniformity). An always-on shared atom specialising on 50Hz would be
   actively wrong wherever 50Hz is absent, so gradient will not let it. A routed
   atom, selected only when relevant, is the structurally correct home.

The second point is the general lesson: **conditional content belongs in the
conditional pool.** The oscillator was always in the wrong pool, and no amount of
priority, grading, or exclusivity moves it.

### What replaced it

Nothing — the goal was already met. `mp_loss` (docs/adr/0011) makes a routed atom
specialise on line noise on its own, reaching purity 0.75-0.80 and ownership up to
0.66 in every configuration tested. A dedicated line-noise stamp exists; it simply
lives in the routed pool.

Deleted outright: `_eval_oscillators`, the oscillator branches of
`_template_tables`, the grouped branch of `_gather_by_idx`, the `osc_f_raw`/
`osc_phi` parameters, `sample_freq` on StampBank, the rank-0 pinning branch, the
`'osc'` exclusivity mode, and the entire `t=` parameter chain threaded through
`forward`/`decode_selected`/`_template_tables` — roughly 195 lines. No repo caller
ever passed `t=`; that chain existed only for this feature.

## Consequences

- The beating artifact can return for 50Hz specifically. `patch_len=50` still gives
  `Δf = 4Hz`, which still does not divide 50Hz evenly. Accepted knowingly: a routed
  atom covers line noise adaptively at purity ~0.78, at the cost of the within-patch
  envelope wobble.
- 60Hz is fine at this grid (60/4 = 15 bins exactly). The problem is 50Hz-specific
  at the current `patch_len`.
- If the artifact returns and matters, reach first for a `patch_len` whose grid
  divides 50Hz (`patch_len=40` at `fs=200` gives `Δf=5Hz`), not a parallel shape
  mechanism.
- Restoring the feature means restoring the `t=` plumbing as well; that chain
  existed only for this, and nothing else asked for absolute time.

## Measurement trap, for whoever tests line noise next

The first round of this investigation reported "the 60Hz oscillator works
perfectly, purity 1.000" and "there is no 50Hz energy at all". Both were artifacts
of measuring on a **per-patch** FFT: at `patch_len=50`, `fs=200` the bins are 4Hz
apart, so 60Hz lands exactly on bin 15 (all its energy in one bin -> "pure") while
50Hz has no bin at all (nearest 48/52 -> "absent"). That measures grid alignment,
not ownership, and it points the opposite way from the truth.

Measure line noise on the **stitched trial** (overlap-add, `T=800` -> 0.25Hz
resolution), never per patch. Also prefer absolute in-band energy normalized into an
ownership share over a per-stamp fraction: a fraction says "this stamp is pure 50Hz"
but cannot say whether any other stamp shed its 50Hz.
