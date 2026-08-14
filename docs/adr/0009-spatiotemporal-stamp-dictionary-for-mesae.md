# Spatiotemporal stamp dictionary replaces Filter+SAE+decoder chain (MeSAE)

## Context

Today's MeSAE pipeline nests two separate sparsity mechanisms with two separate
decode steps:

```
z [B,C,N,D]
  -> ExpertChannelPool         (n_filters=36 static channel-unmix rows, spatial only)
  -> FilterRouter               (top-4-of-36 Filters, softmax-normalized gate)
  -> TopKSAE                    (shared tied-linear dict, D-dim, top-32-of-800 atoms)
  -> MultiHeadDecoder            (per-Filter LINEAR, D -> C*patch_len)
  -> sum over Filters -> recon
```

Two problems motivate collapsing this:

1. **Two independent atom vocabularies stacked** (36 spatial Filters x 800 shared
   SAE atoms) means the real per-patch "vocabulary size" is a product, not a sum,
   and `docs/adr/0007`'s router-collapse history shows the Filter layer already
   struggles to specialize spatially on its own.
2. **Both decode steps are linear by design** (`TopKSAE`'s tied-weight decoder,
   `MultiHeadDecoder`'s per-Filter linear map — the latter's docstring records an
   explicit revert from a nonlinear 2-layer version, reasoning "a nonlinear
   per-Filter decoder can bend around arbitrary patch content on its own"). Linear
   decode can't represent amplitude-dependent waveform *shape* change (e.g. a
   spike that broadens as it grows), only amplitude scaling of a fixed shape.

## Decision

Replace `ExpertChannelPool` + `FilterRouter` + `TopKSAE` + `MultiHeadDecoder` with
one `StampBank` module: a dictionary of `n_stamps=800` **rank-1 spatiotemporal
atoms**, each `stamp_i = (u_i in R^C, g_i: R -> R^patch_len)` — a static spatial
topography (identical role/init to today's `ExpertChannelPool.spatial_logit`, just
800 rows instead of 36) paired with its own small generator MLP. One selection
stage replaces the two nested ones.

**Routed/shared split, carried over from `docs/adr/0007`'s Filter design**:
`n_stamps=800` = `n_routed_stamps=796` (compete via score+top-32) +
`n_shared_stamps=4` (always included, fixed `shared_weight=0.2`, never enters
the score/top-k competition). A shared stamp's `h_i` is the constant
`shared_weight`, not content-derived — same convention as today's
`gate_shared = pooled.new_full(..., self.shared_weight)`. Since `phi_i`'s only
input is `h_i` (see step 5), a shared stamp's `phi_i(shared_weight)` is
literally constant across every patch — the stamp degenerates into a learned
fixed spatiotemporal pattern, a natural fit for "shared similarity across EEG
patches" (baseline structure common to most patches, not patch-specific).
Total atoms generated per patch: `32 routed + 4 shared = 36` — coincidentally
the same total as today's `n_filters=36`, despite a completely different
vocabulary structure underneath.

**Forward, per patch:**

1. **Spatial pool, dense over all 800** (reuse `ExpertChannelPool` math unchanged,
   `n_experts=800`): `u: [800, C]`, `attn = softmax(u, dim=-1)`,
   `pooled = einsum('hc,mcd->mhd', attn, z_bnc)` -> `[M, 800, D]`. Cheap, same
   einsum already in the codebase, no MLP yet.
2. **Score, dense over the 796 routed only** (reuse `FilterRouter`'s scoring math,
   `score = einsum('mhd,hd->mh', pooled_routed, w) * scale` -> `[M, 796]`,
   `w: [796, D]` a second, separate learned param from `u`). Still cheap, no MLP
   yet. Shared stamps skip scoring entirely.
3. **Select, TopKSAE convention exactly** (not `FilterRouter`'s
   softmax-normalized gate — see deviation below): `topk_val, topk_idx =
   score.topk(32)`; `h_routed = relu(topk_val)` -> `[M, 32]`, raw unbounded
   strength, matching `TopKSAE.forward`'s `F.relu(topk_val)` exactly.
   `h_shared = full([M, 4], shared_weight)` — always present, constant.
   `h = cat([h_routed, h_shared])` -> `[M, 36]`.
4. **Gather selected atoms' params** (sparse-dispatch boundary — see
   Consequences): `u_sel = u[cat(topk_idx, shared_idx)]` -> `[M,36,C]`,
   `phi_sel` gathered per-atom MLP weights the same way, `pooled_sel =
   pooled.gather(...)` -> `[M,36,D]` (kept only for the finetune head, step 6b —
   never fed to `phi`). The 4 shared indices are fixed/constant across the
   batch, so their gather is effectively free (no per-example indexing needed).
5. **Generate, selected atoms only, scalar input only**:
   `waveform = phi_sel(h)` -> `[M,36,patch_len]`. `phi_i`'s *only* input is the
   scalar `h_i` — never `pooled_i`/`z` — the deliberate anti-shortcut
   restriction (see Stamp capacity below and the flagged risk in Consequences).
6. **Recon, unweighted sum** (no external multiply by `h_i` — `phi_i` alone
   decides its own output amplitude): `contribution = einsum('mkc,mkp->mkcp',
   u_sel, waveform)` -> `[M,36,C,patch_len]`; `recon = contribution.sum(dim=1)`
   -> `[M,C,patch_len]`. Unselected routed atoms contribute nothing because they
   were never gathered/computed (hard top-k), not because of a zero-at-origin
   trick — no such trick is needed under this hard-selection scheme.

**Stamp capacity**: `phi_i` is a 1-hidden-layer MLP, width `<= patch_len/4`
(leaning toward the narrow end deliberately — no strong reason found to go
wider). Note the shortcut-memorization risk that killed the old nonlinear
`MultiHeadDecoder` does **not** transfer here and narrow width isn't a guard
against it: that decoder took a full D-dim code as input, high-dimensional
enough to carry arbitrary patch-specific content; `phi_i` takes a single scalar
(`h_i`, this atom's own firing strength) — there is no patch-specific
information in a scalar, so `phi_i` structurally cannot memorize per-patch
content regardless of width. Width is instead a plain capacity/param-count
tradeoff (how much amplitude-dependent shape-bending is useful per atom at
`n_stamps=800`), and the smaller side was chosen there, not for safety.

**Risk flag — scalar-only `phi_i` input may be too restrictive.** Confirmed
deliberately (not `pooled_i`, to preserve the anti-shortcut property above), but
this is a real bottleneck: `phi_i` must reconstruct an entire `patch_len`
waveform's *shape* from one scalar, with zero information about which patch,
which trial, or what the rest of the signal looks like — all shape variation
has to come from `h_i`'s magnitude alone. If training shows the reconstruction
loss plateauing far above today's baseline, or `phi_i` outputs collapsing to a
near-constant shape regardless of `h_i` (i.e. the nonlinearity buys nothing and
this reduces to a linear atom in practice), that's this constraint biting, not
a bug — the fix then is loosening the input (letting `phi_i` see a small
low-dimensional summary of `pooled_i` instead of the raw scalar) at the direct
cost of reopening the shortcut-memorization question this constraint exists to
avoid. Treat as the first thing to check if this architecture underperforms.

**Diversity**: extend `ExpertChannelPool.decorrelation_loss` (pairwise cosine sim
of `u_i` rows, off-diagonal, unchanged formula) to run at `n=800`. No tied-weight
decoder anymore, so no RICA-style free decorrelation pressure on the *temporal*
side (`g_i`) — open question, not blocking: start with spatial-only decorrelation
(cheap, proven) and watch `filter_relation.png`-equivalent panels before adding a
temporal-diversity term.

**Dead-atom handling**: port `TopKSAE`'s `fire_ema` / `dead_threshold_frac` /
aux-k rescue mechanism onto the 796-wide routed selection only — shared stamps
are always "alive" by construction (constant `h_i`, never subject to the
top-k competition dead atoms lose), so they're excluded from `fire_ema`
tracking entirely, mirroring how today's shared Filters sit outside the
router's load-balance accounting too. Not a byte-identical port: today's rescue
is cheap because `TopKSAE`'s decode is a tied **linear** matrix
(`residual_hat = h_aux @ dec_n`, one matmul regardless of atom count); here
decode is per-atom-nonlinear, so rescuing dead atoms means actually running
their `phi_i` (real, if capped and small, extra generator forwards —
`aux_k_cap_frac`-scaled to 796, same formula, typically a few dozen atoms).
Rescue input is the dead stamp's own pre-topk `score` (step 2), same role as
today's `aux_val`; rescue target is `outer(u_i, phi_i(score_rescue))` per
rescued atom instead of the linear tied decode, same residual-normalized MSE
objective otherwise.

**Finetune head**: `PerChannelHeadAttn` is **unchanged** — finalized as
pre-generator, not post-generator. It reads `z_h = h_i * pooled_i` (the D-dim,
per-stamp view *before* `phi_i`/`g_i` ever runs — same shape contract as today's
`encode_post_sae_expert` output) for the 36 selected stamps (32 routed + 4
shared), zero elsewhere. `phi_i` never runs during finetune (backbone frozen, no
reconstruction target) — cheaper than today's finetune path, which currently
still runs the full decode. Rejected post-generator (head reading the actual
generated waveform): would tie classifier capacity to reconstruction-generator
capacity and force `PerChannelHeadAttn` to accept patch-space input instead of
D-dim, for no benefit identified.

## Consequences

- **Collapses two nested sparsity stages into one**: top-4-of-36 x top-32-of-800
  -> top-32-of-796-routed (+4 always-on shared), single selection stage, single
  vocabulary. Removes the Filter-layer specialization problem `docs/adr/0007`
  fought, by removing the Filter layer as a separate concept — spatial pattern
  now lives per-stamp, alongside that stamp's own temporal generator, not in a
  separate upstream pool. Routed/shared split itself survives unchanged from
  `docs/adr/0007`, just applied at the stamp level instead of the Filter level.
- **`ExpertChannelPool` math is reused as-is** (just resized), `FilterRouter`'s
  scoring math is reused but its softmax-normalize step is deliberately dropped —
  document this divergence at the call site so it doesn't read as an oversight.
- **`TopKSAE` and `MultiHeadDecoder` are retired** for MeSAE's main path (their
  dead-atom-EMA and "sum after decode" logic survive conceptually, ported into
  `StampBank`).
- **New failure mode to watch**: sparse dispatch. Naively evaluating all 800
  `g_i` then masking (the "compute-all-then-gate" convention `MoEFFN` already
  uses elsewhere in this codebase, see its `ponytail:` comment) does not scale
  here — 800 real MLP forwards per patch vs. today's single 800-wide matmul (a
  single dense linear layer, not 800 separate MLPs). Resolved via plain
  `nn.Parameter` gather (`u[topk_idx]`, `W1[topk_idx]`, etc. — standard
  embedding-lookup-style indexing, no custom kernel/vmap needed) followed by a
  batched einsum over only the 36 selected atoms per example — see the worked
  example in-thread. Genuinely different from every other MoE-ish path in this
  codebase (all dense-then-mask), but not exotic to implement.
- **Tokenizer-stage-only change**, same as `docs/adr/0007`: Masked stage already
  freezes the whole SAE apparatus (`freeze_sae`); `StampBank` (spatial rows +
  scorer + generators) freezes the same way, same rationale.
- Loss/get_loss wiring (`filter_lb_loss`, `decorr_loss`, `aux_loss`) carries over
  with renamed sources (`StampBank`'s own load-balance/decorrelation/aux terms
  replacing `FilterRouter`'s/`ExpertChannelPool`'s/`TopKSAE`'s).
- Monitoring/viz impact is large enough to need its own section — see below.
- **Param count drops sharply**: today's `MultiHeadDecoder` alone is
  `[n_filters=36, embed_dim=100, C*patch_len]` — at C=64, patch_len=20 that's
  `36x100x1280 ~ 4.6M` params in one tensor. `StampBank` at `n_stamps=800`:
  `800 x (u:64 + w:100 + phi:~210) ~ 800x374 ~ 300K` total — roughly 15x fewer
  params despite the redesign, a consequence of factoring each atom into three
  narrow low-rank roles (spatial/recognizer/generator) instead of one dense
  per-Filter matrix touching the full `C*patch_len` output space directly.

## Monitoring impact

Every panel/metric that currently keys on `Filter`/`SAE` internals, swept file
by file. Most are mechanical renames; three are real rewrites, called out
explicitly.

**`model/MeSAE/plugin.py` (`MeSAETrainer`/`MeSAEChecker`/`MeSAEPlotter`)**
- `compute_loss`: `out.filter_lb_loss` -> `out.stamp_lb_loss`. Mechanical.
- `update_diagnostics`: currently `model.update_head_metrics(out.gate[:,
  :model.n_routed_filters])` feeds `FilterRouter`'s dense **softmax-normalized**
  gate into entropy formulas that assume a probability distribution (rows sum
  to ~1). The stamp equivalent (dense scatter of `h_routed`, `[M,796]`, zeros at
  unselected) is **raw and unbounded** — entropy computed on it directly is not
  meaningful. **Real fix, not a rename**: renormalize
  (`h_routed / h_routed.sum(-1, keepdim=True)`) purely for this diagnostic,
  never touching the actual reconstruction path.
- `on_pretrain_start`: `model.freeze_sae()` -> `model.freeze_stamps()`.
- `unit_label = 'Filter'` -> `'Stamp'` on both `MeSAEChecker` and
  `MeSAECodebookChecker` — touches every panel title/output filename either
  produces.
- `compute_unit_colors`: same black(routed)/red(shared) split, resized to
  `model.n_stamps` (800) from `model.n_filters` (36). Mechanical.
- `render_finetune_attn`: `backbone.encode_post_sae_expert` ->
  `encode_post_stamp_expert`. Shapes `[1,N,Q,D]`/`[1,N,Q,C]` stay compatible
  since selected `Q=36` (32 routed + 4 shared) matches today's `n_filters=36`
  exactly — no downstream shape changes in this function despite the rename.

**`viz/extract.py` — `extract_filter_psd`/`extract_filter_spectra`, real
rewrite.** Both call `model.filter_pool(...)`, `model.sae(...)`,
`model.decoder(...)` directly (bypassing `forward()`), deliberately **un-gated**
— every Filter's natural PSD/spectrum regardless of whether the router selected
it (see `decoder`'s "un-gated by design" comment). Stamp equivalent: run
**every stamp's `phi_i` at a fixed canonical probe** (`h=1.0` — same probe
already used for the temporal-decorrelation fingerprint below), not just the 36
actually selected for a given patch — `800` tiny-MLP forwards, fine at
diagnostic-call cadence (this runs per checker invocation, not per training
step, unlike the sparse-dispatch requirement in the main forward path).

**`model/base_codebook_checker.py` (`MeSAECodebookChecker`)**
- `extract_usage`: reads `out.sae_hidden` `[M,Q,F]` (per-Filter x per-SAE-feature
  activation). No `F` axis exists anymore. Stamp equivalent: the dense
  stamp-hidden tensor `[M,800]` (zeros at unselected, `h_routed` at
  selected-routed, constant `0.2` at the 4 shared) — same object needed for
  `update_diagnostics` above, reused here.
- `decoder_fingerprint_matrix`: reads `model.decoder.w` `[Q,embed_dim,patch_len]`,
  cosine-sims flattened rows — this **is** `filter_relation.png`, the exact
  diagnostic that motivated `docs/adr/0007` in the first place. Stamp
  equivalent: `flatten(outer(u_i, phi_i(probe=1.0)))` per stamp, same pairwise
  cosine formula. This panel doubles as the empirical test for the
  Diversity section's still-open "is spatial-only decorrelation enough, or do we
  need a temporal term too" question — if it stays high the way the original
  ADR-0007 problem did, that's the trigger to add one.
- `rank_ceiling`: `min(model.sae.k, model.head_dim)` ->
  `min(model.stamps.top_k, model.head_dim)` (32, value unchanged).

**`MeSAEPlotter.plot_pretrain` panels**
- `l0_sparsity_routed` keeps real meaning (count of the 32 routed picks with
  `h_i>0`, can be `<32` since `relu` zeroes weak scores). `l0_sparsity_shared`
  becomes **trivial** — shared stamps have no on/off dynamics anymore (always
  on, fixed weight), so this series would just plot a flat `4`. Drop it from
  the panel rather than plot a constant.
- `dead_feature_rate`: unchanged mechanism, scoped to the 796 routed stamps.
- "Filter Router Health" -> "Stamp Router Health", same 3-metric panel shape,
  sourced through the renormalization fix above.
- Architecture panels (skip gates, block norms): **untouched** — encoder-level,
  `StampBank` doesn't touch `TSAEncoder`.

## Update: no load-balance loss (dropped, not deferred)

`StampBank.forward` originally carried a Switch-Transformer-style `lb_loss`
(`n_routed * sum((f/sum(f)) * (p/sum(p)))`, `f`=hard fire frequency, `p`=softmax
of `score` — same formula `FilterRouter`/`FFNRouter` use). Removed entirely, for
reasons specific to this module that don't apply to those two:

1. **The compute-balance rationale it exists for doesn't apply here.** `MoEFFN`
   (docs/adr/0008) dense-computes every Expert then masks — lb_loss spreads
   traffic so no Expert's compute sits wasted. `StampBank` dispatches sparsely
   (gather by `idx`), so there's no compute-balance problem to solve in the
   first place.
2. **It fights the score-unification decision above.** Since selection score
   was tied to the generator's own reconstruction quality
   (`score_i = -mean((pooled_i - z_hat_i)^2)`), `p = softmax(score)` is no
   longer an arbitrary router weight — it *is* content-fit quality. Pushing `p`
   toward uniform means punishing atoms for winning too often on merit, pulling
   directly against the fit-quality signal in the same forward pass.
3. **Uniform usage isn't actually the healthy target for a dictionary.** Unlike
   MoE-FFN Experts (interchangeable compute units, uniform load is the correct
   equilibrium), stamps are content-addressed atoms — legitimate sparse-coding
   dictionaries have power-law usage (some atoms broadly useful, most
   specialized/rare). Forcing uniformity fights that on purpose.

Dead-atom collapse — the actual pathology lb_loss was guarding against — is
handled by the existing `fire_ema`/`dead_threshold_frac`/`aux_loss` rescue
instead: narrower (only intervenes on atoms below a floor), and doesn't
homogenize the healthy majority's natural usage distribution the way lb_loss
did. `stamp_lb_weight` removed from `config.json`'s `loss` blocks;
`get_loss`/`compute_loss`/`epoch_metrics` no longer take or log a stamp
`lb_loss` term. "Stamp Router Health" panel's LB-loss twin-axis line simply
stops appearing (guarded by `key in self.history['train']` in
`router_health_series`) rather than needing its own panel edit.
