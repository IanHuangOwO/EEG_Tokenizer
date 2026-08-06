# DeepSeekMoE-style shared+routed Expert FFN for MeSAE's TSABlock

## Context

`TSABlock.ffn` (`model/MeSAE/MeSAE_modules.py`) was a single dense 2-layer `FFN` applied
identically to every `(B, C, N)` token — no competitive pressure, no specialization,
unlike the SAE Filter pool which already got a routed+shared split (`FilterRouter`,
`docs/adr/0007-routed-filter-gating-for-mesae.md`). Goal: apply the same DeepSeekMoE
routed+shared Expert pattern to the FFN sub-layer itself, at per-token granularity —
matching DeepSeek's own MoE-FFN design (routed Experts top-k gated, competing for a fixed
per-token budget, plus always-on shared Experts summed at full weight).

Scope deliberately narrower than `docs/adr/0007`: MeSAE only (not MeFSQ), a **new**
dedicated router (not `FilterRouter` reuse — that one scores a small fixed pool of
pre-pooled Filter Views via a dot-product weight; this one scores every raw token directly
via a plain linear gate, since token count here is `B*C*N`, not `n_filters`).

## Decision

1. **`FFNRouter(dim, n_routed, top_k)`** (`MeSAE_modules.py`, alongside `FilterRouter`) —
   `nn.Linear(dim, n_routed)` gate logits, top-k softmax-normalized weights scattered into
   a dense `[T, n_routed]` gate mask, plus the same Switch-Transformer-style load-balance
   loss formula `FilterRouter` already uses.

2. **`MoEFFN(dim, hidden_dim, n_routed, n_shared, top_k, expert_hidden=None, dropout=0.0)`**
   — `routed_experts`/`shared_experts` are `nn.ModuleList`s of the existing `FFN` class
   (each Expert is just a smaller instance of the same 2-layer MLP). `expert_hidden`
   defaults to `max(8, hidden_dim // (n_shared + top_k))` — standard DeepSeekMoE
   fine-grained-Expert sizing, keeping *active* per-token compute (n_shared + top_k Experts
   firing) roughly at parity with the old single dense FFN. Routed Experts: dense-compute
   every Expert on every token then combine via the gate mask (same "compute all, mask by
   gate" convention `FilterRouter`/`ExpertChannelPool` already use in this file — not real
   sparse dispatch; fine at this Expert count, revisit if it becomes the bottleneck). Shared
   Experts: always run, summed at **full weight** — true DeepSeekMoE shared-Expert
   semantics, deliberately *not* `ExpertChannelPool`'s 0.2x-down-weighted shared-Filter
   convention (a different, already-established choice for the VQ side, not what a
   DeepSeekMoE FFN calls for).

3. **`TSABlock`/`TSAEncoder`** thread the MoE hyperparams down and now return
   `(x, ffn_lb_loss)` instead of just `x` — `TSAEncoder.forward` sums each block's
   `ffn_lb_loss` into one scalar per encoder pass.

4. **Two distinct load-balance losses, two distinct names** — `MeSAEPretrain.forward`'s
   output namespace carries both `filter_lb_loss` (the SAE Filter router, renamed from the
   previous generic `lb_loss`) and `ffn_lb_loss` (summed across all `TSABlock`s' `MoEFFN`
   routers). `get_loss` takes both as separate weighted terms (`lb_weight` for
   `filter_lb_loss`, `ffn_lb_weight` for `ffn_lb_loss`, both default `0.01`) — kept apart
   rather than merged into one number so each MoE's health is independently visible in the
   training dashboard and independently tunable.

5. **`filter_lb_loss` is dropped once `sae_frozen`** (same as before — it lives in the
   frozen Filter/router/SAE path, see `docs/adr/0007`). **`ffn_lb_loss` is added
   unconditionally, both stages** — it comes from the encoder's `TSABlock`s, and
   `freeze_sae()` never locks the encoder (only `filter_pool`/`sae`/`decoder`/`router`), so
   the MoE-FFN keeps training straight through the Masked stage.

## Consequences

- Breaking architecture change: `TSABlock.ffn`'s state-dict keys change entirely (single
  `FFN`'s `fc1`/`fc2` → `MoEFFN`'s `routed_experts.*`/`shared_experts.*`/`router.gate`) —
  old MeSAE tokenizer checkpoints will not load into the new `TSABlock`. Retraining from
  scratch is expected; no backward-compat shim.
- `config/config.json`: new `model_params.MeSAE.pretrain.moe_ffn` block
  (`n_routed_experts`, `n_shared_experts`, `top_k`, `expert_hidden`) and
  `model_params.MeSAE.pretrain.loss.ffn_lb_weight`.
- Training dashboard's Router Health panel now plots both `filter_lb_loss` and
  `ffn_lb_loss` side by side — watch both for the standard MoE collapse failure mode
  (a few Experts absorbing all top-k picks), same as `docs/adr/0007`'s existing concern for
  the Filter router.
- MeFSQ is untouched — its own `Router`/Expert-pool design (`docs/adr/0001`,
  `docs/adr/0002`) is unrelated to this change.
