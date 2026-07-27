# Per-Expert channel-attention pooling, replacing the concatenated Token

Amends `0001-moe-style-quantization.md`. That ADR chose a single concatenated Token (all C channels' embeddings, C\*D wide) shared identically by the Router and every Expert, on the grounds that "a single joint code can capture cross-channel structure that independent per-channel quantization can't."

Concatenation ties each channel to a fixed position in the C\*D vector. That's fine for a single fixed-channel-layout dataset, but breaks under multi-dataset training (`CLAUDE.md`: "Channels are unified from the first dataset; other datasets are mapped onto that channel space (missing channels zero-padded)") — a zero-padded slot still occupies a fixed position in the concatenated vector and pollutes it, and the same channel can land in a different slot across datasets with different channel sets.

Replaced concatenation with per-Expert content-based channel attention: every Expert (Routed and Shared pool alike) has its own learnable query attending over the C per-channel embeddings (keys), producing its own pooled D-wide **Expert View** (see `CONTEXT.md`). Padded/invalid channels (`valid_channels`, already produced by `IO/dataset.py` but previously only consumed by the finetune path) are masked to −inf before softmax, so channel count and ordering can vary freely across datasets without corrupting the pooled result.

The Router scores each routed Expert against that Expert's own View (`dot(router.weight[e], expert_view_e)`), not a separate shared input — the gating decision and the quantized input are always the same vector, so "why did this patch route to Expert e" stays answerable from what Expert e actually encoded.

Trade-offs accepted:
- No longer one shared input across all Experts — each Expert may weight channels differently for the same patch. This is a deliberate reversal of 0001's "one shared code must explain all channels jointly" framing.
- More attention parameters than one shared pooling (a per-Expert query/key pair per Expert instead of one), though far smaller than the C\*D-wide `Router.weight`/`MeFSQ.A` matrices this replaces.
- `valid_channels` masking must now be threaded through the pretrain path (`MeFSQPretrain.forward`/`stage_features`), not just finetune — a real increase in plumbing surface.

Hard to reverse (retrain required, changes what "Router" and "Code" mean throughout the pipeline) and a genuine trade-off (uniform shared input vs. per-Expert specialized input), so recorded here rather than left implicit in code.
