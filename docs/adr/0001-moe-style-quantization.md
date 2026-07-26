# DeepSeekMoE-style routed+shared Experts over a fused cross-channel Token

Quantization moved from one independent Code per channel to a single Token per (batch, patch) — all C channels' embeddings concatenated (C*D wide) — quantized jointly by a pool of Experts. A `Router` top-k gates a Routed pool for specialization; a Shared pool stays always-on as a down-weighted (0.2x) baseline contribution. Both pools are still densely computed (unselected routed Experts masked to zero, not sparsely dispatched) — fine at patch-level EEG scale, but sparse dispatch is the noted upgrade path if `n_routed_experts` grows large.

This was chosen over per-channel independent Codes because a single joint code can capture cross-channel structure that independent per-channel quantization can't. Trade-off: harder to reverse (retrain required, changes the meaning of "one code" throughout the pipeline) and less obvious to a reader expecting classic per-channel VQ.
