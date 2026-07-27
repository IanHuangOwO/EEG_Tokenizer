# MeSAE two-stage training: local tokenizer stage, frozen-SAE masked stage

MeSAE currently trains single-phase, unmasked, with a 12-block `TSAEncoder` (same depth as MeFSQ's) in front of the per-filter `TopKSAE` — masking was deliberately deferred at design time (`MeSAEPretrain` docstring). We want MeSAE to gain a masked-patch pretext task. Two structures were on the table: (a) mirror MeFSQ's `freeze_vq_and_decoder()` pattern — train tokenizer and backbone *jointly* through a warmup phase, then freeze and switch on masking; (b) train the tokenizer to convergence standalone first (LaBraM-style "neural tokenizer"), freeze it, then train only the backbone against masked input with the frozen SAE's output as reconstruction target.

Chose **(b)**. Two reasons:

1. **Target leakage.** A 12-block encoder with cross-patch temporal attention and cross-channel spatial attention already contextualizes each patch against the whole unmasked sequence before the SAE pools it. Using that as a frozen masked-prediction target lets the backbone partly copy leaked context instead of inferring it — the same failure mode BEiT's dVAE and LaBraM's neural tokenizer avoid by keeping the tokenizer local/shallow. MeSAE's tokenizer-stage encoder must therefore shrink toward per-patch locality (much lower depth, and/or drop cross-channel spatial attention), leaving relationship-learning entirely to the masked stage.
2. **Sparsity loss becomes pointless once frozen.** `TopKSAE.aux_loss` exists to rescue *this SAE's own* dead features (EMA-gated, second-chance decode). Once `.sae`'s weights stop updating in the masked stage, rescuing its dead features is meaningless — `aux_loss`/`aux_weight` is dropped entirely from the masked-stage loss, not just left at zero.

The joint-warmup pattern MeFSQ uses (b's rejected alternative) was justified there by VQ/router codebook stability needing warmup before mask noise hits — that reasoning doesn't transfer to a non-discrete, non-routed SAE, and joint training would reintroduce exactly the target-leakage problem (1) is avoiding.

## Consequences

- `model/MeSAE/MeSAE.py` tokenizer-stage encoder config shrinks (lower `enc_depth`, possibly drop spatial cross-channel attention) — a new architecture, not a hyperparameter tweak, so old `mesae_v1` checkpoints are not compatible with the retrained tokenizer.
- `train_tokenizer.py` (standalone, single-phase, no masking) is retired — its role is absorbed as "stage 1" of a unified two-phase script, structurally parallel to how `train_pretrain.py` already does `vq_warmup_epochs` → `freeze_vq_and_decoder()` for MeFSQ. `training_params.tokenizer` config section is retired with it.
- Retraining the tokenizer stage was going to happen anyway (EEGMMIdb needs adding to the training set), so no completed run is being discarded by this change.
