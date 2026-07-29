# Re-split train_tokenizer.py / train_pretrain.py

`docs/adr/0003-mesae-two-stage-masked-training.md` merged the standalone tokenizer script
into `train_pretrain.py` as `vq_warmup_epochs` (stage 1) → `on_stage2_start()` (stage 2),
one continuous run per model. That coupled two things that don't need to move together:
when the encoder's spatial/temporal mixing turns on, and when masking turns on. Both
stages' `enable_spatial()`/`enable_temporal()` calls only ever fired at the stage1→stage2
boundary, so the encoder never got to train its cross-channel/cross-patch mixing jointly
with the VQ/SAE except unmasked-and-about-to-freeze — you couldn't tune "how long does
mixing train" independently of "how long does masking stay off", or rerun one stage
without repeating the other.

Split back into two scripts, each producing its own checkpoint:

- **`train_tokenizer.py`**: builds the model, calls `trainer.on_tokenizer_start()`
  (enables spatial/temporal immediately — no waiting on stage 2), trains encoder+VQ/SAE
  jointly, unmasked, for `training_params.tokenizer.epochs`. Saves
  `output/<name>/tokenizer/best_tokenizer.pth`.
- **`train_pretrain.py`**: builds the same architecture, loads that checkpoint
  (`training_params.pretrain.tokenizer_checkpoint`), re-enables spatial/temporal (plain
  flags, not persisted in the state dict — same re-enable `build_finetune_from_config`
  already needed), calls `trainer.on_pretrain_start()` to freeze the VQ/SAE, then trains
  only the transformer against masked reconstruction.

`BaseTrainer.on_stage2_start` (enable + freeze, one call) is now two hooks:
`on_tokenizer_start` (generic — enable spatial/temporal, no per-model override) and
`on_pretrain_start` (per-model — freeze whatever the Tokenizer stage owns). Reusing
`on_tokenizer_start` inside `train_pretrain.py` to re-enable the flags after checkpoint
load, rather than duplicating that logic in the script, keeps the "how do I turn mixing
on" knowledge in one place.

## Consequences

- `training_params.tokenizer` config section is un-retired. `vq_warmup_epochs` is gone;
  Tokenizer-stage length is just `training_params.tokenizer.epochs`.
- Two checkpoints per run instead of one — `train_pretrain.py` now requires
  `training_params.pretrain.tokenizer_checkpoint` to point at a completed Tokenizer run.
- MeSAE's target-leakage concern (0003, reason 1) still applies to the *Tokenizer stage's
  own* encoder depth/spatial-attention choice — nothing here changes that; it changes only
  where the stage boundary lives (two scripts vs. one script's internal epoch counter).
