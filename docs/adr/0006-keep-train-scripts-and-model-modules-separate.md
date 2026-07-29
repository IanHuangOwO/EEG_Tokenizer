# Keep train_tokenizer.py/train_pretrain.py and MeFSQ_modules.py/MeSAE_modules.py duplicated, not factored

An architecture review (`/improve-codebase-architecture`) flagged two large near-duplicated
surfaces as deepening candidates:

- `train_tokenizer.py` / `train_pretrain.py`: `setup_logger`, `_unpack_batch`,
  `train_one_epoch`, `validate_one_epoch`, dataset-split, and the epoch loop are
  line-for-line duplicated between the two scripts.
- `model/MeFSQ/MeFSQ_modules.py` / `model/MeSAE/MeSAE_modules.py`: 8 of 10 classes
  (`SpatialTemporalEmbeddings`, `ConvolutionalAdditiveAttention`, `FFN`, `TSABlock`,
  `TSAEncoder`, `MultiHeadDecoder`, `ExpertChannelPool`, `PerChannelHeadAttn`) are
  near-verbatim duplicated between the two model dirs.

Decided **not** to factor either into a shared module. Both scripts and both model dirs
stay separately owned on purpose — MeFSQ and MeSAE (and future models/stages) are allowed
to diverge in their training loop and encoder internals independently, without a shared
module forcing them to stay in lockstep or requiring a change in one to be reasoned about
against the other's constraints.

## Consequences

- Bugfixes to the training loop or shared encoder blocks (grad clipping, AMP handling,
  `TSABlock` mixing, etc.) must be applied by hand in each copy — no compiler/test catches
  drift between them.
- Future architecture reviews should not re-flag `train_tokenizer.py`/`train_pretrain.py`
  duplication or `MeFSQ_modules.py`/`MeSAE_modules.py` duplication as candidates; this ADR
  records that the duplication is deliberate.
- This does not extend to `model/base_trainer.py`/`base_checker.py`/`base_plotter.py`
  (ADR-0004) — those stay shared; only the per-script training loop and per-model encoder
  internals are exempted here.
