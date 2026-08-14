"""BaseTrainer: per-step loss, per-epoch metrics, stage1->stage2 transition — the
per-model hook a model dir (model/<Name>/plugin.py) implements so train_pretrain.py's
loop has no if/elif model_type dispatch. See docs/adr/0004-model-plugin-base-classes.md.
"""


class BaseTrainer:
    """One instance per model_type, stateless except for whatever the model itself holds."""

    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        """Returns (l_total, l_masked, l_unmasked). Pure loss math — does not mutate
        model state. See update_diagnostics for the per-step EMA side effect this used
        to carry."""
        raise NotImplementedError

    def update_diagnostics(self, model, out):
        """Called once per step (train and val alike), right after compute_loss, with
        that same step's `out`. Mutates model-side EMA buffers (e.g. MeFSQ/MeSAE's
        update_head_metrics) — the one place per-step diagnostic state gets touched.
        Default: no-op, for models with nothing to track here."""
        pass

    def epoch_metrics(self, model, out):
        """Returns a dict of extra metrics to merge into the epoch's averaged totals
        (codebook perplexity, STE gap, head diversity, ...)."""
        raise NotImplementedError

    def on_tokenizer_start(self, model, logger=None):
        """Called once, before train_tokenizer.py's epoch loop starts. Enables
        spatial (+temporal, if the model has it) mixing so the encoder trains with
        full mixing capability from the start of the Tokenizer stage, unmasked —
        same pattern model/factory.py's build_finetune_from_config uses. Generic
        across models, no per-plugin override needed."""
        model.enable_spatial()
        if hasattr(model, 'enable_temporal'):
            model.enable_temporal()
        if logger:
            logger.info("  [Tokenizer] spatial" + (" + temporal" if hasattr(model, 'enable_temporal') else "") + " enabled")

    def on_pretrain_start(self, model, logger=None):
        """Called once, at the start of train_pretrain.py, right after loading the
        Tokenizer-stage checkpoint. Freezes whatever this model's tokenizer-only
        component is (VQ+decoder for MeFSQ, SAE for MeSAE) so only the main
        transformer keeps learning through masked reconstruction."""
        raise NotImplementedError
