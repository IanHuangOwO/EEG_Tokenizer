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


def nonfinite_step_report(l_total, model, out=None):
    """Returns None when `l_total` is finite (the normal path), else a short string
    naming WHERE the non-finiteness lives, for the caller to log before skipping the
    step.

    Exists because a single NaN loss used to be terminal: v13's tokenizer run went NaN
    at epoch 8 and every later epoch reported nan for every metric (topk on NaN scores
    returns the leading indices, so router entropy also read a fake ~0 collapse). The
    caller must skip backward/step on a non-finite loss — AMP's GradScaler skips a step
    whose GRADIENTS are non-finite, but nothing stops a poisoned forward from being
    backwarded through, and nothing un-poisons parameters once they are NaN.

    The report distinguishes the two causes that need different fixes:
      - 'params' non-finite  -> the weights are already dead; skipping cannot save the
        run, the instability is upstream (lower LR / stronger clipping / fp32 island).
      - params fine, forward tensor non-finite -> a per-batch fp16 overflow; skipping
        that batch IS the fix.
    """
    import torch

    if torch.isfinite(l_total):
        return None

    bad_params = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    bad_bufs = [n for n, b in model.named_buffers()
                if b.is_floating_point() and not torch.isfinite(b).all()]
    bad_out = []
    for name in ('recon', 'aux_loss', 'ffn_lb_loss', 'h', 'dense_routed', 'k_eff'):
        t = getattr(out, name, None) if out is not None else None
        if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
            bad_out.append(name)

    return (f"non-finite loss ({l_total.item()}): "
            f"params={bad_params[:5] or 'ok'} buffers={bad_bufs[:5] or 'ok'} "
            f"out={bad_out or 'ok'}")
