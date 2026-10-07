"""long_context (on request): masked MSE on held-out inputs 1x / 2x / 4x the 5 s training window, joined from
contiguous cache rows (tools/analysis/long_context.py) -> analysis/long_context.json. Compares how backbones' time
position handles inputs longer than training (FoPE vs table / Fourier, 2026-10-07)."""
import os

from tools.analysis.long_context import evaluate

STAGES = frozenset({'pretrain'})


def run(ctx):
    evaluate(ctx.model, ctx.config, os.path.join(ctx.out_dir, 'long_context.json'),
             name=ctx.config['training_params']['pretrain']['model_name'])
    ctx.model.to(ctx.device)   # evaluate() runs on the CPU
