"""backbone_eval: held-out reconstruction under fixed test masks vs interpolation baselines, embedding
ablations and structure checks (tools/analysis/backbone_eval.py) -> analysis/backbone_eval.json, which
analysis_finetune.py's report panel reads. Options: --max-windows (default 512)."""
import os

from tools.analysis.backbone_eval import evaluate

STAGES = frozenset({'pretrain'})


def run(ctx):
    evaluate(ctx.model, ctx.config, os.path.join(ctx.out_dir, 'backbone_eval.json'),
             max_windows=ctx.args.max_windows, name=ctx.config['training_params']['pretrain']['model_name'])
    ctx.model.to(ctx.device)   # evaluate() runs on the CPU
