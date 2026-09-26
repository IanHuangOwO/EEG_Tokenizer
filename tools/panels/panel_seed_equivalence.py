"""seed_equivalence: for multi-seed finetunes (output/<backbone>/finetune/<head>_seed<s>/<cell>), each
group vs --ref: seed spread, paired difference and a TOST equivalence test (tools/analysis/
seed_equivalence.py) -> seed_equivalence.md. Option: --margin (default 0.02 = 2 points)."""
import os

from tools.analysis.seed_equivalence import compare

STAGES = frozenset({'finetune'})


def run(ctx):
    if not ctx.ref:
        raise RuntimeError('seed_equivalence needs --ref')
    text = ''.join(compare(ctx.groups[ctx.ref], bb, ctx.head, ctx.args.margin, ctx.args.metric)
                   for g, bb in ctx.groups.items() if g != ctx.ref)
    if not text:
        raise RuntimeError(f"no seeded runs (output/<backbone>/finetune/{ctx.head}_seed*/) for these groups")
    out = os.path.join(ctx.out_dir, 'seed_equivalence.md')
    with open(out, 'w') as f:
        f.write(text + '\n')
    print(text)
    print(f"  -> {out}")
