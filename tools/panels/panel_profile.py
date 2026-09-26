"""profile: parameter counts and per-component forward timing of a fresh model built from the config
(no checkpoint needed; without --run the pretrain template). Option: --train (train-mode timing)."""
from tools.analysis.profile import run_profile

STAGES = frozenset({'pretrain'})


def run(ctx):
    r = run_profile(ctx.config, ctx.device, train_mode=ctx.args.train)
    print(f"Model {r.model_type} on {ctx.device}, {'train' if ctx.args.train else 'eval'} mode; input "
          f"B={r.batch} C={r.channels} N={r.patches} L={r.patch_len}")
    print(f"{'component':<22} | {'params (M)':<10} | {'time (ms)':<10} | % total")
    for name, _ in r.children:
        t, p = r.time_stats.get(name, 0.0), r.param_map.get(name, 0)
        print(f"{name:<22} | {p / 1e6:<10.2f} | {t:<10.2f} | {100 * t / r.total_ms:6.1f}%")
    print(f"{'get_loss':<22} | {'-':<10} | {r.loss_ms:<10.2f} | {100 * r.loss_ms / r.total_ms:6.1f}%")
    print(f"{'total':<22} | {sum(r.param_map.values()) / 1e6:<10.2f} | {r.total_ms:<10.2f} | 100.0%")
