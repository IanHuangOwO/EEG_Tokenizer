"""profile panel: parameter counts and per-component forward-pass timing for a fresh
(untrained) model built from config/config.json -- no checkpoint, no dataset. Replaces
the old standalone profile_model.py; run via
`python analysis_pretrain.py --panel profile [--train]` or the same from
analysis_finetune.py -- both stages support it, see STAGES below."""
from tools.analysis.profile import run_profile

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = False
NEEDS_DATASET = False


def run(ctx):
    train_mode = getattr(ctx.args, 'train', False)
    mode_str = 'TRAIN (eigh skipped)' if train_mode else 'EVAL (eigh active)'
    print(f"Profiling on device: {ctx.device}  |  Mode: {mode_str}")

    result = run_profile(ctx.config, ctx.device, train_mode=train_mode)

    print(f"\nModel: {result.model_type}")
    print(f"Input: Batch={result.batch}, Channels={result.channels}, "
          f"Patches={result.patches}, Samples={result.patch_len}")
    print("-" * 60)
    print("Detected Components:")
    for name, _ in result.children:
        p = result.param_map.get(name, 0)
        print(f"  - {name:<20} : {p / 1e6:>6.2f} M params")
    print("-" * 60)

    print("\nPerformance Summary (Avg of 20 runs):")
    print(f"{'Component':<22} | {'Params (M)':<10} | {'Time (ms)':<10} | {'% Total'}")
    print("-" * 70)
    for name, _ in result.children:
        t_ms = result.time_stats.get(name, 0.0)
        p = result.param_map.get(name, 0)
        print(f"{name:<22} | {p / 1e6:<10.2f} | {t_ms:<10.2f} | "
              f"{(t_ms / result.total_ms) * 100:>6.1f}%")
    print(f"{'Method: get_loss':<22} | {'-':<10} | {result.loss_ms:<10.2f} | "
          f"{(result.loss_ms / result.total_ms) * 100:>6.1f}%")
    print("-" * 70)
    total_params = sum(result.param_map.values())
    print(f"{'Total (Fwd + Loss + Rec)':<22} | {total_params / 1e6:<10.2f} | "
          f"{result.total_ms:<10.2f} | 100.0%")
    print("-" * 70)
