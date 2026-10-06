"""
Per-layer size and speed of a trained pretrain backbone, and what fp32 would cost vs the fp16 AMP it trains with.

    python -m tools.misc.profile_precision [--run qtome_tiny_p50_s16_real_s1] [--dataset Lee2019_MI] [--rounds 5]

Modes: fp16-amp (train_pretrain.py: fp16 autocast + GradScaler, TF32 on), fp32+tf32 (no autocast, TF32 matmuls),
fp32 (no autocast, TF32 off). Every mode trains a copy of the checkpoint's weights on the same real masked-phase
batches (val-style subjects of one pretrain dataset, masks at masked epoch 40). Reports:
  - parameters per module (weights are fp32 in every mode: AMP keeps fp32 master weights and casts per op),
  - forward ms per module and per block part (hooks sync the GPU at every module, so these add up to more than
    the unhooked step),
  - full train step ms (forward + loss + backward + clip + AdamW, no hooks), median over --rounds rounds that
    alternate the modes, so load from other GPU jobs hits each mode alike,
  - peak GPU memory of a train step,
  - eval masked MSE from the checkpoint's weights per mode, and its largest per-batch relative difference to fp32.
Unlike the profile panel (tools/analysis/profile.py: a fresh model, dummy input, forward only), this uses a
trained checkpoint, real batches and the train step.
"""
import argparse
import copy
import json
import statistics
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from IO.dataset import build_dataset_from_config, MontageBatchSampler
from IO.masking import build_masking_strategy_from_config
from model.factory import build_from_checkpoint, MODEL_REGISTRY, optimizer_param_groups
from train_pretrain import _unpack_batch
from tools.analysis import QTOME_OUTPUT

MODES = ('fp16-amp', 'fp32+tf32', 'fp32')


def set_mode(mode):
    tf32 = mode != 'fp32'
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    return mode == 'fp16-amp'


def module_groups(model):
    """Top-level children, with the encoder split into its blocks."""
    out = []
    for n, mod in model.named_children():
        subs = list(mod.named_children())
        if n == 'encoder' and subs:
            for sn, sm in subs:
                out += ([(f'encoder.{sn}.{i}', b) for i, b in enumerate(sm)] if isinstance(sm, nn.ModuleList)
                        else [(f'encoder.{sn}', sm)])
        else:
            out.append((n, mod))
    return out


def block_parts(model):
    """(part name, module) for every block's children with parameters: timings summed over blocks."""
    return [(pn, p) for n, b in module_groups(model) if n.startswith('encoder.blocks')
            for pn, p in b.named_children() if sum(q.numel() for q in p.parameters())]


class Hooks:
    def __init__(self, named):
        self.ms, self.t0, self.h = defaultdict(float), {}, []
        for n, m in named:
            self.h += [m.register_forward_pre_hook(self._pre(n)), m.register_forward_hook(self._post(n))]

    def _pre(self, n):
        def f(mod, inp):
            torch.cuda.synchronize()
            self.t0[(n, id(mod))] = time.perf_counter()
        return f

    def _post(self, n):
        def f(mod, inp, out):
            torch.cuda.synchronize()
            self.ms[n] += (time.perf_counter() - self.t0[(n, id(mod))]) * 1e3
        return f

    def remove(self):
        for h in self.h:
            h.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='qtome_tiny_p50_s16_real_s1')
    ap.add_argument('--dataset', default='Lee2019_MI')
    ap.add_argument('--subjects', nargs='+', default=['2', '7', '8', '9', '15', '16', '18', '41'])
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--iters', type=int, default=10, help='train steps per timing round')
    ap.add_argument('--rounds', type=int, default=5)
    a = ap.parse_args()

    dev = torch.device('cuda')
    torch.manual_seed(0)
    cfg = json.load(open(f'{QTOME_OUTPUT}/{a.run}/pretrain/artifacts/config.json'))
    ckpt = torch.load(f'{QTOME_OUTPUT}/{a.run}/pretrain/checkpoint/last.pth', map_location='cpu', weights_only=False)
    c = copy.deepcopy(cfg)
    c['dataset_params']['pretrain'] = {a.dataset: dict(cfg['dataset_params']['pretrain'][a.dataset],
                                                       subject_to_use=a.subjects)}
    ds = build_dataset_from_config(c, transform=None, mode='pretrain')
    ms = build_masking_strategy_from_config(cfg['preprocess_params'].get('mask', {}))
    ms.set_epoch(40)
    ds.set_masking(ms)
    batches = [_unpack_batch(b, dev) for b in DataLoader(ds, batch_sampler=MontageBatchSampler(ds, a.batch_size, True))]
    trainer = MODEL_REGISTRY[ckpt['build_config']['model_type']].trainer_cls()
    lh = dict(ckpt['build_config']['model_params'].get('loss', {}))

    # ---- sizes
    m = build_from_checkpoint(ckpt)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"\n{a.run}: {n_par / 1e6:.3f} M parameters = {n_par * 4 / 2**20:.2f} MiB as fp32 "
          f"({n_par * 2 / 2**20:.2f} MiB if stored fp16); {len(batches)} batches of x "
          f"{tuple(batches[0][0].shape)} (B, C, patches, patch_len)")
    print(f"\n{'module':<24}{'params':>10}{'share':>8}")
    for n, mod in module_groups(m):
        k = sum(p.numel() for p in mod.parameters())
        if k:
            print(f'{n:<24}{k:>10,}{100 * k / n_par:>7.1f}%')
    parts = defaultdict(int)
    for pn, p in block_parts(m):
        parts[pn] += sum(q.numel() for q in p.parameters())
    print('  one block: ' + ', '.join(f'{pn} {sum(q.numel() for q in p.parameters()):,}'
                                      for pn, p in block_parts(m)[:len(parts)]))

    # ---- models / optimizers per mode
    state = {}
    for mode in MODES:
        model = build_from_checkpoint(ckpt).to(dev).train()
        state[mode] = (model, torch.optim.AdamW(optimizer_param_groups(model, 0.01), lr=1e-5),
                       torch.amp.GradScaler(enabled=mode == 'fp16-amp'))

    def step(mode, b):
        model, opt, scaler = state[mode]
        amp = set_mode(mode)
        x, co, ti, mp, vc = b
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out = model(x, co, ti, bool_masked_pos=mp, valid_channels=vc)
            loss = trainer.compute_loss(model, x, out, mp, **lh)[0]
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

    fwd, peak, step_ms = {}, {}, defaultdict(list)
    for mode in MODES:
        for b in batches[:3]:
            step(mode, b)                                       # warm-up (cuDNN / allocator)
        hk = Hooks(module_groups(state[mode][0]) + block_parts(state[mode][0]))
        torch.cuda.reset_peak_memory_stats()
        for i in range(a.iters):
            step(mode, batches[i % len(batches)])
        torch.cuda.synchronize()
        peak[mode] = torch.cuda.max_memory_allocated() / 2**20
        fwd[mode] = {k: v / a.iters for k, v in hk.ms.items()}
        hk.remove()
    for r in range(a.rounds):                                    # alternate modes: shared-GPU load hits each alike
        for mode in (MODES if r % 2 == 0 else MODES[::-1]):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(a.iters):
                step(mode, batches[i % len(batches)])
            torch.cuda.synchronize()
            step_ms[mode].append((time.perf_counter() - t0) / a.iters * 1e3)

    # ---- numerical agreement from the checkpoint's weights
    ev = {}
    model = build_from_checkpoint(ckpt).to(dev).eval()
    for mode in MODES:
        amp = set_mode(mode)
        with torch.no_grad():
            ev[mode] = []
            for x, co, ti, mp, vc in batches:
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    out = model(x, co, ti, bool_masked_pos=mp, valid_channels=vc)
                    ev[mode].append(float(trainer.compute_loss(model, x, out, mp, **lh)[1]))

    hdr = f"{'':<28}" + ''.join(f'{k:>12}' for k in MODES)
    print(f"\nforward ms per train step, per module (hooks sync at every module)\n{hdr}")
    part_names = list(parts)
    for k in MODES:  # spatial attention runs its weights through F.scaled_dot_product_attention, not a module
        fwd[k]['rest of blocks'] = (sum(v for n, v in fwd[k].items() if n.startswith('encoder.blocks'))
                                    - sum(fwd[k].get(n, 0) for n in part_names))
    for n in list(fwd['fp32']):
        label = f'  {n}' if n in part_names + ['rest of blocks'] else n
        print(f'{label:<28}' + ''.join(f"{fwd[k].get(n, 0):>12.2f}" for k in MODES))
    print('  (indented: summed over all blocks; rest = spatial attention core, residuals, pooling)')
    print(f"\n{'train step ms (median)':<24}" + ''.join(f"{statistics.median(step_ms[k]):>12.1f}" for k in MODES))
    print(f"{'  min / max':<24}" + ''.join(f"{f'{min(step_ms[k]):.0f}/{max(step_ms[k]):.0f}':>12}" for k in MODES))
    print(f"{'peak GPU MiB':<24}" + ''.join(f"{peak[k]:>12.0f}" for k in MODES))
    print(f"{'eval masked MSE':<24}" + ''.join(f"{statistics.mean(ev[k]):>12.5f}" for k in MODES))
    print(f"{'max rel diff vs fp32':<24}" + ''.join(
        f"{max(abs(p - q) / abs(q) for p, q in zip(ev[k], ev['fp32'])):>12.1e}" for k in MODES))


if __name__ == '__main__':
    main()
