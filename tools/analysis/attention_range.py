"""
Measured attention range per encoder block on held-out pretrain windows (unmasked input, CPU): where
each block's temporal and spatial attention actually looks, content and bias together.

Per block (heads averaged, only real channels / real patches as queries):
  temporal  mean_dt_s     attention-weighted mean |t_query - t_key| in seconds (patch spacing
                          doubles every stage: patch_stride / sample_freq * 2^stage)
            uniform_dt_s  the same under uniform attention over the real patches (the reference)
            self_w        weight on the query's own patch
            entropy       attention entropy / log(#keys): 0 = one key, 1 = uniform
  spatial   dist_ratio    attention-weighted mean electrode distance / uniform mean distance over the
                          real channels (< 1 favours neighbours, > 1 favours far channels)
            self_w, entropy as above, over channels
  scale_t / scale_s       the branch's mean |LayerScale|: how much of it reaches the residual stream

Panel: `python analysis_pretrain.py --run <backbone> --panel attention_range` writes
output/<backbone>/pretrain/analysis/attention_range.json.
"""
import json
import math

import torch
import torch.nn.functional as F

from tools.analysis.backbone_eval import batches, eval_windows


def _row_stats(w, dist, rows):
    """w [R, K] attention rows, dist [R, K] query-key distance, rows [R] bool -> sums over real rows."""
    w, dist = w[rows], dist[rows]
    keys = (w > 0).sum(-1).clamp_min(2)
    ent = -(w * w.clamp_min(1e-12).log()).sum(-1) / keys.float().log()
    return float((w * dist).sum()), float(ent.sum()), int(rows.sum())


@torch.no_grad()
def attention_range(model, config, out_path, max_windows=256):
    model = model.cpu().eval()
    pp = config['preprocess_params']
    dt = pp.get('patch_stride', pp['patch_length']) / float(pp['sample_freq'])
    enc = model.encoder
    ds, idx = eval_windows(config, max_windows)
    acc = [dict(t=[0.0, 0.0, 0.0, 0.0, 0], s=[0.0, 0.0, 0.0, 0.0, 0]) for _ in enc.blocks]   # w*d, uniform d, self, entropy, rows
    cur, hooks = {}, []

    def pre_hook(i):
        def f(block, args):
            x, vc, _, vp = args
            B, C, N, _ = x.shape
            cur.update(i=i, B=B, C=C, N=N, vc=vc, vp=vp)
        return f

    def temporal_hook(block, args, out):
        w = out[1]                                                           # [B*C, N, N], heads averaged
        if w is None:
            return
        B, C, N, i = cur['B'], cur['C'], cur['N'], cur['i']
        stage = sum(p < i for p in enc.pool_after)
        pos = torch.arange(N, dtype=torch.float32)
        d = (pos[:, None] - pos[None, :]).abs() * dt * 2 ** stage           # [N, N] seconds
        vp = cur['vp'] if cur['vp'] is not None else torch.ones(B, N, dtype=torch.bool)
        rows = (cur['vc'][:, :, None] & vp[:, None, :]).reshape(B * C * N)
        keyv = vp[:, None, :].expand(B, C, N).reshape(B * C, 1, N).float()
        uni = (d[None] * keyv).sum(-1) / keyv.sum(-1)                       # [B*C, N] uniform mean |dt|
        s_wd, s_ent, n = _row_stats(w.reshape(-1, N), d.expand(B * C, N, N).reshape(-1, N), rows)
        a = acc[i]['t']
        a[0] += s_wd; a[1] += float(uni.reshape(-1)[rows].sum())
        a[2] += float(w.diagonal(dim1=-2, dim2=-1).reshape(-1)[rows].sum()); a[3] += s_ent; a[4] += n

    def spatial_wrap(i, block):
        orig = block._spatial_attention
        def f(x, valid_channels, spatial_bias):
            B, N, C, D = x.shape
            mha = block.spatial_attn
            H = mha.num_heads
            q, k, _ = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
            q, k = (t.reshape(B, N, C, H, D // H).transpose(2, 3) for t in (q, k))    # [B, N, H, C, d]
            logit = q @ k.transpose(-1, -2) / math.sqrt(D // H)
            if spatial_bias is not None:
                logit = logit + spatial_bias[:, None]
            vc = valid_channels.bool()
            logit = logit.masked_fill(~vc[:, None, None, None, :], float('-inf'))
            w = logit.softmax(-1).mean(2)                                    # [B, N, C, C]
            coords = cur['coords']
            d = torch.cdist(coords, coords)                                  # [B, C, C]
            kv = vc[:, None, :].float()
            uni = (d * kv).sum(-1) / kv.sum(-1)                              # [B, C]
            vp = cur['vp'] if cur['vp'] is not None else torch.ones(B, N, dtype=torch.bool)
            rows = (vp[:, :, None] & vc[:, None, :]).reshape(-1)
            s_wd, s_ent, n = _row_stats(w.reshape(-1, C), d[:, None].expand(B, N, C, C).reshape(-1, C), rows)
            a = acc[i]['s']
            a[0] += s_wd; a[1] += float(uni[:, None].expand(B, N, C).reshape(-1)[rows].sum())
            a[2] += float(w.diagonal(dim1=-2, dim2=-1).reshape(-1)[rows].sum()); a[3] += s_ent; a[4] += n
            return orig(x, valid_channels, spatial_bias)
        return f

    for i, block in enumerate(enc.blocks):
        hooks.append(block.register_forward_pre_hook(pre_hook(i)))
        hooks.append(block.temporal_attn.register_forward_hook(temporal_hook))
        block._spatial_attention = spatial_wrap(i, block)
    try:
        for _, x, coords, t, valid in batches(ds, idx):
            cur['coords'] = coords
            model.stage_features(x, coords, time_idx=t, valid_channels=valid)
    finally:
        for h in hooks:
            h.remove()
        for block in enc.blocks:
            del block._spatial_attention                                     # back to the class method

    res = []
    for i, (block, a) in enumerate(zip(enc.blocks, acc)):
        r = {'block': i, 'stage': sum(p < i for p in enc.pool_after),
             'scale_t': float(block.scale_t.abs().mean()), 'scale_s': float(block.scale_s.abs().mean())}
        t, s = a['t'], a['s']
        if t[4]:
            r['temporal'] = {'mean_dt_s': t[0] / t[4], 'uniform_dt_s': t[1] / t[4], 'self_w': t[2] / t[4], 'entropy': t[3] / t[4]}
        if s[4]:
            r['spatial'] = {'dist_ratio': s[0] / s[1], 'self_w': s[2] / s[4], 'entropy': s[3] / s[4]}
        res.append(r)
    with open(out_path, 'w') as f:
        json.dump({'windows': len(idx), 'blocks': res}, f, indent=2)

    print(f'  {"block":>5} {"stage":>5} | {"t: dt s":>8} {"unif s":>7} {"self":>5} {"ent":>5} {"scale":>6} | '
          f'{"s: dist":>8} {"self":>5} {"ent":>5} {"scale":>6}')
    for r in res:
        t, s = r.get('temporal'), r.get('spatial')
        tt = f'{t["mean_dt_s"]:8.2f} {t["uniform_dt_s"]:7.2f} {t["self_w"]:5.2f} {t["entropy"]:5.2f}' if t else f'{"off":>8} {"":7} {"":5} {"":5}'
        ss = f'{s["dist_ratio"]:8.2f} {s["self_w"]:5.2f} {s["entropy"]:5.2f}' if s else f'{"off":>8} {"":5} {"":5}'
        print(f'  {r["block"]:5d} {r["stage"]:5d} | {tt} {r["scale_t"]:6.3f} | {ss} {r["scale_s"]:6.3f}')
    return res
