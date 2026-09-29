"""Combined head vs its parents, per backbone and cell, on whatever has finished."""
import json, os, sys
import numpy as np
BB = ['mesae_tiny_p50_s16_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3']
CELLS = sys.argv[1:] or [f'{d}_{s}' for s in ('loso', 'fewshot') for d in ('BNCI2014004', 'BNCI2014001', 'BNCI2014008')]
def acc(bb, head, cell):
    out = []
    for s in (1, 2, 3):
        p = f'output/{bb}/finetune/{head}/{cell}_seed{s}/artifacts/group_eval.json'
        if os.path.exists(p):
            d = json.load(open(p))
            out.append(np.mean([v['tail'] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]) * 100)
    return f'{np.mean(out):.1f}' + ('' if len(out) == 3 else f'[{len(out)}]') if out else '--'
print('| Cell | backbone | z probe | stamp head | combined |')
for c in CELLS:
    for b in BB:
        print(f"| {c} | {b[-2:]} | {acc(b, 'patch_probe', c)} | {acc(b, 'cw_stamp', c)} | {acc(b, 'combined', c)} |")
