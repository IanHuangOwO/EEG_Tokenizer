"""Convergence check from training logs. Usage: python probes/epoch_curve.py <run_dir> [block=5]
Mean [Val]/[Train] bal_acc per <block>-epoch block over the fold tags found in
<run_dir>/artifacts/train_*.log; also prints tail (last 10) vs the previous 10 epochs."""
import glob, os, re, sys
import numpy as np

run = sys.argv[1]
block = int(sys.argv[2]) if len(sys.argv) > 2 else 5
val, trn = {}, {}
for path in sorted(glob.glob(os.path.join(run, 'artifacts', 'train_*.log'))):
    tag = ep = None
    for line in open(path):
        m = re.search(r'\[([^\]]+)\] Epoch (\d+)/(\d+) Summary', line)
        if m:
            tag, ep, total = m.group(1), int(m.group(2)), int(m.group(3))
            continue
        m = re.search(r'\[(Val|Train)\].*bal_acc: ([\d.]+)', line)
        if m and tag:
            (val if m.group(1) == 'Val' else trn).setdefault(tag, {})[ep] = float(m.group(2))
tags = [t for t in val if len(val[t]) >= total]
print(f'{run}: {len(tags)} complete fold tags, {total} epochs')
for a in range(1, total + 1, block):
    b = min(a + block - 1, total)
    v = np.mean([np.mean([val[t][e] for e in range(a, b + 1)]) for t in tags])
    r = np.mean([np.mean([trn[t][e] for e in range(a, b + 1)]) for t in tags])
    print(f' ep{a:3}-{b:3}  val {v:.3f}  train {r:.3f}')
if total >= 20:
    last = np.mean([np.mean([val[t][e] for e in range(total - 9, total + 1)]) for t in tags])
    prev = np.mean([np.mean([val[t][e] for e in range(total - 19, total - 9)]) for t in tags])
    print(f' tail(last 10) {last:.3f} vs previous 10 {prev:.3f}  gain {last - prev:+.3f}')
