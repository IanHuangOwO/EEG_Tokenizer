"""Gate for the small-backbone run: exit 0 if the combined head is not beaten by the z probe on BNCI2014008 loso
over the 3 tiny seeds (diff > -2 x sqrt(SE_a^2 + SE_b^2)), else 1."""
import json, sys
import numpy as np
BB = ['mesae_tiny_p50_s16_s1', 'mesae_tiny_p50_s16_s2', 'mesae_tiny_p50_s16_s3']
def acc(bb, head):
    out = []
    for s in (1, 2, 3):
        d = json.load(open(f'output/{bb}/finetune/{head}/BNCI2014008_loso_seed{s}/artifacts/group_eval.json'))
        out.append(np.mean([v['tail'] for f in d.values() for g in f['groups'].values() for v in g['subjects'].values()]))
    return np.mean(out) * 100
z, c = np.array([acc(b, 'patch_probe') for b in BB]), np.array([acc(b, 'combined') for b in BB])
se = lambda v: v.std(ddof=1) / np.sqrt(3)
d, t = c.mean() - z.mean(), 2 * np.sqrt(se(z) ** 2 + se(c) ** 2)
print(f'P300 loso gate: combined {c.mean():.1f} vs z probe {z.mean():.1f}, diff {d:+.1f}, threshold {t:.1f}')
sys.exit(0 if d > -t else 1)
