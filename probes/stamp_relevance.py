"""Per-stamp task relevance on BCICIV2a, from probe_v10.py's stamp_dump_<run>.npz.
Usage: python stamp_relevance.py <run_name> [onset_sample=200] [post_end_s=4.0]

Three measures per stamp s (alive routed + shared), trials cue-aligned (cue at sample
200 = 1.0 s; MI window 0.5-4.0 s after cue):
  1. decod  - per-subject shrinkage LDA on log post-cue power per channel [Cv]
              (and dB change vs pre-cue, decod_bl). One-sided t of per-subject acc vs
              chance 0.25 across subjects, BH-FDR over stamps.
  2. erd    - post/pre power change, dB, mean over channels/trials/subjects (event-related?)
     lri    - lateralization: [dB(C3,R)-dB(C3,L)] - [dB(C4,R)-dB(C4,L)]; classic
              contralateral ERD makes this negative. t across subjects.
  3. itc    - inter-trial phase coherence of atan2(b, a), post minus pre, max over
              channels, mean over subjects (ERP-like phase locking).
Plus sel (approx top-k selection rate, routed only) and the template's peak frequency.

dense amps = response, not use: read decod next to sel.
"""
import os, sys
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
RUN = sys.argv[1]
HERE = os.path.join('output', RUN, 'probes')   # written by probes/probe_v10.py
ONSET = int(sys.argv[2]) if len(sys.argv) > 2 else 200
POST_END = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
d = np.load(os.path.join(HERE, f'stamp_dump_{RUN}.npz'))
amp = d['amp'].astype(np.float32)            # [T, N, Cv, S, 2]
sel, y, g = d['sel'], d['y'], d['g']
ids, n_alive = d['stamp_ids'], int(d['n_alive'])
PS, PL, FS = int(d['patch_stride']), int(d['patch_len']), int(d['fs'])
names = d['chan_names'][d['valid'].astype(bool)]
T, N, Cv, S, _ = amp.shape
starts = np.arange(N) * PS
pre = starts + PL <= ONSET                                  # patch fully before cue
post = (starts >= ONSET + 0.5 * FS) & (starts + PL <= ONSET + POST_END * FS)
assert pre.any() and post.any(), (pre.sum(), post.sum())
print(f'{RUN}: T={T} N={N} (pre {pre.sum()}, post {post.sum()} patches) Cv={Cv} stamps={S} '
      f'({n_alive} routed alive + {S - n_alive} shared)')

pw = (amp ** 2).sum(-1) + 1e-12                             # [T, N, Cv, S]
lp_post = np.log(pw[:, post].mean(1))                       # [T, Cv, S]
db = 10 * np.log10(pw[:, post].mean(1) / pw[:, pre].mean(1))
phase = np.exp(1j * np.arctan2(amp[..., 1], amp[..., 0]))  # [T, N, Cv, S]
subjects = np.unique(g)
c3, c4 = (int(np.flatnonzero(names == n)[0]) if n in names else None for n in ('C3', 'C4'))


def lda_acc(X):
    out = []
    for s_ in subjects:
        mk = g == s_
        pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        out.append(cross_val_score(pipe, X[mk], y[mk], cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())
    return np.array(out)


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


rows = []
for s in range(S):
    acc = lda_acc(lp_post[:, :, s])
    acc_bl = lda_acc(db[:, :, s])
    erd_subj = np.array([db[g == s_, :, s].mean() for s_ in subjects])
    if c3 is not None and c4 is not None:
        def m_(s_, c, lab):
            return db[(g == s_) & (y == lab), c, s].mean()
        lri = np.array([(m_(s_, c3, 1) - m_(s_, c3, 0)) - (m_(s_, c4, 1) - m_(s_, c4, 0)) for s_ in subjects])
    else:
        lri = np.full(len(subjects), np.nan)
    itc = []
    for s_ in subjects:
        ph = phase[g == s_][..., s]                          # [Ts, N, Cv]
        r = np.abs(ph.mean(0))                               # [N, Cv]
        itc.append((r[post].mean(0) - r[pre].mean(0)).max())
    rows.append(dict(
        stamp=int(ids[s]), kind='routed' if s < n_alive else 'shared', peak_hz=float(d['D_peak_hz'][s]),
        sel=float(sel[..., s].mean()) if s < n_alive else 1.0,
        decod=acc.mean(), p_decod=stats.ttest_1samp(acc, 0.25, alternative='greater').pvalue,
        decod_bl=acc_bl.mean(), p_decod_bl=stats.ttest_1samp(acc_bl, 0.25, alternative='greater').pvalue,
        erd_db=erd_subj.mean(), p_erd=stats.ttest_1samp(erd_subj, 0).pvalue,
        lri_db=np.nanmean(lri), p_lri=stats.ttest_1samp(lri, 0).pvalue if np.isfinite(lri).all() else np.nan,
        itc=float(np.mean(itc))))
    print(f'  stamp {s + 1}/{S} done', flush=True)

for k in ('p_decod', 'p_decod_bl', 'p_erd', 'p_lri'):
    q = bh(np.nan_to_num([r[k] for r in rows], nan=1.0))
    for r, qq in zip(rows, q):
        r['q' + k[1:]] = qq

rows.sort(key=lambda r: -r['decod'])
cols = ['stamp', 'kind', 'peak_hz', 'sel', 'decod', 'q_decod', 'decod_bl', 'q_decod_bl',
        'erd_db', 'q_erd', 'lri_db', 'q_lri', 'itc']
print('\nsorted by decod (per-subject LDA on one stamp\'s post-cue power, chance 0.25); q = BH-FDR')
print(' '.join(f'{c:>10s}' for c in cols))
for r in rows:
    print(' '.join(f'{r[c]:>10.3f}' if isinstance(r[c], float) else f'{r[c]:>10}' for c in cols))
with open(os.path.join(HERE, f'stamp_relevance_{RUN}_post{POST_END:g}s.csv'), 'w') as f:
    f.write(','.join(cols) + '\n')
    for r in rows:
        f.write(','.join(str(r[c]) for c in cols) + '\n')
print(f'\nITC bias floor ~ 1/sqrt(trials per subject) = {1 / np.sqrt(T / len(subjects)):.3f}')
