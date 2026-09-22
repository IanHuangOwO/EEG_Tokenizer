"""Reproducible seen/unseen subject subsets for group-holdout finetune evals. Ported from
probes/select_eval_subsets.py -- CPU only, writes config/subject_groups/<name>.json,
idempotent (seed 42). RUN/DATASETS/DS_ROOT/SEED below are the same defaults the original
script hardcoded; select_eval_subsets()'s `names`/`run_config`/`out_dir` params let a
caller override the parts that vary in practice. DATASETS/DS_ROOT/SEED stay module
globals -- add real params for them if a caller ever needs to swap them.
"""
import json
import os

import numpy as np
from scipy.stats import ks_2samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

from IO.dataset import build_dataset_from_config

RUN = 'output/mesae_v10_small/artifacts/config.json'
SEED, BANDS = 42, ((8, 13), (13, 30))
DATASETS = {'physionetmi': 'PhysionetMI', 'beta4s': 'BETA_4s'}
DS_ROOT = {'PhysionetMI': 'datas/pretrain', 'BETA_4s': 'datas/pretrain'}


def _load(cfg, ds, ds_root):
    cfg = json.loads(json.dumps(cfg))
    cfg['dataset_params']['finetune'] = {ds: {'dataset_path': f'{ds_root[ds]}/{ds}',
                                              'subject_to_use': ['all'], 'channels_to_use': ['all']}}
    b = build_dataset_from_config(cfg, mode='finetune').base_dataset
    return b, cfg['preprocess_params']['sample_freq']


def _band_logpow(x, fs):
    """x [n, C, T] -> [n, C, 2]; same statistic as MeSAEFeatureHead._band_logpow."""
    sp = np.abs(np.fft.rfft(x.astype(np.float32), axis=-1)) ** 2
    fr = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    return np.stack([np.log(sp[..., (fr >= lo) & (fr < hi)].sum(-1) + 1e-12) for lo, hi in BANDS], -1)


def _proxy(x, y, fs, seed):
    x = x[:, np.abs(x).sum((0, 2)) > 0]                      # drop all-zero channels
    f = _band_logpow(x, fs).reshape(len(x), -1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                                 f, y, cv=cv).mean())


def _q(v): return {k: float(np.quantile(v, p)) for k, p in (('q10', .1), ('q25', .25), ('q50', .5), ('q75', .75), ('q90', .9))}
def _summ(v): return {'mean': float(np.mean(v)), 'std': float(np.std(v)), **_q(v)}


def _run_one(name, ds, pre_cfg, ds_root, seed, out_dir):
    b, fs = _load(pre_cfg, ds, ds_root)
    sid = b.subject_data.numpy(); y = b.labels.numpy() if hasattr(b.labels, 'numpy') else np.asarray(b.labels)
    subs = sorted(set(sid.tolist()))
    exist = {str(s) for s in subs}
    seen = sorted({str(s) for s in pre_cfg['dataset_params']['pretrain'][ds]['subject_to_use']} & exist, key=int)
    unseen = sorted(exist - set(seen), key=int)
    rng = np.random.RandomState(seed)
    out = {'dataset': ds, 'seed': seed}
    if name == 'physionetmi':
        px = {str(s): _proxy(np.asarray(b.data[sid == s]), y[sid == s], fs, seed) for s in subs}
        order = sorted(unseen, key=lambda s: (px[s], int(s)))
        bins = np.array_split(np.arange(len(order)), 10)
        ev = [order[bn[len(bn) // 2]] for bn in bins]
        rest = [s for s in unseen if s not in ev]
        tr = sorted(rng.choice(rest, 30, replace=False).tolist(), key=int)
        pu = np.array([px[s] for s in unseen]); pe = np.array([px[s] for s in ev])
        ks = ks_2samp(pe, pu)
        out.update(proxy=px, proxy_note='5-fold stratified shrinkage-LDA acc on log mu/beta band power',
                   stats={'all_unseen': _summ(pu), 'eval_unseen': _summ(pe),
                          'seen': _summ([px[s] for s in seen]) if seen else None,
                          'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue), 'n_unseen': len(unseen), 'n_seen': len(seen)})
    else:
        ev = sorted(rng.choice(unseen, 10, replace=False).tolist(), key=int)
        rest = [s for s in unseen if s not in ev]
        tr = sorted(rng.choice(rest, min(30, len(rest)), replace=False).tolist(), key=int)
        out.update(selection='random, seeded (no difficulty proxy for 40-class SSVEP)',
                   stats={'n_unseen': len(unseen), 'n_train': len(tr), 'n_seen_eval': len(seen)})
    out.update(train=tr, eval={'seen': seen, 'unseen': sorted(ev, key=int)})
    assert not set(tr) & set(ev) and not set(tr) & set(seen) and not set(seen) & set(unseen)
    assert set(tr) | set(ev) | set(seen) <= exist
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f'{name}.json')
    json.dump(out, open(p, 'w'), indent=1); open(p, 'a').write('\n')
    print(f'== {ds}: train={tr}\n   eval seen={seen}\n   eval unseen={out["eval"]["unseen"]}')
    print('  stats', json.dumps(out['stats'], indent=1))


def select_eval_subsets(names=None, run_config=RUN, out_dir='config/subject_groups'):
    """Writes out_dir/<name>.json for each name in `names` (default: every key in
    DATASETS). Pure side-effecting print + file write, no return value (matches the
    tools/panels/ contract: the analysis layer prints/saves, the panel just passes
    params through)."""
    names = list(DATASETS) if names is None else names
    cfg = json.load(open(run_config))  # fails loudly if missing
    for n in names:
        _run_one(n, DATASETS[n], cfg, DS_ROOT, SEED, out_dir)
