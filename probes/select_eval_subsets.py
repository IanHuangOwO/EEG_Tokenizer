"""Reproducible seen/unseen subject subsets for group-holdout finetune evals (Phase 2, Task 3).
CPU only. Writes config/subject_groups/<name>.json. Seed 42; idempotent.
Usage: python probes/select_eval_subsets.py [--datasets physionetmi beta4s]"""
import os, sys, json, argparse
import numpy as np
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.getcwd())
from scipy.stats import ks_2samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from IO.dataset import build_dataset_from_config

RUN = 'output/pretrain/mesae_v10_small/artifacts/config.json'
SEED, BANDS = 42, ((8, 13), (13, 30))
DATASETS = {'physionetmi': 'PhysionetMI', 'beta4s': 'BETA_4s'}
DS_ROOT = {'PhysionetMI': 'datas/pretrain', 'BETA_4s': 'datas/pretrain'}  # renamed from EEGMMIdb 2026-09-22


def load(cfg, ds):
    cfg = json.loads(json.dumps(cfg))
    cfg['dataset_params']['finetune'] = {ds: {'dataset_path': f'{DS_ROOT[ds]}/{ds}',
                                              'subject_to_use': ['all'], 'channels_to_use': ['all']}}
    b = build_dataset_from_config(cfg, mode='finetune').base_dataset
    return b, cfg['preprocess_params']['sample_freq']


def band_logpow(x, fs):
    """x [n, C, T] -> [n, C, 2]; same statistic as MeSAEFeatureHead._band_logpow."""
    sp = np.abs(np.fft.rfft(x.astype(np.float32), axis=-1)) ** 2
    fr = np.fft.rfftfreq(x.shape[-1], 1.0 / fs)
    return np.stack([np.log(sp[..., (fr >= lo) & (fr < hi)].sum(-1) + 1e-12) for lo, hi in BANDS], -1)


def proxy(x, y, fs):
    x = x[:, np.abs(x).sum((0, 2)) > 0]                      # drop all-zero channels
    f = band_logpow(x, fs).reshape(len(x), -1)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return float(cross_val_score(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                                 f, y, cv=cv).mean())


def q(v): return {k: float(np.quantile(v, p)) for k, p in (('q10', .1), ('q25', .25), ('q50', .5), ('q75', .75), ('q90', .9))}
def summ(v): return {'mean': float(np.mean(v)), 'std': float(np.std(v)), **q(v)}


def run(name, ds, pre_cfg):
    b, fs = load(pre_cfg, ds)
    sid = b.subject_data.numpy(); y = b.labels.numpy() if hasattr(b.labels, 'numpy') else np.asarray(b.labels)
    subs = sorted(set(sid.tolist()))
    exist = {str(s) for s in subs}
    seen = sorted({str(s) for s in pre_cfg['dataset_params']['pretrain'][ds]['subject_to_use']} & exist, key=int)
    unseen = sorted(exist - set(seen), key=int)
    rng = np.random.RandomState(SEED)
    out = {'dataset': ds, 'seed': SEED}
    if name == 'physionetmi':
        px = {str(s): proxy(np.asarray(b.data[sid == s]), y[sid == s], fs) for s in subs}
        order = sorted(unseen, key=lambda s: (px[s], int(s)))
        bins = np.array_split(np.arange(len(order)), 10)
        ev = [order[bn[len(bn) // 2]] for bn in bins]
        rest = [s for s in unseen if s not in ev]
        tr = sorted(rng.choice(rest, 30, replace=False).tolist(), key=int)
        pu = np.array([px[s] for s in unseen]); pe = np.array([px[s] for s in ev])
        ks = ks_2samp(pe, pu)
        out.update(proxy=px, proxy_note='5-fold stratified shrinkage-LDA acc on log mu/beta band power',
                   stats={'all_unseen': summ(pu), 'eval_unseen': summ(pe),
                          'seen': summ([px[s] for s in seen]) if seen else None,
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
    p = f'config/subject_groups/{name}.json'
    json.dump(out, open(p, 'w'), indent=1); open(p, 'a').write('\n')
    print(f'== {ds}: train={tr}\n   eval seen={seen}\n   eval unseen={out["eval"]["unseen"]}')
    print('  stats', json.dumps(out['stats'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    a = ap.parse_args()
    cfg = json.load(open(RUN))  # fails loudly if missing
    for n in a.datasets: run(n, DATASETS[n], cfg)
