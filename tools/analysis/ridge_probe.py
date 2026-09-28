"""
Closed-form linear probe on the frozen pre-stamp z: a deterministic readout of how linearly usable
the representation is, without SGD head training (the finetune heads' run-to-run noise).

Per dataset (BNCI2014004 / BNCI2014001 / BNCI2014008), Compass LOSO split (the protocol's session
selection, configs/finetune_protocols.json incl. _dataset_split). Per held-out subject: PCA of the
training trials' z tokens (D -> n_pca, like latent_proj 'pca'), flatten [patches, channels, n_pca],
standardise on the training trials, ridge classifier (balanced class weights; alpha by leave-one-subject-
out over the training subjects). Score: balanced accuracy on the held-out subject, mean over subjects.
Same inputs -> same number: differences between backbones are not head-training noise. A relative
ruler between backbones, not a Compass number: on BNCI2014001 it reads ~10 points below the factorised
SGD probe (latent_signed), whose low-rank form regularises 22-channel data better than plain L2.

Panel: `python analysis_pretrain.py --run <backbone> --panel ridge_probe` writes
output/<backbone>/pretrain/analysis/ridge_probe.json (the latent feature cache is built on first use).
"""
import copy
import json

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

from cache_feature import CachedStampDataset, get_stamp_cache
from train_finetune import _load_sessions, apply_protocol

DATASETS = (('BNCI2014004', 'mi_loso'), ('BNCI2014001', 'mi_loso'), ('BNCI2014008', 'p300_loso'))
ALPHAS = np.logspace(-1, 5, 7)


def _pca_axes(tokens, n):
    """tokens [M, D] -> (mean [D], top-n principal axes [D, n]); eigh of the D x D covariance."""
    mu = tokens.mean(0)
    cov = (tokens - mu).T @ (tokens - mu) / max(len(tokens) - 1, 1)
    _, vecs = torch.linalg.eigh(cov.double())
    return mu, vecs[:, -n:].flip(1).to(tokens.dtype)


def ridge_probe(config, checkpoint, out_path, n_pca=8, datasets=DATASETS):
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'loso'}}
        cfg['model_params'].setdefault('MeSAE', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        sessions = cfg['training_params']['finetune']['split'].get('sessions')
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedStampDataset(get_stamp_cache(cfg, ds, subs, latent='output'), subs)
        z, y = data.z.float(), data.labels.numpy()                                 # [T, N', Cv, D]
        subj = data.subject_data.numpy()
        keep = np.ones(len(y), bool)
        if sessions is not None:
            keep = np.isin(_load_sessions(cfg, ds_args, subs, subj), sessions)
        T, Np, Cv, D = z.shape
        scores = {}
        for s in subs:
            tr, te = keep & (subj != int(s)), keep & (subj == int(s))
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, D), n_pca)
            f = ((z - mu) @ W).reshape(T, -1).numpy()                                # [T, N' * Cv * n_pca]
            m, sd = f[tr].mean(0), f[tr].std(0) + 1e-6
            # alpha by leave-one-subject-out on the training subjects (the test is cross-subject; a
            # leave-one-trial-out choice rewards subject-specific fit)
            clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                               scoring='balanced_accuracy').fit((f[tr] - m) / sd, y[tr], groups=subj[tr])
            scores[s] = float(balanced_accuracy_score(y[te], clf.predict((f[te] - m) / sd)))
        res[ds] = {'split': 'loso', 'sessions': sessions, 'per_subject': scores,
                   'mean': float(np.mean(list(scores.values())))}
        print(f'  {ds:12s} loso ridge probe: {res[ds]["mean"] * 100:5.1f}  (sessions {sessions}; '
              + ' '.join(f'{k}:{v * 100:.0f}' for k, v in scores.items()) + ')')
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


# ---------- stamp power vs raw band power: does the stamp code carry more than a filterbank? ----------

N_SEG = 4                                                   # time segments per trial (mean power in each)
BANDS_HZ = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def _loso_ridge(X, y, subj, keep, subs):
    """Closed-form ridge per held-out subject (alpha by leave-one-subject-out on the training
    subjects), features standardised on the training trials -> {subject: balanced accuracy}."""
    out = {}
    for s in subs:
        tr, te = keep & (subj != int(s)), keep & (subj == int(s))
        m, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                           scoring='balanced_accuracy').fit((X[tr] - m) / sd, y[tr], groups=subj[tr])
        out[s] = float(balanced_accuracy_score(y[te], clf.predict((X[te] - m) / sd)))
    return out


def stamp_vs_raw(config, checkpoint, out_path, datasets=DATASETS):
    """Per dataset, the same closed-form loso ridge on two fixed feature sets of the same trials:
    log stamp power (a^2 + b^2 per channel x stamp, mean over N_SEG time segments) and log raw band
    power (per channel x band, same segments). stamp - raw > 0: the stamp code carries class
    information a raw spectral filterbank does not."""
    from train_finetune import RawSource
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'loso'}}
        cfg['model_params'].setdefault('MeSAE', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        sessions = cfg['training_params']['finetune']['split'].get('sessions')
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedStampDataset(get_stamp_cache(cfg, ds, subs), subs)
        raw = RawSource(cfg, ds, subs)
        assert torch.equal(data.labels, raw.labels), 'stamp cache and raw trials are not in the same order'
        y, subj = data.labels.numpy(), data.subject_data.numpy()
        keep = np.isin(_load_sessions(cfg, ds_args, subs, subj), sessions) if sessions is not None else np.ones(len(y), bool)
        amp = data.amp.float()                                                    # [T, N', Cv, S, 2]
        P = amp.pow(2).sum(-1)                                                    # [T, N', Cv, S]
        seg = torch.stack([c.mean(1) for c in P.tensor_split(min(N_SEG, P.shape[1]), dim=1)], 1)
        X_stamp = (seg + 1e-6).log().reshape(len(y), -1).numpy()
        sf = float(cfg['preprocess_params']['sample_freq'])
        xr = raw.x.float()                                                        # [T, Cv, samples]
        spans = xr.tensor_split(N_SEG, dim=-1)
        f = torch.fft.rfftfreq(spans[0].shape[-1], 1 / sf)
        bp = torch.stack([torch.stack([(torch.fft.rfft(sp, dim=-1).abs().pow(2)[..., (f >= lo) & (f < hi)]).mean(-1)
                                       for lo, hi in BANDS_HZ], -1) for sp in spans], 1)   # [T, seg, Cv, bands]
        X_raw = (bp + 1e-6).log().reshape(len(y), -1).numpy()
        st, rw = _loso_ridge(X_stamp, y, subj, keep, subs), _loso_ridge(X_raw, y, subj, keep, subs)
        d = np.array([st[s] - rw[s] for s in subs])
        res[ds] = {'sessions': sessions, 'stamp_power': float(np.mean(list(st.values()))),
                   'raw_band': float(np.mean(list(rw.values()))), 'diff': float(d.mean()),
                   'subjects_better': int((d > 0).sum()), 'n_subjects': len(subs),
                   'per_subject': {'stamp_power': st, 'raw_band': rw}}
        print(f'  {ds:12s} loso ridge: stamp power {res[ds]["stamp_power"] * 100:5.1f} | raw band power '
              f'{res[ds]["raw_band"] * 100:5.1f} | stamp - raw {d.mean() * 100:+.1f} ({(d > 0).sum()}/{len(subs)} subjects)')
    json.dump(res, open(out_path, 'w'), indent=2)
    return res
