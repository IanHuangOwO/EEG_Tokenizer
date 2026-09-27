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
