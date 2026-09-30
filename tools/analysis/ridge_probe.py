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
import torch.nn.functional as F
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


def _stamp_bank(checkpoint):
    """The trained StampBank of a pretrain checkpoint (rebuilt from its build_config, legacy names mapped)."""
    from model.factory import build_from_checkpoint
    return build_from_checkpoint(torch.load(checkpoint, map_location='cpu', weights_only=False)).stamps.eval()


@torch.no_grad()
def stamp_hidden(z, bank, chunk=256):
    """z [T, N', Cv, D] (the cached encoder output the stamps read) -> stamp hidden u [T, N', Cv, S * K]:
    every stamp's MLP hidden GELU(LN(z) W_down_s + b_down_s), stamps concatenated (StampBank._amp's first map)."""
    out = []
    for c in z.split(chunk):
        u = F.gelu(torch.einsum('tncd,sdk->tncsk', bank.input_norm(c.float()), bank.W_down) + bank.b_down)
        out.append(u.flatten(-2))
    return torch.cat(out)


def _features(feature, z, checkpoint):
    """The token features a probe reads: 'z' (as cached) or 'stamp_hidden'."""
    if feature == 'z':
        return z
    assert feature == 'stamp_hidden', feature
    return stamp_hidden(z, _stamp_bank(checkpoint))


def ridge_probe(config, checkpoint, out_path, n_pca=8, datasets=DATASETS, pool=1, feature='z'):
    """pool > 1: average `pool` adjacent z tokens first (train_finetune.pool_tokens). feature: 'z' or
    'stamp_hidden' (the probe reads every stamp's MLP hidden instead of z; same PCA pipeline)."""
    from train_finetune import pool_tokens
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
        z, y = _features(feature, data.z.float(), checkpoint), data.labels.numpy()   # [T, N', Cv, D]
        if pool > 1:
            z = pool_tokens(z, pool)
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
        nfft = max(spans[0].shape[-1], int(sf))                                   # >= 1 Hz bins: short spans (008, 1 s trials) leave no bin in 0.5-4 Hz
        f = torch.fft.rfftfreq(nfft, 1 / sf)
        bp = torch.stack([torch.stack([(torch.fft.rfft(sp, n=nfft, dim=-1).abs().pow(2)[..., (f >= lo) & (f < hi)]).mean(-1)
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


# ---------- few-shot closed-form probe: shrinkage chosen inside the calibration trials ----------

FEWSHOT = (('BNCI2014004', 'mi_fewshot'), ('BNCI2014001', 'mi_fewshot'), ('BNCI2014008', 'p300_fewshot'))


def _segment_means(t, n_seg):
    """t [T, N', ...] -> [T, n_seg * ...]: mean over n_seg consecutive stretches of the token axis, flattened."""
    return torch.stack([c.mean(1) for c in t.tensor_split(min(n_seg, t.shape[1]), dim=1)], 1).reshape(len(t), -1)


def fewshot_ridge(config, checkpoint, out_path, n_pca=8, n_seg=4, datasets=FEWSHOT, feature='z'):
    """Within-subject few-shot (the protocol's chronological calibration split, per subject) with a closed-form
    probe instead of a trained head: RidgeClassifierCV, its shrinkage picked by efficient leave-one-out on the
    subject's calibration trials only. Two compact fixed feature sets per trial: 'z' = z tokens projected on
    n_pca principal axes (fit on the calibration tokens), averaged over n_seg time segments; 'z_power' = log mean
    square of the same projections per segment (power, what MI carries); 'stamp' = log stamp
    power (a^2 + b^2) averaged over the same segments. -> {dataset: {feature: {mean, per_subject}}}."""
    from sklearn.linear_model import RidgeClassifierCV
    from train_finetune import make_runs
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'fewshot', 'train_fraction': 0.3}}
        cfg['model_params'].setdefault('MeSAE', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        split = cfg['training_params']['finetune']['split']
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedStampDataset(get_stamp_cache(cfg, ds, subs, latent='output'), subs)
        z, y, subj = _features(feature, data.z.float(), checkpoint), data.labels.numpy(), data.subject_data.numpy()
        stamp = _segment_means((data.amp.float().pow(2).sum(-1) + 1e-6).log(), n_seg).numpy()
        runs = make_runs(split, subs, subj, y, _load_sessions(cfg, ds_args, subs, subj))
        scores = {'z': {}, 'z_power': {}, 'stamp': {}}
        for run in runs:
            tr = run['train']
            (s, te), = run['eval']['heldout'].items()
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, z.shape[-1]), n_pca)
            proj = (z[np.concatenate([tr, te])] - mu) @ W                              # [T, N', Cv, n_pca]
            feats = {'z': _segment_means(proj, n_seg).numpy(),
                     'z_power': _segment_means(proj.pow(2), n_seg).add(1e-6).log().numpy(),
                     'stamp': stamp[np.concatenate([tr, te])]}
            for k, f in feats.items():
                ftr, fte = f[:len(tr)], f[len(tr):]
                m, sd = ftr.mean(0), ftr.std(0) + 1e-6
                clf = RidgeClassifierCV(alphas=ALPHAS, class_weight='balanced').fit((ftr - m) / sd, y[tr])
                scores[k][s] = float(balanced_accuracy_score(y[te], clf.predict((fte - m) / sd)))
        res[ds] = {k: {'mean': float(np.mean(list(v.values()))), 'per_subject': v} for k, v in scores.items()}
        res[ds]['split'] = split
        print(f"  {ds:12s} few-shot closed-form ridge: z {res[ds]['z']['mean'] * 100:5.1f} | z log-power "
              f"{res[ds]['z_power']['mean'] * 100:5.1f} | stamp power {res[ds]['stamp']['mean'] * 100:5.1f}"
              f"  (train_fraction {split.get('train_fraction')})")
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


# ---------- stamp hidden: does each stamp's MLP hidden carry more than z? (docs/cards/2026-09-29-stamp-hidden.md) ----------

@torch.no_grad()
def stamp_hidden_stats(config, checkpoint, out_path, datasets=DATASETS, n_tokens=20000, seed=0):
    """Per dataset, on n_tokens random (trial, patch, channel) tokens of the cached z:
    check -- the cached amp is u w_amp + b_amp times one per-token scalar (the rms), i.e. u is what the StampBank
    computed (relative residual, fp16-level expected);
    free_share -- per stamp, share of centred u_s variance outside the column space of w_amp_s (never read by the
    reconstruction; random directions: 1 - 2 / hidden_width);
    copy_r2 -- held-out R^2 of a least-squares map [z, 1] -> u (1 = a linear copy of z);
    rank -- participation-ratio effective rank of u and of z;
    pair_cc -- mean over stamp pairs of the top canonical correlation between u_s and u_t (1 = same subspace)."""
    bank = _stamp_bank(checkpoint)
    S, K = bank.W_down.shape[0], bank.W_down.shape[2]
    g = torch.Generator().manual_seed(seed)
    res = {}
    for ds, proto in datasets:
        ds_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds: ds_args}
        cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                              'split': {'type': 'loso'}}
        cfg['model_params'].setdefault('MeSAE', {}).setdefault('finetune', {})
        cfg = apply_protocol(cfg)
        subs = list(json.load(open(f"{ds_args['dataset_path']}/metadata.json"))['data_structure'])
        data = CachedStampDataset(get_stamp_cache(cfg, ds, subs, latent='output'), subs)
        z = data.z.float().flatten(0, 2)                                         # [T*N'*Cv, D]
        amp = data.amp.float().flatten(0, 2)                                     # [T*N'*Cv, S, 2]
        idx = torch.randperm(len(z), generator=g)[:n_tokens]
        z, amp = z[idx], amp[idx]
        u = stamp_hidden(z[:, None, None], bank)[:, 0, 0].view(-1, S, K)        # [M, S, K]
        r = torch.einsum('msk,skp->msp', u, bank.w_amp) + bank.b_amp             # [M, S, 2], no rms
        a, r = amp.flatten(1), r.flatten(1)
        rms = (a * r).sum(1, keepdim=True) / r.pow(2).sum(1, keepdim=True).clamp(min=1e-12)
        check = float(((a - rms * r).norm(dim=1) / a.norm(dim=1).clamp(min=1e-12)).median())
        uc = u - u.mean(0)
        free = []
        for s in range(S):
            Q, _ = torch.linalg.qr(bank.w_amp[s])                               # [K, 2] readout column space
            free.append(float(1 - (uc[:, s] @ Q).pow(2).sum() / uc[:, s].pow(2).sum().clamp(min=1e-12)))
        U, h = uc.flatten(1), len(z) // 2
        X = torch.cat([z, torch.ones(len(z), 1)], 1).double()
        B = torch.linalg.lstsq(X[:h], U[:h].double()).solution
        copy_r2 = float(1 - (U[h:].double() - X[h:] @ B).pow(2).sum() / (U[h:] - U[h:].mean(0)).double().pow(2).sum())
        pr = lambda m: float((lambda ev: ev.sum() ** 2 / ev.pow(2).sum())(torch.linalg.eigvalsh(torch.cov(m.T.double()))))
        Qs = [torch.linalg.qr(uc[:, s].double())[0] for s in range(S)]
        cc = [float(torch.linalg.svdvals(Qs[s].T @ Qs[t])[0]) for s in range(S) for t in range(s + 1, S)]
        res[ds] = {'check_rel_residual': check, 'free_share': {'mean': float(np.mean(free)), 'min': min(free),
                   'max': max(free), 'per_stamp': free, 'random': 1 - 2 / K}, 'copy_r2': copy_r2,
                   'rank_u': pr(U), 'rank_z': pr(z), 'dim_u': S * K, 'dim_z': z.shape[1],
                   'pair_cc': float(np.mean(cc)), 'n_tokens': len(z)}
        print(f"  {ds:12s} stamp hidden: check {check:.1e} | free share {np.mean(free):.2f} ({min(free):.2f}-{max(free):.2f}, "
              f"random {1 - 2 / K:.2f}) | z->u R^2 {copy_r2:.3f} | rank u {res[ds]['rank_u']:.1f}/{S * K} vs z "
              f"{res[ds]['rank_z']:.1f}/{z.shape[1]} | pair cc {np.mean(cc):.2f}")
        assert check < 0.02, f'{ds}: cached amp is not u w_amp + b_amp times rms (rel residual {check:.3g})'
    json.dump(res, open(out_path, 'w'), indent=2)
    return res


# ---------- coordinate robustness: is the coordinate embedding a per-site lookup? ----------

COORD_TRANSFORMS = ('jitter_2mm', 'jitter_5mm', 'jitter_10mm', 'mirror')


def coord_robustness(config, checkpoint, out_path, n_pca=8, transforms=COORD_TRANSFORMS,
                     datasets=(('BNCI2014004', 'mi_loso'), ('BNCI2014001', 'mi_loso'))):
    """docs/cards/2026-10-01-coordinate-lookup.md: the loso ridge probe (as ridge_probe) trained on the training
    subjects' normal-coordinate z, tested on the held-out subject's z extracted with transformed coordinates
    (cache_feature.transform_coords, data untouched). Per transform: balanced accuracy, the fraction of held-out
    predictions that change vs normal coordinates, and the fraction of hand trials predicted as the other hand."""
    res = {}
    for ds, proto in datasets:
        base_args = {'dataset_path': f'datas/finetune/{ds}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}
        meta = json.load(open(f"{base_args['dataset_path']}/metadata.json"))
        subs = list(meta['data_structure'])
        tg = meta['data_metadata']['targets']
        hand = {side: next(int(k) for k, v in tg.items() if k.isdigit() and side in json.dumps(v).lower())
                for side in ('left', 'right')}

        def load(tf):
            args = dict(base_args, **({'coords_transform': tf} if tf else {}))
            cfg = copy.deepcopy(config)
            cfg['dataset_params']['finetune'] = {ds: args}
            cfg['training_params']['finetune'] = {'pretrained_checkpoint': checkpoint, 'protocol': proto,
                                                  'split': {'type': 'loso'}}
            cfg['model_params'].setdefault('MeSAE', {}).setdefault('finetune', {})
            cfg = apply_protocol(cfg)
            data = CachedStampDataset(get_stamp_cache(cfg, ds, subs, latent='output'), subs)
            return cfg, args, data

        cfg, args, data = load(None)
        z, y, subj = data.z.float(), data.labels.numpy(), data.subject_data.numpy()
        zt = {tf: load(tf)[2].z.float() for tf in transforms}
        sessions = cfg['training_params']['finetune']['split'].get('sessions')
        keep = np.isin(_load_sessions(cfg, args, subs, subj), sessions) if sessions is not None else np.ones(len(y), bool)
        T, Np, Cv, D = z.shape
        per = {k: {} for k in ('normal',) + tuple(transforms)}
        changed, flipped = {tf: [] for tf in transforms}, {tf: [] for tf in transforms}
        for s in subs:
            tr, te = keep & (subj != int(s)), keep & (subj == int(s))
            mu, W = _pca_axes(z[torch.from_numpy(tr)].reshape(-1, D), n_pca)
            feat = lambda zz: ((zz - mu) @ W).reshape(T, -1).numpy()
            f = feat(z)
            m, sd = f[tr].mean(0), f[tr].std(0) + 1e-6
            clf = GridSearchCV(RidgeClassifier(class_weight='balanced'), {'alpha': ALPHAS}, cv=LeaveOneGroupOut(),
                               scoring='balanced_accuracy').fit((f[tr] - m) / sd, y[tr], groups=subj[tr])
            p0 = clf.predict((f[te] - m) / sd)
            per['normal'][s] = float(balanced_accuracy_score(y[te], p0))
            hand_te = np.isin(y[te], list(hand.values()))
            for tf in transforms:
                p = clf.predict((feat(zt[tf])[te] - m) / sd)
                per[tf][s] = float(balanced_accuracy_score(y[te], p))
                changed[tf].append(float((p != p0).mean()))
                other = np.where(y[te] == hand['left'], hand['right'], hand['left'])
                flipped[tf].append(float((p[hand_te] == other[hand_te]).mean()))
        res[ds] = {'sessions': sessions, 'hand_classes': hand,
                   'balanced_accuracy': {k: float(np.mean(list(v.values()))) for k, v in per.items()},
                   'per_subject': per,
                   'prediction_changed': {tf: float(np.mean(v)) for tf, v in changed.items()},
                   'hand_predicted_as_other_hand': {tf: float(np.mean(v)) for tf, v in flipped.items()}}
        r = res[ds]
        print(f'  {ds:12s} normal {r["balanced_accuracy"]["normal"] * 100:5.1f} | ' + ' | '.join(
            f'{tf} {r["balanced_accuracy"][tf] * 100:5.1f} (changed {r["prediction_changed"][tf] * 100:.0f}%, '
            f'other hand {r["hand_predicted_as_other_hand"][tf] * 100:.0f}%)' for tf in transforms))
    json.dump(res, open(out_path, 'w'), indent=2)
    return res
