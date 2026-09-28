"""Finetune a FeatureHead on a frozen MeSAE backbone (ADR 0016; finetune restructure, sub-project C).

The backbone never trains: stamp features come from the amplitude cache (cache_feature.py), raw
features from the compiled dataset; only the head is optimised (fp32, batches indexed from RAM).
training_params.finetune.split picks a split type (SPLITS: loso, subject_kfold, eval_subjects,
kfold, blocked_kfold, fewshot). Every run writes artifacts/group_eval.json: per-subject tail (mean of the last 10
epochs) and last-epoch balanced accuracy."""
import argparse, copy, json, logging, os, random, subprocess, sys, warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold

from IO.dataset import build_dataset_from_config
from IO.preprocessing import cache_suffix, num_patches, slice_patches
from cache_feature import CachedStampDataset, get_stamp_cache
from model.factory import MODEL_REGISTRY, load_backbone
from model.MeSAE.MeSAE_modules import (FeatureHead, StampExtractor, make_head_checkpoint,
                                       resolve_head_config, needs_stamp, needs_raw, needs_latent, feature_names)
from tools.analysis import apply_overrides, load_config

torch.set_float32_matmul_precision('high')


def setup_logger(output_dir):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(output_dir, f'train_{timestamp}.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, timestamp



def _resolve_all_subjects(data_root):
    with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    all_available_subjects = list(meta.get('data_structure', {}).keys())
    try:
        return sorted([int(s) for s in all_available_subjects])
    except ValueError:
        return sorted(all_available_subjects)



def resolve_subjects(entry, pool):
    """Subject entry -> list of pool subjects: 'all' | ['all'] | explicit list | {'random': n, 'seed': s}."""
    if entry in ('all', ['all']):
        return list(pool)
    if isinstance(entry, dict):
        return sorted(random.Random(entry.get('seed', 42)).sample(list(pool), entry['random']))
    by_str = {str(s): s for s in pool}
    missing = [s for s in entry if str(s) not in by_str]
    if missing:
        raise ValueError(f"subjects {missing} are not in the pool {sorted(by_str, key=str)}")
    return [by_str[str(s)] for s in entry]


def _load_sessions(config, ds_args, pool, subject_data):
    """Per-trial session index (0 = the subject's first recorded session) for the pool, read
    from the compiled cache's 'session' array (MoabbLoader datasets); None if any subject's
    cache predates it. Both sources keep each subject's trials in cache order, so the
    subject's array maps onto its rows of subject_data directly."""
    pp = config['preprocess_params']
    suffix = cache_suffix(pp['sample_freq'], pp['bandpass_filter'],
                          pp.get('pre_event_seconds', 0.0), pp.get('post_event_seconds', 0.0))
    session = np.zeros(len(subject_data), dtype=np.int64)
    for s in pool:
        z = np.load(os.path.join(ds_args['dataset_path'], 'cache', f'{s}_{suffix}.npz'))
        if 'session' not in z:
            return None
        rows = np.flatnonzero(subject_data == int(s))
        assert len(rows) == len(z['session']), f"subject {s}: {len(rows)} trials vs {len(z['session'])} cached sessions"
        session[rows] = z['session']
    return session


# ---------- split types: training_params.finetune.split = {"type": <name>, ...} ----------
# Every type takes `sessions` (keep only those session indices, 0 = each subject's first recorded
# session -- EEG-FM-Compass uses session 0 for MI/P300) and `seed`; the within-subject types also
# take `purge` (drop eval trials within that many recording positions of a train trial, for
# datasets whose neighbouring trials overlap in time). A new split pattern is a new function
# plus a SPLITS entry.
#   loso           every subject held out once (subject_kfold with n_folds = number of subjects)
#   subject_kfold  n_folds folds of whole subjects, subjects shuffled by seed; train_subjects
#                  optionally limits the training pool
#   eval_subjects  one run: eval_subjects (list, dict of named groups, {"random": n, "seed": s})
#                  vs the rest (or train_subjects)
#   kfold          per subject, n_folds shuffled stratified folds (optimistic: neighbouring
#                  trials land on both sides)
#   blocked_kfold  per subject, n_folds contiguous chronological blocks
#   fewshot        per subject, per class the first ceil(train_fraction * n) trials in recording
#                  order train, the rest evaluate (Compass within-subject calibration)

def _inter_runs(evals, pool, train_pool, trials, disjoint_check):
    runs = []
    for name, groups in evals:
        ev_subs = {s for v in groups.values() for s in v}
        train_subs = [s for s in (train_pool if train_pool is not None else pool) if s not in ev_subs]
        if not train_subs or any(not v for v in groups.values()):
            raise ValueError(f"run {name}: empty training set or evaluation group")
        if disjoint_check and train_pool is not None and set(train_pool) & ev_subs:
            raise ValueError(f"run {name}: train_subjects and evaluation subjects overlap: {sorted(set(train_pool) & ev_subs)}")
        runs.append(dict(name=name, train=trials(train_subs), train_subjects=[str(s) for s in train_subs],
                         eval={g: {str(s): trials([s]) for s in v} for g, v in groups.items()}))
    return runs


def _subject_kfold(split, pool, trials, labels, seed):
    k = int(split['n_folds'])
    if not 2 <= k <= len(pool):
        raise ValueError(f"n_folds must be in [2, {len(pool)}], got {k}")
    subs = list(pool)
    random.Random(seed).shuffle(subs)
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    return _inter_runs([(f'fold{i}', {'heldout': subs[i::k]}) for i in range(k)], pool, train_pool, trials, False)


def _loso(split, pool, trials, labels, seed):
    return _subject_kfold({**split, 'n_folds': len(pool)}, pool, trials, labels, seed)


def _eval_subjects(split, pool, trials, labels, seed):
    ev = split['eval_subjects']
    ev = {'heldout': ev} if not isinstance(ev, dict) or 'random' in ev else ev
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    return _inter_runs([('main', {g: resolve_subjects(v, pool) for g, v in ev.items()})], pool, train_pool, trials, True)


def _per_subject(folds_of):
    """Within-subject type: folds_of(split, idx, labels, seed) -> [(name, train, eval)] per subject."""
    def make(split, pool, trials, labels, seed):
        purge, runs = int(split.get('purge', 0)), []
        for s in pool:
            idx = trials([s])                      # recording order
            for name, tr, va in folds_of(split, idx, labels, seed):
                if purge:
                    pos = np.searchsorted(idx, tr)
                    near = np.zeros(len(idx), dtype=bool)
                    for d in range(-purge, purge + 1):
                        near[np.clip(pos + d, 0, len(idx) - 1)] = True
                    va = va[~near[np.searchsorted(idx, va)]]
                runs.append(dict(name=f'{s}_{name}', train=tr, train_subjects=[str(s)],
                                 eval={'heldout': {str(s): va}}))
        return runs
    return make


def _kfold_folds(split, idx, labels, seed):
    skf = StratifiedKFold(n_splits=int(split['n_folds']), shuffle=True, random_state=seed)
    return [(f'fold{i}', np.sort(idx[tr]), np.sort(idx[va])) for i, (tr, va) in enumerate(skf.split(idx, labels[idx]))]


def _blocked_folds(split, idx, labels, seed):
    return [(f'fold{i}', np.setdiff1d(idx, b), b) for i, b in enumerate(np.array_split(idx, int(split['n_folds'])))]


def _fewshot_folds(split, idx, labels, seed):
    f = float(split['train_fraction'])
    tr = np.concatenate([ic[:max(1, int(np.ceil(f * len(ic))))]
                         for ic in (idx[labels[idx] == c] for c in np.unique(labels[idx]))])
    return [('fewshot', np.sort(tr), np.setdiff1d(idx, tr))]


_COMMON = {'type', 'sessions', 'seed'}
SPLITS = {   # name -> (make_runs function, required keys, optional keys)
    'loso':          (_loso, set(), set()),
    'subject_kfold': (_subject_kfold, {'n_folds'}, {'train_subjects'}),
    'eval_subjects': (_eval_subjects, {'eval_subjects'}, {'train_subjects'}),
    'kfold':         (_per_subject(_kfold_folds), {'n_folds'}, {'purge'}),
    'blocked_kfold': (_per_subject(_blocked_folds), {'n_folds'}, {'purge'}),
    'fewshot':       (_per_subject(_fewshot_folds), {'train_fraction'}, {'purge'}),
}


def apply_protocol(config, path='configs/finetune_protocols.json'):
    """training_params.finetune.protocol = <name>: apply that entry of the protocol table (its
    dotted keys, in order) on top of the merged config, then the table's _dataset_split settings
    for the finetune dataset(s). No protocol key: unchanged."""
    name = config['training_params']['finetune'].get('protocol')
    if not name:
        return config
    raw = json.load(open(path))
    table = {k: v for k, v in raw.items() if not k.startswith('_')}
    if name not in table:
        raise ValueError(f"unknown finetune protocol {name!r}, known: {sorted(table)} ({path})")
    config = apply_overrides(config, [f'{k}={json.dumps(v)}' for k, v in table[name].items()])
    # per-dataset split settings (Compass: which session, few-shot fraction); train_fraction only for fewshot
    split = config['training_params']['finetune']['split']
    for ds in config['dataset_params']['finetune']:
        for k, v in raw.get('_dataset_split', {}).get(ds, {}).items():
            if k != 'train_fraction' or split.get('type') == 'fewshot':
                split[k] = v
    return config


def make_runs(split, pool, subject_data, labels, session=None):
    """split block -> runs [{name, train, train_subjects, eval}], see SPLITS above."""
    if split.get('type') not in SPLITS:
        raise ValueError(f"split.type must be one of {sorted(SPLITS)}, got {split!r}")
    make, required, optional = SPLITS[split['type']]
    if required - set(split) or set(split) - required - optional - _COMMON:
        raise ValueError(f"split type {split['type']!r} takes {sorted(required)} (required) and "
                         f"{sorted(optional | _COMMON - {'type'})} (optional), got {sorted(set(split) - {'type'})}")
    keep = np.ones(len(subject_data), dtype=bool)
    if split.get('sessions') is not None:
        if session is None:
            raise ValueError("split.sessions needs a cache with per-trial sessions -- recompile the dataset")
        keep = np.isin(session, split['sessions'])
    for s in pool:
        if not keep[subject_data == int(s)].any():
            raise ValueError(f"subject {s} has no trials in sessions {split.get('sessions')}")

    def trials(subjects):
        return np.flatnonzero(np.isin(subject_data, [int(x) for x in subjects]) & keep)

    return make(split, pool, trials, labels, split.get('seed', 42))


class StampSource:
    """Cached stamp amplitudes of the pool (cache_feature.py), moved to the device once (a
    BNCI2014008 64-channel impute cache is ~1 GB fp16): the head is tiny, so per-batch CPU
    indexing and host-to-device copies were most of a step's time."""
    kind = 'stamp'

    def __init__(self, config, ds_name, pool, device, latent=False):
        subs = [str(s) for s in pool]
        self.data = CachedStampDataset(get_stamp_cache(config, ds_name, subs, device=device, latent=latent), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx = self.data.channel_idx
        self.num_patches, self.num_stamps = self.data.num_patches, self.data.num_stamps
        self.amp, self.labels_dev = self.data.amp.to(device), self.labels.to(device)
        self.z = self.data.z.to(device) if latent else None                 # [T, N', Cv, D] fp16
        self.latent_dim = self.z.shape[-1] if latent else None

    def get(self, idx):
        idx = idx.to(self.amp.device)
        out = {'stamp': self.amp[idx].float()}
        if self.z is not None:
            out['latent'] = self.z[idx].float()
        return out, self.labels_dev[idx]


class RawSource:
    """Compiled raw trials of the pool, real channels only; patched on the fly."""
    kind = 'raw'

    def __init__(self, config, ds_name, pool):
        cfg = copy.deepcopy(config)
        cfg['dataset_params']['finetune'] = {ds_name: {**config['dataset_params']['finetune'][ds_name],
                                                       'subject_to_use': list(pool)}}
        base = build_dataset_from_config(cfg, mode='finetune').base_dataset
        assert len({tuple(v.tolist()) for v in base.all_valid_channels}) == 1, "one real-channel set per dataset"
        self.channel_idx = torch.nonzero(base.all_valid_channels[0]).flatten().tolist()
        self.x = base.data[:, self.channel_idx].contiguous()                  # [N, C_valid, T]
        self.labels, self.subject_data = base.labels.long(), base.subject_data.long()
        pp = config['preprocess_params']
        self.patch_len = pp.get('patch_length', 100)
        self.patch_stride = pp.get('patch_stride', self.patch_len)
        self.num_patches = num_patches(self.x.shape[-1], self.patch_len, self.patch_stride)
        self.num_stamps = 0

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return {'raw': xp}, self.labels[idx]


class CombinedSource:
    """Serves a StampSource and a RawSource together, for a head whose features list needs
    both (e.g. features=['stamp_power', 'raw_band']). Exposes the union of attributes either
    single source exposes (num_patches/num_stamps/channel_idx/labels/subject_data) --
    both sources are built from the SAME (ds_name, pool), so their per-trial ordering,
    labels and channel_idx must already agree; asserted once at construction, not re-checked
    per batch."""
    kind = 'combined'

    def __init__(self, stamp_source, raw_source):
        assert stamp_source.channel_idx == raw_source.channel_idx, \
            "StampSource/RawSource channel_idx mismatch -- same dataset/pool should agree"
        assert torch.equal(stamp_source.labels, raw_source.labels), \
            "StampSource/RawSource label order mismatch -- same dataset/pool should agree"
        self.stamp, self.raw = stamp_source, raw_source
        self.labels, self.subject_data = stamp_source.labels, stamp_source.subject_data
        self.channel_idx = stamp_source.channel_idx
        self.num_patches, self.num_stamps = stamp_source.num_patches, stamp_source.num_stamps
        self.z, self.latent_dim = stamp_source.z, stamp_source.latent_dim

    def get(self, idx):
        stamp_d, y = self.stamp.get(idx)
        raw_d, _ = self.raw.get(idx)
        return {**stamp_d, **raw_d}, y


def make_source(config, ds_name, pool, device):
    ft_cfg = dict(config['model_params']['MeSAE']['finetune'])
    want_stamp, want_raw, latent = needs_stamp(ft_cfg), needs_raw(ft_cfg), needs_latent(ft_cfg)
    if latent:   # latent_source: which z the latent_* entries read ('output' = what the stamps read)
        latent = ft_cfg.get('latent_source', 'output')
        if latent not in ('output', 'bottleneck'):
            raise ValueError(f"latent_source must be output|bottleneck, got {latent!r}")
    if want_stamp and want_raw:
        return CombinedSource(StampSource(config, ds_name, pool, device, latent), RawSource(config, ds_name, pool))
    if want_stamp:
        return StampSource(config, ds_name, pool, device, latent)
    return RawSource(config, ds_name, pool)


def iter_batches(source, idx, batch_size, device, shuffle, gen=None):
    idx = torch.as_tensor(idx, dtype=torch.long)
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen)]
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        if shuffle and len(j) < 2:
            continue
        x, y = source.get(j)
        yield {k: v.to(device) for k, v in x.items()}, y.to(device)


def env_stamp():
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    try:
        import mne
        mne_version = mne.__version__
    except ImportError:
        mne_version = None
    return {'git_commit': sh('git', 'rev-parse', 'HEAD'), 'git_dirty': bool(sh('git', 'status', '--porcelain')),
            'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'mne': mne_version}


def _metrics(labels, preds, loss):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'loss': float(loss), 'acc': float((labels == preds).mean()),
                'f1': f1_score(labels, preds, average='macro', zero_division=0),
                'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
                'balanced_acc': balanced_accuracy_score(labels, preds), 'kappa': cohen_kappa_score(labels, preds)}


@torch.no_grad()
def _predict(head, source, idx, batch_size, device):
    """logits [n, C] (numpy) and mean loss over the trials idx, in the given (sorted) order."""
    head.eval()
    logits, ys = [], []
    for x, y in iter_batches(source, idx, batch_size, device, shuffle=False):
        logits.append(head(x).float()); ys.append(y)
    logits, ys = torch.cat(logits).cpu(), torch.cat(ys).cpu()
    return logits.numpy(), F.cross_entropy(logits, ys).item()


def build_head_factory(config, source, num_classes):
    """(resolved head config, function returning a freshly initialised FeatureHead)."""
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    cfg = resolve_head_config(
        config['model_params']['MeSAE']['finetune'], num_classes=num_classes, num_patches=source.num_patches,
        num_channels=len(source.channel_idx), num_stamps=source.num_stamps, patch_len=patch_len,
        latent_dim=getattr(source, 'latent_dim', None),
        patch_stride=pp.get('patch_stride', patch_len), sample_freq=float(pp['sample_freq']))
    tables = None
    if 'stamp_band' in feature_names(cfg):   # the template spectra need the backbone, once
        tables = StampExtractor(load_backbone(config), [0]).band_tables(cfg['sample_freq'])

    def new_head():
        head = FeatureHead(cfg)
        if tables is not None:
            head.entries['stamp_band'].E_D.copy_(tables[0])
            head.entries['stamp_band'].E_H.copy_(tables[1])
        return head
    return cfg, new_head


@torch.no_grad()
def _pca_axes(z, train_idx, m, chunk=256):
    """Top-m principal axes [m, D] of z [T, N', C, D] over the given trials (every patch and channel a row)."""
    idx = torch.as_tensor(train_idx, device=z.device)
    D = z.shape[-1]
    s, ss, n = z.new_zeros(D, dtype=torch.float64), z.new_zeros(D, D, dtype=torch.float64), 0
    for i in range(0, len(idx), chunk):
        x = z[idx[i:i + chunk]].reshape(-1, D).double()
        s += x.sum(0); ss += x.T @ x; n += len(x)
    mean = s / n
    cov = ss / n - torch.outer(mean, mean)
    return torch.linalg.eigh(cov)[1][:, -m:].flip(1).T.float()


def run_one(config, run, source, head_cfg, new_head, tag, out_dir, logger, device):
    """Train one head on run['train'], evaluate on the union of run['eval'] every epoch."""
    tp = config['training_params']['finetune']
    E, bs, seed, warm = tp['epochs'], tp['batch_size'], tp.get('seed', 42), tp['warmup_epochs']
    torch.manual_seed(seed)
    head = new_head().to(device)
    for mod in head.entries.values():            # latent_proj 'pca': fixed projection fit on this run's training trials
        if getattr(mod, 'e', {}).get('latent_proj') == 'pca':
            mod.proj.weight.data.copy_(_pca_axes(source.z, run['train'], mod.proj.out_features))
    opt = optim.AdamW(head.parameters(), lr=tp['learning_rate'], weight_decay=tp['weight_decay'])
    sched = optim.lr_scheduler.SequentialLR(
        opt, schedulers=[optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
                         optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, E - warm), eta_min=tp['min_learning_rate'])],
        milestones=[warm])
    ev_idx = np.unique(np.concatenate([i for g in run['eval'].values() for i in g.values()]))
    pos = {(g, s): np.searchsorted(ev_idx, i) for g, subs in run['eval'].items() for s, i in subs.items()}
    y_all = source.labels.numpy()
    y_ev = y_all[ev_idx]
    # training_params.finetune.class_weight "balanced": weight the loss by inverse class frequency
    # of this run's training trials (BNCI2014008 is 5:1 non-target, so plain CE predicts the majority)
    cw = None
    if tp.get('class_weight') == 'balanced':
        counts = np.bincount(y_all[run['train']], minlength=head_cfg['num_classes']).astype(float)
        cw = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    gen = torch.Generator().manual_seed(seed)
    plotter = MODEL_REGISTRY[tp.get('model_type', 'MeSAE')].plotter_cls(output_dir=out_dir['vis'])
    tail_start, hist = max(0, E - 10), {}
    logger.info(f"[{tag}] train={len(run['train'])} eval={len(ev_idx)} head_params={sum(p.numel() for p in head.parameters())}")
    for epoch in range(1, E + 1):
        head.train()
        losses, preds, ys = [], [], []
        for x, y in iter_batches(source, run['train'], bs, device, shuffle=True, gen=gen):
            opt.zero_grad()
            logits = head(x)
            loss = F.cross_entropy(logits, y, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.detach()); preds.append(logits.argmax(1)); ys.append(y)   # no per-step GPU sync
        sched.step()
        train_metrics = _metrics(torch.cat(ys).cpu().numpy(), torch.cat(preds).cpu().numpy(),
                                 torch.stack(losses).mean().item())
        # eval batch size doesn't change predictions (eval mode: BatchNorm uses running stats)
        logits, val_loss = _predict(head, source, ev_idx, max(bs, 1024), device)
        val_pred = logits.argmax(1)
        val_metrics = _metrics(y_ev, val_pred, val_loss)
        if epoch > tail_start:                       # per-subject balanced accuracy + kappa, from the same pass
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for key, p in pos.items():
                    hist.setdefault(key, []).append(
                        (balanced_accuracy_score(y_ev[p], val_pred[p]), cohen_kappa_score(y_ev[p], val_pred[p])))
        logger.info(f"--- [{tag}] Epoch {epoch}/{E} Summary ---")
        for name, m in (('Train', train_metrics), ('Val  ', val_metrics)):
            logger.info(f"  [{name}] loss: {m['loss']:.4f} | acc: {m['acc']:.4f} | f1: {m['f1']:.4f} | f1_w: {m['f1_weighted']:.4f}"
                        f" | bal_acc: {m['balanced_acc']:.4f} | kappa: {m['kappa']:.4f}")
        logger.info("-" * 40)
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
    plotter.plot_finetune(freeze_backbone=True)       # once, at the end (the 'Backbone Recon MSE' panel stays empty)
    torch.save(make_head_checkpoint(head, head_cfg, source.channel_idx, tp['pretrained_checkpoint']),
               os.path.join(out_dir['ckpt'], 'head.pth'))
    out = {}
    for g, subs in run['eval'].items():
        sd = {}
        for s, i in subs.items():
            bal_acc_hist, kappa_hist = zip(*hist[(g, s)])
            sd[s] = {'tail': float(np.mean(bal_acc_hist)), 'last': float(bal_acc_hist[-1]),
                      'kappa_tail': float(np.mean(kappa_hist)), 'kappa_last': float(kappa_hist[-1]),
                      'n_trials': int(len(i))}
        out[g] = {'subjects': sd, 'n_subjects': len(sd),
                  'mean_tail': float(np.mean([v['tail'] for v in sd.values()])),
                  'mean_last': float(np.mean([v['last'] for v in sd.values()])),
                  'mean_kappa_tail': float(np.mean([v['kappa_tail'] for v in sd.values()])),
                  'mean_kappa_last': float(np.mean([v['kappa_last'] for v in sd.values()]))}
    return out


def main():
    ap = argparse.ArgumentParser(description='Finetune a FeatureHead on a frozen MeSAE backbone')
    ap.add_argument('--config', default='configs/finetune.template.json')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override a config value after base_config merging, dotted path, JSON value '
                         '(repeatable), e.g. --set training_params.finetune.learning_rate=0.003')
    args = ap.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    config = apply_overrides(apply_protocol(config), args.set)      # a named protocol, then --set still wins
    tp = config['training_params']['finetune']
    if tp.get('num_threads'):          # CPU threads for this process (parallel runs share the cores)
        torch.set_num_threads(int(tp['num_threads']))
    if 'split' not in tp:
        raise ValueError(f"training_params.finetune.split is required (type: one of {sorted(SPLITS)})")
    # output_path: where this run writes under output/ -- separate from model_name (a
    # clean identity string), same split as train_pretrain.py's. Falls back to
    # model_name for configs that don't set it.
    base = f"output/{tp.get('output_path', tp.get('model_name', 'default_finetune_run'))}"
    artifact_dir = os.path.join(base, 'artifacts')
    os.makedirs(artifact_dir, exist_ok=True)
    logger, timestamp = setup_logger(artifact_dir)
    snapshot = dict(config, env=env_stamp())
    for name in ('config.json', f'config_{timestamp}.json'):
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(snapshot, f, indent=2)
    dp = config['dataset_params']['finetune']
    if len(dp) != 1:
        raise ValueError("finetune runs one dataset at a time: dataset_params.finetune must have exactly one entry")
    ds_name, ds_args = next(iter(dp.items()))
    pool = resolve_subjects(ds_args['subject_to_use'], _resolve_all_subjects(ds_args['dataset_path']))
    device = tp.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    source = make_source(config, ds_name, pool, device)
    labels = source.labels.numpy()
    num_classes = int(labels.max()) + 1
    assert set(labels.tolist()) == set(range(num_classes)), f"labels must be contiguous 0..{num_classes - 1}"
    head_cfg, new_head = build_head_factory(config, source, num_classes)
    logger.info(f"dataset={ds_name} pool={len(pool)} subjects, {len(labels)} trials, classes={num_classes}, head={head_cfg}")
    subject_data = source.subject_data.numpy()
    runs = make_runs(tp['split'], pool, subject_data, labels, _load_sessions(config, ds_args, pool, subject_data))
    result = {}
    for run in runs:
        tag = f"{ds_name}_{run['name']}"
        dirs = {'ckpt': os.path.join(base, 'finetune', f"run_{run['name']}"), 'vis': os.path.join(base, 'visualization', f"run_{run['name']}")}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        groups = run_one(config, run, source, head_cfg, new_head, tag, dirs, logger, device)
        for g, d in groups.items():
            logger.info(f"  [{run['name']}] group {g}: n={d['n_subjects']} mean_tail={d['mean_tail']:.4f} mean_last={d['mean_last']:.4f}")
        result[run['name']] = {'train_subjects': run['train_subjects'], 'epochs': tp['epochs'],
                               'tail_epochs': min(10, tp['epochs']), 'groups': groups}
        with open(os.path.join(artifact_dir, 'group_eval.json'), 'w') as f:      # rewritten per run: partial results survive
            json.dump(result, f, indent=2)
    logger.info("Finetuning complete.")


if __name__ == '__main__':
    main()
