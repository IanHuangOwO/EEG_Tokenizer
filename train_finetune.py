"""Finetune a FeatureHead on a frozen MeSAE backbone (ADR 0016; finetune restructure, sub-project C).

The backbone never trains: stamp features come from the amplitude cache (cache_feature.py), raw
features from the compiled dataset; only the head is optimised (fp32, batches indexed from RAM).
training_params.finetune.split has two modes, intra_subject and inter_subject (see the plan /
CLAUDE.md). Every run writes artifacts/group_eval.json: per-subject tail (mean of the last 10
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
from IO.preprocessing import num_patches, slice_patches
from cache_feature import CachedStampDataset, get_stamp_cache
from model.factory import MODEL_REGISTRY, load_backbone
from model.MeSAE.MeSAE_modules import (FeatureHead, StampExtractor, make_head_checkpoint,
                                       resolve_head_config, needs_stamp, needs_raw,
                                       _normalize_features)
from tools.analysis import load_config

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


def _trials_of(subject_data, subjects):
    return np.flatnonzero(np.isin(subject_data, [int(s) for s in subjects]))


def _resolve_auto_split(split, ds_name, pretrained_checkpoint):
    """split['eval_subjects'] == 'auto' -> the cached (or freshly generated)
    configs/finetune_eval_splits/<ds_name.lower()>.json seen/unseen split, filled into
    eval_subjects/train_subjects. Any other eval_subjects value (a list, or an explicit
    dict) passes through unchanged -- 'auto' is opt-in, not the default.

    Errors, deliberately, rather than falling back to a different split mode, if ds_name
    isn't one tools.analysis.select_eval_subsets.DATASETS knows how to generate: two runs
    both saying eval_subjects='auto' must mean the same split *mode* regardless of which
    dataset ends up plugged in, or they stop being comparable to each other. Registering a
    new dataset there is a deliberate step (see docs/agents/adding-a-tool.md), not
    something 'auto' should paper over by silently choosing n_folds instead."""
    if split.get('eval_subjects') != 'auto':
        return split
    from tools.analysis.select_eval_subsets import DATASETS, select_eval_subsets
    try:
        key = next(k for k, v in DATASETS.items() if v == ds_name)
    except StopIteration:
        raise ValueError(
            f"split.eval_subjects='auto' needs '{ds_name}' registered in "
            f"tools.analysis.select_eval_subsets.DATASETS (currently: {sorted(DATASETS)}) "
            "-- add it there first, 'auto' does not fall back to a different split mode")
    cache_path = os.path.join('configs', 'finetune_eval_splits', f'{key}.json')
    if not os.path.exists(cache_path):
        run_config = os.path.join(os.path.dirname(os.path.dirname(pretrained_checkpoint)),
                                  'artifacts', 'config.json')
        select_eval_subsets(names=[key], run_config=run_config)
    with open(cache_path) as f:
        cached = json.load(f)
    return {**split, 'eval_subjects': cached['eval'], 'train_subjects': cached['train']}


def make_runs(split, pool, subject_data, labels):
    """split block -> runs [{name, train, train_subjects, eval}] (see the plan's contract)."""
    mode, seed = split.get('mode'), split.get('seed', 42)
    if mode == 'intra_subject':
        k = int(split['n_folds'])
        runs = []
        for s in pool:
            idx = np.flatnonzero(subject_data == int(s))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for i, (tr, va) in enumerate(skf.split(idx, labels[idx])):
                runs.append(dict(name=f'{s}_fold{i}', train=np.sort(idx[tr]), train_subjects=[str(s)],
                                 eval={'heldout': {str(s): np.sort(idx[va])}}))
        return runs
    if mode != 'inter_subject':
        raise ValueError(f"split.mode must be 'intra_subject' or 'inter_subject', got {mode!r}")
    if ('n_folds' in split) == ('eval_subjects' in split):
        raise ValueError("inter_subject needs exactly one of n_folds / eval_subjects")
    train_pool = resolve_subjects(split['train_subjects'], pool) if 'train_subjects' in split else None
    if 'n_folds' in split:
        k = int(split['n_folds'])
        if not 2 <= k <= len(pool):
            raise ValueError(f"n_folds must be in [2, {len(pool)}], got {k}")
        subs = list(pool)
        random.Random(seed).shuffle(subs)
        evals = [(f'fold{i}', {'heldout': subs[i::k]}) for i in range(k)]
    else:
        ev = split['eval_subjects']
        ev = {'heldout': ev} if not isinstance(ev, dict) or 'random' in ev else ev
        evals = [('main', {g: resolve_subjects(v, pool) for g, v in ev.items()})]
    runs = []
    for name, groups in evals:
        ev_subs = {s for v in groups.values() for s in v}
        train_subs = [s for s in (train_pool if train_pool is not None else pool) if s not in ev_subs]
        if not train_subs or any(not v for v in groups.values()):
            raise ValueError(f"run {name}: empty training set or evaluation group")
        if 'eval_subjects' in split and train_pool is not None and set(train_pool) & ev_subs:   # n_folds: the fold is subtracted instead
            raise ValueError(f"run {name}: train_subjects and evaluation subjects overlap: {sorted(set(train_pool) & ev_subs)}")
        runs.append(dict(name=name, train=_trials_of(subject_data, train_subs), train_subjects=[str(s) for s in train_subs],
                         eval={g: {str(s): _trials_of(subject_data, [s]) for s in v} for g, v in groups.items()}))
    return runs


class StampSource:
    """Cached stamp amplitudes of the pool (cache_feature.py), in RAM."""
    kind = 'stamp'

    def __init__(self, config, ds_name, pool, device):
        subs = [str(s) for s in pool]
        self.data = CachedStampDataset(get_stamp_cache(config, ds_name, subs, device=device), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx, self.keep = self.data.channel_idx, self.data.keep
        self.num_patches, self.num_stamps = self.data.num_patches, self.data.num_stamps

    def get(self, idx):
        return {'stamp': self.data.amp[idx].float()}, self.labels[idx]


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
        self.num_stamps, self.keep = 0, None

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return {'raw': xp}, self.labels[idx]


class CombinedSource:
    """Serves a StampSource and a RawSource together, for a head whose features list needs
    both (e.g. features=['stamp_power', 'raw_band']). Exposes the union of attributes either
    single source exposes (num_patches/num_stamps/channel_idx/keep/labels/subject_data) --
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
        self.channel_idx, self.keep = stamp_source.channel_idx, stamp_source.keep
        self.num_patches, self.num_stamps = stamp_source.num_patches, stamp_source.num_stamps

    def get(self, idx):
        stamp_d, y = self.stamp.get(idx)
        raw_d, _ = self.raw.get(idx)
        return {**stamp_d, **raw_d}, y


def make_source(config, ds_name, pool, device):
    ft_cfg = dict(config['model_params']['MeSAE']['finetune'])
    _normalize_features(ft_cfg)
    want_stamp, want_raw = needs_stamp(ft_cfg), needs_raw(ft_cfg)
    if want_stamp and want_raw:
        return CombinedSource(StampSource(config, ds_name, pool, device), RawSource(config, ds_name, pool))
    if want_stamp:
        return StampSource(config, ds_name, pool, device)
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
        logits.append(head(x).float().cpu()); ys.append(y.cpu())
    logits, ys = torch.cat(logits), torch.cat(ys)
    return logits.numpy(), F.cross_entropy(logits, ys).item()


def build_head_factory(config, source, num_classes):
    """(resolved head config, function returning a freshly initialised FeatureHead)."""
    pp = config['preprocess_params']
    patch_len = pp.get('patch_length', 100)
    cfg = resolve_head_config(
        config['model_params']['MeSAE']['finetune'], num_classes=num_classes, num_patches=source.num_patches,
        num_channels=len(source.channel_idx), num_stamps=source.num_stamps, patch_len=patch_len,
        patch_stride=pp.get('patch_stride', patch_len), sample_freq=float(pp['sample_freq']))
    tables = None
    if 'stamp_band' in cfg['features']:   # the template spectra need the backbone, once
        tables = StampExtractor(load_backbone(config), [0]).band_tables(cfg['sample_freq'])

    def new_head():
        head = FeatureHead(cfg)
        if tables is not None:
            head.entries['stamp_band'].E_D.copy_(tables[0])
            head.entries['stamp_band'].E_H.copy_(tables[1])
        return head
    return cfg, new_head


def run_one(config, run, source, head_cfg, new_head, tag, out_dir, logger, device):
    """Train one head on run['train'], evaluate on the union of run['eval'] every epoch."""
    tp = config['training_params']['finetune']
    E, bs, seed, warm = tp['epochs'], tp['batch_size'], tp.get('seed', 42), tp['warmup_epochs']
    torch.manual_seed(seed)
    head = new_head().to(device)
    opt = optim.AdamW(head.parameters(), lr=tp['learning_rate'], weight_decay=tp['weight_decay'])
    sched = optim.lr_scheduler.SequentialLR(
        opt, schedulers=[optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warm),
                         optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, E - warm), eta_min=tp['min_learning_rate'])],
        milestones=[warm])
    ev_idx = np.unique(np.concatenate([i for g in run['eval'].values() for i in g.values()]))
    pos = {(g, s): np.searchsorted(ev_idx, i) for g, subs in run['eval'].items() for s, i in subs.items()}
    y_all = source.labels.numpy()
    y_ev = y_all[ev_idx]
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
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item()); preds.append(logits.argmax(1).cpu()); ys.append(y.cpu())
        sched.step()
        train_metrics = _metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy(), float(np.mean(losses)))
        logits, val_loss = _predict(head, source, ev_idx, bs, device)
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
    torch.save(make_head_checkpoint(head, head_cfg, source.channel_idx, source.keep, tp['pretrained_checkpoint']),
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
    args = ap.parse_args()
    config = load_config(args.config)
    tp = config['training_params']['finetune']
    if 'split' not in tp:
        raise ValueError("training_params.finetune.split is required (mode: intra_subject | inter_subject)")
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
    tp['split'] = _resolve_auto_split(tp['split'], ds_name, tp['pretrained_checkpoint'])
    runs = make_runs(tp['split'], pool, source.subject_data.numpy(), labels)
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
