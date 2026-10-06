"""Finetune a FeatureHead on a frozen Qtome backbone.

The backbone never trains: Q-atom features come from the amplitude cache (cache_feature.py), raw
features from the compiled dataset; only the head is optimised (fp32, batches indexed from RAM).
training_params.finetune.split picks a split type (SPLITS: loso, subject_kfold, eval_subjects,
kfold, blocked_kfold, fewshot). Every run writes artifacts/group_eval.json: per-subject tail (mean of the last 10
epochs) and last-epoch balanced accuracy."""
import argparse, copy, json, logging, os, subprocess, sys, warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

from IO.dataset import build_dataset_from_config
from IO.preprocessing import num_patches, slice_patches
from cache_feature import CachedAtomDataset, get_atom_cache
from model.factory import MODEL_REGISTRY, load_backbone
from model.Qtome.Qtome_modules import (FeatureHead, AtomExtractor, make_head_checkpoint,
                                       resolve_head_config, needs_atom, needs_raw, needs_latent, feature_names)
from tools.analysis import apply_overrides, load_config
from IO.splits import SPLITS, all_subjects, load_sessions, make_runs, protocol_split, resolve_subjects

QTOME_HEADS = 'configs/Qtome/protocol_heads.json'

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



def apply_protocol(config, heads=QTOME_HEADS):
    """training_params.finetune.protocol = <name>: the protocol's split (IO/splits.py protocol_split: the shared table
    plus per-dataset settings), then Qtome's head and optimiser settings for it (configs/Qtome/protocol_heads.json,
    dotted keys in order). No protocol key: unchanged."""
    name = config['training_params']['finetune'].get('protocol')
    if not name:
        return config
    split = protocol_split(name, list(config['dataset_params']['finetune']))
    table = {k: v for k, v in json.load(open(heads)).items() if not k.startswith('_')}
    if name not in table:
        raise ValueError(f"protocol {name!r} has no Qtome head in {heads}")
    return apply_overrides(config, [f'training_params.finetune.split={json.dumps(split)}'] +
                           [f'{k}={json.dumps(v)}' for k, v in table[name].items()])


class AtomSource:
    """Cached Q-atom amplitudes of the pool (cache_feature.py), moved to the device once (a
    BNCI2014008 64-channel impute cache is ~1 GB fp16): the head is tiny, so per-batch CPU
    indexing and host-to-device copies were most of a step's time."""
    kind = 'atom'

    def __init__(self, config, ds_name, pool, device, latent=False):
        subs = [str(s) for s in pool]
        self.data = CachedAtomDataset(get_atom_cache(config, ds_name, subs, latent=latent), subs)
        self.labels, self.subject_data = self.data.labels, self.data.subject_data
        self.channel_idx = self.data.channel_idx
        self.num_patches, self.num_atoms = self.data.num_patches, self.data.num_atoms
        self.amp, self.labels_dev = self.data.amp.to(device), self.labels.to(device)
        self.z = self.data.z.to(device) if latent else None                 # [T, N', Cv, D] fp16
        self.latent_dim = self.z.shape[-1] if latent else None

    def get(self, idx):
        idx = idx.to(self.amp.device)
        out = {'atom': self.amp[idx].float()}
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
        self.num_atoms = 0

    def get(self, idx):
        xp, _ = slice_patches(self.x[idx], self.patch_len, self.patch_stride)  # [B, C_valid, N', L]
        return {'raw': xp}, self.labels[idx]


class CombinedSource:
    """Serves a AtomSource and a RawSource together, for a head whose features list needs
    both (e.g. features=['atom_power', 'raw_band']). Exposes the union of attributes either
    single source exposes (num_patches/num_atoms/channel_idx/labels/subject_data) --
    both sources are built from the SAME (ds_name, pool), so their per-trial ordering,
    labels and channel_idx must already agree; asserted once at construction, not re-checked
    per batch."""
    kind = 'combined'

    def __init__(self, atom_source, raw_source):
        assert atom_source.channel_idx == raw_source.channel_idx, \
            "AtomSource/RawSource channel_idx mismatch -- same dataset/pool should agree"
        assert torch.equal(atom_source.labels, raw_source.labels), \
            "AtomSource/RawSource label order mismatch -- same dataset/pool should agree"
        self.atom, self.raw = atom_source, raw_source
        self.labels, self.subject_data = atom_source.labels, atom_source.subject_data
        self.channel_idx = atom_source.channel_idx
        self.num_patches, self.num_atoms = atom_source.num_patches, atom_source.num_atoms
        self.z, self.latent_dim = atom_source.z, atom_source.latent_dim

    def get(self, idx):
        atom_d, y = self.atom.get(idx)
        raw_d, _ = self.raw.get(idx)
        return {**atom_d, **raw_d}, y


def make_source(config, ds_name, pool, device):
    ft_cfg = dict(config['model_params']['Qtome']['finetune'])
    want_atom, want_raw, latent = needs_atom(ft_cfg), needs_raw(ft_cfg), needs_latent(ft_cfg)
    if latent:   # latent_source: which z the latent_* entries read ('output' = what the Q-atoms read)
        latent = ft_cfg.get('latent_source', 'output')
        if latent not in ('output', 'bottleneck'):
            raise ValueError(f"latent_source must be output|bottleneck, got {latent!r}")
    k = int(config['training_params']['finetune'].get('latent_pool', 1))
    if k > 1:   # experiment (2026-09-29): average k adjacent z tokens, for a head that reads only z
        if not all(n.startswith('latent_') for n in feature_names(ft_cfg)):
            raise ValueError("training_params.finetune.latent_pool needs a head with only latent_* entries")
        src = AtomSource(config, ds_name, pool, device, latent)
        src.z = pool_tokens(src.z, k)
        src.num_patches = src.z.shape[1]
        return src
    if want_atom and want_raw:
        return CombinedSource(AtomSource(config, ds_name, pool, device, latent), RawSource(config, ds_name, pool))
    if want_atom:
        return AtomSource(config, ds_name, pool, device, latent)
    return RawSource(config, ds_name, pool)


def pool_tokens(z, k):
    """z [T, N, ...] -> [T, N // k, ...]: mean of k adjacent tokens (a trailing N % k is dropped). Splits only the
    token axis, so the reshape is safe (docs/agents/reshape-pitfalls.md)."""
    n = z.shape[1] // k * k
    return z[:, :n].reshape(z.shape[0], n // k, k, *z.shape[2:]).float().mean(2).to(z.dtype)


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


def env_atom():
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
        config['model_params']['Qtome']['finetune'], num_classes=num_classes, num_patches=source.num_patches,
        num_channels=len(source.channel_idx), num_atoms=source.num_atoms, patch_len=patch_len,
        latent_dim=getattr(source, 'latent_dim', None),
        patch_stride=pp.get('patch_stride', patch_len), sample_freq=float(pp['sample_freq']))
    tables = None
    if 'atom_band' in feature_names(cfg):   # the template spectra need the backbone, once
        tables = AtomExtractor(load_backbone(config), [0]).band_tables(cfg['sample_freq'])

    def new_head():
        head = FeatureHead(cfg)
        if tables is not None:
            head.entries['atom_band'].E_D.copy_(tables[0])
            head.entries['atom_band'].E_H.copy_(tables[1])
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


def run_closed_form(config, run, source, tag, logger):
    """training_params.finetune.fit 'closed_form': fit ClosedFormHead (model/Qtome/closed_form.py) on run['train'],
    no SGD; per-subject balanced accuracy + kappa on run['eval'], in run_one's format (tail = last: one fit)."""
    from model.Qtome.closed_form import ClosedFormHead
    tp = config['training_params']['finetune']
    if source.kind != 'atom':
        raise ValueError("fit 'closed_form' reads the atom code: the head needs a atom entry")
    amp, y = source.data.amp, source.labels.numpy()
    head = ClosedFormHead(**tp.get('closed_form', {})).fit(amp[run['train']].float().numpy(), y[run['train']])
    logger.info(f"[{tag}] closed_form {tp.get('closed_form', {})} train={len(run['train'])}")
    out = {}
    for g, subs in run['eval'].items():
        sd = {}
        for s, i in subs.items():
            pred = head.predict(amp[i].float().numpy())
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                acc, kappa = float(balanced_accuracy_score(y[i], pred)), float(cohen_kappa_score(y[i], pred))
            sd[s] = {'tail': acc, 'last': acc, 'kappa_tail': kappa, 'kappa_last': kappa, 'n_trials': int(len(i))}
        out[g] = {'subjects': sd, 'n_subjects': len(sd),
                  **{f'mean_{k}': float(np.mean([v[k] for v in sd.values()]))
                     for k in ('tail', 'last', 'kappa_tail', 'kappa_last')}}
    return out


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
    plotter = MODEL_REGISTRY[tp.get('model_type', 'Qtome')].plotter_cls(output_dir=out_dir['vis'])
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
    ap = argparse.ArgumentParser(description='Finetune a FeatureHead on a frozen Qtome backbone')
    ap.add_argument('--config', default='configs/Qtome/finetune.template.json')
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
    snapshot = dict(config, env=env_atom())
    for name in ('config.json', f'config_{timestamp}.json'):
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(snapshot, f, indent=2)
    dp = config['dataset_params']['finetune']
    if len(dp) != 1:
        raise ValueError("finetune runs one dataset at a time: dataset_params.finetune must have exactly one entry")
    ds_name, ds_args = next(iter(dp.items()))
    pool = resolve_subjects(ds_args['subject_to_use'], all_subjects(ds_args['dataset_path']))
    device = tp.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    source = make_source(config, ds_name, pool, device)
    labels = source.labels.numpy()
    num_classes = int(labels.max()) + 1
    assert set(labels.tolist()) == set(range(num_classes)), f"labels must be contiguous 0..{num_classes - 1}"
    head_cfg, new_head = build_head_factory(config, source, num_classes)
    logger.info(f"dataset={ds_name} pool={len(pool)} subjects, {len(labels)} trials, classes={num_classes}, head={head_cfg}")
    subject_data = source.subject_data.numpy()
    runs = make_runs(tp['split'], pool, subject_data, labels, load_sessions(config, ds_args, pool, subject_data))
    result = {}
    for run in runs:
        tag = f"{ds_name}_{run['name']}"
        dirs = {'ckpt': os.path.join(base, 'finetune', f"run_{run['name']}"), 'vis': os.path.join(base, 'visualization', f"run_{run['name']}")}
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        groups = (run_closed_form(config, run, source, tag, logger) if tp.get('fit') == 'closed_form'
                  else run_one(config, run, source, head_cfg, new_head, tag, dirs, logger, device))
        for g, d in groups.items():
            logger.info(f"  [{run['name']}] group {g}: n={d['n_subjects']} mean_tail={d['mean_tail']:.4f} mean_last={d['mean_last']:.4f}")
        result[run['name']] = {'train_subjects': run['train_subjects'], 'epochs': tp['epochs'],
                               'tail_epochs': min(10, tp['epochs']), 'groups': groups}
        with open(os.path.join(artifact_dir, 'group_eval.json'), 'w') as f:      # rewritten per run: partial results survive
            json.dump(result, f, indent=2)
    logger.info("Finetuning complete.")


if __name__ == '__main__':
    main()
