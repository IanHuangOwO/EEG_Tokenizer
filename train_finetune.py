import os
import sys
import json
import argparse
import shutil

import copy
import random
import logging
import warnings
import statistics
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

from IO.dataset import build_dataset_from_config
from IO.preprocessing import slice_patches
from model.factory import build_finetune_from_config, MODEL_REGISTRY
from viz import pick_trial

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


class FinetuneCollate:
    """FinetuneDataset yields raw (x [C,T], coords [C,3], label, valid_channels [C], valid_length).
    Patchify here via IO.preprocessing.slice_patches, the same helper PretrainDataset/
    TokenizerDataset use, since the backbone expects [B,C,N,L].
    valid_channels is passed through separately (backbone's own channel-attention pool masks
    zero-padded channels internally). pad_mask [B,N] (True=valid) covers only zero-padded
    trailing time (subjects shorter than the batch's max_T) so it doesn't get pooled into the
    classification head as if it were real signal.
    A class (not a closure) so it's picklable for num_workers > 0 on Windows."""
    def __init__(self, patch_len):
        self.patch_len = patch_len

    def __call__(self, batch):
        xs, coords, labels, valid_channels, valid_length = zip(*batch)
        xs             = torch.stack(xs)                       # [B, C, T]
        coords         = torch.stack(coords)                   # [B, C, 3]
        valid_channels = torch.stack(valid_channels)           # [B, C]
        valid_length   = torch.as_tensor(valid_length, dtype=torch.long)  # [B]
        labels = torch.as_tensor([l.item() if torch.is_tensor(l) else l for l in labels], dtype=torch.long)

        B, C, T = xs.shape
        x_patches, _ = slice_patches(xs, self.patch_len)
        P = x_patches.shape[2]
        time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0).expand(B, P).contiguous()

        # ponytail: a patch counts valid only if fully inside the real (non-padded) length —
        # conservative (drops at most one boundary patch per trial) rather than tracking partial overlap
        n_valid_patches = (valid_length // self.patch_len).clamp(max=P)             # [B]
        pad_mask = torch.arange(P).unsqueeze(0) < n_valid_patches.unsqueeze(1)      # [B, P]

        return x_patches, coords, time_idx, labels, valid_channels, pad_mask


def _unpack_batch(batch, device):
    x_patches, coords, time_idx, labels, valid_channels, pad_mask = batch
    return (x_patches.to(device), coords.to(device), time_idx.to(device),
            labels.to(device), valid_channels.to(device), pad_mask.to(device))


def _classification_metrics(all_labels, all_preds):
    """sklearn warns 'looks like regression' when a partial batch has few samples vs
    many classes — a false positive here since labels are always integer class ids."""
    labels = torch.cat(all_labels).numpy()
    preds  = torch.cat(all_preds).numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return {
            'f1':           f1_score(labels, preds, average='macro', zero_division=0),
            'f1_weighted':  f1_score(labels, preds, average='weighted', zero_division=0),
            'balanced_acc': balanced_accuracy_score(labels, preds),
            'kappa':        cohen_kappa_score(labels, preds),
        }


def _finalize_epoch(totals, n, all_labels, all_preds):
    """Average running totals and add classification metrics."""
    metrics = {k: v / n for k, v in totals.items()}
    metrics.update(_classification_metrics(all_labels, all_preds))
    return metrics


@torch.no_grad()
def _recon_mse(model, x, coords, time_idx, valid_channels):
    """Backbone's own unmasked reconstruction MSE against the raw patches — a frozen (or
    slow-lr) backbone that reconstructs poorly on this dataset's real channels caps how much
    class-relevant signal the finetune head can possibly extract, independent of the
    classification loss/acc curve. Masked to real (non-zero-padded) channels only — recon
    on zero-padded channels is meaningless and would just dilute the number toward 0."""
    out = model.backbone(x, coords, time_idx=time_idx, bool_masked_pos=None, valid_channels=valid_channels)
    mask = valid_channels[:, :, None, None].expand_as(x).float()
    return ((out.recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, freeze_backbone=False):
    model.train()
    if freeze_backbone:
        # backbone weights don't update when frozen, but nn.Module.train() still leaves its
        # dropout stochastic — noisy features (and a shifting SAE top-k selection) then churn
        # per step for the same trial, starving the head of a stable signal to learn from.
        model.backbone.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "acc": 0.0, "recon_mse": 0.0}
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, labels, valid_channels, pad_mask = _unpack_batch(batch, device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            logits = model(x, coords, time_idx=time_idx, valid_channels=valid_channels, pad_mask=pad_mask)[0]
            loss = nn.functional.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.amp.autocast(device_type='cuda'):
            recon_mse = _recon_mse(model, x, coords, time_idx, valid_channels)

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        totals["loss"] += loss.item()
        totals["acc"]  += acc.item()
        totals["recon_mse"] += recon_mse.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({'L': f"{totals['loss'] / n:.4f}", 'acc': f"{totals['acc'] / n:.4f}",
                               'rMSE': f"{totals['recon_mse'] / n:.4f}"})

    n = batch_idx + 1
    return _finalize_epoch(totals, n, all_labels, all_preds)


def validate_one_epoch(model, data_loader, device):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "acc": 0.0, "recon_mse": 0.0}
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, labels, valid_channels, pad_mask = _unpack_batch(batch, device)

            with torch.amp.autocast(device_type='cuda'):
                logits = model(x, coords, time_idx=time_idx, valid_channels=valid_channels, pad_mask=pad_mask)[0]
                loss = nn.functional.cross_entropy(logits, labels)
                recon_mse = _recon_mse(model, x, coords, time_idx, valid_channels)

            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            totals["loss"] += loss.item()
            totals["acc"]  += acc.item()
            totals["recon_mse"] += recon_mse.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                pbar.set_postfix({'L': f"{totals['loss'] / n:.4f}", 'acc': f"{totals['acc'] / n:.4f}",
                                   'rMSE': f"{totals['recon_mse'] / n:.4f}"})

    n = batch_idx + 1
    return _finalize_epoch(totals, n, all_labels, all_preds)


def _underlying_base_dataset(ds):
    return ds.base_dataset


def _resolve_all_subjects(data_root):
    with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    all_available_subjects = list(meta.get('data_structure', {}).keys())
    try:
        return sorted([int(s) for s in all_available_subjects])
    except ValueError:
        return sorted(all_available_subjects)


def _resolve_requested_subjects(ds_args, all_available_subjects):
    """all_available_subjects is int-typed when every metadata subject key int-converts
    (see _resolve_all_subjects), so a JSON config's string subject ids ("1", not 1) never
    matched by raw `in` membership -- try both the requested value as-is and int-converted
    against whichever type all_available_subjects actually holds."""
    requested_subjects = ds_args['subject_to_use']
    if requested_subjects in (["all"], "all"):
        return list(all_available_subjects)
    available = set(all_available_subjects)
    resolved = []
    for s in requested_subjects:
        if s in available:
            resolved.append(s)
            continue
        try:
            si = int(s)
        except (TypeError, ValueError):
            continue
        if si in available:
            resolved.append(si)
    return resolved


def build_subject_split_datasets(config, dataset_params, split_ratio, logger):
    """Subject-level holdout: shuffle subjects per dataset, split by ratio.
    Never leaks trials from the same subject across train/val."""
    random.seed(42)
    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    for ds_name, ds_args in dataset_params.items():
        all_available_subjects = _resolve_all_subjects(ds_args['dataset_path'])
        subjects_to_split = _resolve_requested_subjects(ds_args, all_available_subjects)

        random.shuffle(subjects_to_split)
        n_train = int(len(subjects_to_split) * split_ratio)
        if n_train == len(subjects_to_split) and len(subjects_to_split) > 1:
            n_train -= 1
        if n_train == 0 and len(subjects_to_split) > 0:
            n_train = 1

        train_config['dataset_params']['finetune'][ds_name]['subject_to_use'] = subjects_to_split[:n_train]
        val_config['dataset_params']['finetune'][ds_name]['subject_to_use']   = subjects_to_split[n_train:]
        logger.info(f"Dataset {ds_name}: {n_train} Train, {len(subjects_to_split) - n_train} Val subjects (subject split)")

    train_dataset = build_dataset_from_config(train_config, transform=None, mode='finetune')
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='finetune')
    return train_dataset, val_dataset


def _loso_fold_configs(config, dataset_params, ds_name, held_out_subject):
    """Train config: every subject except held_out_subject in ds_name (other datasets
    in dataset_params, if any, contribute fully to train — held-out is only ever
    pulled from its own dataset). Val config: just held_out_subject in ds_name."""
    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    for name, ds_args in dataset_params.items():
        all_available_subjects = _resolve_all_subjects(ds_args['dataset_path'])
        subjects = _resolve_requested_subjects(ds_args, all_available_subjects)

        if name == ds_name:
            train_config['dataset_params']['finetune'][name]['subject_to_use'] = [
                s for s in subjects if s != held_out_subject
            ]
            val_config['dataset_params']['finetune'][name]['subject_to_use'] = [held_out_subject]
        else:
            train_config['dataset_params']['finetune'][name]['subject_to_use'] = subjects
            val_config['dataset_params']['finetune'][name]['subject_to_use']   = []

    return train_config, val_config


def _loso_folds(dataset_params):
    """List of (ds_name, subject) pairs, one per LOSO fold."""
    folds = []
    for ds_name, ds_args in dataset_params.items():
        all_available_subjects = _resolve_all_subjects(ds_args['dataset_path'])
        subjects = _resolve_requested_subjects(ds_args, all_available_subjects)
        folds.extend((ds_name, subject) for subject in subjects)
    return folds


def run_training_loop(config, train_dataset, val_dataset, checkpoint_dir, vis_dir,
                       artifact_dir, logger, patch_len, fold_tag=""):
    """Builds the finetune model/optimizer/scheduler and runs the full epoch loop for one
    train/val dataset pair. Returns the metrics dict (train+val) from the best-val-acc epoch."""
    train_params = config['training_params']['finetune']
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    viz_every_n = config.get('training_params', {}).get('visualize', {}).get('finetune', {}).get('every_n_epochs', 5)

    collate_fn = FinetuneCollate(patch_len)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=train_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)

    train_base = _underlying_base_dataset(train_dataset)
    val_base   = _underlying_base_dataset(val_dataset)

    train_labels = set(train_base.labels.tolist())
    val_labels   = set(val_base.labels.tolist())
    all_labels   = train_labels | val_labels
    num_classes  = len(all_labels)
    assert all_labels == set(range(num_classes)), \
        f"labels must be contiguous 0..{num_classes - 1}, got {sorted(all_labels)}"
    if val_labels - train_labels:
        logger.info(f"  WARNING [{fold_tag}]: classes {sorted(val_labels - train_labels)} only appear in val, never trained on")
    logger.info(f"[{fold_tag}] num_classes={num_classes}")

    model_type = train_params.get('model_type', 'MeFSQ')
    entry      = MODEL_REGISTRY[model_type]
    checker    = entry.checker_cls()

    logger.info(f"[{fold_tag}] Loading pretrained backbone...")
    model = build_finetune_from_config(config, num_classes, mode='finetune')
    logger.info(f"[{fold_tag}] Loaded backbone weights from {train_params['pretrained_checkpoint']}")
    freeze_backbone = config['model_params'][model_type].get('finetune', {}).get('freeze_backbone', False)
    model.to(device)

    scaler = torch.amp.GradScaler('cuda')
    # backbone_lr_mult: unfrozen backbone shares gradients with a head that has orders of
    # magnitude fewer params and far less signal-to-noise (few subjects) — full-speed backbone
    # updates memorize per-subject artifacts within a handful of epochs (see finetune log:
    # train acc 98% by epoch 32, val loss 3.7 -> 12.5). Default 0.1x keeps backbone adaptation
    # slow relative to the head instead of turning it off entirely (freeze_backbone=True).
    backbone_lr_mult = train_params.get('backbone_lr_mult', 0.1)
    head_params     = list(model.head.parameters())
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    param_groups = [{'params': head_params, 'lr': train_params['learning_rate']}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': train_params['learning_rate'] * backbone_lr_mult})
    optimizer = optim.AdamW(param_groups, weight_decay=train_params['weight_decay'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = entry.plotter_cls(output_dir=vis_dir)

    topo_trial_idx = topo_subject_id = None
    try:
        _first_val_subject = val_base.subject_data[0].item()
        topo_trial_idx, topo_subject_id = pick_trial(val_dataset, _first_val_subject, trial=1)
        logger.info(f"[{fold_tag}] Topo viz: subject={topo_subject_id}, trial_idx={topo_trial_idx}")
    except Exception as e:
        logger.warning(f"[{fold_tag}] Topo trial pick failed: {e}")

    best_val_acc = 0.0
    best_metrics = None
    total_epochs = train_params['epochs']
    logger.info(f"[{fold_tag}] Starting Finetuning ({total_epochs} epochs, freeze_backbone={freeze_backbone})")

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, freeze_backbone=freeze_backbone)
        val_metrics   = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        logger.info(f"--- [{fold_tag}] Epoch {epoch}/{total_epochs} Summary ---")
        logger.info(f"  [Train] loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.4f} | f1: {train_metrics['f1']:.4f} | f1_w: {train_metrics['f1_weighted']:.4f} | bal_acc: {train_metrics['balanced_acc']:.4f} | kappa: {train_metrics['kappa']:.4f} | recon_mse: {train_metrics['recon_mse']:.4f}")
        logger.info(f"  [Val]   loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.4f} | f1: {val_metrics['f1']:.4f} | f1_w: {val_metrics['f1_weighted']:.4f} | bal_acc: {val_metrics['balanced_acc']:.4f} | kappa: {val_metrics['kappa']:.4f} | recon_mse: {val_metrics['recon_mse']:.4f}")
        logger.info("-" * 40)

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_metrics = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics}
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_finetune.pth'))
            logger.info(f"  > [{fold_tag}] Saved Best Checkpoint")

        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot_finetune(freeze_backbone=freeze_backbone)

        if epoch % viz_every_n == 0 and topo_trial_idx is not None:
            try:
                checker.check_finetune(
                    config, vis_dir, model, val_dataset,
                    topo_trial_idx, subject_id=topo_subject_id, epoch=epoch,
                )
            except Exception as e:
                logger.warning(f"  Topomap viz failed (epoch {epoch}): {e}")

    logger.info(f"[{fold_tag}] Finetuning Complete. Best val acc: {best_val_acc:.4f} (epoch {best_metrics['epoch'] if best_metrics else 'n/a'})")
    return best_metrics


def _run_loso(config, dataset_params, base_output_dir, artifact_dir, logger, patch_len):
    folds = _loso_folds(dataset_params)
    logger.info(f"LOSO: {len(folds)} folds ({', '.join(f'{ds}:{s}' for ds, s in folds)})")

    fold_results = {}
    for ds_name, subject in folds:
        fold_tag = f"{ds_name}_{subject}"
        logger.info(f"===== LOSO fold {fold_tag} =====")
        train_config, val_config = _loso_fold_configs(config, dataset_params, ds_name, subject)

        train_dataset = build_dataset_from_config(train_config, transform=None, mode='finetune')
        val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='finetune')
        logger.info(f"[{fold_tag}] Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

        checkpoint_dir = os.path.join(base_output_dir, "finetune", f"fold_{fold_tag}")
        vis_dir        = os.path.join(base_output_dir, "visualization", f"fold_{fold_tag}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        best_metrics = run_training_loop(
            config, train_dataset, val_dataset, checkpoint_dir, vis_dir,
            artifact_dir, logger, patch_len, fold_tag=fold_tag,
        )
        fold_results[fold_tag] = best_metrics

    logger.info("===== LOSO Summary (best-val-acc epoch per fold) =====")
    metric_keys = ['acc', 'f1', 'f1_weighted', 'balanced_acc', 'kappa']
    per_metric = {k: [] for k in metric_keys}
    for fold_tag, best_metrics in fold_results.items():
        if best_metrics is None:
            logger.info(f"  {fold_tag}: no valid epoch")
            continue
        v = best_metrics['val']
        logger.info(f"  {fold_tag}: " + " | ".join(f"{k}={v[k]:.4f}" for k in metric_keys))
        for k in metric_keys:
            per_metric[k].append(v[k])

    summary = {'folds': {}, 'aggregate': {}}
    for fold_tag, best_metrics in fold_results.items():
        summary['folds'][fold_tag] = best_metrics
    for k in metric_keys:
        vals = per_metric[k]
        if vals:
            mean = statistics.mean(vals)
            std  = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            summary['aggregate'][k] = {'mean': mean, 'std': std}
            logger.info(f"  MEAN {k}: {mean:.4f} +/- {std:.4f}")

    summary_path = os.path.join(artifact_dir, 'loso_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"LOSO summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='MeFSQ Finetuning')
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_params = config['training_params']['finetune']
    model_name = train_params.get('model_name', 'default_finetune_run')

    base_output_dir = f"output/{model_name}"
    checkpoint_dir  = os.path.join(base_output_dir, "finetune")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")
    vis_dir         = os.path.join(base_output_dir, "visualization")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger, timestamp = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))
    shutil.copy(args.config, os.path.join(artifact_dir, f'config_{timestamp}.json'))

    dataset_params = config['dataset_params']['finetune']
    split_ratio = train_params.get('train_val_split', 0.9)
    split_mode  = train_params.get('split_mode', 'subject')

    pp = config.get('preprocess_params', {})
    patch_len = pp.get('patch_length', 100)

    logger.info(f"split_mode={split_mode}")

    if split_mode == 'loso':
        _run_loso(config, dataset_params, base_output_dir, artifact_dir, logger, patch_len)
        return

    if split_mode == 'subject':
        train_dataset, val_dataset = build_subject_split_datasets(config, dataset_params, split_ratio, logger)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r} (expected 'subject' or 'loso')")

    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")
    run_training_loop(config, train_dataset, val_dataset, checkpoint_dir, vis_dir, artifact_dir, logger, patch_len, fold_tag=split_mode)
    logger.info("Finetuning Complete.")


if __name__ == '__main__':
    main()
