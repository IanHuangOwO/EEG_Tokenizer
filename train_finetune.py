import os
import sys
import json
import argparse
import shutil

import copy
import random
import logging
import warnings
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config
from model.MeFSQ.MeFSQ import MeFSQFinetune
from viz.train import Plotter
from viz import pick_trial
from viz.check_epoch_finetune import run as run_recon_analysis_finetune

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
    return logger


class FinetuneCollate:
    """FinetuneDataset yields raw (x [C,T], coords [C,3], label, valid_channels [C], valid_length).
    Patchify here the same way MaskedPretrainDataset does, since the backbone expects [B,C,N,L].
    Also builds pad_mask [B,C,N] (True=valid) so zero-padded channels (multi-dataset channel
    unification) and zero-padded trailing time (subjects shorter than the batch's max_T) don't
    get pooled into the classification head as if they were real signal.
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
        P = T // self.patch_len
        x_patches = xs[:, :, :P * self.patch_len].reshape(B, C, P, self.patch_len)
        time_idx  = torch.arange(P, dtype=torch.long).unsqueeze(0).expand(B, P).contiguous()

        # ponytail: a patch counts valid only if fully inside the real (non-padded) length —
        # conservative (drops at most one boundary patch per trial) rather than tracking partial overlap
        n_valid_patches = (valid_length // self.patch_len).clamp(max=P)             # [B]
        patch_valid = torch.arange(P).unsqueeze(0) < n_valid_patches.unsqueeze(1)   # [B, P]
        pad_mask = valid_channels.unsqueeze(-1) & patch_valid.unsqueeze(1)          # [B, C, P]

        return x_patches, coords, time_idx, labels, pad_mask


def _unpack_batch(batch, device):
    x_patches, coords, time_idx, labels, pad_mask = batch
    return (x_patches.to(device), coords.to(device), time_idx.to(device),
            labels.to(device), pad_mask.to(device))


def _macro_f1(labels, preds):
    """sklearn warns 'looks like regression' when a partial batch has few samples vs
    many classes — a false positive here since labels are always integer class ids."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return f1_score(torch.cat(labels), torch.cat(preds), average='macro', zero_division=0)


def _finalize_epoch(model, totals, n, all_labels, all_preds, v_q_gated):
    """Average running totals, add macro F1, and merge backbone head/fingerprint metrics."""
    metrics = {k: v / n for k, v in totals.items()}
    metrics['f1'] = _macro_f1(all_labels, all_preds)
    metrics.update(model.backbone.get_metrics(v_q_gated.detach()))
    return metrics


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, load_balance_weight=0.0, diversity_weight=0.0):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "acc": 0.0}
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, labels, pad_mask = _unpack_batch(batch, device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            logits, _, _, lb_loss, v_q_gated, gate_mask, B_, C_ = model(
                x, coords, time_idx=time_idx, pad_mask=pad_mask, return_head_stats=True)
            loss = nn.functional.cross_entropy(logits, labels)
            if load_balance_weight > 0:
                loss = loss + load_balance_weight * lb_loss
            model.backbone.update_head_metrics(gate_mask)
            if diversity_weight > 0:
                loss = loss + diversity_weight * model.backbone.get_diversity_loss(v_q_gated, gate_mask, B_, C_)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        totals["loss"] += loss.item()
        totals["acc"]  += acc.item()
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            running_f1 = _macro_f1(all_labels, all_preds)
            pbar.set_postfix({'L': f"{totals['loss'] / n:.4f}", 'acc': f"{totals['acc'] / n:.4f}", 'f1': f"{running_f1:.4f}"})

    n = batch_idx + 1
    return _finalize_epoch(model, totals, n, all_labels, all_preds, v_q_gated)


def validate_one_epoch(model, data_loader, device):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "acc": 0.0}
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, labels, pad_mask = _unpack_batch(batch, device)

            with torch.amp.autocast(device_type='cuda'):
                logits, _, _, _, v_q_gated, _, _, _ = model(
                    x, coords, time_idx=time_idx, pad_mask=pad_mask, return_head_stats=True)
                loss = nn.functional.cross_entropy(logits, labels)

            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
            totals["loss"] += loss.item()
            totals["acc"]  += acc.item()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                running_f1 = _macro_f1(all_labels, all_preds)
                pbar.set_postfix({'L': f"{totals['loss'] / n:.4f}", 'acc': f"{totals['acc'] / n:.4f}", 'f1': f"{running_f1:.4f}"})

    n = batch_idx + 1
    return _finalize_epoch(model, totals, n, all_labels, all_preds, v_q_gated)


def main():
    parser = argparse.ArgumentParser(description='MeFSQ Finetuning')
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_params = config['training_params']['finetune']
    device     = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_params.get('model_name', 'default_finetune_run')

    base_output_dir = f"output/{model_name}"
    checkpoint_dir  = os.path.join(base_output_dir, "finetune")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")
    vis_dir         = os.path.join(base_output_dir, "visualization")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))

    dataset_params = config['dataset_params']['finetune']
    split_ratio = train_params.get('train_val_split', 0.9)
    random.seed(42)

    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    for ds_name, ds_args in dataset_params.items():
        data_root = ds_args['dataset_path']
        with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
            meta = json.load(f)

        all_available_subjects = list(meta.get('data_structure', {}).keys())
        try:
            all_available_subjects = sorted([int(s) for s in all_available_subjects])
        except:
            all_available_subjects = sorted(all_available_subjects)

        requested_subjects = ds_args['subject_to_use']
        if requested_subjects in (["all"], "all"):
            subjects_to_split = all_available_subjects
        else:
            subjects_to_split = [s for s in requested_subjects if s in all_available_subjects or str(s) in all_available_subjects]

        random.shuffle(subjects_to_split)
        n_train = int(len(subjects_to_split) * split_ratio)
        if n_train == len(subjects_to_split) and len(subjects_to_split) > 1:
            n_train -= 1
        if n_train == 0 and len(subjects_to_split) > 0:
            n_train = 1

        train_config['dataset_params']['finetune'][ds_name]['subject_to_use'] = subjects_to_split[:n_train]
        val_config['dataset_params']['finetune'][ds_name]['subject_to_use']   = subjects_to_split[n_train:]
        logger.info(f"Dataset {ds_name}: {n_train} Train, {len(subjects_to_split) - n_train} Val subjects")

    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='finetune')
    logger.info("Building Validation Dataset...")
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='finetune')
    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    _first_subject = train_dataset.base_dataset.subject_data[0].item()
    topo_trial_idx, topo_subject_id = pick_trial(train_dataset, _first_subject, trial=1)
    logger.info(f"Topo viz: subject={topo_subject_id}, trial_idx={topo_trial_idx}")

    train_labels = set(train_dataset.base_dataset.labels.tolist())
    val_labels   = set(val_dataset.base_dataset.labels.tolist())
    all_labels   = train_labels | val_labels
    num_classes  = len(all_labels)
    assert all_labels == set(range(num_classes)), \
        f"labels must be contiguous 0..{num_classes - 1}, got {sorted(all_labels)}"
    if val_labels - train_labels:
        logger.info(f"WARNING: classes {sorted(val_labels - train_labels)} only appear in val, never trained on")
    logger.info(f"num_classes={num_classes}")

    pp = config.get('preprocess_params', {})
    patch_len = pp.get('patch_length', 100)
    collate_fn = FinetuneCollate(patch_len)

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=train_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True, collate_fn=collate_fn)

    logger.info(f"Loading pretrained backbone (Run: {model_name})...")
    backbone = build_model_from_config(config, src_output_dir=artifact_dir, mode='finetune')
    ckpt_path = train_params['pretrained_checkpoint']
    state = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(state['model_state_dict'])
    backbone.enable_spatial()
    logger.info(f"Loaded backbone weights from {ckpt_path}")

    ft_params = config['model_params']['MeFSQ'].get('finetune', {})
    freeze_backbone = ft_params.get('freeze_backbone', False)
    num_channels = train_dataset.base_dataset.Nc
    model = MeFSQFinetune(
        backbone, num_channels, num_classes,
        hidden=ft_params.get('hidden', 128),
        freeze_backbone=freeze_backbone,
    )
    model.to(device)

    scaler    = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = Plotter(output_dir=vis_dir)
    ft_loss_params = config.get('loss_params', {}).get('finetune', {})
    load_balance_weight = ft_loss_params.get('load_balance_weight', 0.0)
    diversity_weight     = ft_loss_params.get('diversity_weight', 0.0)

    best_val_acc = 0.0
    total_epochs = train_params['epochs']
    logger.info(f"Starting Finetuning ({total_epochs} epochs, freeze_backbone={freeze_backbone}, "
                f"lb_weight={load_balance_weight}, diversity_weight={diversity_weight})")

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch,
                                        load_balance_weight=load_balance_weight, diversity_weight=diversity_weight)
        val_metrics   = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        logger.info(f"--- Epoch {epoch}/{total_epochs} Summary ---")
        logger.info(f"  [Train] loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.4f} | f1: {train_metrics['f1']:.4f}")
        logger.info(f"  [Val]   loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.4f} | f1: {val_metrics['f1']:.4f}")
        logger.info("-" * 40)

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_finetune.pth'))
            logger.info("  > Saved Best Checkpoint")

        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot(mode='finetune', freeze_backbone=freeze_backbone)

        if epoch % 10 == 0:
            try:
                run_recon_analysis_finetune(
                    config, output_dir=vis_dir,
                    model=model,
                    dataset=train_dataset,
                    trial_idx=topo_trial_idx,
                    subject_id=topo_subject_id,
                    epoch=epoch,
                )
            except Exception as e:
                logger.warning(f"  Topomap viz failed (epoch {epoch}): {e}")

    logger.info("Finetuning Complete.")


if __name__ == '__main__':
    main()
