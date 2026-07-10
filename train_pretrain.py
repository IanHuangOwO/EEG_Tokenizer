import os
import sys
import json
import argparse
import shutil

import copy
import random
import logging
import matplotlib
matplotlib.use('Agg')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config
from viz.train import Plotter
from viz import pick_trial
from viz.check_epoch import run as run_recon_analysis

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


def _unpack_batch(batch, device):
    x_patches, coords, mask, time_indices, _, _ = batch
    x        = x_patches.to(device)      # [B, C, N, L]
    coords   = coords.to(device)         # [B, C, 3]
    time_idx = time_indices.to(device)   # [B, N]
    B, C, N, L = x.shape
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    return x, coords, time_idx, bool_masked_pos



def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, mask_weight=1.0, vq_warmup=False,
                     load_balance_weight=0.0, diversity_weight=0.0, current_k=None):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0, "lb_loss": 0.0}

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, bool_masked_pos = _unpack_batch(batch, device)
        B, C = x.shape[0], x.shape[1]
        optimizer.zero_grad()

        mp = None if vq_warmup else bool_masked_pos
        use_routing = current_k < model.vq_head_num
        with torch.amp.autocast(device_type='cuda'):
            recon, _, v_q, gate_mask, lb_loss = model(x, coords, time_idx, bool_masked_pos=mp, use_routing=use_routing, k_active_override=current_k)
            l_total, l_masked, l_unmasked = model.get_loss(x, recon, mp)
            if use_routing:
                model.update_head_metrics(gate_mask)
                if diversity_weight > 0:
                    l_total = l_total + diversity_weight * model.get_diversity_loss(v_q, gate_mask, B, C)
            if use_routing and load_balance_weight > 0:
                l_total = l_total + load_balance_weight * lb_loss

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"]     += l_total.item()
        totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
        totals["unmasked"] += l_unmasked.item()
        totals["lb_loss"]  += lb_loss.item() if hasattr(lb_loss, 'item') else float(lb_loss)

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({
                'L':   f"{totals['loss']     / n:.4f}",
                'msk': f"{totals['masked']   / n:.4f}",
                'vis': f"{totals['unmasked'] / n:.4f}",
                'lb':  f"{totals['lb_loss']  / n:.3f}",
            })

    n = batch_idx + 1
    epoch_metrics = {k: v / n for k, v in totals.items()}
    if hasattr(model, 'get_metrics'):
        epoch_metrics.update(model.get_metrics(v_q.detach()))

    return epoch_metrics


def validate_one_epoch(model, data_loader, device, mask_weight, vq_warmup=False, current_k=None):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, bool_masked_pos = _unpack_batch(batch, device)

            mp = None if vq_warmup else bool_masked_pos
            with torch.amp.autocast(device_type='cuda'):
                recon, _, v_q, _, _ = model(x, coords, time_idx, bool_masked_pos=mp, use_routing=current_k < model.vq_head_num, k_active_override=current_k)
                l_total, l_masked, l_unmasked = model.get_loss(x, recon, mp, mask_weight=mask_weight)

            totals["loss"]     += l_total.item()
            totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
            totals["unmasked"] += l_unmasked.item()

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'L':   f"{totals['loss']     / n:.4f}",
                    'msk': f"{totals['masked']   / n:.4f}",
                    'vis': f"{totals['unmasked'] / n:.4f}",
                })

    n = batch_idx + 1
    val_metrics = {k: v / n for k, v in totals.items()}
    if hasattr(model, 'get_metrics'):
        val_metrics.update(model.get_metrics(v_q.detach()))

    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='MeFSQ Masked Pretraining')
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_params = config['training_params']['pretrain']
    device     = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_params.get('model_name', 'default_run')

    base_output_dir = f"output/{model_name}"
    checkpoint_dir  = os.path.join(base_output_dir, "pretrain")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")
    vis_dir         = os.path.join(base_output_dir, "visualization")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))

    dataset_params = config['dataset_params']['pretrain']
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

        train_config['dataset_params']['pretrain'][ds_name]['subject_to_use'] = subjects_to_split[:n_train]
        val_config['dataset_params']['pretrain'][ds_name]['subject_to_use']   = subjects_to_split[n_train:]
        logger.info(f"Dataset {ds_name}: {n_train} Train, {len(subjects_to_split) - n_train} Val subjects")

    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='pretrain')
    logger.info("Building Validation Dataset...")
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='pretrain')
    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    _first_subject = val_dataset.base_dataset.subject_data[0].item()
    topo_trial_idx, topo_subject_id = pick_trial(val_dataset, _first_subject, trial=0)
    logger.info(f"Topo viz: subject={topo_subject_id}, trial_idx={topo_trial_idx}")

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=train_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_model_from_config(config, src_output_dir=artifact_dir)
    model.to(device)

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, bool_masked_pos = _unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=bool_masked_pos)

    scaler    = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = Plotter(output_dir=vis_dir)

    pp          = config.get('preprocess_params', {})
    strat_name  = pp.get('masking_strategy', 'random')
    mask_ratio  = 0.5 if strat_name == 'complementary' else pp.get(strat_name, {}).get('mask_ratio', 0.5)
    loss_params          = config.get('loss_params', {}).get('pretrain', {})
    mask_weight          = loss_params.get('mask_weight', (1.0 - mask_ratio) / mask_ratio)
    load_balance_weight  = loss_params.get('load_balance_weight', 0.0)
    diversity_weight     = loss_params.get('diversity_weight', 0.0)
    logger.info(f"mask_ratio={mask_ratio}  mask_weight={mask_weight:.4f}  lb_weight={load_balance_weight}  diversity_weight={diversity_weight}")

    best_val_loss    = float('inf')
    total_epochs     = train_params['epochs']
    vq_warmup_epochs = train_params.get('vq_warmup_epochs', 0)
    H                = model.vq_head_num
    k_target         = model.k_active
    logger.info(f"Starting Masked Pretraining ({total_epochs} epochs, VQ warmup={vq_warmup_epochs})")
    logger.info(f"Routing: k={k_target}/{H} from epoch 1")

    for epoch in range(1, total_epochs + 1):
        vq_warmup  = epoch <= vq_warmup_epochs
        current_k  = k_target
        if vq_warmup:
            logger.info(f"  [VQ Warmup] epoch {epoch}/{vq_warmup_epochs} — masking disabled, spatial locked | k_active={current_k}/{H}")
        elif epoch == vq_warmup_epochs + 1:
            model.enable_spatial()
            logger.info(f"  [Spatial Enabled] epoch {epoch} — coord_proj + spatial_attn unlocked | k_active={current_k}/{H}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, mask_weight=mask_weight, vq_warmup=vq_warmup, load_balance_weight=load_balance_weight, diversity_weight=diversity_weight, current_k=current_k)
        val_metrics   = validate_one_epoch(model, val_loader, device, mask_weight=mask_weight, vq_warmup=vq_warmup, current_k=current_k)
        scheduler.step()

        loss_keys = {'loss', 'masked', 'unmasked'}
        other_metrics = {k: v for k, v in train_metrics.items() if k not in loss_keys}

        logging.info(f"--- Epoch {epoch}/{total_epochs} Summary ---")
        logging.info(f"  [Train] " + " | ".join([f"{k}: {train_metrics.get(k, 0.0):.4f}" for k in ['loss', 'masked', 'unmasked']]))
        logging.info(f"  [Val]   " + " | ".join([f"{k}: {val_metrics.get(k, 0.0):.4f}"   for k in ['loss', 'masked', 'unmasked']]))

        if other_metrics:
            o_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in other_metrics.items()])
            logging.info(f"  [Other] {o_str}")

        logging.info("-" * 40)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_pretrain.pth'))
            logger.info("  > Saved Best Checkpoint")

        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot()

        if epoch % 10 == 0:
            try:
                run_recon_analysis(
                    config, output_dir=vis_dir,
                    args=argparse.Namespace(), model=model,
                    dataset=val_dataset,
                    trial_idx=topo_trial_idx,
                    subject_id=topo_subject_id,
                    epoch=epoch,
                )
            except Exception as e:
                logger.warning(f"  Topomap viz failed (epoch {epoch}): {e}")

    logger.info("Pretraining Complete.")


if __name__ == '__main__':
    main()
