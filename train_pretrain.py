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
from model.factory import build_pretrain_from_config
from viz.train import Plotter
from viz import pick_trial
from viz.check_epoch_pretrain import run as run_recon_analysis

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
    x_patches, coords, mask, time_indices, _, _, valid_channels = batch
    x              = x_patches.to(device)      # [B, C, N, L]
    coords         = coords.to(device)         # [B, C, 3]
    time_idx       = time_indices.to(device)   # [B, N]
    valid_channels = valid_channels.to(device) # [B, C]
    B, C, N, L = x.shape
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    return x, coords, time_idx, bool_masked_pos, valid_channels



def _step_loss(model, model_type, x, out, mp, mask_weight, load_balance_weight, aux_weight):
    """Model-specific forward-loss glue — the two models' get_loss signatures and
    per-step extra losses genuinely differ (router load-balance vs SAE dead-feature
    aux), so this branches on model_type rather than forcing a fake-shared interface."""
    if model_type == 'MeFSQ':
        l_total, l_masked, l_unmasked = model.get_loss(x, out.recon, mp, mask_weight=mask_weight)
        if load_balance_weight > 0:
            l_total = l_total + load_balance_weight * out.lb_loss
        model.update_head_metrics(out.gate_mask_routed)
    else:  # MeSAE
        l_total, l_masked, l_unmasked = model.get_loss(
            x, out.recon, out.aux_loss, bool_masked_pos=mp, mask_weight=mask_weight, aux_weight=aux_weight)
    return l_total, l_masked, l_unmasked


def _epoch_metrics(model, model_type, out):
    if model_type == 'MeFSQ':
        return model.get_metrics(out.v_q_routed.detach(), out.v_q_shared.detach())
    return model.get_metrics(out.sae_hidden.detach())


def train_one_epoch(model, data_loader, optimizer, scaler, device, epoch, model_type, mask_weight=1.0, warmup=False,
                     load_balance_weight=0.0, aux_weight=0.03):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)
        optimizer.zero_grad()

        mp = None if warmup else bool_masked_pos
        with torch.amp.autocast(device_type='cuda'):
            out = model(x, coords, time_idx, bool_masked_pos=mp, valid_channels=valid_channels)
            l_total, l_masked, l_unmasked = _step_loss(model, model_type, x, out, mp, mask_weight, load_balance_weight, aux_weight)

        scaler.scale(l_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["loss"]     += l_total.item()
        totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
        totals["unmasked"] += l_unmasked.item()
        if hasattr(out, 'lb_loss'):
            totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({
                'L':   f"{totals['loss']     / n:.4f}",
                'msk': f"{totals['masked']   / n:.4f}",
                'vis': f"{totals['unmasked'] / n:.4f}",
            })

    n = batch_idx + 1
    epoch_metrics = {k: v / n for k, v in totals.items()}
    epoch_metrics.update(_epoch_metrics(model, model_type, out))

    return epoch_metrics


def validate_one_epoch(model, data_loader, device, model_type, mask_weight=1.0, warmup=False, load_balance_weight=0.0, aux_weight=0.03):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)

            mp = None if warmup else bool_masked_pos
            with torch.amp.autocast(device_type='cuda'):
                out = model(x, coords, time_idx, bool_masked_pos=mp, valid_channels=valid_channels)
                l_total, l_masked, l_unmasked = _step_loss(model, model_type, x, out, mp, mask_weight, load_balance_weight, aux_weight)

            totals["loss"]     += l_total.item()
            totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
            totals["unmasked"] += l_unmasked.item()
            if hasattr(out, 'lb_loss'):
                totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'L':   f"{totals['loss']     / n:.4f}",
                    'msk': f"{totals['masked']   / n:.4f}",
                    'vis': f"{totals['unmasked'] / n:.4f}",
                })

    n = batch_idx + 1
    val_metrics = {k: v / n for k, v in totals.items()}
    val_metrics.update(_epoch_metrics(model, model_type, out))

    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='Masked Pretraining (MeFSQ or MeSAE, by training_params.pretrain.model_type)')
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

        # metadata.json subject keys are always strings (JSON object keys); keep them as
        # strings throughout instead of int-converting — a requested subject_to_use given
        # as either "1" or 1 in config then normalizes to the same "1" for comparison,
        # so either representation works instead of silently matching nothing.
        all_available_subjects = sorted(meta.get('data_structure', {}).keys())

        requested_subjects = ds_args['subject_to_use']
        if requested_subjects in (["all"], "all"):
            subjects_to_split = all_available_subjects
        else:
            requested_strs = {str(s) for s in requested_subjects}
            subjects_to_split = [s for s in all_available_subjects if s in requested_strs]

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

    viz_params  = config.get('training_params', {}).get('visualize', {})
    viz_target_cfg = viz_params.get('targets') or [{'subject': val_dataset.base_dataset.subject_data[0].item(), 'trial': 0}]
    viz_every_n = viz_params.get('every_n_epochs', 2)
    viz_targets = [pick_trial(val_dataset, t['subject'], trial=t['trial'], dataset_name=t.get('dataset')) for t in viz_target_cfg]
    logger.info(f"Recon viz targets (subject, trial_idx): {viz_targets} every {viz_every_n} epochs")

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=train_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_pretrain_from_config(config, src_output_dir=artifact_dir)
    model.to(device)

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)

    scaler    = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = Plotter(output_dir=vis_dir)

    pp          = config.get('preprocess_params', {}).get('mask', {})
    strat_name  = pp.get('masking_strategy', 'random')
    mask_ratio  = 0.5 if strat_name == 'complementary' else pp.get(strat_name, {}).get('mask_ratio', 0.5)
    model_type           = train_params.get('model_type', 'MeFSQ')
    loss_params          = config.get('model_params', {}).get(model_type, {}).get('pretrain', {}).get('loss', {})
    mask_weight          = loss_params.get('mask_weight', (1.0 - mask_ratio) / mask_ratio)
    load_balance_weight  = loss_params.get('load_balance_weight', 0.0)
    aux_weight           = loss_params.get('aux_weight', 0.03)
    logger.info(f"model_type={model_type}  mask_ratio={mask_ratio}  mask_weight={mask_weight:.4f}  lb_weight={load_balance_weight}  aux_weight={aux_weight}")

    best_val_loss  = float('inf')
    total_epochs   = train_params['epochs']
    # Stage-1 length: MeFSQ's VQ-warmup (encoder+VQ trained jointly, unmasked) and
    # MeSAE's Tokenizer stage (encoder+SAE trained jointly, unmasked, local features
    # only — see docs/adr/0003-mesae-two-stage-masked-training.md) share the same shape
    # (N unmasked epochs, then a one-time transition into masked training), so they share
    # this config field even though "VQ warmup" isn't literally accurate for MeSAE.
    stage1_epochs = train_params.get('vq_warmup_epochs', 0)
    logger.info(f"Starting {model_type} Pretraining ({total_epochs} epochs, stage1={stage1_epochs})")

    for epoch in range(1, total_epochs + 1):
        stage1 = epoch <= stage1_epochs
        if stage1:
            logger.info(f"  [Stage 1] epoch {epoch}/{stage1_epochs} — masking disabled, temporal/spatial locked")
        elif epoch == stage1_epochs + 1:
            if model_type == 'MeFSQ':
                model.enable_spatial()
                model.freeze_vq_and_decoder()
                logger.info(f"  [Stage 2] epoch {epoch} — spatial enabled, VQ+decoder frozen, only main transformer trains from here")
            else:  # MeSAE
                model.enable_temporal()
                model.enable_spatial()
                model.freeze_sae()
                logger.info(f"  [Stage 2] epoch {epoch} — temporal+spatial enabled, SAE+decoder frozen, only main transformer trains from here")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, model_type,
                                        mask_weight=mask_weight, warmup=stage1,
                                        load_balance_weight=load_balance_weight, aux_weight=aux_weight)
        val_metrics   = validate_one_epoch(model, val_loader, device, model_type,
                                           mask_weight=mask_weight, warmup=stage1,
                                           load_balance_weight=load_balance_weight, aux_weight=aux_weight)
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

        if viz_every_n > 0 and epoch % viz_every_n == 0:
            for topo_trial_idx, topo_subject_id in viz_targets:
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
                    logger.warning(f"  Topomap viz failed (epoch {epoch}, subject={topo_subject_id}, trial_idx={topo_trial_idx}): {e}")

    logger.info("Pretraining Complete.")


if __name__ == '__main__':
    main()
