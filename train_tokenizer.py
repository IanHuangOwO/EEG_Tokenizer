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
from model.factory import build_pretrain_from_config, optimizer_param_groups, MODEL_REGISTRY
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


def _unpack_batch(batch, device):
    x_patches, coords, mask, time_indices, _, _, valid_channels = batch
    x              = x_patches.to(device)      # [B, C, N, L]
    coords         = coords.to(device)         # [B, C, 3]
    time_idx       = time_indices.to(device)   # [B, N]
    valid_channels = valid_channels.to(device) # [B, C]
    B, C, N, L = x.shape
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    return x, coords, time_idx, bool_masked_pos, valid_channels


def train_one_epoch(model, trainer, data_loader, optimizer, scaler, device, epoch,
                     masked_mse_weight=1.0, unmasked_mse_weight=1.0, **loss_hparams):
    """Tokenizer stage: always unmasked (bool_masked_pos=None) — encoder + VQ/SAE
    train jointly on patch-local features, no masking, no masked/unmasked split.
    (masked_mse_weight only ever multiplies get_loss's l_masked=1.0 placeholder here —
    a constant offset on the logged total, no gradient effect — kept for parity with
    train_pretrain.py's logged loss scale.)"""
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, _, valid_channels = _unpack_batch(batch, device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            out = model(x, coords, time_idx, bool_masked_pos=None, valid_channels=valid_channels)
            l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, None, masked_mse_weight,
                                                                  unmasked_mse_weight, True, **loss_hparams)
        trainer.update_diagnostics(model, out)

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
        for i, v in enumerate(getattr(model, '_last_pyramid_levels', None) or []):
            key = f'mse_level_{i}'
            totals[key] = totals.get(key, 0.0) + v

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({
                'L':   f"{totals['loss']     / n:.4f}",
                'msk': f"{totals['masked']   / n:.4f}",
                'vis': f"{totals['unmasked'] / n:.4f}",
            })

    n = batch_idx + 1
    epoch_metrics = {k: v / n for k, v in totals.items()}
    epoch_metrics.update(trainer.epoch_metrics(model, out))

    return epoch_metrics


def validate_one_epoch(model, trainer, data_loader, device,
                        masked_mse_weight=1.0, unmasked_mse_weight=1.0, **loss_hparams):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, _, valid_channels = _unpack_batch(batch, device)

            with torch.amp.autocast(device_type='cuda'):
                out = model(x, coords, time_idx, bool_masked_pos=None, valid_channels=valid_channels)
                l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, None, masked_mse_weight,
                                                                      unmasked_mse_weight, True, **loss_hparams)
            trainer.update_diagnostics(model, out)

            totals["loss"]     += l_total.item()
            totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
            totals["unmasked"] += l_unmasked.item()
            if hasattr(out, 'lb_loss'):
                totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))
            for i, v in enumerate(getattr(model, '_last_pyramid_levels', None) or []):
                key = f'mse_level_{i}'
                totals[key] = totals.get(key, 0.0) + v

            if batch_idx % 5 == 0:
                n = batch_idx + 1
                pbar.set_postfix({
                    'L':   f"{totals['loss']     / n:.4f}",
                    'msk': f"{totals['masked']   / n:.4f}",
                    'vis': f"{totals['unmasked'] / n:.4f}",
                })

    n = batch_idx + 1
    val_metrics = {k: v / n for k, v in totals.items()}
    val_metrics.update(trainer.epoch_metrics(model, out))

    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='Tokenizer Stage (MeFSQ or MeSAE) — unmasked joint '
                                                   'encoder+VQ/SAE training, spatial/temporal mixing enabled '
                                                   'from the start, by training_params.tokenizer.model_type')
    parser.add_argument('--config', type=str, default='config/config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_params = config['training_params']['tokenizer']
    device     = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_params.get('model_name', 'default_tokenizer_run')

    base_output_dir = f"output/{model_name}"
    checkpoint_dir  = os.path.join(base_output_dir, "tokenizer")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")
    vis_dir         = os.path.join(base_output_dir, "visualization")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger, timestamp = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))
    shutil.copy(args.config, os.path.join(artifact_dir, f'config_{timestamp}.json'))

    dataset_params = config['dataset_params']['pretrain']
    split_ratio = train_params.get('train_val_split', 0.9)
    random.seed(42)

    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    for ds_name, ds_args in dataset_params.items():
        data_root = ds_args['dataset_path']
        with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
            meta = json.load(f)

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
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='tokenizer')
    logger.info("Building Validation Dataset...")
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='tokenizer')
    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    def _first_subject(dataset_name=None):
        sub_data = val_dataset.base_dataset.subject_data
        if dataset_name is not None:
            names = val_dataset.base_dataset.dataset_names
            idx = next((i for i, n in enumerate(names) if n == dataset_name), None)
            if idx is not None:
                return sub_data[idx].item()
        return sub_data[0].item()

    viz_params  = config.get('training_params', {}).get('visualize', {}).get('pretrain', {})
    viz_target_cfg = viz_params.get('targets') or [{'subject': None, 'trial': 0}]
    viz_every_n = viz_params.get('every_n_epochs', 2)
    viz_targets = [
        pick_trial(val_dataset, t.get('subject') if t.get('subject') is not None else _first_subject(t.get('dataset')),
                   trial=t.get('trial'), dataset_name=t.get('dataset'))
        for t in viz_target_cfg
    ]
    logger.info(f"Recon viz targets (subject, trial_idx): {viz_targets} every {viz_every_n} epochs")

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=train_params['batch_size'], shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    model_type = train_params.get('model_type', 'MeFSQ')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()
    checker    = entry.checker_cls()

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_pretrain_from_config(config, mode='tokenizer')
    model.to(device)

    # Tokenizer stage: spatial (+temporal, if the model has it) mixing enabled from the
    # start — the whole point of splitting this out of train_pretrain.py is to let the
    # encoder's spatial/temporal attention train jointly with the VQ/SAE, unmasked,
    # instead of only turning on once masked pretraining begins.
    trainer.on_tokenizer_start(model, logger=logger)

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, _, valid_channels = _unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=None, valid_channels=valid_channels)

    scaler    = torch.amp.GradScaler('cuda')
    optimizer = optim.AdamW(optimizer_param_groups(model, train_params['weight_decay']), lr=train_params['learning_rate'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = entry.plotter_cls(output_dir=vis_dir)

    pp          = config.get('preprocess_params', {}).get('mask', {})
    strat_name  = pp.get('masking_strategy', 'random')
    mask_ratio  = 0.5 if strat_name == 'complementary' else pp.get(strat_name, {}).get('mask_ratio', 0.5)
    model_params_mt = config.get('model_params', {}).get(model_type, {})
    loss_params = model_params_mt.get('tokenizer', {}).get('loss') or model_params_mt.get('pretrain', {}).get('loss', {})
    masked_mse_weight   = loss_params.get('masked_mse_weight', (1.0 - mask_ratio) / mask_ratio)
    unmasked_mse_weight = loss_params.get('unmasked_mse_weight', 1.0)
    loss_hparams = {k: v for k, v in loss_params.items() if k not in ('masked_mse_weight', 'unmasked_mse_weight')}
    logger.info(f"model_type={model_type}  loss_hparams={loss_hparams}")

    best_val_loss  = float('inf')
    total_epochs   = train_params['epochs']
    logger.info(f"Starting {model_type} Tokenizer training ({total_epochs} epochs, unmasked)")

    for epoch in range(1, total_epochs + 1):
        train_metrics = train_one_epoch(model, trainer, train_loader, optimizer, scaler, device, epoch,
                                        masked_mse_weight=masked_mse_weight, unmasked_mse_weight=unmasked_mse_weight,
                                        **loss_hparams)
        val_metrics   = validate_one_epoch(model, trainer, val_loader, device,
                                           masked_mse_weight=masked_mse_weight, unmasked_mse_weight=unmasked_mse_weight,
                                           **loss_hparams)
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
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_tokenizer.pth'))
            logger.info("  > Saved Best Checkpoint")

        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot_pretrain()

        if viz_every_n > 0 and epoch % viz_every_n == 0:
            for topo_trial_idx, topo_subject_id in viz_targets:
                try:
                    checker.check_pretrain(
                        config, vis_dir, model, val_dataset,
                        topo_trial_idx, subject_id=topo_subject_id, epoch=epoch,
                    )
                except Exception as e:
                    logger.warning(f"  Topomap viz failed (epoch {epoch}, subject={topo_subject_id}, trial_idx={topo_trial_idx}): {e}")

    logger.info("Tokenizer Training Complete.")


if __name__ == '__main__':
    main()
