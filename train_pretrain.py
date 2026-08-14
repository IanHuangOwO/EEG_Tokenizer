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
from IO.masking import RandomMaskingStrategy, ComplementaryMaskingStrategy
from model.factory import build_pretrain_from_config, optimizer_param_groups, MODEL_REGISTRY
from viz import pick_trial

torch.set_float32_matmul_precision('high')


def _masking_for_epoch(epoch, strat_name, target_ratio, curriculum):
    """Returns (masking_strategy, ratio) for this epoch. Only ramps when the configured
    strategy is 'complementary' (which structurally ignores any ratio argument, always
    pairing mask+~mask at a fixed 0.5 — see IO/masking.py) and
    preprocess_params.mask.mask_curriculum.enabled is true; otherwise passes the
    configured strategy/ratio through unchanged from epoch 1, same as before this existed.

    Ramps a RandomMaskingStrategy from start_ratio up to target_ratio in coarse steps (one
    new ratio every step_every epochs) over ramp_epochs total, then switches to the real
    configured strategy once the ramp completes. Exists because masking is the one
    un-softened distribution shock at the Tokenizer->Masked stage boundary: spatial/
    temporal mixing already ramps in gradually via zero-init LayerScale
    (TSABlock.enable_temporal/enable_spatial), but bool_masked_pos substitutes mask_token
    wholesale with no ramp, and MeSAE's StampBank generator is now content-conditioned
    (docs/adr/0009's later revision) rather than just amplitude-conditioned — a sudden,
    never-seen-in-Tokenizer-stage input distribution is a sharper shock to a nonlinear
    generator than it was to the old linear decode chain.
    """
    if strat_name != 'complementary' or not curriculum.get('enabled', False):
        strategy = ComplementaryMaskingStrategy() if strat_name == 'complementary' else RandomMaskingStrategy()
        return strategy, target_ratio

    ramp_epochs = curriculum.get('ramp_epochs', 25)
    step_every  = max(1, curriculum.get('step_every', 5))
    start_ratio = curriculum.get('start_ratio', 0.1)

    if epoch > ramp_epochs:
        return ComplementaryMaskingStrategy(), ComplementaryMaskingStrategy.MASK_RATIO

    n_steps = max(1, ramp_epochs // step_every)
    step = min((epoch - 1) // step_every, n_steps - 1)
    ratio = start_ratio if n_steps <= 1 else start_ratio + (target_ratio - start_ratio) * step / (n_steps - 1)
    return RandomMaskingStrategy(), ratio


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
    # (C, N) order is implicit — mask itself carries no axis labels. Must match x_patches'
    # (C, N, L) layout in IO/dataset.py; any consumer reshaping (N, C) instead silently
    # misaligns masked/unmasked patches with no shape error.
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    return x, coords, time_idx, bool_masked_pos, valid_channels



def train_one_epoch(model, trainer, data_loader, optimizer, scaler, device, epoch,
                     masked_mse_weight=1.0, unmasked_mse_weight=1.0,
                     **loss_hparams):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
            l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, masked_mse_weight,
                                                                  unmasked_mse_weight, False, **loss_hparams)
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
            x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)

            with torch.amp.autocast(device_type='cuda'):
                out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
                l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, masked_mse_weight,
                                                                      unmasked_mse_weight, False, **loss_hparams)
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
    parser = argparse.ArgumentParser(description='Masked Pretraining (MeFSQ or MeSAE, by training_params.pretrain.model_type) — '
                                                   'loads a Tokenizer-stage checkpoint (train_tokenizer.py), freezes its VQ/SAE, '
                                                   'and trains the encoder against masked reconstruction')
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

    def _first_subject(dataset_name=None):
        """subject: null in a target config -> first val subject (of that dataset, if
        dataset_name is given — subject ids aren't unique across datasets)."""
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

    def _make_loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=shuffle,
                           num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader   = _make_loader(val_dataset,   shuffle=False)

    model_type = train_params.get('model_type', 'MeFSQ')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()
    checker    = entry.checker_cls()

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_pretrain_from_config(config, mode='pretrain')

    tokenizer_ckpt = train_params['tokenizer_checkpoint']
    state = torch.load(tokenizer_ckpt, map_location='cpu')
    # The Tokenizer stage can now run its own (e.g. shallower) encoder architecture --
    # see model_params.<type>.tokenizer in config.json -- so its checkpoint's encoder/embed
    # weights won't shape-match this Pretrain-stage model at all. Only the frozen VQ/SAE
    # apparatus (filter_pool/sae/decoder/router for MeSAE, the Expert pools/router/fusion
    # params for MeFSQ) is meant to carry over; everything else (embed/encoder/mask_token)
    # trains fresh here anyway once on_pretrain_start freezes the former. Filtering to
    # shape-matching keys only, rather than hardcoding submodule prefixes per model type,
    # keeps this loader working for both without a model-type branch.
    # encoder.*/mask_token are excluded outright, not just left to the shape filter: their
    # per-tensor shapes (embed_dim, spatial_heads) are identical between the tokenizer and
    # pretrain architecture blocks even when depth/pool_after_blocks differ, so shape
    # matching alone would silently transplant e.g. Tokenizer's block-1 weights (trained
    # already-pooled, since tokenizer pools after every block) into Pretrain's block-1
    # (trained at full resolution, since pretrain only pools after block 2) -- same shape,
    # wrong resolution context, worse than random init.
    own_state = model.state_dict()
    ckpt_state = state['model_state_dict']
    to_load = {k: v for k, v in ckpt_state.items()
               if k in own_state and own_state[k].shape == v.shape
               and not k.startswith('encoder.') and k != 'mask_token'}
    skipped = [k for k in ckpt_state if k not in to_load]
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    logger.info(f"Loaded Tokenizer-stage checkpoint from {tokenizer_ckpt} "
                f"({len(to_load)} tensors loaded, {len(skipped)} skipped on shape/name mismatch: {skipped})")
    model.to(device)

    # enable_spatial/enable_temporal are plain flags, not persisted in the state dict —
    # re-enable them (same pattern as model/factory.py's build_finetune_from_config), then
    # freeze the VQ/SAE apparatus the Tokenizer stage already trained.
    trainer.on_tokenizer_start(model, logger=logger)
    trainer.on_pretrain_start(model, logger=logger)

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)

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
    mask_curriculum = pp.get('mask_curriculum', {})
    loss_params = config.get('model_params', {}).get(model_type, {}).get('pretrain', {}).get('loss', {})
    masked_mse_weight   = loss_params.get('masked_mse_weight', (1.0 - mask_ratio) / mask_ratio)
    unmasked_mse_weight = loss_params.get('unmasked_mse_weight', 1.0)
    loss_hparams = {k: v for k, v in loss_params.items() if k not in ('masked_mse_weight', 'unmasked_mse_weight')}
    logger.info(f"model_type={model_type}  mask_ratio={mask_ratio}  masked_mse_weight={masked_mse_weight:.4f}  "
                f"unmasked_mse_weight={unmasked_mse_weight:.4f}  loss_hparams={loss_hparams}")
    if mask_curriculum.get('enabled', False) and strat_name == 'complementary':
        logger.info(f"  [mask curriculum] ramping random-mask ratio {mask_curriculum.get('start_ratio', 0.1)} -> "
                    f"{mask_ratio} over {mask_curriculum.get('ramp_epochs', 25)} epochs "
                    f"(step every {mask_curriculum.get('step_every', 5)}), then switching to complementary")

    best_val_loss  = float('inf')
    total_epochs   = train_params['epochs']
    logger.info(f"Starting {model_type} Pretraining ({total_epochs} epochs, masked)")

    current_mask_state = None  # (strategy class name, ratio) — tracks the last (strategy, ratio) applied
    for epoch in range(1, total_epochs + 1):
        mask_strategy, epoch_ratio = _masking_for_epoch(epoch, strat_name, mask_ratio, mask_curriculum)
        new_mask_state = (type(mask_strategy).__name__, epoch_ratio)
        if new_mask_state != current_mask_state:
            train_dataset.set_masking(mask_strategy, epoch_ratio)
            val_dataset.set_masking(mask_strategy, epoch_ratio)
            # rebuild, don't just re-iterate: persistent_workers=True means worker
            # subprocesses hold their own copy of the dataset from spawn time and never
            # see the mutation above otherwise (see MaskedPretrainDataset.set_masking).
            train_loader = _make_loader(train_dataset, shuffle=True)
            val_loader   = _make_loader(val_dataset,   shuffle=False)
            logger.info(f"  [mask curriculum] epoch {epoch}: strategy={new_mask_state[0]} ratio={epoch_ratio:.3f}")
            current_mask_state = new_mask_state

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
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_pretrain.pth'))
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

    logger.info("Pretraining Complete.")


if __name__ == '__main__':
    main()
