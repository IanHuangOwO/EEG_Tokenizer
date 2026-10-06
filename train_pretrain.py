import os
import sys
import json
import argparse
import shutil

import copy
import logging
import matplotlib
matplotlib.use('Agg')
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import MontageBatchSampler, build_dataset_from_config, split_pretrain_subjects
from IO.masking import build_masking_strategy_from_config
from model.base_trainer import nonfinite_step_report
from model.factory import build_pretrain_from_config, checkpoint_build_config, optimizer_param_groups, MODEL_REGISTRY
from tools.analysis import apply_overrides, pick_trial, resolve_output_path
from tools.analysis.snapshot import build_pretrain_bundle
from tools.viz.snapshot import render_recon, render_stamp_gallery

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
    x_patches, coords, mask, time_indices, _, valid_channels = batch
    x              = x_patches.to(device)      # [B, C, N, L]
    coords         = coords.to(device)         # [B, C, 3]
    time_idx       = time_indices.to(device)   # [B, N]
    valid_channels = valid_channels.to(device) # [B, C]
    B, C, N, L = x.shape
    # (C, N) order is implicit — mask itself carries no axis labels. Must match x_patches'
    # (C, N, L) layout in IO/dataset.py; any consumer reshaping (N, C) instead silently
    # misaligns masked/unmasked patches with no shape error.
    bool_masked_pos = mask.view(B, C, N).to(device)  # [B, C*N] -> [B, C, N]
    # Batches hold one montage (MontageBatchSampler): drop the canonical channels no window
    # in the batch has -- the model is channel-count agnostic, and they are pure padding.
    keep = valid_channels.any(0)
    if not keep.all():
        x, coords, bool_masked_pos, valid_channels = x[:, keep], coords[:, keep], bool_masked_pos[:, keep], valid_channels[:, keep]
    return x, coords, time_idx, bool_masked_pos, valid_channels



def train_one_epoch(model, trainer, data_loader, optimizer, scaler, device, epoch, masked,
                     **loss_hparams):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}
    skipped = 0

    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)
        # Tokenizer phase: the dataset still generates masks, they're just not used. Must be
        # None, not all-False — get_loss's None branch is what keeps mp_loss on.
        if not masked:
            bool_masked_pos = None
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
            l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, **loss_hparams)
        # A non-finite loss must never reach backward (see nonfinite_step_report):
        # backwarding a non-finite loss can leave NaN parameters, which nothing recovers
        # from. update_diagnostics is skipped too — it folds `out` into EMA buffers.
        report = nonfinite_step_report(l_total, model, out)
        if report is not None:
            skipped += 1
            if skipped <= 3:
                logging.warning(f"epoch {epoch} batch {batch_idx} skipped: {report}")
            optimizer.zero_grad(set_to_none=True)
            continue

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
        for name, v in (getattr(model, '_last_loss_terms', None) or {}).items():
            key = f'mse_{name}'
            totals[key] = totals.get(key, 0.0) + v

        if batch_idx % 5 == 0:
            n = batch_idx + 1
            pbar.set_postfix({
                'L':   f"{totals['loss']     / n:.4f}",
                'msk': f"{totals['masked']   / n:.4f}",
                'vis': f"{totals['unmasked'] / n:.4f}",
            })

    n = max(batch_idx + 1 - skipped, 1)
    epoch_metrics = {k: v / n for k, v in totals.items()}
    epoch_metrics.update(trainer.epoch_metrics(model, out))
    if skipped:
        logging.warning(f"epoch {epoch}: skipped {skipped}/{batch_idx + 1} non-finite batches")
        epoch_metrics['skipped_batches'] = skipped

    return epoch_metrics


def validate_one_epoch(model, trainer, data_loader, device, masked, **loss_hparams):
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation",
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    totals = {"loss": 0.0, "masked": 0.0, "unmasked": 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(batch, device)
            if not masked:
                bool_masked_pos = None

            with torch.amp.autocast(device_type='cuda'):
                out = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos, valid_channels=valid_channels)
                l_total, l_masked, l_unmasked = trainer.compute_loss(model, x, out, bool_masked_pos, **loss_hparams)
            trainer.update_diagnostics(model, out)

            totals["loss"]     += l_total.item()
            totals["masked"]   += float(l_masked) if not hasattr(l_masked, 'item') else l_masked.item()
            totals["unmasked"] += l_unmasked.item()
            if hasattr(out, 'lb_loss'):
                totals["lb_loss"] = totals.get("lb_loss", 0.0) + (out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss))
            for name, v in (getattr(model, '_last_loss_terms', None) or {}).items():
                key = f'mse_{name}'
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
    parser = argparse.ArgumentParser(description='MeSAE pretraining, one run, two phases: unmasked tokenizer phase '
                                                   '(training_params.pretrain.tokenizer_epochs), then masked phase.')
    parser.add_argument('--config', type=str, default='configs/pretrain.template.json')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config value, dotted path, JSON value (repeatable), '
                             'e.g. --set training_params.pretrain.seed=2')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = apply_overrides(json.load(f), args.set)

    train_params = config['training_params']['pretrain']
    if train_params.get('num_threads'):   # CPU threads for this process (parallel runs share the cores)
        torch.set_num_threads(int(train_params['num_threads']))
    device     = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_name = train_params.get('model_name', 'default_run')
    train_params.setdefault('model_name', model_name)

    base_output_dir = f"output/{resolve_output_path(config, mode='pretrain')}"
    checkpoint_dir  = os.path.join(base_output_dir, "checkpoint")
    artifact_dir    = os.path.join(base_output_dir, "artifacts")
    vis_dir         = os.path.join(base_output_dir, "visualization")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    logger, timestamp = setup_logger(artifact_dir)
    for name in ('config.json', f'config_{timestamp}.json'):     # the effective config, overrides applied
        with open(os.path.join(artifact_dir, name), 'w') as f:
            json.dump(config, f, indent=2)

    dataset_params = config['dataset_params']['pretrain']
    split_ratio = train_params.get('train_val_split', 0.9)
    # training_params.pretrain.seed (optional): weight init, masking and batch order all draw
    # from torch's RNG. The subject train/val split has its own fixed seed, so runs that
    # differ only in this seed share the same split. Unset = unseeded, as before.
    if 'seed' in train_params:
        torch.manual_seed(int(train_params['seed']))

    train_config = copy.deepcopy(config)
    val_config   = copy.deepcopy(config)

    # person-disjoint, order-independent subject split (IO/dataset.py's split_pretrain_subjects)
    for ds_name, (train_subs, val_subs) in split_pretrain_subjects(dataset_params, split_ratio).items():
        train_config['dataset_params']['pretrain'][ds_name]['subject_to_use'] = train_subs
        val_config['dataset_params']['pretrain'][ds_name]['subject_to_use'] = val_subs
        logger.info(f"Dataset {ds_name}: {len(train_subs)} Train, {len(val_subs)} Val subjects")

    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='pretrain')
    logger.info("Building Validation Dataset...")
    val_dataset   = build_dataset_from_config(val_config,   transform=None, mode='pretrain')
    logger.info(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Separate assemble_trials=False dataset(s) just for periodic snapshot viz --
    # val_dataset itself stays assembled (assemble_trials=True, continuous windows) for
    # real train/val loss. A snapshot built from an assembled window mixes multiple real
    # trials with no single event to mark, so _lookup_event_onset
    # (tools/analysis/snapshot.py) silently drops recon_signal's onset line whenever it's
    # handed one. Same assemble_trials=False dataset analysis_pretrain.py's own snapshot
    # path already uses.
    #
    # ONE ISOLATED single-dataset build per distinct target dataset name -- NOT one
    # combined multi-dataset build (even restricted to just the wanted datasets isn't
    # enough, see below). EEGDataset standardizes every trial's length to the single
    # longest trial across EVERY dataset it loads in that one build (its "Standardize
    # temporal length" step); combining a short-trial dataset (PhysionetMI, 800-sample
    # trials -- its loader deliberately hardcodes pre_event_seconds=0, see
    # datas/finetune/PhysionetMI/loader.py) with a long-trial one in the SAME build (e.g.
    # SRM_RestingState's one 48000-sample recording, itself one of this run's viz targets)
    # pads the short one to 98%+ zero -- a near-empty recon_signal snapshot and, per the
    # "CUDA out of memory" viz-failure log lines this caused, a real crash risk from the
    # oversized padded tensor. Building each target dataset on its own keeps its max_T its
    # own longest trial only.
    viz_params  = config.get('training_params', {}).get('visualize_params', {}).get('pretrain', {})
    viz_target_cfg = viz_params.get('targets') or [{'subject': None, 'trial': 0}]
    viz_every_n = viz_params.get('every_n_epochs', 2)

    default_viz_dataset_name = next(iter(val_config['dataset_params']['pretrain']))
    wanted_viz_datasets = sorted({t.get('dataset') or default_viz_dataset_name for t in viz_target_cfg})

    viz_datasets_by_name = {}
    for ds_name in wanted_viz_datasets:
        single_config = copy.deepcopy(val_config)
        single_config['dataset_params']['pretrain'] = {ds_name: val_config['dataset_params']['pretrain'][ds_name]}
        logger.info(f"Building Viz Snapshot Dataset for {ds_name} (assemble_trials=False)...")
        viz_datasets_by_name[ds_name] = build_dataset_from_config(
            single_config, transform=None, mode='pretrain', assemble_trials=False)

    def _first_subject(dataset_name):
        return viz_datasets_by_name[dataset_name].base_dataset.subject_data[0].item()

    viz_targets = []
    for t in viz_target_cfg:
        ds_name = t.get('dataset') or default_viz_dataset_name
        viz_ds = viz_datasets_by_name[ds_name]
        subject = t.get('subject') if t.get('subject') is not None else _first_subject(ds_name)
        trial_idx, subject_id = pick_trial(viz_ds, subject, trial=t.get('trial'), dataset_name=t.get('dataset'))
        viz_targets.append((ds_name, trial_idx, subject_id))
    logger.info(f"Recon viz targets (dataset, trial_idx, subject): {viz_targets} every_n_epochs={viz_every_n}")

    def _make_loader(dataset, shuffle):
        return DataLoader(dataset, batch_sampler=MontageBatchSampler(dataset, train_params['batch_size'], shuffle),
                          num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader   = _make_loader(val_dataset,   shuffle=False)

    model_type = train_params.get('model_type', 'MeSAE')
    entry      = MODEL_REGISTRY[model_type]
    trainer    = entry.trainer_cls()

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing model for {Nc} channels (Run: {model_name})...")
    model = build_pretrain_from_config(config, mode='pretrain')
    build_config = checkpoint_build_config(config, mode='pretrain')   # saved in every checkpoint

    tokenizer_epochs = train_params['tokenizer_epochs']
    freeze_stamps    = train_params.get('freeze_stamps', True)
    total_epochs     = train_params['epochs']
    model.enter_tokenizer_phase()
    model.to(device)
    logger.info(f"  [Tokenizer phase] epochs 1-{tokenizer_epochs}: every block, temporal only, unmasked; "
                f"pool after blocks {model.encoder.pool_after}. Masked phase from epoch {tokenizer_epochs + 1}, "
                f"freeze_stamps={freeze_stamps}")

    logger.info("Warming up with dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, bool_masked_pos, valid_channels = _unpack_batch(dummy_batch, device)
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx, bool_masked_pos=None, valid_channels=valid_channels)

    # Built once, while every param is trainable: bypassed blocks and (later) frozen
    # Q-atoms just get grad=None, which AdamW skips.
    scaler    = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(optimizer_param_groups(model, train_params['weight_decay']), lr=train_params['learning_rate'])
    cosine_t_max     = max(1, train_params['epochs'] - train_params['warmup_epochs'])
    main_scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler        = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = entry.plotter_cls(output_dir=vis_dir)

    mask_pp     = config.get('preprocess_params', {}).get('mask', {})
    loss_params = config.get('model_params', {}).get(model_type, {}).get('pretrain', {}).get('loss', {})
    loss_hparams = dict(loss_params)
    # IO/masking.py: the strategy (with its optional channel subsampler) owns its schedule;
    # this loop only tells it the masked-phase epoch and redraws every masked epoch.
    mask_strategy = build_masking_strategy_from_config(mask_pp)
    logger.info(f"model_type={model_type}  masking={mask_pp.get('masking_strategy', 'random')}  "
                f"subsample={'on' if mask_strategy.subsampler else 'off'}  loss_hparams={loss_hparams}")

    best_val_loss = float('inf')  # reset at the phase boundary: masked loss isn't comparable
    logger.info(f"Starting {model_type} Pretraining ({total_epochs} epochs)")

    for epoch in range(1, total_epochs + 1):
        masked = epoch > tokenizer_epochs
        if epoch == tokenizer_epochs + 1:
            model.enter_masked_phase(freeze_stamps=freeze_stamps)
            best_val_loss = float('inf')
            logger.info(f"  [Masked phase] epoch {epoch}: all blocks, spatial attention + coord embedding on, "
                        f"StampBank {'frozen' if freeze_stamps else 'TRAINING (mp loss stays on)'}")
        # Curriculum counts from the first masked epoch; masks are redrawn every masked epoch.
        # Tokenizer-phase epochs ignore the dataset's masks.
        mask_strategy.set_epoch(max(1, epoch - tokenizer_epochs))
        if masked:
            train_dataset.set_masking(mask_strategy)
            val_dataset.set_masking(mask_strategy)
            # rebuild, don't just re-iterate: persistent_workers=True means worker
            # subprocesses hold their own copy of the dataset from spawn time and never
            # see the mutation above otherwise (see PretrainDataset.set_masking).
            train_loader = _make_loader(train_dataset, shuffle=True)
            val_loader   = _make_loader(val_dataset,   shuffle=False)
            logger.info(f"  [mask curriculum] epoch {epoch}: {mask_strategy.describe()}")

        train_metrics = train_one_epoch(model, trainer, train_loader, optimizer, scaler, device, epoch, masked,
                                        **loss_hparams)
        val_metrics   = validate_one_epoch(model, trainer, val_loader, device, masked, **loss_hparams)
        scheduler.step()

        loss_keys = {'loss', 'masked', 'unmasked'}
        other_metrics = {k: v for k, v in train_metrics.items() if k not in loss_keys}

        logging.info(f"--- Epoch {epoch}/{total_epochs} ({'masked' if masked else 'tokenizer'}) Summary ---")
        logging.info(f"  [Train] " + " | ".join([f"{k}: {train_metrics.get(k, 0.0):.4f}" for k in ['loss', 'masked', 'unmasked']]))
        logging.info(f"  [Val]   " + " | ".join([f"{k}: {val_metrics.get(k, 0.0):.4f}"   for k in ['loss', 'masked', 'unmasked']]))

        if other_metrics:
            o_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in other_metrics.items()])
            logging.info(f"  [Other] {o_str}")

        logging.info("-" * 40)

        # Unconditional, every epoch: a mask-ratio curriculum (or any other future metric
        # surprise) can make val_metrics['loss'] structurally incomparable across epochs —
        # e.g. the mask-ratio ramp makes the task itself harder over
        # time, so best_val_loss below can freeze on an early, easy-ratio epoch and never
        # update again even while the model keeps genuinely improving within each step.
        # That leaves ONLY that early checkpoint on disk if training is later interrupted —
        # real instance: mesae_pretrain_v4 froze "best" at epoch 4/50, losing every epoch's
        # progress after that when the run was stopped at epoch 32. last.pth is the
        # insurance: whatever epoch you actually stopped at is always recoverable.
        torch.save({'model_state_dict': model.state_dict(), 'build_config': build_config}, os.path.join(checkpoint_dir, 'last.pth'))

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            # Tokenizer-phase best is overwritten by the first masked epoch (reset above).
            torch.save({'model_state_dict': model.state_dict(), 'build_config': build_config}, os.path.join(checkpoint_dir, 'best.pth'))
            logger.info(f"  > Saved Best Checkpoint ({'masked' if masked else 'tokenizer'} phase)")

        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot_pretrain()

        # every_n_epochs: int (0 = never) or "last" (final epoch only, for quick tests)
        if epoch == total_epochs if viz_every_n == 'last' else (viz_every_n > 0 and epoch % viz_every_n == 0):
            # Training holds the GPU close to capacity; the caching allocator doesn't
            # return freed blocks to the OS on its own, so a big new alloc here (recon
            # panels build separate activations) can OOM even though nothing is really
            # using that memory anymore. One empty_cache() before the batch of panels
            # releases it back -- cheap relative to the epoch itself, only paid every
            # every_n_epochs.
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
            for topo_dataset_name, topo_trial_idx, topo_subject_id in viz_targets:
                try:
                    topo_dataset = viz_datasets_by_name[topo_dataset_name]
                    bundle, _metrics = build_pretrain_bundle(
                        model, topo_dataset, topo_trial_idx, config, device,
                        subject_id=topo_subject_id, epoch=epoch)
                    recon_dir = os.path.join(vis_dir, 'recon')
                    render_recon(bundle, config, recon_dir)
                    render_stamp_gallery(bundle, config, recon_dir,
                                         cmap=config.get('training_params', {}).get('visualize_params', {}).get('cmap', 'YlOrRd'))
                except Exception as e:
                    logger.warning(f"  Topomap viz failed (epoch {epoch}, subject={topo_subject_id}, trial_idx={topo_trial_idx}): {e}")

    logger.info("Pretraining Complete.")


if __name__ == '__main__':
    main()
