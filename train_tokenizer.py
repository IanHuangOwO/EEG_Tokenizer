import os
import sys
import json
import argparse
import shutil
import copy
import random
import logging
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for server/multi-thread safety
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_preprocessing_from_config
from utils.reconstruction import visualize_reconstruction
from utils.plotter import Plotter

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    file_handler = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    metrics = {"loss": 0.0, "sub": 0.0, "recon": 0.0, "temp": 0.0, "amp": 0.0, "phase": 0.0, "temp_mse": 0.0}
    last_x, last_recon = None, None
    # No graphical bar, just percent and stats
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    for batch in pbar:
        x, coords, time_idx, _ = [t.to(device) for t in batch]
        optimizer.zero_grad()

        p1, p2, p3, sub_loss, _, _ = model(x, coords, time_idx)
        recon_loss, l_amp, l_phs, l_tmp, l_mse = model.get_loss(x, p1, p2, p3, x_fft=None)

        loss = recon_loss + sub_loss
        loss.backward()
        optimizer.step()

        metrics["loss"] += loss.item()
        metrics["sub"] += sub_loss.item()
        metrics["recon"] += recon_loss.item()
        metrics["temp"] += l_tmp.item()
        metrics["amp"] += l_amp.item()
        metrics["phase"] += l_phs.item()
        metrics["temp_mse"] += l_mse.item()

        last_x, last_recon = x, model.reconstruct(p1, p2, p3, n_samples=x.shape[-1]).detach()

        # Simple progress bar
        pbar.set_postfix({'L': f"{loss.item():.2f}", 'MSE': f"{l_mse.item():.4f}"})

    N = len(data_loader)
    epoch_metrics = {k: v/N for k, v in metrics.items()}
    
    # Calculate health metrics once at end of epoch
    if hasattr(model, 'get_current_metrics'):
        health = model.get_current_metrics()
        epoch_metrics.update(health)
    elif hasattr(model, 'attnvq'):
        health = model.attnvq.get_current_metrics()
        epoch_metrics.update(health)
        
    return epoch_metrics, (last_x, last_recon)

def validate_one_epoch(model, data_loader, device):
    model.eval()
    metrics = {"loss": 0.0, "sub": 0.0, "recon": 0.0, "temp": 0.0, "amp": 0.0, "phase": 0.0, "temp_mse": 0.0}
    last_x, last_recon = None, None
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    with torch.no_grad():
        for batch in pbar:
            x, coords, time_idx, _ = [t.to(device) for t in batch]
            p1, p2, p3, sub_loss, _, _ = model(x, coords, time_idx)
            recon_loss, l_amp, l_phs, l_tmp, l_mse = model.get_loss(x, p1, p2, p3, x_fft=None)
            
            metrics["loss"] += (recon_loss + sub_loss).item()
            metrics["sub"] += sub_loss.item()
            metrics["recon"] += recon_loss.item()
            metrics["temp"] += l_tmp.item()
            metrics["amp"] += l_amp.item()
            metrics["phase"] += l_phs.item()
            metrics["temp_mse"] += l_mse.item()
            
            last_x, last_recon = x, model.reconstruct(p1, p2, p3, n_samples=x.shape[-1]).detach()
            pbar.set_postfix({'L': f"{(recon_loss + sub_loss).item():.2f}", 'MSE': f"{l_mse.item():.4f}"})

    N = len(data_loader)
    return {k: v/N for k, v in metrics.items()}, (last_x, last_recon)

def main():
    parser = argparse.ArgumentParser(description='EEG Tokenizer Training')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to config file')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract params
    train_params = config['training_params']
    
    # Override from config
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_type = train_params.get('model_type', 'NeuroRVQ')
    model_name = train_params.get('model_name', 'default_run')
    
    # New structured paths: ./output/{model_name}/{category}
    base_output_dir = f"output/{model_name}"
    checkpoint_dir = os.path.join(base_output_dir, "tokenizer")
    # RENAMED: config -> artifacts
    artifact_dir = os.path.join(base_output_dir, "artifacts")
    vis_dir = os.path.join(base_output_dir, "visualization")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Setup Logger
    logger = setup_logger(artifact_dir)
    
    # Copy config for reproducibility to artifacts dir
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))
    
    # 2. Dataset Setup (Subject-Level Split)
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    # Build unified transform
    transform = build_preprocessing_from_config(config)
    
    # Split Subjects
    all_subjects = config['dataset_params']['subjects']
    split_ratio = train_params.get('train_val_split', 0.9)
    
    # Deterministic shuffle for reproducibility
    random.seed(42)
    shuffled_subjects = list(all_subjects)
    random.shuffle(shuffled_subjects)
    
    n_train = int(len(shuffled_subjects) * split_ratio)
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:]
    
    logger.info(f"Subject Split: {len(train_subjects)} Train, {len(val_subjects)} Val")
    
    # Create Config Copies
    train_config = copy.deepcopy(config)
    val_config = copy.deepcopy(config)
    
    train_config['dataset_params']['subjects'] = train_subjects
    val_config['dataset_params']['subjects'] = val_subjects
    
    # Build Datasets
    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=transform, mode='tokenizer')
    
    logger.info("Building Validation Dataset...")
    # Reuse transform (it's stateless or config-based)
    val_dataset = build_dataset_from_config(val_config, transform=transform, mode='tokenizer')
    
    logger.info(f"Dataset Sizes: Train={len(train_dataset)} patches, Val={len(val_dataset)} patches")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_params['batch_size'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 3. Initialize Model
    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing {model_type} Tokenizer for {Nc} channels (Run: {model_name})...")
    
    model = build_model_from_config(config, src_output_dir=artifact_dir)
    model.to(device)

    # --- Initialization Step for Lazy Modules ---
    # Since AttnVQ uses nn.LazyLinear, we must perform one forward pass to initialize parameter shapes
    # before we can create the optimizer, as it needs to see all parameters.
    logger.info("Warming up Lazy Modules with a dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, _ = [t.to(device) for t in dummy_batch]
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx)
    # --------------------------------------------

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    
    # Main LR scheduler
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=train_params['epochs'] - train_params['warmup_epochs'], 
        eta_min=train_params['min_learning_rate']
    )
    
    # Warmup LR scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01, # Start at 1% of base LR
        total_iters=train_params['warmup_epochs']
    )

    # Chained Scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[train_params['warmup_epochs']]
    )

    plotter = Plotter(output_dir=vis_dir)

    # 5. Training Loop
    best_val_loss = float('inf')
    total_epochs = train_params['epochs']
    
    logger.info(f"Starting Training ({total_epochs} epochs)")
    for epoch in range(1, total_epochs + 1):
        train_metrics, train_last_batch = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{total_epochs}:")
        logger.info(f"  > Train [L:{train_metrics['loss']:.4f}, MSE:{train_metrics['temp_mse']:.4f}, Rec:{train_metrics['recon']:.4f}, Sub:{train_metrics['sub']:.4f}]")
        logger.info(f"  > Val   [L:{val_metrics['loss']:.4f}, MSE:{val_metrics['temp_mse']:.4f}, Rec:{val_metrics['recon']:.4f}, Sub:{val_metrics['sub']:.4f}]")
        if 'subspace_loss' in train_metrics:
            logger.info(f"  > Subspace [Loss:{train_metrics['subspace_loss']:.4f}, Div:{train_metrics['head_cross_corr']:.4f}]")
            logger.info(f"  > Codebook [Perp:{train_metrics['codebook_perplexity']:.2f}, Sharp:{train_metrics['codebook_sharpness']:.3f}, Util:{train_metrics['active_rank_ratio']:.1%}]")
            logger.info(f"  > Matrix   [A_S:{train_metrics['A_sing_val_avg']:.3f}, A_C:{train_metrics['A_cond']:.1f}]")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_tokenizer.pth'))
            logger.info("  > Saved Best Tokenizer")
        
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot(); plotter.plot_metrics()
        
        if epoch % 5 == 0:
            visualize_reconstruction(train_last_batch, val_last_batch, epoch, output_dir=os.path.join(vis_dir, 'reconstruction'))

    logger.info("Training Complete.")

if __name__ == '__main__':
    main()
