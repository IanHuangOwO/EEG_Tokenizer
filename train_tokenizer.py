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
    metrics = {"loss": 0.0, "vq": 0.0, "recon": 0.0, "temp": 0.0, "amp": 0.0, "phase": 0.0, "temp_mse": 0.0}
    last_x, last_recon = None, None
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}")
    
    for batch in pbar:
        x, coords, _ = [t.to(device) for t in batch]
        optimizer.zero_grad()
        
        p1, p2, p3, vq_loss, _, _ = model(x, coords)
        recon_loss, l_amp, l_phs, l_tmp, l_mse = model.get_loss(x, p1, p2, p3, x_fft=None)
        
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        
        metrics["loss"] += loss.item()
        metrics["vq"] += vq_loss.item()
        metrics["recon"] += recon_loss.item()
        metrics["temp"] += l_tmp.item()
        metrics["amp"] += l_amp.item()
        metrics["phase"] += l_phs.item()
        metrics["temp_mse"] += l_mse.item()
        
        last_x, last_recon = x, model.reconstruct(p1, p2, p3, n_samples=x.shape[-1]).detach()
        pbar.set_postfix({'L': f"{loss.item():.2f}", 'MSE': f"{l_mse.item():.4f}", 'Temp': f"{l_tmp.item():.2f}"})

    N = len(data_loader)
    return tuple(v/N for v in metrics.values()), (last_x, last_recon)

def validate_one_epoch(model, data_loader, device):
    model.eval()
    metrics = {"loss": 0.0, "vq": 0.0, "recon": 0.0, "temp": 0.0, "amp": 0.0, "phase": 0.0, "temp_mse": 0.0}
    last_x, last_recon = None, None
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation")
    
    with torch.no_grad():
        for batch in pbar:
            x, coords, _ = [t.to(device) for t in batch]
            p1, p2, p3, vq_loss, _, _ = model(x, coords)
            recon_loss, l_amp, l_phs, l_tmp, l_mse = model.get_loss(x, p1, p2, p3, x_fft=None)
            
            metrics["loss"] += (recon_loss + vq_loss).item()
            metrics["vq"] += vq_loss.item()
            metrics["recon"] += recon_loss.item()
            metrics["temp"] += l_tmp.item()
            metrics["amp"] += l_amp.item()
            metrics["phase"] += l_phs.item()
            metrics["temp_mse"] += l_mse.item()
            
            last_x, last_recon = x, model.reconstruct(p1, p2, p3, n_samples=x.shape[-1])
            pbar.set_postfix({'MSE': f"{l_mse.item():.4f}", 'Temp': f"{l_tmp.item():.2f}"})

    N = len(data_loader)
    return tuple(v/N for v in metrics.values()), (last_x, last_recon)

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
    checkpoint_dir = os.path.join(base_output_dir, "checkpoints")
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
    refine_epochs = train_params.get('decoder_refine_epochs', 0)
    
    logger.info(f"Starting Stage 1: Joint Training ({total_epochs} epochs)")
    for epoch in range(1, total_epochs + 1):
        # ... (unchanged loop content)
        train_metrics, train_last_batch = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(model, val_loader, device)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}/{total_epochs}:")
        logger.info(f"  > Train [L:{train_metrics[0]:.4f}, MSE:{train_metrics[6]:.4f}, Rec:{train_metrics[2]:.4f}, VQ:{train_metrics[1]:.4f}]")
        logger.info(f"  > Val   [L:{val_metrics[0]:.4f}, MSE:{val_metrics[6]:.4f}, Rec:{val_metrics[2]:.4f}, VQ:{val_metrics[1]:.4f}]")
        
        if val_metrics[0] < best_val_loss:
            best_val_loss = val_metrics[0]
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_model.pth'))
            logger.info("  > Saved Best Model (Stage 1)")
        
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot(); plotter.plot_metrics()
        
        if epoch % 5 == 0:
            visualize_reconstruction(train_last_batch, val_last_batch, epoch, output_dir=os.path.join(vis_dir, 'reconstruction'))

    # --- Stage 2: Decoder Refinement ---
    if refine_epochs > 0:
        logger.info(f"\nStarting Stage 2: Decoder-Only Refinement ({refine_epochs} epochs)")
        
        # Freeze everything except decoders
        for name, param in model.named_parameters():
            if 'scale_decoders' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Re-initialize optimizer for refinement (optional but recommended)
        refine_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=train_params['learning_rate'] * 0.5, 
            weight_decay=train_params['weight_decay']
        )
        
        for epoch in range(total_epochs + 1, total_epochs + refine_epochs + 1):
            train_metrics, train_last_batch = train_one_epoch(model, train_loader, refine_optimizer, device, epoch)
            val_metrics, val_last_batch = validate_one_epoch(model, val_loader, device)
            
            logger.info(f"Refine Epoch {epoch}:")
            logger.info(f"  > Train [MSE:{train_metrics[6]:.4f}, Rec:{train_metrics[2]:.4f}]")
            logger.info(f"  > Val   [MSE:{val_metrics[6]:.4f}, Rec:{val_metrics[2]:.4f}]")
            
            if val_metrics[0] < best_val_loss:
                best_val_loss = val_metrics[0]
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(checkpoint_dir, 'best_model_refined.pth'))
                logger.info("  > Saved Best Refined Model")
            
            plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
            plotter.plot(); plotter.plot_metrics()
            
            if epoch % 2 == 0:
                visualize_reconstruction(train_last_batch, val_last_batch, epoch, output_dir=os.path.join(vis_dir, 'reconstruction'))

    logger.info("Training Complete.")

if __name__ == '__main__':
    main()
