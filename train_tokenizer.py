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
from model.factory import build_model_from_config
from utils.reconstruction import visualize_reconstruction
from utils.plotter import Plotter

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    last_batch = None
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    running_metrics = torch.zeros(6, device=device) # loss, sub, recon, real, imag, temp_mse
    
    for batch_idx, batch in enumerate(pbar):
        x, coords, time_idx, _, x_fft = [t.to(device) if (t is not None and t.numel() > 0) else t for t in batch]
        optimizer.zero_grad()
        
        # 🚀 AMP: Autocast forward pass
        with torch.amp.autocast(device_type='cuda'):
            p_real, p_imag, l_sub, _, _ = model(x, coords, time_idx)
            l_total, l_real, l_imag, l_rec, l_sub_eval, l_mse = model.get_loss(x, p_real, p_imag, l_sub, x_fft=x_fft)
            
        # 🚀 AMP: Scaled backward pass
        scaler.scale(l_total).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate on GPU (No .item() sync)
        with torch.no_grad():
            running_metrics += torch.stack([l_total, l_sub_eval, l_rec, l_real, l_imag, l_mse])
            
        if batch_idx % 5 == 0:
            # Only sync every 5 steps for progress bar
            m = running_metrics / (batch_idx + 1)
            pbar.set_postfix({'L': f"{m[0]:.2f}", 'Spec': f"{m[2]:.4f}", 'MSE': f"{m[5]:.4f}"})
        
        last_batch = batch
    
    # Sync once at end of epoch
    N = len(data_loader)
    m_final = (running_metrics / N).cpu().tolist()
    metrics_keys = ["loss", "sub", "recon", "real", "imag", "temp_mse"]
    epoch_metrics = {k: v for k, v in zip(metrics_keys, m_final)}
    epoch_metrics.update({"temp": epoch_metrics["recon"], "grid": 0.0})

    if hasattr(model, 'get_current_metrics'): 
        epoch_metrics.update(model.get_current_metrics())
        
    with torch.no_grad():
        x, coords, time_idx, _, _ = [t.to(device) if isinstance(t, torch.Tensor) else t for t in last_batch]
        with torch.amp.autocast(device_type='cuda'):
            p_real, p_imag, _, _, _ = model(x, coords, time_idx)
            x_recon = model.reconstruct(p_real, p_imag, n_samples=x.shape[-1])
        last_samples = (x.detach(), x_recon.detach())
    return epoch_metrics, last_samples

def validate_one_epoch(model, data_loader, device):
    model.eval()
    last_batch = None
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    running_metrics = torch.zeros(6, device=device)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x, coords, time_idx, _, x_fft = [t.to(device) if (t is not None and t.numel() > 0) else t for t in batch]
            
            with torch.amp.autocast(device_type='cuda'):
                p_real, p_imag, l_sub, _, _ = model(x, coords, time_idx)
                l_total, l_real, l_imag, l_rec, l_sub_eval, l_mse = model.get_loss(x, p_real, p_imag, l_sub, x_fft=x_fft)
            
            running_metrics += torch.stack([l_total, l_sub_eval, l_rec, l_real, l_imag, l_mse])
            
            if batch_idx % 5 == 0:
                m = running_metrics / (batch_idx + 1)
                pbar.set_postfix({'L': f"{m[0]:.2f}", 'Spec': f"{m[2]:.4f}", 'MSE': f"{m[5]:.4f}"})
            last_batch = batch
            
        x, coords, time_idx, _, _ = [t.to(device) if isinstance(t, torch.Tensor) else t for t in last_batch]
        with torch.amp.autocast(device_type='cuda'):
            p_real, p_imag, _, _, _ = model(x, coords, time_idx)
            x_recon = model.reconstruct(p_real, p_imag, n_samples=x.shape[-1])
        last_samples = (x.detach(), x_recon.detach())
        
    N = len(data_loader)
    m_final = (running_metrics / N).cpu().tolist()
    metrics_keys = ["loss", "sub", "recon", "real", "imag", "temp_mse"]
    val_metrics = {k: v for k, v in zip(metrics_keys, m_final)}
    val_metrics.update({"temp": val_metrics["recon"], "grid": 0.0, "grid_loss": 0.0})
    
    return val_metrics, last_samples

def main():
    parser = argparse.ArgumentParser(description='EEG Tokenizer Training')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_params = config['training_params']
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_type = train_params.get('model_type', 'AttnVQ')
    model_name = train_params.get('model_name', 'default_run')
    
    base_output_dir = f"output/{model_name}"
    checkpoint_dir = os.path.join(base_output_dir, "tokenizer")
    artifact_dir = os.path.join(base_output_dir, "artifacts")
    vis_dir = os.path.join(base_output_dir, "visualization")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))
    
    # --- Dataset Setup (Multi-Dataset Subject Split) ---
    dataset_params = config['dataset_params']
    split_ratio = train_params.get('train_val_split', 0.9)
    random.seed(42)

    train_config = copy.deepcopy(config)
    val_config = copy.deepcopy(config)
    
    for ds_name, ds_args in dataset_params.items():
        data_root = ds_args['dataset_path']
        with open(os.path.join(data_root, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        
        all_available_subjects = list(meta.get('data_structure', {}).keys())
        # Convert to list of ints if possible, otherwise keep as strings
        try:
            all_available_subjects = sorted([int(s) for s in all_available_subjects])
        except:
            all_available_subjects = sorted(all_available_subjects)
            
        requested_subjects = ds_args['subject_to_use']
        if requested_subjects == ["all"] or requested_subjects == "all":
            subjects_to_split = all_available_subjects
        else:
            subjects_to_split = [s for s in requested_subjects if s in all_available_subjects or str(s) in all_available_subjects]

        random.shuffle(subjects_to_split)
        n_train = int(len(subjects_to_split) * split_ratio)
        
        # Ensure at least one subject in val if there are at least 2 total
        if n_train == len(subjects_to_split) and len(subjects_to_split) > 1:
            n_train -= 1
        # Ensure at least one subject in train if there are any
        if n_train == 0 and len(subjects_to_split) > 0:
            n_train = 1
            
        train_config['dataset_params'][ds_name]['subject_to_use'] = subjects_to_split[:n_train]
        val_config['dataset_params'][ds_name]['subject_to_use'] = subjects_to_split[n_train:]
        
        logger.info(f"Dataset {ds_name}: {n_train} Train, {len(subjects_to_split) - n_train} Val subjects")
    
    logger.info("Building Training Dataset...")
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='tokenizer')
    logger.info("Building Validation Dataset...")
    val_dataset = build_dataset_from_config(val_config, transform=None, mode='tokenizer')
    
    logger.info(f"Dataset Sizes: Train={len(train_dataset)} patches, Val={len(val_dataset)} patches")
    
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    Nc = train_dataset.base_dataset.Nc
    logger.info(f"Initializing {model_type} Tokenizer for {Nc} channels (Run: {model_name})...")
    model = build_model_from_config(config, src_output_dir=artifact_dir)
    model.to(device)

    logger.info("Warming up Lazy Modules with a dummy pass...")
    dummy_batch = next(iter(train_loader))
    x, coords, time_idx, _, x_fft = [t.to(device) if isinstance(t, torch.Tensor) else t for t in dummy_batch]
    model.eval()
    with torch.no_grad():
        model(x, coords, time_idx)

    optimizer = optim.AdamW(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'] - train_params['warmup_epochs'], eta_min=train_params['min_learning_rate'])
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=train_params['warmup_epochs'])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_params['warmup_epochs']])

    plotter = Plotter(output_dir=vis_dir)

    best_val_loss = float('inf')
    total_epochs = train_params['epochs']
    logger.info(f"Starting Training ({total_epochs} epochs)")
    for epoch in range(1, total_epochs + 1):
        train_metrics, train_last_batch = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(model, val_loader, device)
        scheduler.step()
        logger.info(f"Epoch {epoch}/{total_epochs}:")
        logger.info(f"  > Train [L:{train_metrics['loss']:.4f}, MSE:{train_metrics['temp_mse']:.4f}, Sub:{train_metrics['sub']:.4f}]")
        logger.info(f"  > Val   [L:{val_metrics['loss']:.4f}, MSE:{val_metrics['temp_mse']:.4f}, Sub:{val_metrics['sub']:.4f}]")
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
