import os
import sys
import json
import argparse
import shutil
import copy
import random
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_backbone_from_config, build_preprocessing_from_config
from utils.plotter import Plotter
from utils.reconstruction import visualize_masked_reconstruction

# Enable TF32 for faster matrix multiplication on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def setup_logger(output_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(os.path.join(output_dir, 'pretrain.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def get_latent_dist(z, attnvq):
    """
    Projects raw backbone features z (B, T, D) into codebook probability distributions.
    z: (B, Tokens, D)
    Returns p: (H, B, Tokens, r)
    """
    B, Tokens, D = z.shape
    H = attnvq.num_heads
    r = attnvq.r
    
    # 1. Project student latent using FROZEN A matrix
    z_flat = z.reshape(B * Tokens, D)
    q = torch.matmul(z_flat, attnvq.A) # (BT, Hr)
    
    # 2. Normalize and Scale identically to Tokenizer
    q = q.view(B, Tokens, H, r)
    q_norm = F.normalize(q, p=2, dim=-1)
    
    # Student Logits (H, B, T, r)
    logits = q_norm.permute(2, 0, 1, 3) 
    return logits

def train_one_epoch(teacher_vq, student, data_loader, optimizer, device, epoch):
    student.train()
    
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    H, r = teacher_vq.vq_head_num, teacher_vq.attnvq.r
    last_batch = None

    for batch_idx, batch in enumerate(pbar):
        x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
        B, C, P, T_patch = x_patches.shape
        Tokens = C * P
        last_batch = batch
        
        # --- Get Tokenizer Targets (The Anchor) ---
        with torch.no_grad():
            # Teacher is blind to temporal trial context (time_idx=None)
            *_, teacher_weights = teacher_vq(x_patches, coords, time_idx=None) # (B, P, C, H, r)
            # Reshape to (H, B, Tokens, r)
            teacher_weights = teacher_weights.permute(3, 0, 2, 1, 4).reshape(H, B, Tokens, r)
            
        optimizer.zero_grad()
        
        # 1. Forward Pass (Single Masked Pass)
        z = student(x_patches, coords, time_indices=time_indices, bool_masked_pos=mask) # (B, Tokens, D)
        p = get_latent_dist(z, teacher_vq.attnvq) # (H, B, Tokens, r)
        
        # --- Loss Calculation ---
        log_p = F.log_softmax(p, dim=-1)
        
        # Calculate KL Divergence (H, B, Tokens)
        kl_div = F.kl_div(log_p, teacher_weights, reduction='none').sum(-1) 
        
        # Simple mean over Heads: (B, Tokens)
        avg_kl = kl_div.mean(dim=0)
        
        # Masked vs Visible Masks
        m_flat = mask.view(B, Tokens)
        v_flat = ~m_flat

        # Individual Head KL (averaged over batch/tokens)
        with torch.no_grad():
            for h in range(H):
                metrics[f"kl_h{h}"] = metrics.get(f"kl_h{h}", 0.0) + kl_div[h].mean().item()
                metrics[f"kl_masked_h{h}"] = metrics.get(f"kl_masked_h{h}", 0.0) + kl_div[h][m_flat].mean().item() if m_flat.any() else 0.0
                metrics[f"kl_visible_h{h}"] = metrics.get(f"kl_visible_h{h}", 0.0) + kl_div[h][v_flat].mean().item() if v_flat.any() else 0.0
        
        # Now average ONLY over the Batch and Token dimensions
        # Total Loss
        loss_masked = avg_kl[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
        loss_visible = avg_kl[~m_flat].mean() if (~m_flat).any() else torch.tensor(0.0, device=device)
        
        # Total Loss
        loss = loss_masked * 0.8 + loss_visible * 0.2
        
        loss.backward()
        optimizer.step()
        
        metrics["loss"] += loss.item()
        metrics["distill_masked"] += loss_masked.item()
        metrics["distill_visible"] += loss_visible.item()
        
        avg_loss = metrics["loss"] / (batch_idx + 1)
        pbar.set_postfix({'L': f"{avg_loss:.4f}", 'M': f"{loss_masked.item():.4f}", 'V': f"{loss_visible.item():.4f}"})

    N = len(data_loader)
    return {k: v/N for k, v in metrics.items()}, last_batch

def validate_one_epoch(teacher_vq, student, data_loader, device):
    student.eval()
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0}
    
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    H, r = teacher_vq.vq_head_num, teacher_vq.attnvq.r
    last_batch = None
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
            B, C, P, _ = x_patches.shape
            Tokens = C * P
            last_batch = batch
            
            # Teacher pass
            *_, teacher_weights = teacher_vq(x_patches, coords, time_idx=None)
            teacher_weights = teacher_weights.permute(3, 0, 2, 1, 4).reshape(H, B, Tokens, r)
            
            # Student pass
            z = student(x_patches, coords, time_indices, mask)
            p = get_latent_dist(z, teacher_vq.attnvq)
            
            log_p = F.log_softmax(p, dim=-1)
            
            # Calculate KL Divergence (H, B, Tokens)
            kl_div = F.kl_div(log_p, teacher_weights, reduction='none').sum(-1)
            
            # Simple mean over Heads: (B, Tokens)
            avg_kl = kl_div.mean(dim=0)
            
            # Masked vs Visible Masks
            m_flat = mask.view(B, Tokens)
            v_flat = ~m_flat

            # Individual Head KL
            for h in range(H):
                metrics[f"kl_h{h}"] = metrics.get(f"kl_h{h}", 0.0) + kl_div[h].mean().item()
                metrics[f"kl_masked_h{h}"] = metrics.get(f"kl_masked_h{h}", 0.0) + kl_div[h][m_flat].mean().item() if m_flat.any() else 0.0
                metrics[f"kl_visible_h{h}"] = metrics.get(f"kl_visible_h{h}", 0.0) + kl_div[h][v_flat].mean().item() if v_flat.any() else 0.0
            
            # Now average ONLY over the Batch and Token dimensions
            loss_masked = avg_kl[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
            loss_visible = avg_kl[~m_flat].mean() if (~m_flat).any() else torch.tensor(0.0, device=device)
            
            loss = loss_masked * 0.8 + loss_visible * 0.2
            
            metrics["loss"] += loss.item()
            metrics["distill_masked"] += loss_masked.item()
            metrics["distill_visible"] += loss_visible.item()
            
            avg_loss = metrics["loss"] / (batch_idx + 1)
            pbar.set_postfix({'L': f"{avg_loss:.4f}"})

    N = len(data_loader)
    return {k: v/N for k, v in metrics.items()}, last_batch

def main():
    parser = argparse.ArgumentParser(description='Masked Distillation Pretraining')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--teacher_ckpt', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f: config = json.load(f)
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'default_run')
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    base_output_dir = f"output/{model_name}"
    checkpoint_dir = os.path.join(base_output_dir, "backbone")
    artifact_dir = os.path.join(base_output_dir, "artifacts")
    vis_dir = os.path.join(base_output_dir, "visualization")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    teacher_ckpt = args.teacher_ckpt or os.path.join(base_output_dir, "tokenizer", "best_tokenizer.pth")
    if not os.path.exists(teacher_ckpt):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_ckpt}.")
    
    logger = setup_logger(artifact_dir)
    logger.info(f"Pretraining (Single Pass) using Teacher: {teacher_ckpt}")
    shutil.copy(args.config, os.path.join(artifact_dir, 'config_pretrain.json'))
    # 2. Dataset Setup
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']

    transform = build_preprocessing_from_config(config)

    # Calculate Patches per Trial to adjust Batch Size
    # This keeps VRAM usage similar to Tokenizer training
    window_s = config['dataset_params'].get('window_size_to_use', 1.0) # Default to 1s if not set
    patches_per_trial = int(window_s) # Assuming 1s patches

    base_batch_size = train_params['batch_size']
    safe_batch_size = max(1, base_batch_size // patches_per_trial)

    logger.info(f"Window Size: {window_s}s ({patches_per_trial} patches/trial)")
    logger.info(f"Adjusting Batch Size: {base_batch_size} -> {safe_batch_size} (to maintain VRAM parity)")

    all_subjects = config['dataset_params']['subjects']
    random.seed(42); random.shuffle(all_subjects)
    n_train = int(len(all_subjects) * train_params.get('train_val_split', 0.9))

    train_config = copy.deepcopy(config); val_config = copy.deepcopy(config)
    train_config['dataset_params']['subjects'] = all_subjects[:n_train]
    val_config['dataset_params']['subjects'] = all_subjects[n_train:]

    logger.info("Building Masked Datasets...")
    train_loader = DataLoader(build_dataset_from_config(train_config, transform=transform, mode='pretrain'), 
                              batch_size=safe_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(build_dataset_from_config(val_config, transform=transform, mode='pretrain'), 
                            batch_size=safe_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    logger.info("Loading Frozen Tokenizer VQ Experts...")
    teacher = build_model_from_config(config)
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location='cpu')['model_state_dict'], strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    logger.info("Initializing Student Backbone...")
    student = build_backbone_from_config(config).to(device)
    
    with torch.no_grad():
        # Copy shared structure weights (SpatialTemporalEncoder & Encoder)
        # We need to map teacher.spatial_temporal_encoder -> student.spatial_temporal_encoder
        logger.info("  > Warming up student SpatialTemporalEncoder with teacher weights...")
        student.spatial_temporal_encoder.load_state_dict(teacher.spatial_temporal_encoder.state_dict())
        
        # Map teacher.encoder -> student.encoder
        logger.info("  > Warming up student Encoder (Transformer Stack) with teacher weights...")
        # Note: Tokenizer and Backbone might have different depths, but we load what matches
        student.encoder.load_state_dict(teacher.encoder.state_dict(), strict=False)
    
    optimizer = optim.AdamW(student.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=train_params['min_learning_rate'])
    plotter = Plotter(output_dir=vis_dir)
    
    best_val_loss = float('inf')
    for epoch in range(1, train_params['epochs'] + 1):
        train_metrics, train_last_batch = train_one_epoch(teacher, student, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(teacher, student, val_loader, device)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train_L={train_metrics['loss']:.4f}, Val_L={val_metrics['loss']:.4f}, Masked={train_metrics['distill_masked']:.4f}")
        
        # Update Plotter with current metrics
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot(filename='pretrain_curves.png')        
        plotter.plot_metrics(filename='pretrain_metrics.png') 
        
        if epoch % 5 == 0:
            visualize_masked_reconstruction(val_last_batch, teacher, student, epoch, output_dir=os.path.join(vis_dir, 'reconstruction_pretrain'))

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({'model_state_dict': student.state_dict()}, os.path.join(checkpoint_dir, 'best_backbone.pth'))
            logger.info("  > Saved Best Backbone")
            
    logger.info("Pretraining Complete.")

if __name__ == '__main__':
    main()
