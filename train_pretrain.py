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

def train_one_epoch(teacher_vq, student, data_loader, optimizer, device, epoch):
    student.train()
    
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0, "acc": 0.0, "acc_m": 0.0, "acc_v": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    total_sub_dim, N = student.total_sub_dim, student.num_discrete
    
    for batch_idx, batch in enumerate(pbar):
        x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
        B, C, P, T_patch = x_patches.shape
        Tokens = C * P
        
        # --- Get Tokenizer Targets ---
        with torch.no_grad():
            *_, teacher_indices, _ = teacher_vq(x_patches, coords, time_idx=None)
            teacher_indices = teacher_indices.permute(0, 2, 1, 3, 4).reshape(B, Tokens, -1)
            
        optimizer.zero_grad()
        logits = student(x_patches, coords, time_indices=time_indices, bool_masked_pos=mask)
        
        # --- Loss Calculation ---
        ce_all = F.cross_entropy(logits.reshape(-1, N), teacher_indices.reshape(-1), reduction='none')
        ce_all = ce_all.reshape(B, Tokens, total_sub_dim).mean(dim=-1)
        
        m_flat = mask.reshape(B, Tokens)
        v_flat = ~m_flat
        loss_masked = ce_all[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
        loss_visible = ce_all[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
        
        loss = loss_masked * 0.8 + loss_visible * 0.2
        loss.backward()
        optimizer.step()
        
        # --- Metrics Calculation ---
        with torch.no_grad():
            preds = logits.argmax(dim=-1) 
            correct = (preds == teacher_indices).float() # (B, Tokens, total_sub_dim)
            correct_tokens = correct.mean(dim=-1) # (B, Tokens)
            
            acc_total = correct.mean()
            acc_m = correct_tokens[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
            acc_v = correct_tokens[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
            
            metrics["acc"] += acc_total.item()
            metrics["acc_m"] += acc_m.item()
            metrics["acc_v"] += acc_v.item()

        metrics["loss"] += loss.item()
        metrics["distill_masked"] += loss_masked.item()
        metrics["distill_visible"] += loss_visible.item()
        
        avg_loss = metrics["loss"] / (batch_idx + 1)
        pbar.set_postfix({'L': f"{avg_loss:.4f}", 'AccM': f"{acc_m.item():.4f}"})

    N_batches = len(data_loader)
    return {k: v/N_batches for k, v in metrics.items()}, batch

def validate_one_epoch(teacher_vq, student, data_loader, device):
    student.eval()
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0, 
               "mse": 0.0, "mse_m": 0.0, "mse_v": 0.0, 
               "acc": 0.0, "acc_m": 0.0, "acc_v": 0.0}
    
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    total_sub_dim, N = student.total_sub_dim, student.num_discrete
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
            B, C, P, T_patch = x_patches.shape
            Tokens = C * P
            
            # Teacher pass
            p1, p2, p3, _, teacher_indices, _ = teacher_vq(x_patches, coords, time_idx=None)
            teacher_indices = teacher_indices.permute(0, 2, 1, 3, 4).reshape(B, Tokens, -1)
            teacher_recon = teacher_vq.reconstruct(p1, p2, p3, n_samples=T_patch)
            if teacher_recon.dim() == 3: teacher_recon = teacher_recon.reshape(B, P, C, T_patch)
            
            # Student pass
            logits = student(x_patches, coords, time_indices, mask)
            
            # Calculate CE
            ce_all = F.cross_entropy(logits.reshape(-1, N), teacher_indices.reshape(-1), reduction='none')
            ce_all = ce_all.reshape(B, Tokens, total_sub_dim).mean(dim=-1)
            
            m_flat = mask.reshape(B, Tokens)
            v_flat = ~m_flat
            loss_masked = ce_all[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
            loss_visible = ce_all[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
            
            loss = loss_masked * 0.8 + loss_visible * 0.2
            
            # --- Accuracy ---
            preds = logits.argmax(dim=-1)
            correct = (preds == teacher_indices).float()
            correct_tokens = correct.mean(dim=-1)
            
            acc_total = correct.mean()
            acc_m = correct_tokens[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
            acc_v = correct_tokens[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
            
            metrics["acc"] += acc_total.item()
            metrics["acc_m"] += acc_m.item()
            metrics["acc_v"] += acc_v.item()
            
            # Reconstruction MSE Calculation
            v_q = preds.float() - (N - 1) / 2.0
            z_q = torch.matmul(v_q.reshape(B * Tokens, -1), teacher_vq.attnvq.A.t()).reshape(B, Tokens, -1)
            z_q_reshaped = z_q.reshape(B, C, P, -1).permute(0, 2, 1, 3).reshape(B * P, C, -1)
            
            p1_s, p2_s, p3_s = teacher_vq.decoder(z_q_reshaped)
            student_recon = teacher_vq.reconstruct(p1_s, p2_s, p3_s, n_samples=T_patch).reshape(B, P, C, T_patch)
            
            m_expanded = mask.reshape(B, C, P).permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, T_patch)
            
            metrics["mse"] += F.mse_loss(student_recon, teacher_recon, reduction='mean').item()
            metrics["mse_m"] += F.mse_loss(student_recon[m_expanded], teacher_recon[m_expanded], reduction='mean').item() if m_expanded.any() else 0.0
            metrics["mse_v"] += F.mse_loss(student_recon[~m_expanded], teacher_recon[~m_expanded], reduction='mean').item() if (~m_expanded).any() else 0.0

            metrics["loss"] += loss.item()
            metrics["distill_masked"] += loss_masked.item()
            metrics["distill_visible"] += loss_visible.item()
            
            pbar.set_postfix({'L': f"{metrics['loss']/(batch_idx+1):.4f}", 'AccM': f"{acc_m.item():.4f}"})

    N_batches = len(data_loader)
    return {k: v/N_batches for k, v in metrics.items()}, batch

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
    window_s = config['dataset_params'].get('window_size_to_use', 1.0) 
    patches_per_trial = int(window_s) 

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

    train_loader = DataLoader(build_dataset_from_config(train_config, transform=transform, mode='pretrain'), 
                              batch_size=safe_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(build_dataset_from_config(val_config, transform=transform, mode='pretrain'), 
                            batch_size=safe_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    teacher = build_model_from_config(config).to(device)
    student = build_backbone_from_config(config).to(device)
    
    # Warmup Lazy Modules
    dummy_batch = next(iter(train_loader))
    x_patches, coords, mask, time_indices, _ = [t.to(device) for t in dummy_batch]
    teacher.eval(); student.eval()
    with torch.no_grad():
        teacher(x_patches, coords, time_idx=None)
        student(x_patches, coords, time_indices, mask)

    # Load teacher weights
    teacher_state_dict = torch.load(teacher_ckpt, map_location='cpu')['model_state_dict']
    model_state_dict = teacher.state_dict()
    filtered_state_dict = {k: v for k, v in teacher_state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
    teacher.load_state_dict(filtered_state_dict, strict=False)
    for p in teacher.parameters(): p.requires_grad = False
    
    with torch.no_grad():
        student.spatial_temporal_encoder.load_state_dict(teacher.spatial_temporal_encoder.state_dict())
        student.encoder.load_state_dict(teacher.encoder.state_dict(), strict=False)
    
    optimizer = optim.AdamW(student.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=train_params['min_learning_rate'])
    plotter = Plotter(output_dir=vis_dir)
    
    best_val_loss = float('inf')
    total_epochs = train_params['epochs']
    
    logger.info(f"Starting Pretraining ({total_epochs} epochs)")
    for epoch in range(1, total_epochs + 1):
        train_metrics, train_last_batch = train_one_epoch(teacher, student, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(teacher, student, val_loader, device)
        scheduler.step()
        
        # --- Detailed Logging (Match train_tokenizer style) ---
        logger.info(f"Epoch {epoch}/{total_epochs}:")
        logger.info(f"  > Train [Loss:{train_metrics['loss']:.4f}, Acc:{train_metrics['acc']:.4f} (M:{train_metrics['acc_m']:.4f}, V:{train_metrics['acc_v']:.4f})]")
        logger.info(f"  > Val   [Loss:{val_metrics['loss']:.4f}, Acc:{val_metrics['acc']:.4f} (M:{val_metrics['acc_m']:.4f}, V:{val_metrics['acc_v']:.4f})]")
        if 'mse' in val_metrics:
            logger.info(f"  > Recon [T_MSE:{val_metrics['mse']:.6f}, M_MSE:{val_metrics['mse_m']:.6f}, V_MSE:{val_metrics['mse_v']:.6f}]")
        
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
