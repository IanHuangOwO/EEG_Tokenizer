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
from model.factory import build_model_from_config, build_backbone_from_config
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
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0, "acc": 0.0, "acc_m": 0.0, "acc_v": 0.0, "mse": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    total_sub_dim, N_discrete = student.total_sub_dim, student.num_discrete
    
    for batch_idx, batch in enumerate(pbar):
        x_patches, coords, mask, time_indices, _, _ = [t.to(device) if t is not None else None for t in batch]
        B, C, P, T_patch = x_patches.shape
        Tokens = C * P
        
        with torch.no_grad():
            # teacher_vq expects [B, C, N, L]
            *_, teacher_indices, _ = teacher_vq(x_patches)
            # teacher_indices: [B, C, P_coarse, H, r]
            B, C, P_coarse, H, r = teacher_indices.shape
            # Reshape to [B, C * P_coarse, H * r] -> [B, Tokens_coarse, total_sub_dim]
            teacher_indices = teacher_indices.reshape(B, C * P_coarse, -1)
            
        optimizer.zero_grad()
        # student expects [B, C, N, L], [B, C, 3], [B, C, N]
        logits = student(x_patches, coords, bool_masked_pos=mask)
        # logits: [B, C, P_coarse, H, r, N_discrete]
        # Reshape to [B, Tokens_coarse, total_sub_dim, N_discrete]
        logits = logits.reshape(B, C * P_coarse, total_sub_dim, N_discrete)
        
        # Calculate CE Loss
        ce_all = F.cross_entropy(logits.reshape(-1, N_discrete), teacher_indices.reshape(-1), reduction='none')
        ce_all = ce_all.reshape(B, C * P_coarse, total_sub_dim).mean(dim=-1) 
        
        # We need a mask for coarse resolution if we use it, 
        # but for now, let's assume P_coarse matches P or handle downsampling
        if P_coarse != P:
            # Downsample mask to match P_coarse
            mask_coarse = F.interpolate(mask.float(), size=P_coarse, mode='nearest').bool()
        else:
            mask_coarse = mask
            
        m_flat = mask_coarse.reshape(B, -1)
        v_flat = ~m_flat
        
        loss_masked = ce_all[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
        loss_visible = ce_all[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
        loss = loss_masked * 0.8 + loss_visible * 0.2
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1) 
            correct = (preds == teacher_indices).float()
            acc_total = correct.mean()
            
            # Reconstruction for MSE logging
            v_q = preds.float() - (N_discrete - 1) / 2.0
            H_vq, r_vq = teacher_vq.attnvq.num_heads, teacher_vq.attnvq.r
            v_q_heads = v_q.reshape(B, C * P_coarse, H_vq, r_vq)
            
            # Map discrete indices back to subspace vectors via Teacher's A matrix
            A_per_head = teacher_vq.attnvq.A.view(-1, H_vq, r_vq) 
            z_q_heads = torch.einsum('bchr, dhr -> bchd', v_q_heads, A_per_head)
            
            # Decode to spectral domain
            p1_s, p2_s = teacher_vq.decoder(z_q_heads)
            
            # Reconstruct to time domain
            student_recon = teacher_vq.reconstruct(p1_s, p2_s, n_samples=T_patch)
            student_recon = student_recon.reshape(B, C, P, T_patch)
            
            mse = F.mse_loss(student_recon, x_patches)
            
            metrics["acc"] += acc_total.item()
            metrics["mse"] += mse.item()
            
        metrics["loss"] += loss.item()
        metrics["distill_masked"] += loss_masked.item()
        metrics["distill_visible"] += loss_visible.item()
        
        pbar.set_postfix({'L': f"{loss.item():.4f}", 'Acc': f"{acc_total.item():.4f}", 'MSE': f"{mse.item():.4f}"})
        
    N_batches = len(data_loader)
    return {k: v/N_batches for k, v in metrics.items()}, batch

def validate_one_epoch(teacher_vq, student, data_loader, device):
    student.eval()
    metrics = {"loss": 0.0, "distill_masked": 0.0, "distill_visible": 0.0, "mse": 0.0, "acc": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    total_sub_dim, N_discrete = student.total_sub_dim, student.num_discrete
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            x_patches, coords, mask, time_indices, _, _ = [t.to(device) if t is not None else None for t in batch]
            B, C, P, T_patch = x_patches.shape
            
            *_, teacher_indices, _ = teacher_vq(x_patches)
            B, C, P_coarse, H, r = teacher_indices.shape
            teacher_indices = teacher_indices.reshape(B, C * P_coarse, -1)
            
            logits = student(x_patches, coords, bool_masked_pos=mask)
            logits = logits.reshape(B, C * P_coarse, total_sub_dim, N_discrete)
            
            ce_all = F.cross_entropy(logits.reshape(-1, N_discrete), teacher_indices.reshape(-1), reduction='none')
            ce_all = ce_all.reshape(B, C * P_coarse, total_sub_dim).mean(dim=-1)
            
            if P_coarse != P:
                mask_coarse = F.interpolate(mask.float(), size=P_coarse, mode='nearest').bool()
            else:
                mask_coarse = mask
                
            m_flat = mask_coarse.reshape(B, -1)
            v_flat = ~m_flat
            
            loss_masked = ce_all[m_flat].mean() if m_flat.any() else torch.tensor(0.0, device=device)
            loss_visible = ce_all[v_flat].mean() if v_flat.any() else torch.tensor(0.0, device=device)
            loss = loss_masked * 0.8 + loss_visible * 0.2
            
            preds = logits.argmax(dim=-1)
            correct = (preds == teacher_indices).float()
            acc_total = correct.mean()
            
            v_q = preds.float() - (N_discrete - 1) / 2.0
            H_vq, r_vq = teacher_vq.attnvq.num_heads, teacher_vq.attnvq.r
            v_q_heads = v_q.reshape(B, C * P_coarse, H_vq, r_vq)
            
            # Map discrete indices back to subspace vectors via Teacher's A matrix
            A_per_head = teacher_vq.attnvq.A.view(-1, H_vq, r_vq) 
            z_q_heads = torch.einsum('bchr, dhr -> bchd', v_q_heads, A_per_head)
            
            # Decode to spectral domain
            p1_s, p2_s = teacher_vq.decoder(z_q_heads)
            
            # Reconstruct to time domain
            student_recon = teacher_vq.reconstruct(p1_s, p2_s, n_samples=T_patch)
            student_recon = student_recon.reshape(B, C, P, T_patch)
            
            mse = F.mse_loss(student_recon, x_patches)
            
            metrics["acc"] += acc_total.item()
            metrics["mse"] += mse.item()
            metrics["loss"] += loss.item()
            metrics["distill_masked"] += loss_masked.item()
            metrics["distill_visible"] += loss_visible.item()
            
            pbar.set_postfix({'L': f"{loss.item():.4f}", 'Acc': f"{acc_total.item():.4f}", 'MSE': f"{mse.item():.4f}"})
            
    N_batches = len(data_loader)
    return {k: v/N_batches for k, v in metrics.items()}, batch

def main():
    parser = argparse.ArgumentParser(description='Masked Distillation Pretraining')
    parser.add_argument('--config', type=str, default='config/config.json')
    parser.add_argument('--teacher_ckpt', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f: 
        config = json.load(f)
        
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
    shutil.copy(args.config, os.path.join(artifact_dir, 'config_pretrain.json'))
    
    # --- Dataset Setup ---
    dataset_params = config['dataset_params']
    random.seed(42)
    train_config = copy.deepcopy(config)
    val_config = copy.deepcopy(config)
    
    for ds_name, ds_args in dataset_params.items():
        all_subjects = list(ds_args['subject_to_use'])
        random.shuffle(all_subjects)
        n_train = int(len(all_subjects) * train_params.get('train_val_split', 0.9))
        train_config['dataset_params'][ds_name]['subject_to_use'] = all_subjects[:n_train]
        val_config['dataset_params'][ds_name]['subject_to_use'] = all_subjects[n_train:]
    
    train_dataset = build_dataset_from_config(train_config, transform=None, mode='pretrain')
    val_dataset = build_dataset_from_config(val_config, transform=None, mode='pretrain')
    
    batch_size = train_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    teacher = build_model_from_config(config).to(device)
    student = build_backbone_from_config(config).to(device)
    
    # Load teacher weights
    teacher_state_dict = torch.load(teacher_ckpt, map_location='cpu')['model_state_dict']
    teacher.load_state_dict(teacher_state_dict)
    teacher.eval()
    for p in teacher.parameters(): 
        p.requires_grad = False
        
    # Initialize student with teacher's encoder weights
    with torch.no_grad():
        student.load_tokenizer_weights(teacher_state_dict)
    
    optimizer = optim.AdamW(student.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=train_params['min_learning_rate'])
    
    plotter = Plotter(output_dir=vis_dir)
    best_val_loss = float('inf')
    total_epochs = train_params['epochs']
    
    logger.info(f"Starting Pretraining for {total_epochs} epochs...")
    
    for epoch in range(1, total_epochs + 1):
        train_metrics, _ = train_one_epoch(teacher, student, train_loader, optimizer, device, epoch)
        val_metrics, val_last_batch = validate_one_epoch(teacher, student, val_loader, device)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}/{total_epochs}:")
        logger.info(f"  > Train [Loss:{train_metrics['loss']:.4f}, Acc:{train_metrics['acc']:.4f}, MSE:{train_metrics['mse']:.4f}]")
        logger.info(f"  > Val   [Loss:{val_metrics['loss']:.4f}, Acc:{val_metrics['acc']:.4f}, MSE:{val_metrics['mse']:.4f}]")
        
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot(filename='pretrain_curves.png')
        plotter.plot_metrics(filename='pretrain_metrics.png') 
        
        if epoch % 5 == 0:
            visualize_masked_reconstruction(val_last_batch, teacher, student, epoch, channel_names=val_dataset.base_dataset.channel_names, output_dir=os.path.join(vis_dir, 'reconstruction_pretrain'))
            
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({'model_state_dict': student.state_dict()}, os.path.join(checkpoint_dir, 'best_backbone.pth'))
            logger.info("  > Saved Best Backbone")
            
    logger.info("Pretraining Complete.")

if __name__ == '__main__':
    main()
