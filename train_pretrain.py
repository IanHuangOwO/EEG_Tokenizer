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
from einops import rearrange

from IO.dataset import build_dataset_from_config
from model.factory import build_model_from_config, build_backbone_from_config, build_preprocessing_from_config
from utils.plotter import Plotter

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

def get_weighted_kl_loss(student_logits, teacher_weights, gate_weights, mask):
    """
    student_logits: (S, H, B, T, r)
    teacher_weights: (S, H, B, T, r)
    gate_weights: (S, H)
    mask: (B, T)
    """
    S, H, B, T, r = student_logits.shape
    
    # log_p: (S, H, B, T, r)
    log_p = F.log_softmax(student_logits, dim=-1)
    
    # kl: (S, H, B, T)
    # F.kl_div(log_p, target) computes target * (log(target) - log_p)
    # We want target * (log(target) - log(student)) but since teacher_weights are probabilities,
    # we use F.kl_div with log_target=False
    kl = F.kl_div(log_p, teacher_weights, reduction='none').sum(-1)
    
    # gate_weights: (S, H, 1, 1)
    weighted_kl = kl * gate_weights.view(S, H, 1, 1)
    
    # mask expanded to (S, H, B, T)
    mask_exp = mask.view(1, 1, B, T).expand(S, H, -1, -1)
    
    # Loss only on masked tokens (standard MAE)
    # We can also compute on all tokens if desired
    loss = (weighted_kl * mask_exp).sum() / (mask_exp.sum() + 1e-6)
    
    # Also track unmasked loss for monitoring
    with torch.no_grad():
        unmasked_loss = (weighted_kl * (1 - mask_exp.float())).sum() / ((1 - mask_exp.float()).sum() + 1e-6)
    
    return loss, unmasked_loss

def train_one_epoch(teacher, student, data_loader, optimizer, device, epoch):
    teacher.eval()
    student.train()
    
    metrics = {"loss": 0.0, "unmasked_kl": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    # Get gate weights from teacher once (constant)
    with torch.no_grad():
        gate_weights = F.softmax(teacher.attnvq.head_weights, dim=3).squeeze() # (S, H)
        if gate_weights.dim() == 1: # Handle single scale case
            gate_weights = gate_weights.unsqueeze(0)

    for batch in pbar:
        x_patches, coords, mask, _ = [t.to(device) for t in batch]
        # x_patches: (B, C, P, T)
        # coords: (B, C, 3)
        # mask: (B, C*P)
        
        B, C, P, T_patch = x_patches.shape
        
        # 1. Teacher Forward (Targets)
        # Teacher expects (B_sz, C, T) where B_sz is B*P
        x_teacher = rearrange(x_patches, 'b c p t -> (b p) c t')
        # We need to repeat coords for each patch
        coords_teacher = coords.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, C, 3)
        
        with torch.no_grad():
            # AttnVQTokenizer returns (pred_amp, pred_sin, pred_cos, vq_loss, top_k_indices, weights)
            *_, teacher_weights = teacher(x_teacher, coords_teacher)
            # teacher_weights: (S, BP, C, H, r)
        
        # 2. Student Forward (Predictions)
        optimizer.zero_grad()
        student_logits = student(x_patches, coords, bool_masked_pos=mask) # (S, H, B, C*P, r)
        
        # 3. Align Teacher Weights to (S, H, B, C*P, r)
        S, BP, C, H, r = teacher_weights.shape
        # (S, BP, C, H, r) -> (S, B, P, C, H, r) -> (S, H, B, C, P, r) -> (S, H, B, CP, r)
        teacher_weights = teacher_weights.view(S, B, P, C, H, r).permute(0, 4, 1, 3, 2, 5).reshape(S, H, B, C*P, r)
        
        # 4. Compute Loss
        loss, unmasked_kl = get_weighted_kl_loss(student_logits, teacher_weights, gate_weights, mask)
        
        loss.backward()
        optimizer.step()
        
        metrics["loss"] += loss.item()
        metrics["unmasked_kl"] += unmasked_kl.item()
        
        pbar.set_postfix({'L': f"{loss.item():.4f}", 'U_KL': f"{unmasked_kl.item():.4f}"})

    N = len(data_loader)
    return {k: v/N for k, v in metrics.items()}

def validate_one_epoch(teacher, student, data_loader, device):
    teacher.eval()
    student.eval()
    
    metrics = {"loss": 0.0, "unmasked_kl": 0.0}
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation", 
                bar_format='{desc}: {percentage:3.0f}%|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    with torch.no_grad():
        gate_weights = F.softmax(teacher.attnvq.head_weights, dim=3).squeeze()
        if gate_weights.dim() == 1: gate_weights = gate_weights.unsqueeze(0)

        for batch in pbar:
            x_patches, coords, mask, _ = [t.to(device) for t in batch]
            B, C, P, T_patch = x_patches.shape
            
            x_teacher = rearrange(x_patches, 'b c p t -> (b p) c t')
            coords_teacher = coords.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, C, 3)
            
            *_, teacher_weights = teacher(x_teacher, coords_teacher)
            student_logits = student(x_patches, coords, bool_masked_pos=mask)
            
            S, BP, C, H, r = teacher_weights.shape
            teacher_weights = teacher_weights.view(S, B, P, C, H, r).permute(0, 4, 1, 3, 2, 5).reshape(S, H, B, C*P, r)
            
            loss, unmasked_kl = get_weighted_kl_loss(student_logits, teacher_weights, gate_weights, mask)
            
            metrics["loss"] += loss.item()
            metrics["unmasked_kl"] += unmasked_kl.item()
            pbar.set_postfix({'L': f"{loss.item():.4f}"})

    N = len(data_loader)
    return {k: v/N for k, v in metrics.items()}

def main():
    parser = argparse.ArgumentParser(description='EEG Backbone Masked Pretraining')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to config file')
    parser.add_argument('--teacher_ckpt', type=str, required=True, help='Path to pretrained teacher checkpoint')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_params = config['training_params']
    model_name = train_params.get('model_name', 'default_pretrain')
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    output_dir = f"output/{model_name}_pretrain"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    artifact_dir = os.path.join(output_dir, "artifacts")
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = setup_logger(artifact_dir)
    shutil.copy(args.config, os.path.join(artifact_dir, 'config.json'))
    
    # 2. Dataset Setup
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    transform = build_preprocessing_from_config(config)
    
    all_subjects = config['dataset_params']['subjects']
    random.seed(42)
    shuffled_subjects = list(all_subjects)
    random.shuffle(shuffled_subjects)
    
    split_ratio = train_params.get('train_val_split', 0.9)
    n_train = int(len(shuffled_subjects) * split_ratio)
    train_subjects = shuffled_subjects[:n_train]
    val_subjects = shuffled_subjects[n_train:]
    
    train_config = copy.deepcopy(config)
    val_config = copy.deepcopy(config)
    train_config['dataset_params']['subjects'] = train_subjects
    val_config['dataset_params']['subjects'] = val_subjects
    
    logger.info("Building Datasets (Pretrain Mode)...")
    train_dataset = build_dataset_from_config(train_config, transform=transform, mode='pretrain')
    val_dataset = build_dataset_from_config(val_config, transform=transform, mode='pretrain')
    
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 3. Models Setup
    logger.info("Initializing Teacher (Tokenizer)...")
    teacher = build_model_from_config(config)
    ckpt = torch.load(args.teacher_ckpt, map_location='cpu')
    teacher.load_state_dict(ckpt['model_state_dict'])
    teacher.to(device)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()
    
    logger.info("Initializing Student (Backbone)...")
    student = build_backbone_from_config(config)
    student.to(device)
    
    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(student.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params['epochs'], eta_min=train_params['min_learning_rate'])
    
    plotter = Plotter(output_dir=vis_dir)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, train_params['epochs'] + 1):
        train_metrics = train_one_epoch(teacher, student, train_loader, optimizer, device, epoch)
        val_metrics = validate_one_epoch(teacher, student, val_loader, device)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}/{train_params['epochs']}:")
        logger.info(f"  > Train [Loss:{train_metrics['loss']:.4f}, Unmasked_KL:{train_metrics['unmasked_kl']:.4f}]")
        logger.info(f"  > Val   [Loss:{val_metrics['loss']:.4f}, Unmasked_KL:{val_metrics['unmasked_kl']:.4f}]")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({'model_state_dict': student.state_dict()}, os.path.join(checkpoint_dir, 'best_backbone.pth'))
            logger.info("  > Saved Best Backbone")
            
        plotter.update(train_metrics=train_metrics, val_metrics=val_metrics)
        plotter.plot()
        plotter.plot_metrics()

    logger.info("Pretraining Complete.")

if __name__ == '__main__':
    main()
