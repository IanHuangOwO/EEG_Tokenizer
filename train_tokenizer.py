import os
import json
import argparse
import shutil
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from IO.dataset import build_dataset_from_config, TokenizerWrapperDataset
from model.factory import build_model_from_config
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing
from model.RecurrentVQ.preprocessing import RecurrentVQProcessing
from model.RecurrentFSQ.preprocessing import RecurrentVQProcessing as RecurrentFSQProcessing
from utils.reconstruction import visualize_reconstruction
from utils.plotter import Plotter

def train_one_epoch(model, model_type, data_loader, optimizer, device, epoch, output_dir, log_freq=10):
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_mse = 0
    
    last_batch_x = None
    last_x_recon = None
    
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}")
    
    for batch_x, batch_fft, batch_coords in pbar:
        batch_x = batch_x.to(device)
        batch_fft = batch_fft.to(device)
        batch_coords = batch_coords.to(device)
        
        # Forward & Loss
        if model_type in ["NeuroRVQ", "RecurrentVQ", "RecurrentFSQ"]:
            pred_amp, pred_sin, pred_cos, vq_loss = model(batch_x, batch_coords)
            recon_loss, _, _, l_temp = model.get_loss(batch_x, pred_amp, pred_sin, pred_cos, x_fft=batch_fft)
            total_amp_loss = 0 # placeholders removed for brevity in this refactor
            total_phase_loss = 0
        elif model_type == "LaBraM":
            pred_amp, pred_angle, vq_loss = model(batch_x)
            recon_loss = model.get_loss(batch_x, pred_amp, pred_angle)
            l_temp = torch.tensor(0.0) # Placeholder
            
        loss = recon_loss + vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics accumulation
        total_loss += loss.item()
        total_vq_loss += vq_loss.item()
        total_recon_loss += recon_loss.item()
        
        # MSE for monitoring (no_grad not strictly needed here as we detached, but good practice)
        with torch.no_grad():
            if model_type in ["NeuroRVQ", "RecurrentVQ", "RecurrentFSQ"]:
                x_recon = model.reconstruct(pred_amp, pred_sin, pred_cos)
                batch_mse = l_temp.item()
                last_batch_x = batch_x
                last_x_recon = x_recon
            else:
                batch_mse = 0.0
            total_mse += batch_mse
        
        # Update Progress Bar
        current_lr = optimizer.param_groups[0]['lr']
        postfix = {
            'L': f"{loss.item():.2f}", 
            'VQ': f"{vq_loss.item():.2f}", 
            'Recon': f"{recon_loss.item():.2f}",
            'lr': f"{current_lr:.1e}"
        }
        pbar.set_postfix(postfix)

    avg_loss = total_loss / len(data_loader)
    avg_recon = total_recon_loss / len(data_loader)
    avg_vq = total_vq_loss / len(data_loader)
    avg_mse = total_mse / len(data_loader)
    
    # Return as tuple matching validate_one_epoch metrics structure
    return (avg_loss, avg_recon, avg_vq, avg_mse), (last_batch_x, last_x_recon)

def validate_one_epoch(model, model_type, data_loader, device):
    model.eval()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_mse = 0
    
    last_batch_x = None
    last_x_recon = None
    
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation")
    
    with torch.no_grad():
        for batch_x, batch_fft, batch_coords in pbar:
            batch_x = batch_x.to(device)
            batch_fft = batch_fft.to(device)
            batch_coords = batch_coords.to(device)
            
            # Forward
            if model_type in ["NeuroRVQ", "RecurrentVQ", "RecurrentFSQ"]:
                pred_amp, pred_sin, pred_cos, vq_loss = model(batch_x, batch_coords)
                recon_loss, _, _, l_temp = model.get_loss(batch_x, pred_amp, pred_sin, pred_cos, x_fft=batch_fft)
            elif model_type == "LaBraM":
                pred_amp, pred_angle, vq_loss = model(batch_x)
                recon_loss = model.get_loss(batch_x, pred_amp, pred_angle)
                l_temp = torch.tensor(0.0) # Placeholder
                vq_loss = torch.tensor(0.0) # Placeholder if not returned
            
            loss = recon_loss + vq_loss
            
            total_loss += loss.item()
            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            
            # MSE / Reconstruct
            if model_type in ["NeuroRVQ", "RecurrentVQ", "RecurrentFSQ"]:
                x_recon = model.reconstruct(pred_amp, pred_sin, pred_cos)
                batch_mse = l_temp.item()
                last_batch_x = batch_x
                last_x_recon = x_recon
            else:
                batch_mse = 0.0
            
            total_mse += batch_mse
            
            # Update Progress Bar
            pbar.set_postfix({'L': f"{loss.item():.2f}", 'MSE': f"{batch_mse:.4f}"})

    avg_loss = total_loss / len(data_loader)
    avg_recon = total_recon_loss / len(data_loader)
    avg_vq = total_vq_loss / len(data_loader)
    avg_mse = total_mse / len(data_loader)
    
    return (avg_loss, avg_recon, avg_vq, avg_mse), (last_batch_x, last_x_recon)

def main():
    parser = argparse.ArgumentParser(description='EEG Tokenizer Training')
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to config file')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract params
    train_params = config['training_params']
    model_params = config['model_params']
    preprocess_params = config.get('preprocess_params', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })
    
    # Override from config
    device = train_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model_type = train_params.get('model_type', 'NeuroRVQ')
    model_name = train_params.get('model_name', 'default_run')
    output_dir = f"output/checkpoints/tokenizer/{model_name}"
    config_output_dir = f"output/config/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Copy config for reproducibility to config output dir
    shutil.copy(args.config, os.path.join(config_output_dir, 'config.json'))
    
    # 2. Dataset
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    fs_orig = meta['data_metadata']['Sample_Frequency']
    if model_type == 'RecurrentVQ':
        transform = RecurrentVQProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )
    elif model_type == 'RecurrentFSQ':
        transform = RecurrentFSQProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )
    else:
        transform = NeuroRVQProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )
    base_dataset = build_dataset_from_config(config, transform=transform)
    
    # Calculate n_fft for pre-computation
    # n_fft = fs / freq_res
    if model_type in ['NeuroRVQ', 'RecurrentVQ', 'RecurrentFSQ']:
        freq_res = model_params[model_type].get('freq_resolution', 1.0)
        n_fft = int(preprocess_params['target_freq'] / freq_res)
    else:
        n_fft = None

    tokenizer_dataset = TokenizerWrapperDataset(
        base_dataset, 
        patch_len=preprocess_params['target_freq'], 
        n_fft=n_fft
    )
    
    # Train/Val Split
    split_ratio = train_params.get('train_val_split', 0.9)
    train_size = int(split_ratio * len(tokenizer_dataset))
    val_size = len(tokenizer_dataset) - train_size
    train_dataset, val_dataset = random_split(tokenizer_dataset, [train_size, val_size])
    
    print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=8)

    # 3. Initialize Model
    Nc = base_dataset.Nc
    print(f"Initializing {model_type} Tokenizer for {Nc} channels (Run: {model_name})...")
    
    model = build_model_from_config(config)
    
    # Save the model architecture source file for reproducibility
    try:
        model_src_path = sys.modules[model.__module__].__file__
        shutil.copy(model_src_path, os.path.join(config_output_dir, 'modeling_tokenizer.py'))
        print(f"Saved model source to {config_output_dir}")
    except Exception as e:
        print(f"Warning: Could not save model source: {e}")
        
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

    vis_dir = f"output/visualization/{model_name}"
    plotter = Plotter(output_dir=vis_dir)

    # 5. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(train_params['epochs']):
        # Train
        train_metrics, train_last_batch = train_one_epoch(
            model, model_type, train_loader, optimizer, device, epoch, output_dir=vis_dir
        )
        
        # Validation
        val_metrics, val_last_batch = validate_one_epoch(model, model_type, val_loader, device)
        val_loss = val_metrics[0]
        
        # Update Plotter
        plotter.update(
            train_metrics=train_metrics,
            val_metrics=val_metrics
        )
        plotter.plot()
        
        print(f"  > Train Loss: {train_metrics[0]:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(output_dir, 'tokenizer_best.pth')
            torch.save(model.state_dict(), ckpt_path)
            print("  > Saved Best Model")
        
        # Periodic Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f'tokenizer_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            
        # Step Scheduler
        scheduler.step()
        
        recon_dir = os.path.join(vis_dir, 'reconstruction')
        visualize_reconstruction(train_last_batch, val_last_batch, epoch, output_dir=recon_dir)

    print("Training Complete.")

if __name__ == '__main__':
    main()
