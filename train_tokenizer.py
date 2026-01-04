import os
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from IO.dataset import build_dataset_from_config
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer
from model.LaBraM.modeling_tokenizer import LaBraMTokenizer
from model.NeuroRVQ.preprocessing import NeuroRVQProcessing
from utils.reconstruction import visualize_reconstruction
from utils.plotter import Plotter

class TokenizerWrapperDataset(Dataset):
    """
    Wraps the standard EEGDataset to yield 1-second patches (200 samples)
    instead of full trials. Returns coordinates directly.
    """
    def __init__(self, base_dataset, patch_len=200):
        self.base_dataset = base_dataset
        self.patch_len = patch_len
        
        sample_x, _ = self.base_dataset[0]
        self.total_len = sample_x.shape[-1]
        self.patches_per_trial = self.total_len // patch_len
        
        # Pre-convert coords to tensor for efficiency
        self.coords_tensor = torch.from_numpy(base_dataset.coords).float()
        
        print(f"Tokenizer Dataset: {len(self.base_dataset)} trials, {self.patches_per_trial} patches each.")

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_trial

    def __getitem__(self, index):
        trial_idx = index // self.patches_per_trial
        patch_offset = (index % self.patches_per_trial) * self.patch_len
        
        x, _ = self.base_dataset[trial_idx]
        patch = x[:, patch_offset : patch_offset + self.patch_len]
        
        # Return full channel coordinates directly
        return patch, self.coords_tensor

def train_one_epoch(model, model_type, data_loader, optimizer, device, epoch, log_freq=10):
    model.train()
    total_loss = 0
    total_vq_loss = 0
    total_recon_loss = 0
    total_mse = 0
    # Add accumulators for loss components
    total_amp_loss = 0
    total_phase_loss = 0
    
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch}")
    
    # Track last batch for visualization
    last_batch_x = None
    last_x_recon = None

    for batch_x, batch_coords in pbar:
        batch_x = batch_x.to(device)
        batch_coords = batch_coords.to(device)
        
        # Forward & Loss
        if model_type == "NeuroRVQ":
            # Pass coordinates directly
            pred_amp, pred_sin, pred_cos, vq_loss = model(batch_x, batch_coords)
            recon_loss, l_amp, l_phase, l_temp = model.get_loss(batch_x, pred_amp, pred_sin, pred_cos)
            total_amp_loss += l_amp.item()
            total_phase_loss += l_phase.item()
        elif model_type == "LaBraM":
            pred_amp, pred_angle, vq_loss = model(batch_x)
            recon_loss = model.get_loss(batch_x, pred_amp, pred_angle)
            l_amp, l_phase, l_temp = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        loss = recon_loss + vq_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        loss_val = loss.item()
        vq_val = vq_loss.item()
        recon_val = recon_loss.item()
        
        total_loss += loss_val
        total_vq_loss += vq_val
        total_recon_loss += recon_val
        
        # Compute MSE for monitoring
        with torch.no_grad():
            if model_type == "NeuroRVQ":
                x_recon = model.reconstruct(pred_amp, pred_sin, pred_cos)
                batch_mse = l_temp.item()
                last_batch_x = batch_x
                last_x_recon = x_recon
            elif model_type == "LaBraM":
                batch_mse = 0.0 
            else:
                batch_mse = 0.0
        
        total_mse += batch_mse
        
        # Update Progress Bar
        current_lr = optimizer.param_groups[0]['lr']
        postfix = {'L': f"{loss_val:.2f}", 'VQ': f"{vq_val:.2f}", 'lr': f"{current_lr:.1e}"}
        if model_type == "NeuroRVQ":
            postfix.update({'Amp': f"{l_amp.item():.2f}", 'Phs': f"{l_phase.item():.2f}", 'Tmp': f"{l_temp.item():.2f}"})
        elif model_type == "LaBraM":
            postfix.update({'Recon': f"{recon_val:.2f}"})
            
        pbar.set_postfix(postfix)

    # End of Epoch Visualization
    if model_type == "NeuroRVQ" and last_batch_x is not None:
        visualize_reconstruction(last_batch_x, last_x_recon, epoch)

    avg_loss = total_loss / len(data_loader)
    avg_mse_epoch = total_mse / len(data_loader)
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  > LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"  > Avg Total Loss: {avg_loss:.4f}")
    print(f"  > Avg VQ Loss: {total_vq_loss / len(data_loader):.4f}")

    if model_type == "NeuroRVQ":
        avg_amp = total_amp_loss / len(data_loader)
        avg_phase = total_phase_loss / len(data_loader)
        # avg_mse_epoch is the temporal loss
        print(f"  > Avg Recon Loss: {total_recon_loss / len(data_loader):.4f} (Amp: {avg_amp:.4f}, Phs: {avg_phase:.4f}, Tmp: {avg_mse_epoch:.4f})")
    else:
        print(f"  > Avg Recon Loss: {total_recon_loss / len(data_loader):.4f}")
    
    return avg_loss, total_recon_loss / len(data_loader), total_vq_loss / len(data_loader), avg_mse_epoch

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
    model_type = train_params.get('model_name', 'NeuroRVQ')
    output_dir = f"output/checkpoints/tokenizer/{model_type.lower()}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Dataset
    dataset_path = config['dataset_params']['dataset_path']
    with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    config['data_metadata'] = meta['data_metadata']
    
    fs_orig = meta['data_metadata']['Sample_Frequency']
    transform = NeuroRVQProcessing(
        original_freq=fs_orig, 
        target_freq=preprocess_params['target_freq'], 
        l_freq=preprocess_params['l_freq'], 
        h_freq=preprocess_params['h_freq'], 
        normalization_type=preprocess_params['normalization_type']
    )
    base_dataset = build_dataset_from_config(config, transform=transform)
    
    tokenizer_dataset = TokenizerWrapperDataset(base_dataset, patch_len=preprocess_params['target_freq']) # patch_len = 1 sec
    data_loader = DataLoader(tokenizer_dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4)

    # 3. Initialize Model
    Nc = base_dataset.Nc
    print(f"Initializing {model_type} Tokenizer for {Nc} channels...")
    
    if model_type == "NeuroRVQ":
        params = model_params['NeuroRVQ']
        model = NeuroRVQTokenizer(
            in_chans=1,
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            enc_heads=params['enc_heads'],
            dec_depth=params['dec_depth'],
            vocab_size=params['vocab_size'],
            n_codebooks=params['num_codebooks'],
            freq_resolution=params.get('freq_resolution', 1.0),
            min_freq=params.get('min_freq', 0.0),
            max_freq=params.get('max_freq', 100.0),
            fs=preprocess_params['target_freq']
        )
    elif model_type == "LaBraM":
        params = model_params['LaBraM']
        model = LaBraMTokenizer(
            in_chans=1,
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            dec_depth=params['dec_depth'],
            n_code=params['vocab_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
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

    plotter = Plotter(output_dir=f"output/visualization/{model_type.lower()}_tokenizer_curves")

    # 5. Training Loop
    best_loss = float('inf')
    for epoch in range(train_params['epochs']):
        avg_loss, avg_recon, avg_vq, avg_mse = train_one_epoch(model, model_type, data_loader, optimizer, device, epoch)
        
        # Step the scheduler at the end of each epoch
        scheduler.step()
        
        plotter.update(avg_loss, avg_recon, avg_vq, avg_mse)
        plotter.plot()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, 'tokenizer_best.pth')
            torch.save(model.state_dict(), ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(output_dir, f'tokenizer_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
    
    print("Training Complete.")

if __name__ == '__main__':
    main()