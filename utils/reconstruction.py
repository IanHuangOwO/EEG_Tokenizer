
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mne

def visualize_reconstruction(train_batch, val_batch, epoch, output_dir='output/visualization/reconstruction'):
    """
    Plots original vs reconstructed signal for 1 Train sample and 3 diverse Validation samples.
    Layout: 
    - Rows: Bands (Raw, Delta, Theta, Alpha, Beta, Gamma)
    - Cols: Train_Ch0, Val_Ch0, Val_ChMid, Val_ChLast
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack
    train_orig, train_recon = train_batch
    val_orig, val_recon = val_batch
    
    if train_orig is None or val_orig is None: return

    num_chans = train_orig.shape[1]
    mid_ch = num_chans // 2
    last_ch = num_chans - 1
    
    fs = 200.0
    time_vec = np.arange(train_orig.shape[-1]) / fs
    
    bands = {
        'Raw': None,
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-80)': (30, 80)
    }
    
    # Grid Setup: 6 Bands (Rows), 4 Samples (Cols)
    n_rows = len(bands)
    n_cols = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 2.5), sharex=True)
    fig.suptitle(f"Reconstruction Analysis - Epoch {epoch}", fontsize=20, fontweight='bold')
    
    # Define Column configurations: (Source_Tensor_Pair, Channel_Index, Label)
    configs = [
        ((train_orig, train_recon), 0, f"Train (Ch 0)"),
        ((val_orig, val_recon), 0, f"Val (Ch 0)"),
        ((val_orig, val_recon), mid_ch, f"Val (Ch {mid_ch})"),
        ((val_orig, val_recon), last_ch, f"Val (Ch {last_ch})")
    ]

    for row_idx, (band_name, freqs) in enumerate(bands.items()):
        for col_idx, ((orig_batch, recon_batch), ch_idx, title) in enumerate(configs):
            ax = axes[row_idx, col_idx]
            
            # Extract single channel
            orig = orig_batch[0, ch_idx].detach().cpu().numpy()
            recon = recon_batch[0, ch_idx].detach().cpu().numpy()
            
            # Filter
            if freqs is None:
                y_o, y_r = orig, recon
            else:
                l_f, h_f = freqs
                o_mne = orig.reshape(1, -1).astype(np.float64)
                r_mne = recon.reshape(1, -1).astype(np.float64)
                try:
                    y_o = mne.filter.filter_data(o_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                    y_r = mne.filter.filter_data(r_mne, fs, l_f, h_f, method='iir', verbose=False)[0]
                except:
                    y_o, y_r = np.zeros_like(orig), np.zeros_like(recon)

            ax.plot(time_vec, y_o, 'k', alpha=0.6, linewidth=0.8, label='Orig')
            ax.plot(time_vec, y_r, 'r--', alpha=0.7, linewidth=0.8, label='Rec')
            
            # Formatting
            if row_idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(band_name, fontsize=12, rotation=0, labelpad=40, fontweight='bold')
            
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(fontsize=10)
                
            ax.grid(True, alpha=0.2)

    # X-Labels on bottom
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    filename = f'recon_epoch_{epoch}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    return save_path

def visualize_masked_reconstruction(batch, teacher, student, epoch, output_dir='output/visualization/reconstruction_pretrain'):
    """
    Plots Original vs Masked vs Student-Reconstructed for pretraining.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(student.parameters()).device
    
    # Unpack batch
    x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
    B, C, P, T_patch = x_patches.shape
    Tokens = C * P
    
    with torch.no_grad():
        # 1. Teacher Ground Truth (Full Reconstruction)
        p1, p2, p3, _, _, _ = teacher(x_patches, coords, time_idx=None)
        teacher_recon = teacher.reconstruct(p1, p2, p3, n_samples=T_patch)
        
        # 2. Student Classification -> Teacher Decoder (Student Reconstruction)
        # student returns: (B, Tokens, total_sub_dim, num_discrete)
        logits = student(x_patches, coords, time_indices=time_indices, bool_masked_pos=mask)
        
        # Get indices from student logits (B, Tokens, total_sub_dim)
        student_indices = logits.argmax(dim=-1)
        
        # Convert indices to discrete values
        N = student.num_discrete
        half_range = (N - 1) / 2.0
        v_q = student_indices.float() - half_range # (B, Tokens, total_sub_dim)
        
        # Map to embedding space using frozen Matrix A.t() from teacher (Weight-Tying)
        # v_q shape: (B, Tokens, total_sub_dim) -> (B*Tokens, Hr)
        v_q_flat = v_q.reshape(B * Tokens, -1)
        # Weight tying: Decoding uses A.t()
        z_q = torch.matmul(v_q_flat, teacher.attnvq.A.t()).reshape(B, Tokens, -1)
        
        # Reshape z_q to teacher decoder format: (B*P, C, D)
        # Backbone tokens are flattened from (B, C, P, D) to (B, C*P, D)
        # We need to reshape back to (B, C, P, D) then permute to (B, P, C, D)
        z_q_reshaped = z_q.reshape(B, C, P, -1).permute(0, 2, 1, 3).reshape(B * P, C, -1)
        
        pred_amp, pred_sin, pred_cos = teacher.decoder(z_q_reshaped)
        student_recon = teacher.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=T_patch)
        # Reshape student_recon back to 4D (B, P, C, T_patch) to match x_patches trial-wise
        student_recon = student_recon.reshape(B, P, C, T_patch)
        
    # Pick first trial, first 8 patches
    fig, axes = plt.subplots(8, 3, figsize=(15, 24))
    fig.suptitle(f"Masked Reconstruction (Epoch {epoch}) - Red=Masked", fontsize=16)
    
    fs = 200.0
    time_vec = np.arange(T_patch) / fs
    
    # Select 4 masked and 4 visible patches from the first trial (searching across all channels)
    masked_list = []
    visible_list = []
    
    # Flatten mask to search efficiently: (C, P)
    m_trial = mask[0].reshape(C, P)
    for c in range(C):
        for p in range(P):
            if m_trial[c, p] and len(masked_list) < 4:
                masked_list.append((c, p))
            elif not m_trial[c, p] and len(visible_list) < 4:
                visible_list.append((c, p))
        if len(masked_list) >= 4 and len(visible_list) >= 4:
            break
            
    display_samples = masked_list + visible_list
    
    for i, (c_idx, p_idx) in enumerate(display_samples):
        # Original
        ax_orig = axes[i, 0]
        ax_orig.plot(time_vec, x_patches[0, c_idx, p_idx].cpu(), 'k', label='Original')
        ax_orig.set_title(f"Ch{c_idx} P{p_idx} Orig")
        
        # Masked view
        ax_mask = axes[i, 1]
        is_masked = m_trial[c_idx, p_idx].item()
        color = 'r' if is_masked else 'g'
        
        # Show actual signal if visible, or zeros if masked
        plot_data = x_patches[0, c_idx, p_idx].cpu()
        if is_masked:
            plot_data = torch.zeros_like(plot_data)
            
        ax_mask.plot(time_vec, plot_data, color, alpha=0.5)
        ax_mask.set_title(f"Input (Masked={is_masked})")
        
        # Student Reconstruction
        ax_rec = axes[i, 2]
        ax_rec.plot(time_vec, x_patches[0, c_idx, p_idx].cpu(), 'k', alpha=0.3, label='GT')
        # student_recon shape: (B, P, C, T)
        ax_rec.plot(time_vec, student_recon[0, p_idx, c_idx].cpu(), 'b--', label='Student')
        ax_rec.set_title(f"Recon (Masked={is_masked})")
        
        if i == 0:
            ax_orig.legend()
            ax_rec.legend()

    # Hide unused rows if we found fewer than 8 samples
    for j in range(len(display_samples), 8):
        for k in range(3):
            axes[j, k].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f'pretrain_recon_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path
