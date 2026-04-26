
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
            
            # Extract and flatten single channel
            orig = orig_batch[0, ch_idx].detach().cpu().numpy().flatten()
            recon = recon_batch[0, ch_idx].detach().cpu().numpy().flatten()
            
            # Ensure recon matches orig length if they differ due to padding/truncation
            min_len = min(len(orig), len(recon))
            orig, recon = orig[:min_len], recon[:min_len]
            t_vec = np.arange(min_len) / fs
            
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

            ax.plot(t_vec, y_o, 'k', alpha=0.6, linewidth=0.8, label='Orig')
            ax.plot(t_vec, y_r, 'r--', alpha=0.7, linewidth=0.8, label='Rec')
            
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

def visualize_masked_reconstruction(batch, teacher, student, epoch, channel_names=None, output_dir='output/visualization/reconstruction_pretrain'):
    """
    Plots Original vs Student-Reconstructed for pretraining.
    Layout: 6 Bands (Rows) x 8 Patches (4 Visible, 4 Masked).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(student.parameters()).device
    
    # Unpack batch
    x_patches, coords, mask, time_indices, _ = [t.to(device) for t in batch]
    B, C, P, T_patch = x_patches.shape
    Tokens = C * P
    
    with torch.no_grad():
        # 1. Student Classification -> Teacher Decoder (Student Reconstruction)
        # student returns: (B, Tokens, total_sub_dim, num_discrete)
        logits = student(x_patches, coords, time_indices=time_indices, bool_masked_pos=mask)
        
        # Get indices from student logits (B, Tokens, total_sub_dim)
        student_indices = logits.argmax(dim=-1)
        
        # Convert indices to discrete values
        N = student.num_discrete
        half_range = (N - 1) / 2.0
        v_q = student_indices.float() - half_range # (B, Tokens, total_sub_dim)
        
        # Map to embedding space per head (Weight-Tying)
        H, r = teacher.attnvq.num_heads, teacher.attnvq.r
        v_q_heads = v_q.reshape(B, Tokens, H, r)
        # Reshape to (B, C, P, H, r) -> (B*P, C, H, r)
        v_q_reshaped = v_q_heads.reshape(B, C, P, H, r).permute(0, 2, 1, 3, 4).reshape(B * P, C, H, r)
        
        A_per_head = teacher.attnvq.A.view(-1, H, r) # (D, H, r)
        z_q_heads = torch.einsum('bchr, dhr -> bchd', v_q_reshaped, A_per_head)
        
        pred_real, pred_imag = teacher.decoder(z_q_heads)
        student_recon = teacher.reconstruct(pred_real, pred_imag, n_samples=T_patch)
        # Reshape student_recon back to 4D (B, P, C, T_patch) to match x_patches trial-wise
        student_recon = student_recon.reshape(B, P, C, T_patch)
        
    # --- Visualization Setup ---
    fs = 200.0
    time_vec = np.arange(T_patch) / fs
    bands = {
        'Raw': None,
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-80)': (30, 80)
    }

    # Select 4 masked and 4 visible patches from the first trial
    masked_list = []
    visible_list = []
    m_trial = mask[0].reshape(C, P)
    for c in range(C):
        for p in range(P):
            if m_trial[c, p] and len(masked_list) < 4:
                masked_list.append((c, p))
            elif not m_trial[c, p] and len(visible_list) < 4:
                visible_list.append((c, p))
        if len(masked_list) >= 4 and len(visible_list) >= 4:
            break
            
    display_samples = visible_list + masked_list # 4 Visible, 4 Masked
    if not display_samples: return ""

    n_rows = len(bands)
    n_cols = len(display_samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5), sharex=True)
    fig.suptitle(f"Masked Reconstruction Analysis (Epoch {epoch})", fontsize=20, fontweight='bold')

    for row_idx, (band_name, freqs) in enumerate(bands.items()):
        for col_idx, (c_idx, p_idx) in enumerate(display_samples):
            ax = axes[row_idx, col_idx]
            
            orig = x_patches[0, c_idx, p_idx].cpu().numpy()
            recon = student_recon[0, p_idx, c_idx].cpu().numpy()
            
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

            is_masked = m_trial[c_idx, p_idx].item()
            status = "MASKED" if is_masked else "VISIBLE"
            color = 'r' if is_masked else 'g'
            
            ax.plot(time_vec, y_o, 'k', alpha=0.6, linewidth=1.0, label='Orig')
            ax.plot(time_vec, y_r, 'r--', alpha=0.8, linewidth=1.0, label='Rec')
            
            if row_idx == 0:
                ch_name = channel_names[c_idx] if channel_names else f"Ch{c_idx}"
                ax.set_title(f"{ch_name} P{p_idx}\n[{status}]", fontsize=12, fontweight='bold', color=color)
            
            if col_idx == 0:
                ax.set_ylabel(band_name, fontsize=12, fontweight='bold', rotation=0, labelpad=40)
            
            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(loc='upper right', fontsize=8)
            
            ax.grid(True, alpha=0.2)

    for ax in axes[-1, :]: ax.set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, f'pretrain_recon_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
    return save_path
