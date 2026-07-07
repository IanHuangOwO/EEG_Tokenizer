"""
Training-time visual callbacks and loss curve tracker.
Called from train_tokenizer.py and train_pretrain.py at fixed epoch intervals.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from viz.draw import _band_filter


# ── Loss curve tracker ────────────────────────────────────────────────────────

class Plotter:
    def __init__(self, output_dir='output/visualization/training_curves'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {'train': {}, 'val': {}}

    def update(self, train_metrics, val_metrics=None):
        for k, v in train_metrics.items():
            if k not in self.history['train']:
                self.history['train'][k] = []
            self.history['train'][k].append(v)
        if val_metrics:
            for k, v in val_metrics.items():
                if k not in self.history['val']:
                    self.history['val'][k] = []
                self.history['val'][k].append(v)

    def plot(self, filename=None):
        self.plot_all()

    def plot_metrics(self, filename=None):
        pass  # merged into plot_all

    def plot_all(self, filename='training_dashboard.png'):
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return

        src_tr  = self.history['train']
        src_val = self.history['val']

        def _ep(d): return range(1, len(d) + 1)
        def _t(k):  return src_tr.get(k)
        def _v(k):  return src_val.get(k)

        fig, axes = plt.subplots(3, 3, figsize=(24, 15), constrained_layout=True)
        fig.suptitle('Training Dashboard', fontsize=14, fontweight='bold')
        (ax_loss,    ax_masked,   ax_unmasked,
         ax_ppl,     ax_ste,      ax_hcos,
         ax_fpsim,   ax_fpstd,    ax_router) = axes.flat

        # ── Row 0: Loss curves ────────────────────────────────────────────────
        tr_loss = _t('loss')
        if tr_loss:
            ax_loss.plot(_ep(tr_loss), tr_loss, 'b-', label='Train')
        vl_loss = _v('loss')
        if vl_loss:
            ax_loss.plot(_ep(vl_loss), vl_loss, 'b--', alpha=0.7, label='Val')
        ax_loss.set_title('Total Loss'); ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize='x-small'); ax_loss.grid(True)

        if _t('masked'): ax_masked.plot(_ep(_t('masked')), _t('masked'), color='crimson', label='Train')
        if _v('masked'): ax_masked.plot(_ep(_v('masked')), _v('masked'), color='crimson', ls='--', alpha=0.7, label='Val')
        ax_masked.set_title('Masked MSE'); ax_masked.set_ylabel('Loss')
        ax_masked.legend(fontsize='x-small'); ax_masked.grid(True)

        if _t('unmasked'): ax_unmasked.plot(_ep(_t('unmasked')), _t('unmasked'), color='steelblue', label='Train')
        if _v('unmasked'): ax_unmasked.plot(_ep(_v('unmasked')), _v('unmasked'), color='steelblue', ls='--', alpha=0.7, label='Val')
        ax_unmasked.set_title('Unmasked MSE'); ax_unmasked.set_ylabel('Loss')
        ax_unmasked.legend(fontsize='x-small'); ax_unmasked.grid(True)

        # ── Row 1: Codebook metrics ───────────────────────────────────────────
        # [1,0] Codebook perplexity
        ppl = _v('codebook_perplexity')
        if ppl:
            ax_ppl.plot(_ep(ppl), ppl, color='green', label='Perplexity')
            ax_ppl.set_ylabel('Perplexity')
            ax_ppl.legend(fontsize='x-small')
        else:
            ax_ppl.axis('off')
        ax_ppl.set_title('Codebook Perplexity'); ax_ppl.grid(True)

        # [1,1] STE gap
        ste = _v('codebook_ste_gap')
        if ste:
            ax_ste.plot(_ep(ste), ste, color='red', label='STE gap')
            ax_ste.set_ylabel('STE gap')
            ax_ste.legend(fontsize='x-small')
        else:
            ax_ste.axis('off')
        ax_ste.set_title('Codebook STE Gap'); ax_ste.grid(True)

        # [1,2] Head projection cosine similarity (lower = more diverse heads)
        hcos = _t('head_cosine_sim')
        if hcos:
            ax_hcos.plot(_ep(hcos), hcos, color='darkorchid', label='Mean |cosine sim|')
            ax_hcos.set_ylabel('Mean |cosine sim|')
            ax_hcos.legend(fontsize='x-small')
        else:
            ax_hcos.axis('off')
        ax_hcos.set_title('Head Projection Diversity\n(lower = more diverse)'); ax_hcos.grid(True)

        # ── Row 2: Head specialization metrics ───────────────────────────────
        # [2,0] Fingerprint mean cosine sim — monitoring only (not a loss term)
        fp_sim_mean = _t('fp_sim_mean')
        if fp_sim_mean and any(v > 0 for v in fp_sim_mean):
            ax_fpsim.plot(_ep(fp_sim_mean), fp_sim_mean, color='darkorange', label='Mean |fp cosine sim|')
            ax_fpsim.set_ylabel('Mean |cosine sim|')
            ax_fpsim.legend(fontsize='x-small')
        else:
            ax_fpsim.axis('off')
        ax_fpsim.set_title('Head Fingerprint Similarity [monitor]\n(lower = more diverse)'); ax_fpsim.grid(True)

        # [2,1] Fingerprint sim std — monitoring only
        fp_sim_std = _t('fp_sim_std')
        if fp_sim_std and any(v > 0 for v in fp_sim_std):
            ax_fpstd.plot(_ep(fp_sim_std), fp_sim_std, color='purple', label='Std fp cosine sim')
            ax_fpstd.set_ylabel('Std cosine sim')
            ax_fpstd.legend(fontsize='x-small')
        else:
            ax_fpstd.axis('off')
        ax_fpstd.set_title('Fingerprint Sim Std [monitor]\n(higher = varied specialization)'); ax_fpstd.grid(True)

        # [2,2] Router health: entropy (higher = balanced) + load std + lb_loss (lower = balanced)
        r_ent = _t('router_entropy')
        r_std = _t('router_load_std')
        r_lb  = _t('lb_loss')
        ax2 = ax_router.twinx()
        if r_ent:
            ax_router.plot(_ep(r_ent), r_ent, color='teal', label='Router entropy')
            ax_router.set_ylabel('Entropy (higher=balanced)', color='teal')
        if r_std:
            ax2.plot(_ep(r_std), r_std, color='salmon', ls='--', label='Load std')
            ax2.set_ylabel('Load std / LB loss', color='salmon')
        if r_lb:
            ax2.plot(_ep(r_lb), r_lb, color='peru', ls=':', label='LB loss (1.0=uniform)')
        _merge_legends(ax_router, ax2)
        ax_router.set_title('Router Health'); ax_router.grid(True)

        for ax in axes.flat:
            ax.set_xlabel('Epoch')

        fig.savefig(os.path.join(self.output_dir, filename), dpi=110, bbox_inches='tight')
        plt.close(fig)


def _merge_legends(ax1, ax2):
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc='upper left', fontsize='x-small')


# ── Tokenizer training snapshot ───────────────────────────────────────────────

def visualize_reconstruction(train_batch, val_batch, epoch,
                             output_dir='output/visualization/reconstruction',
                             channel_names=None,
                             subject_id=None, trial_idx=None,
                             mask=None, patch_len=100):
    """
    Band-filtered orig vs recon for all channels of one val sample.
    Rows: channels. Cols: Raw / Delta / Theta / Alpha / Beta / Gamma.
    Masked patches highlighted in red per channel.
    mask: [C, N] bool numpy array or None.
    """
    os.makedirs(output_dir, exist_ok=True)

    val_orig, val_recon = val_batch
    if val_orig is None:
        return

    fs    = 200.0
    orig  = val_orig[0].detach().cpu().numpy()
    recon = val_recon[0].detach().cpu().numpy()
    C = orig.shape[0]
    n = min(orig.shape[-1], recon.shape[-1])
    orig, recon = orig[:, :n], recon[:, :n]
    t = np.arange(n) / fs

    bands = {
        'Raw':            None,
        'Delta (0.5-4)':  (0.5,  4),
        'Theta (4-8)':    (4,    8),
        'Alpha (8-13)':   (8,   13),
        'Beta (13-30)':   (13,  30),
        'Gamma (30-80)':  (30,  80),
    }

    n_rows, n_cols = C, len(bands)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 1.2),
                             sharex=True)
    if C == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Reconstruction (Val) — Epoch {epoch}",
                 fontsize=14, fontweight='bold')

    for col, (band_name, freqs) in enumerate(bands.items()):
        for row in range(C):
            ax = axes[row, col]
            yo, yr = _band_filter(orig[row], recon[row], freqs, fs)
            ax.plot(t, yo, color='#666666', lw=0.5, alpha=0.7)
            ax.plot(t, yr, 'r--', lw=0.5, alpha=0.8)
            # shade masked patches
            if mask is not None:
                ch_mask = mask[row] if mask.ndim == 2 else mask  # [N]
                for p_idx, is_masked in enumerate(ch_mask):
                    if is_masked:
                        t0 = p_idx * patch_len / fs
                        t1 = (p_idx + 1) * patch_len / fs
                        ax.axvspan(t0, t1, color='red', alpha=0.15, linewidth=0)
            ax.set_yticks([])
            ax.grid(True, alpha=0.08)
            if row == 0:
                ax.set_title(band_name, fontsize=8, fontweight='bold')
            if col == 0:
                ch_label = channel_names[row] if channel_names else f'Ch {row}'
                ax.set_ylabel(ch_label, fontsize=5, rotation=0, labelpad=28, va='center')
            if row < C - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)", fontsize=6)

    plt.tight_layout()
    prefix   = f"sub{subject_id}_trial{trial_idx}_" if subject_id is not None else ""
    ep_tag   = f"ep{epoch:04d}_" if epoch is not None else ""
    path = os.path.join(output_dir, f"{prefix}{ep_tag}recon_signal.png")
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    return path


# ── Pretrain masked-reconstruction snapshot ───────────────────────────────────

def visualize_masked_reconstruction(batch, model, epoch,
                                    channel_names=None,
                                    output_dir='output/visualization/reconstruction_pretrain'):
    """
    Orig vs model-reconstructed for pretraining.
    Rows: 6 bands. Cols: 4 visible + 4 masked patches from the first trial.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = next(model.parameters()).device

    x_patches, coords, mask, time_indices, _, _ = [t.to(device) for t in batch]
    B, C, P, T_patch = x_patches.shape

    with torch.no_grad():
        recon, _, _, _, _ = model(x_patches, coords, time_idx=time_indices, bool_masked_pos=None)
        student_recon = recon.permute(0, 2, 1, 3)  # [B, P, C, T_patch]

    fs = 200.0
    bands = {
        'Raw':            None,
        'Delta (0.5-4)':  (0.5,  4),
        'Theta (4-8)':    (4,    8),
        'Alpha (8-13)':   (8,   13),
        'Beta (13-30)':   (13,  30),
        'Gamma (30-80)':  (30,  80),
    }

    masked_list, visible_list = [], []
    m_trial = mask[0].reshape(C, P)
    for c in range(C):
        for p in range(P):
            if m_trial[c, p] and len(masked_list) < 4:
                masked_list.append((c, p))
            elif not m_trial[c, p] and len(visible_list) < 4:
                visible_list.append((c, p))
        if len(masked_list) >= 4 and len(visible_list) >= 4:
            break

    display = visible_list + masked_list
    if not display:
        return ""

    n_rows, n_cols = len(bands), len(display)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 2.5), sharex=True)
    fig.suptitle(f"Masked Reconstruction Analysis (Epoch {epoch})",
                 fontsize=20, fontweight='bold')
    t_vec = np.arange(T_patch) / fs

    for row, (band_name, freqs) in enumerate(bands.items()):
        for col, (ci, pi) in enumerate(display):
            ax    = axes[row, col]
            orig  = x_patches[0, ci, pi].cpu().numpy()
            recon = student_recon[0, pi, ci].cpu().numpy()
            yo, yr = _band_filter(orig, recon, freqs, fs)
            is_masked = m_trial[ci, pi].item()

            ax.plot(t_vec, yo, 'k',   alpha=0.6, linewidth=1.0, label='Orig')
            ax.plot(t_vec, yr, 'r--', alpha=0.8, linewidth=1.0, label='Rec')
            if row == 0:
                ch_name = channel_names[ci] if channel_names else f"Ch{ci}"
                status  = "MASKED" if is_masked else "VISIBLE"
                color   = 'r' if is_masked else 'g'
                ax.set_title(f"{ch_name} P{pi}\n[{status}]",
                             fontsize=12, fontweight='bold', color=color)
            if col == 0:
                ax.set_ylabel(band_name, fontsize=12, fontweight='bold',
                              rotation=0, labelpad=40)
            if row == 0 and col == n_cols - 1:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.2)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(output_dir, f'pretrain_recon_epoch_{epoch}.png')
    plt.savefig(path)
    plt.close()
    return path
