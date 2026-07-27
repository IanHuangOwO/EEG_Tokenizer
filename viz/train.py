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

    def plot(self, filename=None, mode='pretrain', freeze_backbone=False):
        self.plot_all(mode=mode, freeze_backbone=freeze_backbone)

    def plot_metrics(self, filename=None):
        pass  # merged into plot_all

    def plot_all(self, filename='training_dashboard.png', mode='pretrain', freeze_backbone=False):
        """mode='finetune' drops the MSE plots (never used post-pretrain).
        freeze_backbone=True zeroes codebook metrics that can't move with a frozen backbone,
        but still shows head diversity (reflects classifier-head behavior)."""
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return

        src_tr  = self.history['train']
        src_val = self.history['val']

        def _ep(d): return range(1, len(d) + 1)
        def _t(k):  return src_tr.get(k)
        def _v(k):  return src_val.get(k)
        n_epochs = len(src_tr['loss'])

        def _plot_tr_val(ax, key, color, title, ylabel):
            """Plot train (solid) + val (dashed) series for `key` on `ax`."""
            if _t(key): ax.plot(_ep(_t(key)), _t(key), color=color, label='Train')
            if _v(key): ax.plot(_ep(_v(key)), _v(key), color=color, ls='--', alpha=0.7, label='Val')
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.legend(fontsize='x-small'); ax.grid(True)

        fig, axes = plt.subplots(3, 3, figsize=(24, 15), constrained_layout=True)
        fig.suptitle('Training Dashboard', fontsize=14, fontweight='bold')
        (ax_loss,    ax_masked,   ax_unmasked,
         ax_ppl,     ax_ste,      ax_hcos,
         ax_fpsim,   ax_fpstd,    ax_router) = axes.flat

        # ── Row 0: Loss / task curves ─────────────────────────────────────────
        _plot_tr_val(ax_loss, 'loss', 'b', 'Total Loss', 'Loss')

        if mode == 'finetune':
            _plot_tr_val(ax_masked,   'acc', 'crimson',   'Accuracy',    'Acc')
            _plot_tr_val(ax_unmasked, 'f1',  'steelblue', 'F1 (macro)',  'F1')
        else:
            _plot_tr_val(ax_masked,   'masked',   'crimson',   'Masked MSE',   'Loss')
            _plot_tr_val(ax_unmasked, 'unmasked', 'steelblue', 'Unmasked MSE', 'Loss')

        # ── Row 1: Codebook metrics (routed + shared MoE pools) ──────────────
        # frozen backbone: perplexity/STE can't move post-freeze, so show a flat
        # zero line instead of a stale/misleading real value. Falls back to the old
        # unsuffixed single-pool keys if present (pre-MoE runs).
        def _pool_pairs(prefix):
            routed = _v(f'{prefix}_routed')
            shared = _v(f'{prefix}_shared')
            if routed or shared:
                return [('routed', routed, 'green'), ('shared', shared, 'steelblue')]
            legacy = _v(prefix)
            return [('', legacy, 'green')] if legacy else []

        # [1,0] Codebook perplexity
        for label, ppl, color in _pool_pairs('codebook_perplexity'):
            ppl = [0.0] * n_epochs if freeze_backbone else ppl
            ax_ppl.plot(_ep(ppl), ppl, color=color, label=f'Perplexity {label}'.strip())
        if ax_ppl.lines:
            ax_ppl.set_ylabel('Perplexity')
            ax_ppl.legend(fontsize='x-small')
        else:
            ax_ppl.axis('off')
        ax_ppl.set_title('Codebook Perplexity' + (' [frozen]' if freeze_backbone else '')); ax_ppl.grid(True)

        # [1,1] STE gap
        for label, ste, color in _pool_pairs('codebook_ste_gap'):
            ste = [0.0] * n_epochs if freeze_backbone else ste
            ax_ste.plot(_ep(ste), ste, color=color, label=f'STE gap {label}'.strip())
        if ax_ste.lines:
            ax_ste.set_ylabel('STE gap')
            ax_ste.legend(fontsize='x-small')
        else:
            ax_ste.axis('off')
        ax_ste.set_title('Codebook STE Gap' + (' [frozen]' if freeze_backbone else '')); ax_ste.grid(True)

        # [1,2] Head projection cosine similarity (lower = more diverse heads)
        hcos_routed = _t('head_cosine_sim_routed')
        hcos_shared = _t('head_cosine_sim_shared')
        hcos_legacy = _t('head_cosine_sim')
        if hcos_routed or hcos_shared:
            if hcos_routed: ax_hcos.plot(_ep(hcos_routed), hcos_routed, color='darkorchid', label='Mean |cosine sim| routed')
            if hcos_shared: ax_hcos.plot(_ep(hcos_shared), hcos_shared, color='teal',       label='Mean |cosine sim| shared')
        elif hcos_legacy:
            ax_hcos.plot(_ep(hcos_legacy), hcos_legacy, color='darkorchid', label='Mean |cosine sim|')
        if ax_hcos.lines:
            ax_hcos.set_ylabel('Mean |cosine sim|')
            ax_hcos.legend(fontsize='x-small')
        else:
            ax_hcos.axis('off')
        ax_hcos.set_title('Head Projection Diversity\n(lower = more diverse)'); ax_hcos.grid(True)

        # ── Row 2: Head specialization metrics ───────────────────────────────
        # Decoder-output fingerprint: data-dependent diversity of each Expert's actual
        # decoded output (see MeFSQPretrain._update_fingerprint_stats), complements the
        # static weight-space head_cosine_sim_* above (which never looks at real data).
        # [2,0] Fingerprint mean cosine sim — monitoring only (not a loss term)
        fp_routed = _t('decoder_fingerprint_sim_routed')
        fp_shared = _t('decoder_fingerprint_sim_shared')
        if fp_routed: ax_fpsim.plot(_ep(fp_routed), fp_routed, color='darkorange', label='Mean cosine sim routed')
        if fp_shared: ax_fpsim.plot(_ep(fp_shared), fp_shared, color='teal',       label='Mean cosine sim shared')
        if ax_fpsim.lines:
            ax_fpsim.set_ylabel('Mean |cosine sim|')
            ax_fpsim.legend(fontsize='x-small')
        else:
            ax_fpsim.axis('off')
        ax_fpsim.set_title('Decoder Fingerprint Similarity [monitor]\n(lower = more diverse)'); ax_fpsim.grid(True)

        # [2,1] Fingerprint sim std — monitoring only
        fp_std_routed = _t('decoder_fingerprint_sim_std_routed')
        fp_std_shared = _t('decoder_fingerprint_sim_std_shared')
        if fp_std_routed: ax_fpstd.plot(_ep(fp_std_routed), fp_std_routed, color='darkorange', label='Std cosine sim routed')
        if fp_std_shared: ax_fpstd.plot(_ep(fp_std_shared), fp_std_shared, color='teal',       label='Std cosine sim shared')
        if ax_fpstd.lines:
            ax_fpstd.set_ylabel('Std cosine sim')
            ax_fpstd.legend(fontsize='x-small')
        else:
            ax_fpstd.axis('off')
        ax_fpstd.set_title('Decoder Fingerprint Sim Std [monitor]\n(higher = varied specialization)'); ax_fpstd.grid(True)

        # [2,2] Router health: entropy (higher = balanced) + load std + lb_loss (lower = balanced)
        # top-k softmax routing: gate_entropy is the entropy of the softmax weight distribution
        # AMONG the k selected heads per patch — low = one head dominates (peaked/confident),
        # high (up to log(k)) = weight split near-uniformly. Old hard-mask scheme had no
        # analog (every selected head always got weight exactly 1).
        r_ent = _t('router_entropy')
        r_gate_ent = _t('gate_entropy')
        r_std = _t('router_load_std')
        r_lb  = _t('lb_loss')
        ax2 = ax_router.twinx()
        if r_ent:
            ax_router.plot(_ep(r_ent), r_ent, color='teal', label='Router entropy (selection)')
            ax_router.set_ylabel('Entropy (higher=balanced)', color='teal')
        if r_gate_ent:
            ax_router.plot(_ep(r_gate_ent), r_gate_ent, color='darkgoldenrod', label='Gate entropy (softmax weight)')
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

    def plot_tokenizer(self, filename='training_dashboard.png'):
        """Dashboard for MeSAEPretrain (per-filter SAE tokenizer) — no masked/unmasked
        split (no masking task), no router/codebook panels (no MoE/VQ in this model)."""
        if 'loss' not in self.history['train'] or not self.history['train']['loss']:
            return

        src_tr, src_val = self.history['train'], self.history['val']
        def _ep(d): return range(1, len(d) + 1)
        def _t(k):  return src_tr.get(k)
        def _v(k):  return src_val.get(k)

        def _plot_tr_val(ax, key, color, title, ylabel):
            if _t(key): ax.plot(_ep(_t(key)), _t(key), color=color, label='Train')
            if _v(key): ax.plot(_ep(_v(key)), _v(key), color=color, ls='--', alpha=0.7, label='Val')
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.legend(fontsize='x-small'); ax.grid(True)

        def _plot_tr_val_band(ax, key, color, title, ylabel):
            """Same as _plot_tr_val, plus a shaded mean +/- 1 std (across filters) band
            around each mean line (reads f'{key}_std' if present) — replaces a separate
            std line, the band already shows the spread directly."""
            t_mean, t_std = _t(key), _t(f'{key}_std')
            v_mean, v_std = _v(key), _v(f'{key}_std')
            if t_mean:
                ax.plot(_ep(t_mean), t_mean, color=color, label='Train (mean)')
                if t_std:
                    lo = [m - s for m, s in zip(t_mean, t_std)]
                    hi = [m + s for m, s in zip(t_mean, t_std)]
                    ax.fill_between(_ep(t_mean), lo, hi, color=color, alpha=0.15, label='Train +/-1 std (across filters)')
            if v_mean:
                ax.plot(_ep(v_mean), v_mean, color=color, ls='--', alpha=0.7, label='Val (mean)')
                if v_std:
                    lo = [m - s for m, s in zip(v_mean, v_std)]
                    hi = [m + s for m, s in zip(v_mean, v_std)]
                    ax.fill_between(_ep(v_mean), lo, hi, color=color, alpha=0.08)
            ax.set_title(title); ax.set_ylabel(ylabel)
            ax.legend(fontsize='x-small'); ax.grid(True)

        fig, axes = plt.subplots(2, 4, figsize=(30, 10), constrained_layout=True)
        fig.suptitle('Tokenizer (SAE) Training Dashboard', fontsize=14, fontweight='bold')
        ax_loss, ax_mse, ax_aux, ax_skip, ax_l0, ax_dead, ax_fp, ax_block = axes.flat

        _plot_tr_val(ax_loss, 'loss', 'b',        'Total Loss (MSE + aux*weight)', 'Loss')
        _plot_tr_val(ax_mse,  'mse',  'steelblue', 'Reconstruction MSE',            'MSE')
        # aux loss is 0 in eval by design (dead-feature revival only runs in training) —
        # train-only line, a val line would just be a flat zero.
        if _t('aux'): ax_aux.plot(_ep(_t('aux')), _t('aux'), color='darkorange', label='Train')
        ax_aux.set_title('SAE Aux-K Loss (dead-feature revival)\n[train only, 0 in eval by design]')
        ax_aux.set_ylabel('Aux loss'); ax_aux.legend(fontsize='x-small'); ax_aux.grid(True)

        _plot_tr_val_band(ax_l0, 'l0_sparsity', 'darkorchid',
                          'L0 Sparsity (active features/patch)\nshaded = mean +/-1 std across filters', 'Count')
        _plot_tr_val(ax_dead, 'dead_feature_rate', 'crimson',    'Dead Feature Rate',                    'Fraction')

        _plot_tr_val_band(ax_fp, 'decoder_fingerprint_sim', 'teal',
                          'Per-Filter Decoder Fingerprint\n(lower mean = more diverse; shaded = mean +/-1 std across pairs)', 'Cosine sim')

        # U-Net skip gate(s) on the encoder's residual-add path: sigmoid(g) in [0,1],
        # 0 = drop skip, 1 = plain add. One line per pool_after_blocks stage.
        def _idx_sorted(prefix):
            return sorted((k for k in src_tr if k.startswith(prefix)),
                          key=lambda k: int(k.rsplit('_', 1)[1]))

        skip_keys = _idx_sorted('skip_gate_')
        for k in skip_keys:
            ax_skip.plot(_ep(_t(k)), _t(k), label=k)
        if skip_keys:
            ax_skip.set_ylabel('sigmoid(gate)')
            ax_skip.legend(fontsize='x-small')
        else:
            ax_skip.axis('off')
        ax_skip.set_title('Residual-Add Skip Gates\n(0=drop skip, 1=plain add)'); ax_skip.grid(True)

        # Per-block contribution norm — how much each encoder block actually changes its
        # input, direct measure (not the skip-gate proxy above, which conflates "shallow
        # skip re-injected" with "deep processing did nothing"). One line per block,
        # colored shallow (dark) -> deep (bright) so a "deep layers barely contribute"
        # collapse is visible as flat near-zero lines on the bright end.
        block_keys = _idx_sorted('block_norm_')
        if block_keys:
            cmap_b = plt.get_cmap('viridis')
            for i, k in enumerate(block_keys):
                ax_block.plot(_ep(_t(k)), _t(k), color=cmap_b(i / max(len(block_keys) - 1, 1)),
                               label=f'Block {k.rsplit("_", 1)[1]}')
            ax_block.set_ylabel('Mean |delta| per block')
            ax_block.legend(fontsize='x-small', ncol=2)
        else:
            ax_block.axis('off')
        ax_block.set_title('Per-Block Contribution Norm\n(flat near-zero = block not used)'); ax_block.grid(True)

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
                             sharex=True, constrained_layout=True)
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

    x_patches, coords, mask, time_indices, _, _, _ = [t.to(device) for t in batch]
    B, C, P, T_patch = x_patches.shape

    with torch.no_grad():
        out = model(x_patches, coords, time_idx=time_indices, bool_masked_pos=None)
        student_recon = out.recon.permute(0, 2, 1, 3)  # [B, P, C, T_patch]

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
                             figsize=(n_cols * 4, n_rows * 2.5), sharex=True, constrained_layout=True)
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
