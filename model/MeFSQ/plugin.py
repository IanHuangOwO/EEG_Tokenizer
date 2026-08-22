"""MeFSQ's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import os

import torch

from model.MeFSQ.MeFSQ import MeFSQPretrain, MeFSQFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import extract_head_psd, extract_head_spectra
from viz.panels import plot_attn_topo as render_attn_topo


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    x_patches, coords, mask, time_indices, _, _, _ = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    fs = dataset.base_dataset.config['preprocess_params']['sample_freq']

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)

    out = model(x_in, coords=coords_in, time_idx=t_in, bool_masked_pos=None)
    recon_flat = out.recon.reshape(1, C, N * L)

    return {
        'raw':    x_patches.reshape(C, T_total).cpu().numpy(),
        'recon':  recon_flat[0].cpu().numpy(),
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


def build_model(bp, num_channels):
    """bp: config['model_params']['MeFSQ']['pretrain']."""
    moe = bp.get('moe', {})
    routed = moe.get('routed_expert', {})
    shared = moe.get('shared_expert', {})

    return MeFSQPretrain(
        embed_dim=bp.get('embed_dim', 128),
        enc_depth=bp.get('enc_depth', 8),
        mlp_ratio=bp.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 100),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        pool_after_blocks=bp.get('pool_after_blocks', []),
        upsample_residual_add=bp.get('upsample_residual_add', True),
        n_routed_experts=moe.get('n_routed_experts', 64),
        top_k=moe.get('top_k', 4),
        n_shared_experts=moe.get('n_shared_experts', 2),
        routed_r=routed.get('r', 10),
        routed_num_discrete=routed.get('num_discrete', 3),
        routed_decoder_hidden=routed.get('decoder_hidden'),
        shared_r=shared.get('r', 16),
        shared_num_discrete=shared.get('num_discrete', 5),
        shared_decoder_hidden=shared.get('decoder_hidden'),
        num_channels=num_channels,
    )


class MeFSQTrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        load_balance_weight = hparams.get('load_balance_weight', 0.0)
        l_total, l_masked, l_unmasked = model.get_loss(x, out.recon, mp, masked_mse_weight=masked_mse_weight,
                                                        unmasked_mse_weight=unmasked_mse_weight)
        if load_balance_weight > 0:
            l_total = l_total + load_balance_weight * out.lb_loss
        return l_total, l_masked, l_unmasked

    def update_diagnostics(self, model, out):
        model.update_head_metrics(out.gate_mask_routed)

    def epoch_metrics(self, model, out):
        return model.get_metrics(out.v_q_routed.detach(), out.v_q_shared.detach())

    def on_pretrain_start(self, model, logger=None):
        model.freeze_vq_and_decoder()
        if logger:
            logger.info("  [Pretrain] VQ+decoder frozen, only main transformer trains from here")


class MeFSQChecker(BaseEpochChecker):
    unit_label = 'Expert'

    def extract_psd(self, model, x_in, c_in, t_in, vc_in):
        return extract_head_psd(model, x_in, c_in, t_in)

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution):
        return extract_head_spectra(model, x_in, c_in, t_in, fs=fs, freq_resolution=freq_resolution)

    def run_reconstruction(self, model, dataset, trial_idx, device):
        return _run_reconstruction(model, dataset, trial_idx, device)

    def render_finetune_attn(self, model, x_in, c_in, t_in, vc_in, valid_channels, valid_length,
                              P, patch_len, viz_dir, epoch_tag, subject_id, trial_idx,
                              pos2d, channel_names, unit_colors):
        """MeFSQFinetune's head has no channel dim (already collapsed into each Expert's
        View by the backbone's own channel-attention pool) — so the topomaps read that
        real channel attention straight from the backbone (encode_post_vq_expert's
        return_chan_attn) as-is; scaling it by each Expert's filter-attention score
        (attn_h) was tried to make Experts "visually comparable" but with many
        low-importance Experts sharing one color scale, it just crushes most topomaps
        toward the scale's dark end instead. Topo values are averaged uniformly over
        patches — NOT weighted by the head's temporal attention (attn_n) — since a
        topomap has no time axis to justify a time-weighted average. The big heatmap
        instead shows attn_n (Patch x Filter), replacing the base class's Channel x Unit
        heatmap."""
        pad_mask = self._build_pad_mask_time(valid_length, P, patch_len).to(x_in.device)
        backbone = model.backbone
        z_h, chan_attn = backbone.encode_post_vq_expert(
            x_in, c_in, time_idx=t_in, valid_channels=vc_in, return_chan_attn=True)  # [1, N, H, D], [1, N, H, C]
        chan_attn = chan_attn[0].mean(dim=0).detach().cpu().numpy()  # [H, C] — uniform mean over N, no temporal weight

        # feed the head directly instead of re-running model(...) — that would repeat the
        # same backbone forward pass we just did to get chan_attn above
        _, attn_h, attn_n = model.head(z_h, pad_mask=pad_mask)
        importance = attn_h[0].detach().cpu().numpy()          # [H] — per-Expert filter-attention score
        patch_filter_attn = attn_n[0].detach().cpu().numpy()   # [H, N] — per-Expert temporal attention

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_attn_topo.png")
        render_attn_topo(
            out_path, pos2d, chan_attn, importance, channel_names,
            valid_channels=valid_channels.numpy(),
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=f'{epoch_tag} [finetune]',
            unit_label=self.unit_label, unit_colors=unit_colors,
            heatmap_attn=patch_filter_attn, heatmap_ylabels=list(range(patch_filter_attn.shape[1])),
            heatmap_ylabel='Patch (time)', heatmap_title=f'Patch x {self.unit_label} Attention',
            heatmap_transpose=False,
        )
        print(f"  [epoch] -> {out_path}")


class MeFSQPlotter(BasePlotter):
    def _codebook_panels(self, freeze_backbone=False):
        n_epochs = len(self.history['train'].get('loss', []))

        def _maybe_zero(series_list):
            if not freeze_backbone:
                return series_list
            for s in series_list:
                if s.get('override_val') is not None:
                    s['override_val'] = [0.0] * n_epochs
                if s.get('override_train') is not None:
                    s['override_train'] = [0.0] * n_epochs
            return series_list

        panel_ppl = dict(
            title='Codebook Perplexity' + (' [frozen]' if freeze_backbone else ''),
            ylabel='Perplexity',
            series=_maybe_zero(self.pool_pair_series('codebook_perplexity')),
        )
        panel_ste = dict(
            title='Codebook STE Gap' + (' [frozen]' if freeze_backbone else ''),
            ylabel='STE gap',
            series=_maybe_zero(self.pool_pair_series('codebook_ste_gap')),
        )

        hcos_series = [
            dict(key=f'head_cosine_sim_{suf}', color=c, label=f'Mean |cosine sim| {suf}', train_only=True)
            for suf, c in (('routed', 'darkorchid'), ('shared', 'teal'))
            if f'head_cosine_sim_{suf}' in self.history['train']
        ]
        if not hcos_series and 'head_cosine_sim' in self.history['train']:
            hcos_series = [dict(key='head_cosine_sim', color='darkorchid', label='Mean |cosine sim|', train_only=True)]
        panel_hcos = dict(title='Head Projection Diversity\n(lower = more diverse)',
                          ylabel='Mean |cosine sim|', series=hcos_series)

        panel_fpsim = dict(
            title='Decoder Fingerprint Similarity [monitor]\n(lower = more diverse)',
            ylabel='Mean |cosine sim|',
            series=[dict(key=f'decoder_fingerprint_sim_{suf}', color=c, label=f'Mean cosine sim {suf}', train_only=True)
                    for suf, c in (('routed', 'darkorange'), ('shared', 'teal'))
                    if f'decoder_fingerprint_sim_{suf}' in self.history['train']],
        )
        panel_fpstd = dict(
            title='Decoder Fingerprint Sim Std [monitor]\n(higher = varied specialization)',
            ylabel='Std cosine sim',
            series=[dict(key=f'decoder_fingerprint_sim_std_{suf}', color=c, label=f'Std cosine sim {suf}', train_only=True)
                    for suf, c in (('routed', 'darkorange'), ('shared', 'teal'))
                    if f'decoder_fingerprint_sim_std_{suf}' in self.history['train']],
        )

        router_series, twin_series = self.router_health_series(entropy_label='Router entropy (selection)')
        panel_router = dict(
            title='Router Health', ylabel='Entropy (higher=balanced)', series=router_series,
            twin=dict(ylabel='Load std / LB loss', series=twin_series) if twin_series else None,
        )

        return [panel_ppl, panel_ste, panel_hcos, panel_fpsim, panel_fpstd, panel_router]

    def plot_pretrain(self, filename='training_dashboard.png'):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Masked MSE', ylabel='Loss', series=[dict(key='masked', color='crimson')]),
            dict(title='Unmasked MSE', ylabel='Loss', series=[dict(key='unmasked', color='steelblue')]),
        ] + self._codebook_panels(freeze_backbone=False)
        self.render(panels, filename, suptitle='Training Dashboard')

    def plot_finetune(self, filename='training_dashboard.png', freeze_backbone=False):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Accuracy', ylabel='Acc', series=[dict(key='acc', color='crimson')]),
            dict(title='F1 (macro)', ylabel='F1', series=[dict(key='f1', color='steelblue')]),
            dict(title='F1 (weighted)', ylabel='F1', series=[dict(key='f1_weighted', color='teal')]),
            dict(title='Balanced Accuracy', ylabel='Bal. Acc', series=[dict(key='balanced_acc', color='darkorchid')]),
            dict(title="Cohen's Kappa", ylabel='Kappa', series=[dict(key='kappa', color='seagreen')]),
            dict(title='Backbone Recon MSE', ylabel='MSE', series=[dict(key='recon_mse', color='darkorange')]),
        ] + self._codebook_panels(freeze_backbone=freeze_backbone)
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=MeFSQFinetune,
    trainer_cls=MeFSQTrainer,
    checker_cls=MeFSQChecker,
    plotter_cls=MeFSQPlotter,
)
