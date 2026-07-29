"""MeFSQ's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import torch

from model.MeFSQ.MeFSQ import MeFSQPretrain, MeFSQFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import extract_head_psd, extract_head_spectra


@torch.no_grad()
def _run_reconstruction(model, dataset, trial_idx, device):
    x_patches, coords, mask, time_indices, _, _, _ = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    fs = dataset.base_dataset.config['preprocess_params']['target_freq']

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
        model.update_head_metrics(out.gate_mask_routed)
        return l_total, l_masked, l_unmasked

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

        router_series = [
            dict(key=k, color=c, label=l, train_only=True)
            for k, c, l in (('router_entropy', 'teal', 'Router entropy (selection)'),
                            ('gate_entropy', 'darkgoldenrod', 'Gate entropy (softmax weight)'))
            if k in self.history['train']
        ]
        twin_series = [
            dict(key=k, color=c, style_train=ls, label=l, train_only=True)
            for k, c, ls, l in (('router_load_std', 'salmon', '--', 'Load std'),
                                ('lb_loss', 'peru', ':', 'LB loss (1.0=uniform)'))
            if k in self.history['train']
        ]
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
        ] + self._codebook_panels(freeze_backbone=freeze_backbone)
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=MeFSQFinetune,
    trainer_cls=MeFSQTrainer,
    checker_cls=MeFSQChecker,
    plotter_cls=MeFSQPlotter,
)
