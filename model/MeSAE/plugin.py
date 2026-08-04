"""MeSAE's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import numpy as np
import torch

from model.MeSAE.MeSAE import MeSAEPretrain, MeSAEFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import extract_filter_psd, extract_filter_spectra


@torch.no_grad()
def _run_reconstruction_sae(model, dataset, trial_idx, device):
    """Always runs unmasked (bool_masked_pos not passed) for a clean reconstruction
    snapshot, regardless of whether the model is currently in the Masked training stage."""
    x_patches, coords, _, time_indices, _, _, valid_channels = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    fs = dataset.base_dataset.config['preprocess_params']['target_freq']

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)
    vc_in     = valid_channels.unsqueeze(0).to(device)

    out = model(x_in, coords=coords_in, time_idx=t_in, valid_channels=vc_in)
    recon_flat = out.recon.reshape(1, C, N * L)

    return {
        'raw':    x_patches.reshape(C, T_total).cpu().numpy(),
        'recon':  recon_flat[0].cpu().numpy(),
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


def build_model(bp, num_channels):
    """bp: config['model_params']['MeSAE']['pretrain']."""
    sae = bp.get('sae', {})

    return MeSAEPretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=bp.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        pool_after_blocks=bp.get('pool_after_blocks', []),
        upsample_residual_add=bp.get('upsample_residual_add', True),
        num_channels=num_channels,
        n_routed_filters=sae.get('n_routed_filters', 32),
        n_shared_filters=sae.get('n_shared_filters', 4),
        n_top_k=sae.get('n_top_k', 4),
        channel_pool_hidden=sae.get('channel_pool_hidden', 32),
        channel_pool_temperature=sae.get('channel_pool_temperature', 1.0),
        sae_expansion=sae.get('sae_expansion', 8),
        sae_k=sae.get('sae_k', 32),
        decoder_hidden=sae.get('decoder_hidden'),
    )


class MeSAETrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        # masked_mse_weight/unmasked_mse_weight are computed generically by train_pretrain.py
        # for every model type but MeSAE's loss no longer uses them — see get_loss.
        aux_weight = hparams.get('aux_weight', 0.03)
        lb_weight = hparams.get('lb_weight', 0.01)
        hierarchical_mse_weight = hparams.get('hierarchical_mse_weight', 1.0)
        model.update_head_metrics(out.gate[:, :model.n_routed_filters])
        return model.get_loss(x, out.recon, out.aux_loss, bool_masked_pos=mp,
                               aux_weight=aux_weight, hierarchical_mse_weight=hierarchical_mse_weight,
                               lb_loss=out.lb_loss, lb_weight=lb_weight)

    def epoch_metrics(self, model, out):
        # mse_level_* are accumulated per-batch and epoch-averaged in train_pretrain.py
        # (train_one_epoch/validate_one_epoch), not added here — this function only ever
        # sees the last batch's out, which would make mse_level_* a last-batch snapshot
        # instead of an epoch average like every other loss stat.
        metrics = model.get_metrics(out.sae_hidden.detach())
        metrics['aux'] = out.aux_loss.item() if hasattr(out.aux_loss, 'item') else float(out.aux_loss)
        metrics['lb_loss'] = out.lb_loss.item() if hasattr(out.lb_loss, 'item') else float(out.lb_loss)
        return metrics

    def on_pretrain_start(self, model, logger=None):
        model.freeze_sae()
        if logger:
            logger.info("  [Pretrain] SAE+decoder frozen, only main transformer trains from here")


class MeSAEChecker(BaseEpochChecker):
    unit_label = 'Filter'

    def compute_unit_colors(self, model, out):
        """red = shared Filter (always-on, structural). orange = routed Filter that
        actually got gated on (nonzero gate) for at least one patch in this trial. black =
        routed Filter unselected this trial. See docs/adr/0007-routed-filter-gating-for-mesae.md."""
        colors = ['black'] * model.n_filters
        for q in range(model.n_routed_filters, model.n_filters):
            colors[q] = 'red'
        gate = out.gate.detach().cpu().numpy()  # [M, Q]
        selected = (gate[:, :model.n_routed_filters] > 0).any(axis=0)  # [n_routed]
        for q in range(model.n_routed_filters):
            if selected[q]:
                colors[q] = 'orange'
        return colors

    def extract_psd(self, model, x_in, c_in, t_in, vc_in):
        return extract_filter_psd(model, x_in, c_in, t_in, vc_in)

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution):
        return extract_filter_spectra(model, x_in, c_in, t_in, vc_in, fs=fs, freq_resolution=freq_resolution)

    def run_reconstruction(self, model, dataset, trial_idx, device):
        return _run_reconstruction_sae(model, dataset, trial_idx, device)


class MeSAECodebookChecker(BaseCodebookChecker):
    unit_label = 'Filter'

    @torch.no_grad()
    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        out = model(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
        return out.sae_hidden.detach().cpu().numpy()  # [M, Q, F]

    def decoder_fingerprint_matrix(self, model):
        w1 = model.decoder.w1.detach().cpu().numpy()  # [Q, embed_dim, hidden]
        flat = w1.reshape(w1.shape[0], -1)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        return flat @ flat.T


class MeSAEPlotter(BasePlotter):
    def plot_pretrain(self, filename='training_dashboard.png'):
        panels = [
            dict(title='Total Loss (MSE + aux*weight)', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Masked vs Unmasked MSE\n(finest pyramid level, diagnostic only)', ylabel='Loss',
                 series=[dict(key='masked', color='crimson'), dict(key='unmasked', color='steelblue')]),
            dict(title='Hierarchical MSE Pyramid\n(coarse=whole-trial avg patch shape -> fine=per-patch)',
                 ylabel='MSE', series=self.indexed_series('mse_level_', cmap_name='plasma', train_only=False)),
            dict(title='SAE Aux-K Loss (dead-feature revival)\n[train only, 0 in eval by design]',
                 ylabel='Aux loss', series=[dict(key='aux', color='darkorange', train_only=True)]),
            dict(title='Residual-Add Skip Gates\n(0=drop skip, 1=plain add)',
                 ylabel='sigmoid(gate)', series=self.indexed_series('skip_gate_')),
            dict(title='L0 Sparsity (active features/patch)\nshaded = mean +/-1 std across filters in pool',
                 ylabel='Count', series=[dict(key='l0_sparsity_routed', color='darkorchid', label='Routed', band=True),
                                          dict(key='l0_sparsity_shared', color='red', label='Shared', band=True)]),
            dict(title='Dead Feature Rate', ylabel='Fraction',
                 series=[dict(key='dead_feature_rate', color='crimson')]),
            dict(title='Per-Filter Decoder Fingerprint\n(lower mean = more diverse; shaded = mean +/-1 std across pairs)',
                 ylabel='Cosine sim', series=[dict(key='decoder_fingerprint_sim', color='teal', band=True)]),
            dict(title='Per-Block Contribution Norm\n(flat near-zero = block not used)',
                 ylabel='Mean |delta| per block', series=self.indexed_series('block_norm_', cmap_name='viridis')),
        ]

        # Router Health — see MeSAE.update_head_metrics for what each number means. Same
        # panel shape as MeFSQ's; `if k in self.history['train']` guard kept even though the
        # router is now mandatory, since a freshly started run's history is empty either way.
        router_series = [
            dict(key=k, color=c, label=l, train_only=True)
            for k, c, l in (('router_entropy', 'teal', 'Router entropy (load balance)'),
                            ('gate_entropy', 'darkgoldenrod', 'Gate entropy (softmax weight)'))
            if k in self.history['train']
        ]
        twin_series = [
            dict(key=k, color=c, style_train=ls, label=l, train_only=True)
            for k, c, ls, l in (('router_load_std', 'salmon', '--', 'Load std'),
                                ('lb_loss', 'peru', ':', 'LB loss (1.0=uniform)'))
            if k in self.history['train']
        ]
        panels.append(dict(
            title='Router Health\n(entropy rising = healthy spread; falling = collapse)',
            ylabel='Entropy (higher=balanced)', series=router_series,
            twin=dict(ylabel='Load std / LB loss', series=twin_series) if twin_series else None,
        ))

        self.render(panels, filename, suptitle='Tokenizer (SAE) Training Dashboard', ncols=4)

    def plot_finetune(self, filename='training_dashboard.png', freeze_backbone=False):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Accuracy', ylabel='Acc', series=[dict(key='acc', color='crimson')]),
            dict(title='F1 (macro)', ylabel='F1', series=[dict(key='f1', color='steelblue')]),
        ]
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=MeSAEFinetune,
    trainer_cls=MeSAETrainer,
    checker_cls=MeSAEChecker,
    plotter_cls=MeSAEPlotter,
    codebook_checker_cls=MeSAECodebookChecker,
)
