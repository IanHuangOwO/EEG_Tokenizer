"""MeSAE's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import torch

from model.MeSAE.MeSAE import MeSAEPretrain, MeSAEFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
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
        n_filters=sae.get('n_filters', 8),
        pool_hidden=sae.get('pool_hidden', 32),
        pool_temperature=sae.get('pool_temperature', 1.0),
        sae_expansion=sae.get('sae_expansion', 8),
        sae_k=sae.get('sae_k', 32),
        decoder_hidden=sae.get('decoder_hidden'),
    )


class MeSAETrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        # masked_mse_weight/unmasked_mse_weight are computed generically by train_pretrain.py
        # for every model type but MeSAE's loss no longer uses them — see get_loss.
        aux_weight = hparams.get('aux_weight', 0.03)
        hierarchical_mse_weight = hparams.get('hierarchical_mse_weight', 1.0)
        return model.get_loss(x, out.recon, out.aux_loss, bool_masked_pos=mp,
                               aux_weight=aux_weight, hierarchical_mse_weight=hierarchical_mse_weight)

    def epoch_metrics(self, model, out):
        # mse_level_* are accumulated per-batch and epoch-averaged in train_pretrain.py
        # (train_one_epoch/validate_one_epoch), not added here — this function only ever
        # sees the last batch's out, which would make mse_level_* a last-batch snapshot
        # instead of an epoch average like every other loss stat.
        metrics = model.get_metrics(out.sae_hidden.detach())
        metrics['aux'] = out.aux_loss.item() if hasattr(out.aux_loss, 'item') else float(out.aux_loss)
        return metrics

    def on_pretrain_start(self, model, logger=None):
        model.freeze_sae()
        if logger:
            logger.info("  [Pretrain] SAE+decoder frozen, only main transformer trains from here")


class MeSAEChecker(BaseEpochChecker):
    unit_label = 'Filter'

    def extract_psd(self, model, x_in, c_in, t_in, vc_in):
        return extract_filter_psd(model, x_in, c_in, t_in, vc_in)

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution):
        return extract_filter_spectra(model, x_in, c_in, t_in, vc_in, fs=fs, freq_resolution=freq_resolution)

    def run_reconstruction(self, model, dataset, trial_idx, device):
        return _run_reconstruction_sae(model, dataset, trial_idx, device)


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
            dict(title='L0 Sparsity (active features/patch)\nshaded = mean +/-1 std across filters',
                 ylabel='Count', series=[dict(key='l0_sparsity', color='darkorchid', band=True)]),
            dict(title='Dead Feature Rate', ylabel='Fraction',
                 series=[dict(key='dead_feature_rate', color='crimson')]),
            dict(title='Per-Filter Decoder Fingerprint\n(lower mean = more diverse; shaded = mean +/-1 std across pairs)',
                 ylabel='Cosine sim', series=[dict(key='decoder_fingerprint_sim', color='teal', band=True)]),
            dict(title='Per-Block Contribution Norm\n(flat near-zero = block not used)',
                 ylabel='Mean |delta| per block', series=self.indexed_series('block_norm_', cmap_name='viridis')),
        ]
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
)
