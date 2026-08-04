"""BaseEpochChecker: per-epoch topo/PSD/attention viz snapshot. Template method — this
class owns the panel-building flow, subclasses only implement the extraction hooks —
keeps MeFSQ's and MeSAE's panel format identical by construction, the same guarantee the
old merged check_epoch_*.py scripts existed to provide. See
docs/adr/0004-model-plugin-base-classes.md.

check_pretrain/check_finetune each build a SnapshotBundle (the stage-specific part: how
to patchify/mask/forward the model) and hand it to _render_snapshot, which owns the
shared PSD/FFT/band-crop/panel-render sequence (the stage-agnostic part). See
docs/adr/0004 and the improve-codebase-architecture review that introduced this split.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from viz.extract import PsdResult, SpectraResult
from viz.topomap import project_coords_2d
from viz.panels import plot_topo_psd_filter, plot_attn_topo as render_attn_topo
from viz.timeseries import visualize_reconstruction


@dataclass
class SnapshotBundle:
    """Stage-normalised input to BaseEpochChecker._render_snapshot. Built by
    check_pretrain/check_finetune, each of which knows how to patchify/mask/forward
    its own stage; _render_snapshot knows nothing about pretrain vs. finetune."""
    x_in: torch.Tensor           # [1, C, N, L] patches fed to extract_psd/extract_spectra
    c_in: torch.Tensor           # [1, C, 3] coords
    t_in: torch.Tensor           # [1, N] time indices
    vc_in: torch.Tensor          # [1, C] valid-channel mask
    psd_model: nn.Module         # object extract_psd/extract_spectra actually run on
    raw_t: torch.Tensor          # [1, C, T] full-resolution raw signal, for visualize_reconstruction
    recon_t: torch.Tensor        # [1, C, T] full-resolution reconstruction
    raw_cnl: np.ndarray          # [C, N, L] raw patches, for the raw/recon FFT rows
    recon_cnl: np.ndarray        # [C, N, L] reconstruction patches
    attn: np.ndarray             # attention map ready for plot_attn_topo
    coords: np.ndarray           # [C, 3]
    channel_names: List[str]
    valid_channels: np.ndarray   # [C] bool
    patch_len: int
    mask_np: Optional[np.ndarray] = None   # [C, N] masked-patch overlay, or None
    title_suffix: str = ''                 # e.g. ' [finetune]'
    unit_colors: Optional[List[str]] = None  # [Q] per-unit title/label color override, or None


class BaseEpochChecker:
    """unit_label: 'Expert' | 'Filter' | ... — used in panel titles/axis labels."""
    unit_label = 'Unit'

    def compute_unit_colors(self, model, out):
        """Optional per-unit title/label color override for topo_psd_filter/attn_topo
        (e.g. MeSAE's routed-gating: red = shared Filter, orange = routed Filter that
        actually fired for this trial, see MeSAEChecker). Default: no coloring."""
        return None

    def extract_psd(self, model, x_in, c_in, t_in, vc_in) -> PsdResult:
        """See viz/extract.py extract_head_psd / extract_filter_psd."""
        raise NotImplementedError

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution) -> SpectraResult:
        """See viz/extract.py extract_head_spectra / extract_filter_spectra."""
        raise NotImplementedError

    def run_reconstruction(self, model, dataset, trial_idx, device):
        """Returns dict(raw, recon, coords, T, N, L, fs) for one trial, full (unmasked)
        reconstruction — see each model's plugin.py for the implementation."""
        raise NotImplementedError

    # -- shared snapshot renderer: PSD/FFT/band-crop/panel-render, stage-agnostic ---

    @torch.no_grad()
    def _render_snapshot(self, bundle, config, output_dir, subject_id=None, epoch=None,
                          trial_idx=None, cmap='YlOrRd',
                          plot_recon=True, plot_topo_psd=True, plot_attn_topo=True):
        viz_dir = os.path.join(output_dir, 'recon')
        os.makedirs(viz_dir, exist_ok=True)
        epoch_tag = f'_ep{epoch:04d}' if epoch is not None else ''
        tagged_epoch_tag = f'{epoch_tag}{bundle.title_suffix}'

        pos2d = project_coords_2d(bundle.coords)
        C, N, patch_len = bundle.raw_cnl.shape

        if plot_recon:
            visualize_reconstruction(
                None, (bundle.raw_t, bundle.recon_t), epoch,
                output_dir=viz_dir,
                channel_names=bundle.channel_names,
                subject_id=subject_id, trial_idx=trial_idx,
                mask=bundle.mask_np, patch_len=bundle.patch_len,
            )

        if not (plot_topo_psd or plot_attn_topo):
            return

        try:
            psd_result = self.extract_psd(
                bundle.psd_model, bundle.x_in, bundle.c_in, bundle.t_in, bundle.vc_in)
            psd_ch_x, importance = psd_result.psd_ch_x, psd_result.importance
        except Exception as e:
            print(f"  [epoch] extract_psd failed, skipping topo_psd_filter + attn_topo: {e}")
            return

        if plot_topo_psd:
            try:
                pp = config.get('preprocess_params', {})
                fs = pp.get('target_freq')
                l_freq, h_freq = pp.get('l_freq'), pp.get('h_freq')
                spectra_result = self.extract_spectra(
                    bundle.psd_model, bundle.x_in, bundle.c_in, bundle.t_in, bundle.vc_in, fs, 0.2)
                psd_x, freqs = spectra_result.psd, spectra_result.freqs

                n_fft = max(patch_len, int(round(fs / 0.2))) if fs else patch_len
                fft_raw   = np.fft.rfft(bundle.raw_cnl,   n=n_fft, axis=-1)
                fft_recon = np.fft.rfft(bundle.recon_cnl, n=n_fft, axis=-1)
                psd_raw   = (fft_raw.real**2   + fft_raw.imag**2  ).mean(axis=1)
                psd_recon = (fft_recon.real**2 + fft_recon.imag**2).mean(axis=1)

                if l_freq is not None and h_freq is not None:
                    band = (freqs >= l_freq) & (freqs <= h_freq)
                    freqs, psd_x   = freqs[band], psd_x[:, :, band]
                    psd_raw, psd_recon = psd_raw[:, band], psd_recon[:, band]

                raw_power   = (bundle.raw_cnl   ** 2).mean(axis=(1, 2))
                recon_power = (bundle.recon_cnl ** 2).mean(axis=(1, 2))

                out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_psd_filter.png")
                plot_topo_psd_filter(
                    out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                    psd_ch_x, psd_x, freqs, importance, cmap=cmap,
                    subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag,
                    unit_label=self.unit_label, l_freq=l_freq, h_freq=h_freq,
                    unit_colors=bundle.unit_colors,
                )
                print(f"  [epoch] -> {out_path}")
            except Exception as e:
                print(f"  [epoch] topo_psd_filter failed: {e}")

        if plot_attn_topo:
            try:
                out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_attn_topo.png")
                render_attn_topo(
                    out_path, pos2d, bundle.attn, importance, bundle.channel_names,
                    valid_channels=bundle.valid_channels,
                    subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag, unit_label=self.unit_label,
                    unit_colors=bundle.unit_colors,
                )
                print(f"  [epoch] -> {out_path}")
            except Exception as e:
                print(f"  [epoch] attn_topo failed: {e}")

    # -- pretrain-stage snapshot: recon_signal / topo_psd_filter / attn_topo -------

    @torch.no_grad()
    def check_pretrain(self, config, output_dir, model, dataset, trial_idx,
                        subject_id=None, epoch=None, cmap='YlOrRd',
                        plot_recon=True, plot_topo_psd=True, plot_attn_topo=True, trainer=None):
        device = next(model.parameters()).device
        model.eval()

        x_patches, coords, mask, time_indices, _, _, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)

        data = self.run_reconstruction(model, dataset, trial_idx, device)

        C, N, patch_len = x_patches.shape
        mask_np   = mask.numpy().reshape(C, N)
        recon_cnl = data['recon'].reshape(C, N, patch_len)

        out = model(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        attn = out.attn[0].mean(dim=0).cpu().numpy()

        metrics = {'recon_mse': float(np.mean((data['raw'] - data['recon']) ** 2))}
        if trainer is not None:
            metrics.update(trainer.epoch_metrics(model, out))

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=model,
            raw_t=torch.from_numpy(data['raw']).unsqueeze(0),
            recon_t=torch.from_numpy(data['recon']).unsqueeze(0),
            raw_cnl=x_patches.numpy(), recon_cnl=recon_cnl, attn=attn,
            coords=coords.numpy(), channel_names=dataset.base_dataset.channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=mask_np,
            unit_colors=self.compute_unit_colors(model, out),
        )
        self._render_snapshot(bundle, config, output_dir, subject_id=subject_id,
                               epoch=epoch, trial_idx=trial_idx, cmap=cmap,
                               plot_recon=plot_recon, plot_topo_psd=plot_topo_psd,
                               plot_attn_topo=plot_attn_topo)
        return metrics

    # -- finetune-stage snapshot: same 3 panels, model.backbone + finetune head's attn -

    @staticmethod
    def _patchify(x, patch_len):
        C, T = x.shape
        P = T // patch_len
        x_patches = x[:, :P * patch_len].reshape(C, P, patch_len).unsqueeze(0)
        time_idx = torch.arange(P, dtype=torch.long).unsqueeze(0)
        return x_patches, time_idx

    @staticmethod
    def _build_pad_mask(valid_channels, valid_length, P, patch_len):
        valid_length = valid_length.item() if torch.is_tensor(valid_length) else valid_length
        n_valid_patches = min(valid_length // patch_len, P)
        patch_valid = torch.arange(P) < n_valid_patches
        return (valid_channels.unsqueeze(-1) & patch_valid.unsqueeze(0)).unsqueeze(0)

    @torch.no_grad()
    def check_finetune(self, config, output_dir, model, dataset, trial_idx,
                        subject_id=None, epoch=None, patch_len=None, cmap='YlOrRd',
                        plot_recon=True, plot_topo_psd=True, plot_attn_topo=True, trainer=None):
        backbone = model.backbone
        device = next(model.parameters()).device
        pp = config.get('preprocess_params', {})
        patch_len = patch_len or pp.get('patch_length', 100)

        x_raw, coords, label, valid_channels, valid_length = dataset[trial_idx]
        x_patches, time_idx = self._patchify(x_raw, patch_len)
        x_in = x_patches.to(device)
        c_in = coords.unsqueeze(0).to(device)
        t_in = time_idx.to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)
        pad_mask = self._build_pad_mask(valid_channels, valid_length, x_patches.shape[2], patch_len).to(device)

        channel_names = dataset.base_dataset.channel_names
        C, N, L = x_patches.shape[1], x_patches.shape[2], patch_len

        was_training = model.training
        model.eval()

        _, attn_h, attn_n, attn_c = model(x_in, c_in, time_idx=t_in, pad_mask=pad_mask)
        attn_np = attn_h[0].detach().cpu().numpy()
        importance = attn_np.sum(axis=0)

        out = backbone(x_in, c_in, time_idx=t_in, bool_masked_pos=None, valid_channels=vc_in)
        raw_cnl   = x_patches[0].numpy()
        recon_cnl = out.recon[0].reshape(C, N, L).detach().cpu().numpy()

        metrics = {'recon_mse': float(np.mean((raw_cnl - recon_cnl) ** 2))}
        if trainer is not None:
            metrics.update(trainer.epoch_metrics(backbone, out))

        bundle = SnapshotBundle(
            x_in=x_in, c_in=c_in, t_in=t_in, vc_in=vc_in, psd_model=backbone,
            raw_t=torch.from_numpy(raw_cnl.reshape(1, C, N * L)),
            recon_t=torch.from_numpy(recon_cnl.reshape(1, C, N * L)),
            raw_cnl=raw_cnl, recon_cnl=recon_cnl, attn=attn_np.T,
            coords=coords.numpy(), channel_names=channel_names,
            valid_channels=valid_channels.numpy(), patch_len=patch_len, mask_np=None,
            title_suffix=' [finetune]',
            unit_colors=self.compute_unit_colors(backbone, out),
        )
        self._render_snapshot(bundle, config, output_dir, subject_id=subject_id,
                               epoch=epoch, trial_idx=trial_idx, cmap=cmap,
                               plot_recon=plot_recon, plot_topo_psd=plot_topo_psd,
                               plot_attn_topo=plot_attn_topo)

        model.train(was_training)
        return metrics
