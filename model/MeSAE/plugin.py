"""MeSAE's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import os
import random

import numpy as np
import torch

from model.MeSAE.MeSAE import MeSAEPretrain, build_finetune
from model.MeSAE.MeSAE_modules import overlap_add_patches
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import (extract_flat_stamp_psd, extract_flat_stamp_psd_by_patch,
                          extract_flat_stamp_gallery, extract_filter_spectra)
from viz.panels import (plot_attn_topo as render_attn_topo, plot_stamp_by_patch,
                         plot_stamp_gallery, plot_event_stamp_dynamics)
from viz.codebook import (plot_stamp_similarity, plot_patch_position_consistency,
                           plot_stamp_identity_consistency, plot_fingerprint_similarity,
                           plot_pool_energy_share, plot_stamp_energy_rank,
                           plot_stamp_phase_consistency, plot_topography_distance,
                           plot_pool_ablation, plot_pool_label_probe)
from IO.preprocessing import slice_patches


@torch.no_grad()
def _run_reconstruction_sae(model, dataset, trial_idx, device):
    """Always runs unmasked (bool_masked_pos not passed) for a clean reconstruction
    snapshot, regardless of whether the model is currently in the Masked training stage."""
    x_patches, coords, _, time_indices, _, _, valid_channels = dataset[trial_idx]
    C, N, L = x_patches.shape
    pp = dataset.base_dataset.config['preprocess_params']
    fs = pp['sample_freq']
    stride = pp.get('patch_stride', L)

    x_in      = x_patches.unsqueeze(0).to(device)
    coords_in = coords.unsqueeze(0).to(device)
    t_in      = time_indices.unsqueeze(0).to(device)
    vc_in     = valid_channels.unsqueeze(0).to(device)

    out = model(x_in, coords=coords_in, time_idx=t_in, valid_channels=vc_in)
    raw_stitched   = overlap_add_patches(x_patches.to(device), stride)     # [C, T]
    recon_stitched = overlap_add_patches(out.recon[0], stride)             # [C, T]
    T_total = raw_stitched.shape[-1]

    return {
        'raw':    raw_stitched.cpu().numpy(),
        'recon':  recon_stitched.cpu().numpy(),
        'coords': coords.numpy(),
        'T': T_total, 'N': N, 'L': L, 'fs': fs,
    }


def build_model(bp, num_channels):
    """bp: config['model_params']['MeSAE']['pretrain']."""
    sb = bp.get('stamp_bank', {})
    moe_ffn = bp.get('moe_ffn', {})

    return MeSAEPretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=moe_ffn.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        pool_after_blocks=bp.get('pool_after_blocks', []),
        num_channels=num_channels,
        n_routed_stamps=sb.get('n_routed_stamps', 60),
        n_shared_stamps=sb.get('n_shared_stamps', 4),
        stamp_top_k=sb.get('stamp_top_k', 32),
        stamp_hidden_width=sb.get('stamp_hidden_width', 8),
        stamp_shared_hidden_width=sb.get('stamp_shared_hidden_width', 16),
        dead_threshold_frac=sb.get('dead_threshold_frac', 0.1),
        stamp_ema_decay=sb.get('sae_ema_decay', 0.999),
        # Both None (the default) = fully continuous (a, b), exactly as before
        # quantization existed. See StampBank._quantize_amp_phase.
        stamp_amp_levels=sb.get('amp_levels'),
        stamp_phase_levels=sb.get('phase_levels'),
        stamp_amp_log2_range=tuple(sb.get('amp_log2_range', (-6.0, 3.0))),
        # 'topk' (original) or 'gain' -- see StampBank.__init__.
        stamp_selection_mode=sb.get('selection_mode', 'topk'),
        # None = rescue every dead atom (default). A value bounds the aux
        # block's memory; selection among dead atoms is starvation-first,
        # never score-first -- see StampBank.__init__.
        stamp_aux_k_cap_frac=sb.get('aux_k_cap_frac'),
        n_routed_ffn_experts=moe_ffn.get('n_routed_experts', 4),
        n_shared_ffn_experts=moe_ffn.get('n_shared_experts', 1),
        ffn_top_k=moe_ffn.get('top_k', 2),
        # patch_stride duplicates preprocess_params (same convention as patch_len
        # above) — the shared build_model(bp, num_channels) interface
        # (model/factory.py) doesn't pass preprocess_params through.
        patch_stride=bp.get('patch_stride'),
    )


class MeSAETrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, **hparams):
        if 'hierarchical_mse_weight' in hparams:
            raise ValueError("loss.hierarchical_mse_weight was split into mse_patch_weight / "
                             "mse_trial_weight (+ unmasked_weight), see MeSAE._recon_loss")
        aux_weight = hparams.get('aux_weight', 0.03)
        ffn_lb_weight = hparams.get('ffn_lb_weight', 0.01)
        mp_weight = hparams.get('mp_weight', 0.0)
        return model.get_loss(x, out.recon, out.aux_loss, bool_masked_pos=mp,
                               aux_weight=aux_weight,
                               mse_patch_weight=hparams.get('mse_patch_weight', 1.0),
                               mse_trial_weight=hparams.get('mse_trial_weight', 1.0),
                               unmasked_weight=hparams.get('unmasked_weight', 1.0),
                               ffn_lb_loss=out.ffn_lb_loss, ffn_lb_weight=ffn_lb_weight,
                               valid_channels=out.valid_channels,
                               mp_loss=out.mp_loss, mp_weight=mp_weight, mp_map=out.mp_map)

    def update_diagnostics(self, model, out):
        model.update_stamp_router_metrics(out.dense_routed)
        model.update_ffn_router_metrics(out.ffn_router_entropy, out.ffn_router_load_std, out.ffn_gate_entropy)

    def epoch_metrics(self, model, out):
        # mse_patch/mse_trial are accumulated per-batch and epoch-averaged in
        # train_pretrain.py (train_one_epoch/validate_one_epoch), not added here — this
        # function only ever sees the last batch's out, which would make them a
        # last-batch snapshot instead of an epoch average like every other loss stat.
        metrics = model.get_metrics(out.dense_routed.detach())
        metrics['aux'] = out.aux_loss.item() if hasattr(out.aux_loss, 'item') else float(out.aux_loss)
        metrics['k_eff'] = out.k_eff.item() if hasattr(out.k_eff, 'item') else float(out.k_eff)
        metrics['ffn_lb_loss'] = out.ffn_lb_loss.item() if hasattr(out.ffn_lb_loss, 'item') else float(out.ffn_lb_loss)
        return metrics


class MeSAEChecker(BaseEpochChecker):
    unit_label = 'Stamp'
    # Flat-token StampBank has no cross-channel pool left to produce a channel-attention
    # map from (see MeSAE_modules.StampBank class docstring) — the topo_psd_by_stamp panel
    # (_render_topo_psd override below) now covers per-stamp channel topography instead.
    has_attn_topo = False

    def compute_unit_colors(self, model, out):
        """red = shared stamp (always-on, structural). black = routed stamp. Restricted to
        stamps actually used somewhere in this trial, capped at 100 (see
        MeSAEPretrain.used_stamp_ids) — with n_stamps=800 and hard top-k selection, showing
        every stamp regardless of whether this trial ever touched it is mostly noise, and
        the per-patch top-k axis has no stable cross-patch identity to color consistently
        in the first place (see docs/adr/0009 / render_finetune_attn's docstring)."""
        used_ids = model.used_stamp_ids(out, max_stamps=100)
        colors = ['red' if i >= model.n_routed_stamps else 'black' for i in used_ids.tolist()]
        return colors, used_ids

    def extract_psd(self, model, x_in, c_in, t_in, vc_in):
        return extract_flat_stamp_psd(model, x_in, c_in, t_in, vc_in)

    def extract_spectra(self, model, x_in, c_in, t_in, vc_in, fs, freq_resolution):
        # Unreachable while has_attn_topo=False (its only caller, _render_stamp_panel, is
        # gated off in base_checker.py) — still points at MeSAE's pooled-channel version,
        # which would hit the same missing-_pool_channels crash extract_psd used to if
        # this ever gets called. Needs the same flat-token treatment before has_attn_topo
        # could safely flip back on.
        return extract_filter_spectra(model, x_in, c_in, t_in, vc_in, fs=fs, freq_resolution=freq_resolution)

    def run_reconstruction(self, model, dataset, trial_idx, device):
        return _run_reconstruction_sae(model, dataset, trial_idx, device)

    def _render_topo_psd(self, bundle, pos2d, viz_dir, subject_id, trial_idx, epoch_tag,
                          tagged_epoch_tag, cmap, fs, l_freq, h_freq, psd_ch_x, importance,
                          fft_resolution=0.2):
        """Overrides BaseEpochChecker's default (per-stamp trial-wide dedup, topo_psd_filter.png)
        with two panels instead of the base's one:
        - stamp_by_patch.png — the real per-patch grid (every patch_stride-th patch's
          own union of stamps its C channels individually selected, zero-filled per
          channel that didn't pick a given displayed stamp — see
          viz.extract.extract_flat_stamp_psd_by_patch). The trial-wide dedup can't tell
          "this stamp fired on 1 patch" from "fired on every patch" apart; the per-patch
          grid can.
        - stamp_gallery.png — the whole-trial Raw/Full-Recon view plus every stamp used
          SOMEWHERE in this trial (trial-wide dedup, see
          viz.extract.extract_flat_stamp_gallery), the piece the base default's
          topo_psd_filter.png would have covered — split into its own file rather than
          folded into stamp_by_patch's header, since it's a different (trial-wide, not
          per-patch) view. psd_ch_x/importance (from extract_psd, via _render_snapshot)
          are still computed upstream since _render_snapshot uses that call to gate
          whether to attempt this panel at all, but this method recomputes its own
          (used_ids-carrying) copy via extract_flat_stamp_gallery rather than reusing
          those — see that function's docstring for why."""
        model = bundle.psd_model
        grid = extract_flat_stamp_psd_by_patch(
            model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
            fs=fs, freq_resolution=fft_resolution, patch_stride=5)

        # Same raw/recon full-trial FFT as the base default, see BaseEpochChecker._render_topo_psd
        # AND its _compute_spectra: n_fft here MUST equal grid.freqs' own n_fft (extract_flat_
        # stamp_psd_by_patch's, patch_len-driven, effectively round(fs/fft_resolution) — the `band`
        # mask below is built from grid.freqs and applied to psd_raw/psd_recon too). Previously this
        # used max(T, round(fs/fft_resolution)) — a real (assemble_trials=False) trial longer than
        # that target gave psd_raw/psd_recon a bigger n_fft than grid.freqs, and `psd_raw[:, band]`
        # crashed ("boolean index did not match... size of axis is 751 but ... axis is 501"). Same
        # bug, independently duplicated here — BaseEpochChecker's own _compute_spectra was fixed
        # first, but MeSAEChecker overrides _render_topo_psd entirely, so that fix never covered
        # this method. rfft's `n=` transparently zero-pads short trials, truncates long ones.
        raw_t   = bundle.raw_t[0].numpy()
        recon_t = bundle.recon_t[0].numpy()
        T = raw_t.shape[-1]
        n_fft = int(round(fs / fft_resolution)) if fs else T

        def _demean_hann_rfft_np(x):
            x = x - x.mean(axis=-1, keepdims=True)
            win = np.hanning(x.shape[-1])
            return np.fft.rfft(x * win, n=n_fft, axis=-1)

        fft_raw   = _demean_hann_rfft_np(raw_t)
        fft_recon = _demean_hann_rfft_np(recon_t)
        psd_raw   = fft_raw.real**2   + fft_raw.imag**2
        psd_recon = fft_recon.real**2 + fft_recon.imag**2

        # grid.freqs and raw/recon's freqs share the same n_fft target (freq_resolution=0.2
        # drives both, see extract_flat_stamp_psd_by_patch), so one shared band-crop applies.
        freqs = grid.freqs
        band = None
        if l_freq is not None and h_freq is not None:
            band = (freqs >= l_freq) & (freqs <= h_freq)
            grid.freqs = freqs[band]
            grid.psd = grid.psd[:, :, :, band]
            grid.recon_psd = grid.recon_psd[:, :, band]
            grid.raw_psd = grid.raw_psd[:, :, band]
            psd_raw, psd_recon = psd_raw[:, band], psd_recon[:, band]

        raw_power   = (bundle.raw_cnl   ** 2).mean(axis=(1, 2))
        recon_power = (bundle.recon_cnl ** 2).mean(axis=(1, 2))

        # Real-trial event marker (see BaseEpochChecker._lookup_event_onset): convert the
        # onset from seconds to a displayed-COLUMN index. grid.patch_ids holds the raw
        # patch-n each displayed column represents; patch n's own start time is
        # n * model.patch_stride / fs (same convention slice_patches/time_indices use) --
        # searchsorted finds the first displayed column at or after the onset, i.e. the
        # pre/post boundary. None (the normal case: an assembled continuous window has no
        # single event) draws nothing, see plot_stamp_by_patch's onset_col docstring.
        onset_col = None
        if bundle.event_onset_sec is not None and fs:
            onset_patch_n = bundle.event_onset_sec * fs / model.patch_stride
            onset_col = int(np.searchsorted(grid.patch_ids, onset_patch_n))

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_stamp_by_patch.png")
        plot_stamp_by_patch(
            out_path, pos2d, grid, cmap=cmap,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag,
            unit_label=self.unit_label, n_routed=model.n_routed_stamps,
            signed_stamps=True,  # grid.topo is signed amp (mixing columns), see extract_flat_stamp_psd_by_patch
            onset_col=onset_col,
        )
        print(f"  [epoch] -> {out_path}")

        (used_ids, gal_importance, psd_ch_x_g, psd_x_g, gal_freqs, phase_ch_x_g,
         waveforms_g, iclabel_probs) = extract_flat_stamp_gallery(
            model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
            fs=fs, freq_resolution=fft_resolution)
        if band is not None:
            gal_freqs = gal_freqs[band]
            psd_x_g = psd_x_g[:, :, band]

        gallery_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_stamp_gallery.png")
        plot_stamp_gallery(
            gallery_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
            psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=cmap,
            phase_ch_x=phase_ch_x_g, waveforms=waveforms_g,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag,
            unit_label=self.unit_label, unit_ids=used_ids, n_routed=model.n_routed_stamps,
            iclabel_probs=iclabel_probs,
        )
        print(f"  [epoch] -> {gallery_path}")


class MeSAECodebookChecker(BaseCodebookChecker):
    unit_label = 'Stamp'
    needs_raw_tensors = True  # _render_patch_position_consistency and
    # _render_identity_consistency both need a fresh forward pass per trial
    # (extract_stamp_content / model.stamps) — too expensive for check_codebook's full
    # trial set, see needs_raw_tensors' docstring on the base class.
    # (_render_patch_similarity no longer needs this — it now reads the cheap `usage`
    # already in trial_records.)

    @torch.no_grad()
    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """[N, n_stamps] dense usage, one row per PATCH POSITION — routed axis real
        selection strength (zeros at unselected), shared axis each shared stamp's real
        post-rms amp magnitude (see StampBank.forward's h; see docs/adr/0009's
        Monitoring impact section).

        StampBank selects per patch position (group selection, see its class
        docstring), so out.dense_routed is already [G=N, n_routed] for a B=1 trial.
        Shared stamps sit at fixed positions top_k: in out.h (idx's routed-then-shared
        layout, see StampBank.forward), so no need for the model to expose idx
        separately here."""
        B, C, N, L = x_in.shape
        out = model(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
        shared = out.h[:, model.stamps.top_k:]                       # [N, n_shared]
        dense_full = torch.cat([out.dense_routed, shared], dim=-1)  # [N, n_stamps] (G = N, B=1)
        return dense_full.detach().cpu().numpy()

    def decoder_fingerprint_matrix(self, model):
        """Per-stamp [patch_len] waveform template D_i (see StampBank.fingerprint —
        content-free and exact now, no probe involved: D_i never depends on any input),
        pairwise cosine sim — this is `filter_relation.png`'s direct successor and the
        empirical check on template diversity. mp_loss is what now discourages
        duplicate atoms (docs/adr/0011); this panel is how you verify it held."""
        fp = model.stamps.fingerprint().cpu().numpy()  # [n_stamps, patch_len]
        flat = fp.reshape(fp.shape[0], -1)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        return flat @ flat.T

    def rank_ceiling(self, model):
        return min(model.stamps.top_k, model.head_dim)

    def _render_fingerprint_similarity(self, viz_dir, model):
        plot_fingerprint_similarity(
            os.path.join(viz_dir, 'stamp_fingerprint_similarity.png'),
            self.decoder_fingerprint_matrix(model), unit_label=self.unit_label,
            n_routed=model.n_routed_stamps)

    def _render_pool_energy_share(self, usage_by_dataset, viz_dir, model):
        """"Where does context live" -- direct answer: what fraction of total h^2
        (real reconstruction energy) each dataset draws from the Shared pool. usage's
        [n_routed:] columns are already the Shared pool's post-rms magnitude (see
        extract_usage), [:n_routed] the Routed pool's selection strength (zero where
        unselected) -- both are the same h units, so summing h^2 on either side of the
        n_routed boundary is a real energy split, not an apples-to-oranges comparison.

        cv_routed/cv_shared: coefficient of variation, across datasets, of each pool's
        own per-stamp MEAN usage -- see plot_pool_energy_share's docstring for why this
        is the complementary "is Shared actually generic" read."""
        n_routed = model.n_routed_stamps
        share_by_dataset = {}
        per_ds_mean = {}  # ds -> [n_stamps] mean usage, for the CV computation below
        for ds_name, usage in usage_by_dataset.items():
            energy = usage.astype(np.float64) ** 2                      # [M, n_stamps]
            routed_e = energy[:, :n_routed].sum()
            shared_e = energy[:, n_routed:].sum()
            share_by_dataset[ds_name] = float(shared_e / max(routed_e + shared_e, 1e-12))
            per_ds_mean[ds_name] = usage.mean(axis=0)                   # [n_stamps]

        stacked = np.stack(list(per_ds_mean.values()), axis=0)          # [D, n_stamps]
        mean_ = stacked.mean(axis=0)
        std_ = stacked.std(axis=0)
        cv = np.divide(std_, mean_, out=np.full_like(mean_, np.nan), where=mean_ > 1e-8)
        cv_routed = float(np.nanmean(cv[:n_routed]))
        cv_shared = float(np.nanmean(cv[n_routed:]))

        plot_pool_energy_share(
            os.path.join(viz_dir, 'pool_energy_share.png'), share_by_dataset,
            cv_routed, cv_shared, unit_label=self.unit_label)

    def _render_stamp_energy_and_rank(self, usage_by_dataset, viz_dir, model):
        """Per-unit mean firing strength (loudness) and, for Routed units only, mean
        rank-when-selected -- see plot_stamp_energy_rank's docstring. Rank is recovered
        from usage alone (no fresh forward pass, no StampBank changes needed): each
        patch's routed usage row has exactly stamp_top_k nonzero entries (the routed
        winners for that patch, see StampBank.forward); ranking those descending by
        value reproduces mp_loss's own by-h ordering (docs/adr/0011) without needing
        StampBank to expose it separately."""
        n_routed = model.n_routed_stamps
        usage = np.concatenate(list(usage_by_dataset.values()), axis=0)  # [M_total, n_stamps]
        n_stamps = usage.shape[1]
        mean_h = usage.mean(axis=0)                                      # [n_stamps], zeros count

        routed = usage[:, :n_routed]
        order = np.argsort(-routed, axis=1)                               # [M, n_routed]
        ranks = np.empty_like(order)
        rows = np.arange(routed.shape[0])[:, None]
        ranks[rows, order] = np.arange(n_routed)[None, :]                 # inverse permutation -> rank per column
        fired = routed > 0
        rank_sum = np.where(fired, ranks, 0).sum(axis=0).astype(np.float64)
        fire_count = fired.sum(axis=0)
        mean_rank = np.full(n_routed, np.nan)
        nz = fire_count > 0
        mean_rank[nz] = rank_sum[nz] / fire_count[nz]

        plot_stamp_energy_rank(
            os.path.join(viz_dir, 'stamp_energy_rank.png'), mean_h, mean_rank, n_routed,
            unit_label=self.unit_label)

    @torch.no_grad()
    def _render_pool_ablation(self, trial_records, viz_dir, model, device, seed, max_trials=60):
        """Causal necessity check -- see plot_pool_ablation's docstring for why the
        correlational usage/energy panels above aren't enough on their own. Re-decodes
        with one pool's amp zeroed via StampBank.decode_selected (public, already used by
        extract_stamp_content) -- no StampBank.forward change needed, selection/idx stay
        exactly what training produced, only the summed contribution changes."""
        needing = [t for t in trial_records if 'raw' in t]
        if not needing:
            return
        rng = random.Random(seed)
        sample = needing if len(needing) <= max_trials else rng.sample(needing, max_trials)

        top_k = model.stamps.top_k
        sums = {}  # ds -> [sq_err_baseline, sq_err_no_shared, sq_err_no_routed, n_valid_elems]
        for t in sample:
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])
            B, C, N, L = x_in.shape
            z, _ = model.stage_features(x_in, c_in, time_idx=t_in)
            z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
            x_g = x_in.permute(0, 2, 1, 3).reshape(B * N, C, L)
            rms = x_g.pow(2).mean(-1, keepdim=True).sqrt()
            vg = vc_in.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
            out = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vg)

            def _mse(amp):
                contrib = model.stamps.decode_selected(out.idx, amp).sum(dim=2)  # [G, C, L]
                err = (contrib - x_g).pow(2)
                return (err * vg.unsqueeze(-1)).sum().item(), vg.sum().item() * L

            amp_no_shared = out.amp.clone(); amp_no_shared[:, :, top_k:, :] = 0
            amp_no_routed = out.amp.clone(); amp_no_routed[:, :, :top_k, :] = 0
            se_base, n_base = _mse(out.amp)
            se_ns, _ = _mse(amp_no_shared)
            se_nr, _ = _mse(amp_no_routed)

            acc = sums.setdefault(t['dataset'], [0.0, 0.0, 0.0, 0.0])
            acc[0] += se_base; acc[1] += se_ns; acc[2] += se_nr; acc[3] += n_base

        baseline = {ds: v[0] / max(v[3], 1e-8) for ds, v in sums.items()}
        no_shared = {ds: v[1] / max(v[3], 1e-8) for ds, v in sums.items()}
        no_routed = {ds: v[2] / max(v[3], 1e-8) for ds, v in sums.items()}
        plot_pool_ablation(
            os.path.join(viz_dir, 'pool_ablation.png'), baseline, no_shared, no_routed,
            unit_label=self.unit_label)

    def _render_pool_label_probe(self, trial_usage_by_dataset, trial_labels_by_dataset, trial_records, viz_dir,
                                  model, min_per_class=5, n_folds=5):
        """Which pool's usage actually predicts the task label, as opposed to which pool
        carries more reconstruction mass (_render_pool_energy_share/_render_pool_ablation,
        a different question -- see plot_pool_label_probe's docstring). Logistic
        regression (standardized features, k-fold CV) on trial-level usage
        (trial_usage_by_dataset, patches already mean-pooled by check_codebook), same
        features for all three stamp-usage probes (routed-only / shared-only / both) so
        the comparison isn't confounded by anything but which columns are visible.

        Also probes a RAW-signal baseline (per-channel power of the stitched trial, no
        learned structure at all) on the SAME trials in the SAME order (trial_records is
        appended dataset-by-dataset, trial-by-trial in exactly the loop order that built
        trial_usage_by_dataset -- see check_codebook) -- without it, a chance-level stamp
        probe is ambiguous between "the tokenizer failed to preserve task info" and "this
        task has ~no linearly-decodable info in anything", and a high stamp-probe score is
        ambiguous between "the tokenizer learned something task-relevant" and "the task is
        just trivially decodable from raw power and the tokenizer didn't need to do
        anything clever".

        Datasets with fewer than 2 classes or fewer than min_per_class trials in their
        smallest class are skipped -- a probe on 1-2 examples of a class is noise, not
        signal."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_routed = model.n_routed_stamps
        by_ds_records = {}
        for t in trial_records:
            by_ds_records.setdefault(t['dataset'], []).append(t)

        results = {}
        for ds_name, X in trial_usage_by_dataset.items():
            y = trial_labels_by_dataset[ds_name]
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) < 2 or counts.min() < min_per_class:
                continue
            cv = StratifiedKFold(n_splits=min(n_folds, int(counts.min())), shuffle=True, random_state=0)
            accs = {}
            for name, feats in (('routed', X[:, :n_routed]), ('shared', X[:, n_routed:]), ('both', X)):
                if feats.shape[1] == 0:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                accs[name] = float(cross_val_score(clf, feats, y, cv=cv).mean())

            records = by_ds_records.get(ds_name, [])
            if len(records) == len(y) and all('raw' in t for t in records):
                stride = model.patch_stride
                raw_feats = []
                for t in records:
                    x_in = t['raw'][0]                                          # [1, C, N, L] cpu
                    vc_in = t['raw'][3][0].bool()                                # [C]
                    stitched = overlap_add_patches(x_in[0], stride)             # [C, T]
                    power = stitched.pow(2).mean(dim=-1).numpy()                # [C]
                    power[~vc_in.numpy()] = 0.0
                    raw_feats.append(power)
                raw_X = np.stack(raw_feats)                                     # [n_trials, C]
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                accs['raw'] = float(cross_val_score(clf, raw_X, y, cv=cv).mean())

            results[ds_name] = dict(accs, n_classes=len(classes), n_trials=len(y), chance=1.0 / len(classes))

        if not results:
            print('  [codebook] pool label probe skipped (no dataset with >=2 classes and '
                  f'>={min_per_class} trials/class)')
            return
        plot_pool_label_probe(
            os.path.join(viz_dir, 'pool_label_probe.png'), results, unit_label=self.unit_label)

    @torch.no_grad()
    def extract_stamp_content(self, model, x_in, c_in, t_in, vc_in):
        """Dense per-(channel,patch) DECODED CONTENT [C, N, n_stamps, patch_len],
        zero-filled at stamps that (channel, patch) token didn't select — real content
        where selected, exact 0 elsewhere (same zero-fill convention as viz.extract's
        flat-token panels, e.g. extract_flat_stamp_psd_by_patch). Unlike extract_usage (a
        scalar gating strength h per stamp), this is the actual decoder output — used only
        by _render_patch_position_consistency (viz.codebook.plot_patch_position_consistency),
        which needs real content to compare, not just selection confidence.
        (_render_patch_similarity used to read this too, for a content-based
        stamp_similarity.png — dropped in favor of the cheaper usage/Jaccard-only version,
        see plot_stamp_similarity's docstring.)

        Expensive: T=C*N tokens x n_stamps x patch_len dense per trial (e.g. 64*16*120*50
        ~= 6M floats, ~25MB). _render_patch_position_consistency only calls this for a
        small subsample of trials (see needs_raw_tensors), not every trial check_codebook
        samples up front."""
        B, C, N, L = x_in.shape
        z, _ = model.stage_features(x_in, c_in, time_idx=t_in)
        z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)   # [G, C, D], G = N (B=1)
        x_g = x_in.permute(0, 2, 1, 3).reshape(B * N, C, L)
        # rms must match the training path (see MeSAEPretrain.forward) — without it
        # amp lacks its raw-amplitude factor and every panel shows systematically
        # mis-scaled contributions.
        rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()
        vc_g = vc_in.unsqueeze(1).expand(B, N, C).reshape(B * N, C) if vc_in is not None else None
        out = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vc_g)
        contribution = model.stamps.decode_selected(out.idx, out.amp)  # [G, C, K, patch_len]
        G, _, K, patch_len = contribution.shape
        n_stamps = model.n_stamps

        dense = contribution.new_zeros(G, C, n_stamps, patch_len)
        dense.scatter_(2, out.idx.view(G, 1, K, 1).expand(G, C, K, patch_len), contribution)
        return dense.permute(1, 0, 2, 3).cpu().numpy()  # [C, N, n_stamps, patch_len]

    def _render_patch_similarity(self, trial_records, viz_dir, model, device, seed):
        """Overrides the base's default groupings (Intra-Trial/Inter-Trial/Inter-Subject,
        all 3) with the StampBank-specific version (viz.codebook.plot_stamp_similarity):
        Intra-Trial/Inter-Trial only (Intra-Patch and Inter-Subject dropped — see that
        function's docstring), binary + weighted Jaccard on `usage` instead of Jaccard +
        cosine on decoder content. `usage` is already sitting in trial_records (cheap,
        built by check_codebook's extract_usage pass) — no fresh forward pass needed for
        this panel anymore, unlike the identity-consistency one below which still does."""
        plot_stamp_similarity(
            os.path.join(viz_dir, 'stamp_similarity.png'), trial_records,
            unit_label=self.unit_label, seed=seed)

        max_trials_per_group = 60
        rng = random.Random(seed)
        sample = trial_records if len(trial_records) <= max_trials_per_group else \
            rng.sample(trial_records, max_trials_per_group)
        self._render_identity_consistency(sample, viz_dir, model, device)

    @torch.no_grad()
    def _render_identity_consistency(self, trial_records, viz_dir, model, device):
        """Does one stamp id mean one thing across patches/trials? The waveform half is
        trivially yes (D_i is a fixed parameter), so this measures the TOPOGRAPHY: every
        occurrence's mixing column, compared within-id vs between-id. Nothing in the
        architecture ties a stamp across patches — group selection binds channels within
        a patch only — so this is a real open question, not a formality. See
        viz.codebook.plot_stamp_identity_consistency for the metric's construction (and
        the two biases it has to avoid)."""
        from collections import defaultdict
        from viz.codebook import plot_stamp_identity_consistency
        from viz.iclabel import ICLABEL_CLASSES

        # Keyed by (dataset, stamp id): channel-validity differs per dataset (e.g. Dial
        # maps 8 of 64 channels, BETA_4s 58), so mixing columns from different datasets
        # have different lengths AND live in different channel subspaces — comparing
        # them would be meaningless even if the shapes matched. Statistics are computed
        # within each dataset and pooled.
        cols, labels = defaultdict(list), defaultdict(list)
        # ab_cols: same (dataset, id) keying, but the raw signed (a, b) pair per valid
        # channel per firing (not just magnitude) — feeds _render_topography_distance's
        # coherent per-channel average below. phase_cols: dataset-agnostic (a scalar, not
        # a channel-shaped vector, so pooling across datasets is fine) — every firing's
        # OVERALL phase (channel-summed complex value's angle), feeds the phase
        # consistency panel.
        ab_cols, phase_cols = defaultdict(list), defaultdict(list)
        for t in trial_records:
            ds_name = t.get('dataset', '_')
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])
            B, C, N, L = x_in.shape
            z, _ = model.stage_features(x_in, c_in, time_idx=t_in)
            z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, -1)
            x_g = x_in.permute(0, 2, 1, 3).reshape(B * N, C, L)
            rms = x_g.pow(2).mean(-1, keepdim=True).sqrt()
            vg = vc_in.unsqueeze(1).expand(B, N, C).reshape(B * N, C)
            o = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vg)
            mag = o.amp.pow(2).sum(-1).sqrt()                      # [G, C, K]
            m = vc_in[0].bool()
            ab_sum = o.amp[:, m, :, :].sum(dim=1)                  # [G, K, 2] channel-summed (a, b)
            phase = torch.atan2(ab_sum[..., 1], ab_sum[..., 0])    # [G, K] this firing's overall phase
            for g in range(mag.shape[0]):
                for k in range(o.idx.shape[1]):
                    sid = int(o.idx[g, k])
                    cols[(ds_name, sid)].append(mag[g, m, k].detach().cpu().numpy())
                    ab_cols[(ds_name, sid)].append(o.amp[g, m, k, :].detach().cpu().numpy())
                    phase_cols[sid].append(float(phase[g, k]))
            gal = extract_flat_stamp_gallery(model, x_in, c_in, time_idx=t_in,
                                              valid_channels=vc_in, fs=200, freq_resolution=0.2)
            uids, probs = gal[0], gal[-1]
            if probs is not None:
                for qi, sid in enumerate(uids.tolist()):
                    if np.all(np.isfinite(probs[qi])):
                        labels[int(sid)].append(int(probs[qi].argmax()))  # class is dataset-agnostic

        n_stamps = model.n_stamps
        circ_var = np.full(n_stamps, np.nan)
        fire_count = np.zeros(n_stamps, dtype=np.int64)
        for sid, ph in phase_cols.items():
            ph = np.asarray(ph)
            fire_count[sid] = len(ph)
            circ_var[sid] = 1.0 - np.abs(np.exp(1j * ph).mean())
        plot_stamp_phase_consistency(
            os.path.join(viz_dir, 'stamp_phase_consistency.png'), circ_var, fire_count,
            model.n_routed_stamps, unit_label=self.unit_label)

        def prep(a):
            # center across channels then unit-norm: raw magnitude columns are
            # non-negative, so their cosines sit near 1 regardless of structure
            a = np.asarray(a, dtype=float)
            a = a - a.mean(-1, keepdims=True)
            return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

        keys = [k for k, c in cols.items() if len(c) >= 6]
        by_ds = defaultdict(list)
        for ds_name, sid in keys:
            by_ds[ds_name].append(sid)
        if not any(len(v) >= 2 for v in by_ds.values()):
            print('  [codebook] identity consistency skipped (too few repeated stamps)')
            return
        P = {k: prep(np.stack(cols[k])) for k in keys}

        # Topography distance matrix, per dataset: same coherent per-firing (a, b) average
        # + reference-phase projection _used_flat_stamps uses for one trial's amp_topo, here
        # pooled across every firing in this dataset's sampled trials instead of one trial's
        # N patches — see plot_topography_distance's docstring for what this adds on top of
        # decoder_fingerprint_matrix (raw waveform shape) and the within/between stats above
        # (aggregate only, not a full pairwise view).
        mats = {}
        for ds_name, sids in by_ds.items():
            if len(sids) < 2:
                continue
            topo_vecs = []
            for sid in sids:
                mean_ab = np.stack(ab_cols[(ds_name, sid)]).mean(axis=0)   # [C, 2], coherent average
                ref = mean_ab.mean(axis=0)
                ref = ref / (np.linalg.norm(ref) + 1e-8)
                signed = mean_ab @ ref                                     # [C] signed projection
                topo_vecs.append(signed / (np.linalg.norm(signed) + 1e-8))
            V = np.stack(topo_vecs)                                        # [n, C]
            mats[ds_name] = (1.0 - V @ V.T, sids)
        plot_topography_distance(
            os.path.join(viz_dir, 'stamp_topography_distance.png'), mats, unit_label=self.unit_label)

        rng = np.random.default_rng(0)
        within, between, per_stamp, ids = [], [], [], []
        for ds_name, sids in by_ds.items():
            if len(sids) < 2:
                continue
            for sid in sids:
                U = P[(ds_name, sid)]; S = U @ U.T; n = len(U)
                v = float((S.sum() - n) / (n * (n - 1)))
                within.append(v); per_stamp.append(v); ids.append(sid)
            # between-id pairs drawn WITHIN this dataset, between INDIVIDUAL occurrences
            # (same footing as within — averaged columns would look falsely self-similar)
            for _ in range(4000 // max(1, len(by_ds))):
                a, b = rng.choice(len(sids), 2, replace=False)
                Ua, Ub = P[(ds_name, sids[a])], P[(ds_name, sids[b])]
                between.append(float(Ua[rng.integers(len(Ua))] @ Ub[rng.integers(len(Ub))]))
        agree = {s_: float(np.bincount(ls).max() / len(ls))
                 for s_, ls in labels.items() if len(ls) >= 3}
        plot_stamp_identity_consistency(
            os.path.join(viz_dir, 'stamp_identity_consistency.png'),
            np.asarray(within), np.asarray(between), ids, per_stamp,
            label_agree=agree or None, unit_label=self.unit_label)

    def _render_patch_position_consistency(self, ds_trials, ds_name, viz_dir, model, device, seed):
        """Overrides the base's usage/gating-based panel with a content-based one (real
        decoder output, channel-collapsed to stay a [N, D] per-trial code like the base's
        `usage` — see extract_stamp_content and plot_patch_position_consistency's
        code_label). Same expensive-dense-decode-on-a-subsample tradeoff as
        _render_patch_similarity (see needs_raw_tensors): re-runs a fresh forward pass on
        only a small subsample of this dataset's trials, not every trial check_codebook
        sampled for it."""
        max_trials_per_group = 60
        rng = random.Random(seed)
        sample = ds_trials if len(ds_trials) <= max_trials_per_group else \
            rng.sample(ds_trials, max_trials_per_group)

        content_records = []
        for t in sample:
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])
            content = self.extract_stamp_content(model, x_in, c_in, t_in, vc_in)  # [C, N, n_stamps, patch_len]
            collapsed = content.mean(axis=0).reshape(content.shape[1], -1)  # [N, n_stamps*patch_len]
            content_records.append(dict(usage=collapsed, dataset=t['dataset'], subject=t['subject']))

        plot_patch_position_consistency(
            os.path.join(viz_dir, f'patch_position_consistency_{ds_name}.png'), content_records,
            unit_label=self.unit_label, seed=seed, code_label='decoder output')

    @torch.no_grad()
    def _render_event_stamp_dynamics(self, ds_trials, ds_name, viz_dir, model, device, seed, config):
        """Event-locked stamp-selection / power trajectory -> event_stamp_dynamics_<ds_name>.png.

        The tokenizer was trained at one patch stride; to read selection/power on a finer
        time axis WITHOUT an out-of-distribution token spacing, this does a sliding-window
        eval: for each sub-stride offset it re-patchifies the raw trial at the NATIVE
        stride (every forward pass in-distribution), runs the stamp bank, and places each
        patch's result at its true sample time. Pooled over trials x offsets -> per-time-bin
        selection rate and power. Trials are onset-aligned (assemble_trials=False in the
        analysis path), so a fixed time within the trial is comparable across trials.

        Event onset per dataset: config['check']['event_onset_sample'] ({ds: samples} dict
        or a scalar for all); absent -> trajectory + heatmap only, no pre/post split. Same
        expensive-on-a-subsample tradeoff as _render_patch_position_consistency."""
        if not ds_trials:
            return
        from collections import defaultdict
        pp = config.get('preprocess_params', {})
        fs = pp.get('sample_freq')
        patch_len = pp.get('patch_length', 100)
        native_stride = pp.get('patch_stride', patch_len)

        eo = config.get('check', {}).get('event_onset_sample', {})
        onset = eo.get(ds_name) if isinstance(eo, dict) else eo
        # `is not None`, not truthiness -- an onset of literal 0 (event at trial start,
        # e.g. BCICIV1_Train/Inria_Train/EEGMMIdb/BCICIV2a, all trigger-cut with no
        # pre-event buffer, see config/analysis.json's event_onset_sample comment) is a
        # real, legitimate value. `onset and fs` treated 0 as falsy and silently fell
        # through to "not configured", which would have made every 0 entry a no-op.
        onset_sec = (onset / fs) if (onset is not None and fs) else (float(onset) if onset is not None else None)

        n_off = next((k for k in (5, 4, 6, 3, 2) if native_stride % k == 0), 1)
        fine = native_stride // n_off
        offsets = list(range(0, native_stride, fine))

        # This panel pools over trials, so it benefits from more of them than the
        # dense-content panels can afford — its own knob, defaulting higher. Still bounded
        # by check_codebook's max_trials_per_dataset (the pool it subsamples from).
        max_trials = config.get('check', {}).get('codebook', {}).get('event_max_trials', 200)
        rng = random.Random(seed)
        sample = ds_trials if len(ds_trials) <= max_trials else rng.sample(ds_trials, max_trials)

        n_stamps = int(model.n_stamps)
        # One accumulator per time-bin center `c`, not six co-indexed dicts — keeps the
        # per-bin fields (sel/amp/obs/pow/pow_sq/h) from being able to drift out of sync.
        bins = defaultdict(lambda: {'sel': np.zeros(n_stamps), 'amp': np.zeros(n_stamps),
                                     'obs': 0, 'pow': 0.0, 'pow_sq': 0.0, 'h': 0.0})

        for t in sample:
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])   # x_in [1,C,N,L] native-stride patches
            xp = x_in[0]                                                  # [C, N, L]
            C, N, L = xp.shape
            take = min(native_stride, L)                                 # =native_stride when stride<=len
            # rebuild the raw [C, T] the patches were cut from — same stitcher used
            # everywhere else in this file (see the two call sites above). Overlapping raw
            # patches are byte-identical copies of the same real samples (unlike a model's
            # recon), so overlap_add_patches's crossfade just averages identical values in
            # the overlap zone, same result as the old first-`take`-samples tiling. Samples
            # past the last patch's end were dropped at slice time and are unrecoverable
            # (fine — the model never saw them either).
            raw = overlap_add_patches(xp, take)

            for off in offsets:
                xps, tidx = slice_patches(raw[:, off:], patch_len, native_stride)  # [C, P, L]
                if xps.shape[1] == 0:
                    continue
                grid = extract_flat_stamp_psd_by_patch(
                    model, xps.unsqueeze(0), c_in, time_idx=tidx.unsqueeze(0).to(device),
                    valid_channels=vc_in, fs=fs, freq_resolution=None, patch_stride=1)
                ids, hh = grid.stamp_ids, grid.h                          # [P, K]
                re = (grid.recon_topo ** 2).mean(axis=1)                  # [P]
                for pi in range(ids.shape[0]):
                    c = off + pi * native_stride + patch_len // 2
                    b = bins[c]
                    b['obs'] += 1; b['pow'] += re[pi]; b['pow_sq'] += re[pi] ** 2; b['h'] += hh[pi].mean()
                    for k, sid in enumerate(ids[pi]):
                        b['sel'][sid] += 1; b['amp'][sid] += hh[pi, k]

        centers = np.array(sorted(bins))
        if len(centers) == 0:
            return
        B = len(centers)
        sr, am = np.zeros((n_stamps, B)), np.zeros((n_stamps, B))
        pm, ps_, hm = np.zeros(B), np.zeros(B), np.zeros(B)
        for j, c in enumerate(centers):
            b = bins[c]
            o = b['obs']
            sr[:, j] = b['sel'] / o
            am[:, j] = np.divide(b['amp'], b['sel'], out=np.zeros(n_stamps), where=b['sel'] > 0)
            pm[j] = b['pow'] / o
            ps_[j] = np.sqrt(max(b['pow_sq'] / o - pm[j] ** 2, 0.0))
            hm[j] = b['h'] / o
        t_axis = centers / fs if fs else centers.astype(float)

        plot_event_stamp_dynamics(
            os.path.join(viz_dir, f'event_stamp_dynamics_{ds_name}.png'),
            t_axis, sr, am, pm, ps_, hm, onset_sec=onset_sec, unit_label=self.unit_label,
            title_suffix=f' — {ds_name} ({len(sample)} trials x {len(offsets)} offsets, '
                         f'step {fine} samp)')


class MeSAEPlotter(BasePlotter):
    def plot_pretrain(self, filename='training_dashboard.png'):
        # Grouped: loss/reconstruction -> SAE health -> routing (Stamp/FFN side by side,
        # directly comparable) -> architecture diagnostics. Order is the only grouping lever
        # `render`'s flat ncols grid gives us — no row breaks/section labels, so panels of a
        # group may still straddle a row edge.
        loss_panels = [
            dict(title='Total Loss\n(recon + sparsity + aux + ffn_lb, weighted)', ylabel='Loss',
                 series=[dict(key='loss', color='b')]),
            # One panel: masked/unmasked split only exists at patch level; mse_patch is their
            # mix, mse_trial the overlap-added real-trial MSE (MeSAE._recon_loss). masked is a
            # 1.0 placeholder during the tokenizer phase.
            dict(title='Recon MSE: masked / unmasked / patch / trial\n(masked=1.0 placeholder in tokenizer phase)',
                 ylabel='MSE',
                 series=[dict(key='masked', color='crimson'),
                         dict(key='unmasked', color='steelblue'),
                         dict(key='mse_patch', color='darkorchid', label='mse_patch'),
                         dict(key='mse_trial', color='darkorange', label='mse_trial')]),
        ]

        stamp_health_panels = [
            dict(title='Stamp Aux-K Loss (dead-atom revival)\n[train only, 0 in eval by design]',
                 ylabel='Aux loss', series=[dict(key='aux', color='darkorange', train_only=True)]),
            dict(title='Dead Feature Rate (left) + Effective Atoms per Token (right)\n'
                       '(k_eff = (sum|a|)^2 / sum(a^2) — 1 = one atom carries all, top_k = all equal)',
                 ylabel='Dead fraction',
                 series=[dict(key='dead_feature_rate', color='crimson')],
                 twin=dict(ylabel='k_eff', series=[dict(key='k_eff', color='darkorchid')])),
        ]

        # Stamp Router Health — see MeSAE.update_stamp_router_metrics for what each number
        # means. Same panel shape as MeFSQ's, via router_health_series. No shared-stamp
        # series here — shared stamps have no on/off dynamics (constant weight), so a
        # separate line would just be flat (see docs/adr/0009's Monitoring impact section).
        stamp_router_series, stamp_twin_series = self.router_health_series(
            'stamp', entropy_label='Router entropy (load balance)')

        # FFN Router Health — same 3 metrics as the Stamp router above, but for the MoEFFN
        # routers inside every TSABlock (averaged across blocks), see
        # MeSAE.update_ffn_router_metrics / docs/adr/0008-moe-ffn-for-mesae.md. A distinct
        # MoE from the Stamp router — kept as its own panel rather than merged, so either
        # one collapsing is visible without the other's curves crowding it out.
        ffn_router_series, ffn_twin_series = self.router_health_series(
            'ffn', entropy_label='Router entropy (load balance)')

        routing_panels = [
            dict(title='Stamp Router Health\n(entropy rising = healthy spread; falling = collapse)',
                 ylabel='Entropy (higher=balanced)', series=stamp_router_series,
                 twin=dict(ylabel='Load std / LB loss', series=stamp_twin_series) if stamp_twin_series else None),
            dict(title='FFN Router Health\n(entropy rising = healthy spread; falling = collapse)',
                 ylabel='Entropy (higher=balanced)', series=ffn_router_series,
                 twin=dict(ylabel='Load std / LB loss', series=ffn_twin_series) if ffn_twin_series else None),
        ]

        architecture_panels = [
            dict(title='Residual-Add Skip Gates\n(0=drop skip, 1=plain add)',
                 ylabel='sigmoid(gate)', series=self.indexed_series('skip_gate_')),
            dict(title='Per-Block Contribution Norm\n(flat near-zero = block not used; pool-boundary blocks '
                       'already have their skip gate folded in)',
                 ylabel='Mean |delta| per block', series=self.indexed_series('block_norm_', cmap_name='viridis')),
        ]

        panels = loss_panels + stamp_health_panels + routing_panels + architecture_panels
        self.render(panels, filename, suptitle='Tokenizer (Stamp) Training Dashboard', ncols=4)

    def plot_finetune(self, filename='training_dashboard.png', freeze_backbone=False):
        panels = [
            dict(title='Total Loss', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Accuracy', ylabel='Acc', series=[dict(key='acc', color='crimson')]),
            dict(title='F1 (macro)', ylabel='F1', series=[dict(key='f1', color='steelblue')]),
            dict(title='F1 (weighted)', ylabel='F1', series=[dict(key='f1_weighted', color='teal')]),
            dict(title='Balanced Accuracy', ylabel='Bal. Acc', series=[dict(key='balanced_acc', color='darkorchid')]),
            dict(title="Cohen's Kappa", ylabel='Kappa', series=[dict(key='kappa', color='seagreen')]),
            dict(title='Backbone Recon MSE', ylabel='MSE', series=[dict(key='recon_mse', color='darkorange')]),
        ]
        self.render(panels, filename, suptitle='Training Dashboard')


PLUGIN = BasePlugin(
    build=build_model,
    finetune_cls=build_finetune,
    trainer_cls=MeSAETrainer,
    checker_cls=MeSAEChecker,
    plotter_cls=MeSAEPlotter,
    codebook_checker_cls=MeSAECodebookChecker,
)
