"""MeSAEFlat's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import os
import random

import numpy as np
import torch

from model.MeSAEFlat.MeSAEFlat import MeSAEFlatPretrain, MeSAEFlatFinetune
from model.base_trainer import BaseTrainer
from model.base_checker import BaseEpochChecker
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from viz.extract import (extract_flat_stamp_psd, extract_flat_stamp_psd_by_patch,
                          extract_flat_stamp_gallery, extract_filter_spectra)
from viz.panels import plot_attn_topo as render_attn_topo, plot_topo_psd_by_patch, plot_stamp_gallery
from viz.codebook import plot_stamp_similarity, plot_patch_position_consistency


@torch.no_grad()
def _run_reconstruction_sae(model, dataset, trial_idx, device):
    """Always runs unmasked (bool_masked_pos not passed) for a clean reconstruction
    snapshot, regardless of whether the model is currently in the Masked training stage."""
    x_patches, coords, _, time_indices, _, _, valid_channels = dataset[trial_idx]
    C, N, L = x_patches.shape
    T_total = N * L
    fs = dataset.base_dataset.config['preprocess_params']['sample_freq']

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
    """bp: config['model_params']['MeSAEFlat']['pretrain'] (or ['tokenizer'])."""
    sb = bp.get('stamp_bank', {})
    moe_ffn = bp.get('moe_ffn', {})

    return MeSAEFlatPretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=moe_ffn.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        pool_after_blocks=bp.get('pool_after_blocks', []),
        num_channels=num_channels,
        n_stamps=sb.get('n_stamps', 800),
        n_shared_stamps=sb.get('n_shared_stamps', 4),
        stamp_top_k=sb.get('stamp_top_k', 32),
        stamp_hidden_width=sb.get('stamp_hidden_width', 8),
        stamp_shared_hidden_width=sb.get('stamp_shared_hidden_width', 16),
        stamp_shared_weight=sb.get('shared_weight', 0.2),
        dead_threshold_frac=sb.get('dead_threshold_frac', 0.1),
        aux_k_cap_frac=sb.get('aux_k_cap_frac', 0.04),
        stamp_ema_decay=sb.get('sae_ema_decay', 0.999),
        n_routed_ffn_experts=moe_ffn.get('n_routed_experts', 4),
        n_shared_ffn_experts=moe_ffn.get('n_shared_experts', 1),
        ffn_top_k=moe_ffn.get('top_k', 2),
        ffn_expert_hidden=moe_ffn.get('expert_hidden'),
    )


class MeSAEFlatTrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, masked_mse_weight, unmasked_mse_weight, warmup, **hparams):
        # masked_mse_weight/unmasked_mse_weight are computed generically by train_pretrain.py
        # for every model type but MeSAE's loss no longer uses them — see get_loss.
        aux_weight = hparams.get('aux_weight', 0.03)
        hierarchical_mse_weight = hparams.get('hierarchical_mse_weight', 1.0)
        decorr_weight = hparams.get('decorr_weight', 0.01)
        ffn_lb_weight = hparams.get('ffn_lb_weight', 0.01)
        return model.get_loss(x, out.recon, out.aux_loss, bool_masked_pos=mp,
                               aux_weight=aux_weight, hierarchical_mse_weight=hierarchical_mse_weight,
                               decorr_loss=out.decorr_loss, decorr_weight=decorr_weight,
                               ffn_lb_loss=out.ffn_lb_loss, ffn_lb_weight=ffn_lb_weight,
                               valid_channels=out.valid_channels)

    def update_diagnostics(self, model, out):
        model.update_stamp_router_metrics(out.dense_routed)
        model.update_ffn_router_metrics(out.ffn_router_entropy, out.ffn_router_load_std, out.ffn_gate_entropy)

    def epoch_metrics(self, model, out):
        # mse_level_* are accumulated per-batch and epoch-averaged in train_pretrain.py
        # (train_one_epoch/validate_one_epoch), not added here — this function only ever
        # sees the last batch's out, which would make mse_level_* a last-batch snapshot
        # instead of an epoch average like every other loss stat.
        metrics = model.get_metrics(out.dense_routed.detach())
        metrics['aux'] = out.aux_loss.item() if hasattr(out.aux_loss, 'item') else float(out.aux_loss)
        metrics['ffn_lb_loss'] = out.ffn_lb_loss.item() if hasattr(out.ffn_lb_loss, 'item') else float(out.ffn_lb_loss)
        return metrics

    def on_tokenizer_start(self, model, logger=None):
        """Override BaseTrainer's generic hook (which enables spatial+temporal
        together): MeSAEFlat's Tokenizer stage must train StampBank on patch-local,
        SINGLE-CHANNEL content only. Enabling spatial mixing here would let the
        encoder leak cross-channel signal into each (channel, patch) token before
        StampBank ever sees it, defeating the point of per-channel stamps -- a "stamp"
        would then just encode a mixed vector again, same failure mode flat tokens
        were built to avoid. enable_temporal only (cross-patch, same-channel context
        stays -- needed for the UNet pool/upsample low-frequency reconstruction path);
        enable_spatial deferred to on_pretrain_start, once StampBank is frozen and
        the transformer is the only thing left learning from cross-channel context."""
        if hasattr(model, 'enable_temporal'):
            model.enable_temporal()
        if logger:
            logger.info("  [Tokenizer] temporal enabled, spatial OFF (stamps stay single-channel)")

    def on_pretrain_start(self, model, logger=None):
        """freeze_stamps() first, then enable_spatial(): StampBank must already be
        locked before cross-channel signal ever reaches it, so the frozen dictionary
        never trains on (and can't be re-opened by) mixed content -- only the
        transformer's masked-reconstruction prediction gets to use cross-channel
        context from here on, not the stamps themselves."""
        model.freeze_stamps()
        model.enable_spatial()
        if logger:
            logger.info("  [Pretrain] StampBank frozen, spatial enabled (transformer now sees cross-channel context)")


class MeSAEFlatChecker(BaseEpochChecker):
    unit_label = 'Stamp'
    # Flat-token StampBank has no cross-channel pool left to produce a channel-attention
    # map from (see MeSAEFlat_modules.StampBank class docstring) — the topo_psd_by_stamp panel
    # (_render_topo_psd override below) now covers per-stamp channel topography instead.
    has_attn_topo = False

    def compute_unit_colors(self, model, out):
        """red = shared stamp (always-on, structural). black = routed stamp. Restricted to
        stamps actually used somewhere in this trial, capped at 100 (see
        MeSAEFlatPretrain.used_stamp_ids) — with n_stamps=800 and hard top-k selection, showing
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
        - topo_psd_by_patch.png — the real per-patch grid (every patch_stride-th patch's
          own union of stamps its C channels individually selected, zero-filled per
          channel that didn't pick a given displayed stamp — see
          viz.extract.extract_flat_stamp_psd_by_patch). The trial-wide dedup can't tell
          "this stamp fired on 1 patch" from "fired on every patch" apart; the per-patch
          grid can.
        - stamp_gallery.png — the whole-trial Raw/Full-Recon view plus every stamp used
          SOMEWHERE in this trial (trial-wide dedup, see
          viz.extract.extract_flat_stamp_gallery), the piece the base default's
          topo_psd_filter.png would have covered — split into its own file rather than
          folded into topo_psd_by_patch's header, since it's a different (trial-wide, not
          per-patch) view. psd_ch_x/importance (from extract_psd, via _render_snapshot)
          are still computed upstream since _render_snapshot uses that call to gate
          whether to attempt this panel at all, but this method recomputes its own
          (used_ids-carrying) copy via extract_flat_stamp_gallery rather than reusing
          those — see that function's docstring for why."""
        model = bundle.psd_model
        grid = extract_flat_stamp_psd_by_patch(
            model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
            fs=fs, freq_resolution=fft_resolution)

        # Same raw/recon full-trial FFT as the base default, see BaseEpochChecker._render_topo_psd.
        raw_t   = bundle.raw_t[0].numpy()
        recon_t = bundle.recon_t[0].numpy()
        T = raw_t.shape[-1]
        n_fft = max(T, int(round(fs / fft_resolution))) if fs else T

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

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_topo_psd_by_patch.png")
        plot_topo_psd_by_patch(
            out_path, pos2d, grid, cmap=cmap,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag,
            unit_label=self.unit_label, n_routed=model.n_routed_stamps,
        )
        print(f"  [epoch] -> {out_path}")

        used_ids, gal_importance, psd_ch_x_g, psd_x_g, gal_freqs = extract_flat_stamp_gallery(
            model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
            fs=fs, freq_resolution=fft_resolution)
        if band is not None:
            gal_freqs = gal_freqs[band]
            psd_x_g = psd_x_g[:, :, band]

        gallery_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_stamp_gallery.png")
        plot_stamp_gallery(
            gallery_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
            psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=cmap,
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=tagged_epoch_tag,
            unit_label=self.unit_label, unit_ids=used_ids, n_routed=model.n_routed_stamps,
        )
        print(f"  [epoch] -> {gallery_path}")

    def render_finetune_attn(self, model, x_in, c_in, t_in, vc_in, valid_channels, valid_length,
                              P, patch_len, viz_dir, epoch_tag, subject_id, trial_idx,
                              pos2d, channel_names, unit_colors):
        """MeSAEFlatFinetune's head has no channel dim (already collapsed into each stamp's
        View by the backbone's own channel-attention pool) — so the topomaps read that real
        channel attention straight from the backbone as-is; scaling it by each stamp's
        stamp-attention score (attn_h) was tried to make stamps "visually comparable" but
        with many low-importance stamps sharing one color scale, it just crushes most
        topomaps toward the scale's dark end instead. Topo values are averaged uniformly
        over patches — NOT weighted by the head's temporal attention (attn_n) — since a
        topomap has no time axis to justify a time-weighted average. The big heatmap
        instead shows attn_n (Patch x Stamp), replacing the base class's Channel x Unit
        heatmap.

        Uses encode_used_stamps, NOT encode_post_stamp_expert: hard top-k means a
        per-patch Q axis has no stable identity across patches (patch A's slot 0 and patch
        B's slot 0 can be different physical stamps), so it can't be fed into a trial-wide
        panel directly. encode_used_stamps instead returns a single fixed set of <=100
        global stamp ids actually used somewhere in this trial (see
        MeSAEFlatPretrain.used_stamp_ids) with each patch's real strength for exactly those
        ids. The `unit_colors` param (from check_finetune's own, separate `backbone(...)`
        forward pass) is ignored — colors are recomputed here from THIS call's own
        used_ids so they're guaranteed to match, rather than trusting two independent
        forward passes to agree on ranking/order."""
        pad_mask = self._build_pad_mask_time(valid_length, P, patch_len).to(x_in.device)
        backbone = model.backbone
        z_h, chan_attn, used_ids = backbone.encode_used_stamps(
            x_in, c_in, time_idx=t_in, valid_channels=vc_in, max_stamps=100)  # [1,N,Qu,D], [1,N,Qu,C], [Qu]
        chan_attn = chan_attn[0].mean(dim=0).detach().cpu().numpy()  # [Qu, C] — uniform mean over N, no temporal weight
        colors = ['red' if i >= backbone.n_routed_stamps else 'black' for i in used_ids.tolist()]

        # feed the head directly instead of re-running model(...) — that would repeat the
        # same backbone forward pass we just did to get chan_attn above
        _, attn_h, attn_n = model.head(z_h, pad_mask=pad_mask)
        importance = attn_h[0].detach().cpu().numpy()          # [Qu] — per-stamp stamp-attention score
        patch_filter_attn = attn_n[0].detach().cpu().numpy()   # [Qu, N] — per-stamp temporal attention

        out_path = os.path.join(viz_dir, f"sub{subject_id}_trial{trial_idx}{epoch_tag}_attn_topo.png")
        render_attn_topo(
            out_path, pos2d, chan_attn, importance, channel_names,
            valid_channels=valid_channels.numpy(),
            subject_id=subject_id, trial_idx=trial_idx, epoch_tag=f'{epoch_tag} [finetune]',
            unit_label=self.unit_label, unit_colors=colors,
            heatmap_attn=patch_filter_attn, heatmap_ylabels=list(range(patch_filter_attn.shape[1])),
            heatmap_ylabel='Patch (time)', heatmap_title=f'Patch x {self.unit_label} Attention',
            heatmap_transpose=False,
        )
        print(f"  [epoch] -> {out_path}")


class MeSAEFlatCodebookChecker(BaseCodebookChecker):
    unit_label = 'Stamp'
    needs_raw_tensors = True  # _render_patch_similarity needs a fresh forward pass per
    # trial (extract_stamp_content) — too expensive for check_codebook's full trial set,
    # see needs_raw_tensors' docstring on the base class.

    @torch.no_grad()
    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """[N, n_stamps] dense usage, one row per PATCH POSITION — routed axis real
        selection strength (zeros at unselected), shared axis the fixed constant weight
        every shared stamp always fires at (see docs/adr/0009's Monitoring impact
        section: no per-atom x per-feature F axis exists anymore, so this replaces the
        retired out.sae_hidden).

        Flat-token StampBank dispatches per (channel, patch) token (see
        MeSAEFlat_modules.StampBank), so out.dense_routed's row axis is T=C*N, channel-major
        (row t = c*N + n, same layout viz.extract._stamp_selection uses) — every
        downstream consumer of extract_usage (plot_patch_position_consistency in
        particular) expects one row per patch POSITION, comparable across trials of the
        same dataset, not C*N channel-patch pairs. Mean over channels collapses back to
        that [N, n_stamps] shape instead of silently leaking C*N rows as if they were N
        patch positions (that mismatch is what made patch_position_consistency panels
        ~C times wider than real N, one column per (channel, patch) instead of per patch)."""
        B, C, N, L = x_in.shape
        out = model(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
        shared = out.dense_routed.new_full((out.dense_routed.shape[0], model.n_shared_stamps), model.shared_weight)
        dense_full = torch.cat([out.dense_routed, shared], dim=-1)  # [T, n_stamps], T = C*N
        per_patch = dense_full.reshape(C, N, -1).mean(dim=0)  # [N, n_stamps]
        return per_patch.detach().cpu().numpy()

    def decoder_fingerprint_matrix(self, model):
        """Per-stamp [patch_len] waveform template D_i (see StampBank.fingerprint —
        content-free and exact now, no probe involved: D_i never depends on any input),
        pairwise cosine sim — this is `filter_relation.png`'s direct successor and
        doubles as the empirical test for whether StampBank.decorrelation_loss needs a
        temporal term added (see that method's docstring)."""
        fp = model.stamps.fingerprint().cpu().numpy()  # [n_stamps, patch_len]
        flat = fp.reshape(fp.shape[0], -1)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
        return flat @ flat.T

    def rank_ceiling(self, model):
        return min(model.stamps.top_k, model.head_dim)

    @torch.no_grad()
    def extract_stamp_content(self, model, x_in, c_in, t_in, vc_in):
        """Dense per-(channel,patch) DECODED CONTENT [C, N, n_stamps, patch_len],
        zero-filled at stamps that (channel, patch) token didn't select — real content
        where selected, exact 0 elsewhere (same zero-fill convention as viz.extract's
        flat-token panels, e.g. extract_flat_stamp_psd_by_patch). Unlike extract_usage (a
        scalar gating strength h per stamp), this is the actual decoder output — used only
        by _render_patch_similarity (viz.codebook.plot_stamp_similarity), which needs real
        content to compare, not just selection confidence.

        Expensive: T=C*N tokens x n_stamps x patch_len dense per trial (e.g. 64*16*120*50
        ~= 6M floats, ~25MB). _render_patch_similarity only calls this for a small
        subsample of trials (see needs_raw_tensors), not every trial check_codebook
        samples up front."""
        B, C, N, L = x_in.shape
        z, _ = model.stage_features(x_in, c_in, time_idx=t_in)
        z_flat = z.reshape(B * C * N, -1)
        out = model.stamps(z_flat, x_target=None)
        idx, h = out.idx, out.h
        contribution, _z_h = model.stamps.decode_selected(idx, h, z_flat)  # [T, K, patch_len]
        T, K, patch_len = contribution.shape
        n_stamps = model.n_stamps

        dense = contribution.new_zeros(T, n_stamps, patch_len)
        dense.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, patch_len), contribution)
        return dense.reshape(C, N, n_stamps, patch_len).cpu().numpy()

    def _render_patch_similarity(self, trial_records, viz_dir, model, device, seed):
        """Overrides the base's usage/gating-based hierarchy panel (patch_similarity_
        hierarchy.png, cosine over selection strength h) with a content-based one
        (stamp_similarity.png, cosine over real decoder output — see extract_stamp_content
        and viz.codebook.plot_stamp_similarity, including its new Intra-Patch grouping).
        Dense per-token decoder content is too expensive to compute for every trial
        check_codebook samples (see needs_raw_tensors), so this re-runs a fresh forward
        pass on only a small trial subsample — same max_trials_per_group cap
        plot_stamp_similarity itself would otherwise apply internally, just applied before
        the (expensive) extraction instead of after."""
        max_trials_per_group = 60
        rng = random.Random(seed)
        sample = trial_records if len(trial_records) <= max_trials_per_group else \
            rng.sample(trial_records, max_trials_per_group)

        content_records = []
        for t in sample:
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])
            content = self.extract_stamp_content(model, x_in, c_in, t_in, vc_in)
            content_records.append(dict(content=content, dataset=t['dataset'], subject=t['subject']))

        plot_stamp_similarity(
            os.path.join(viz_dir, 'stamp_similarity.png'), content_records,
            unit_label=self.unit_label, seed=seed)

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


class MeSAEFlatPlotter(BasePlotter):
    def plot_pretrain(self, filename='training_dashboard.png'):
        # Grouped: loss/reconstruction -> SAE health -> routing (Filter/FFN side by side,
        # directly comparable) -> architecture diagnostics. Order is the only grouping lever
        # `render`'s flat ncols grid gives us — no row breaks/section labels, so panels of a
        # group may still straddle a row edge.
        loss_panels = [
            dict(title='Total Loss (MSE + aux*weight)', ylabel='Loss', series=[dict(key='loss', color='b')]),
            dict(title='Masked vs Unmasked MSE\n(patch level, diagnostic only)', ylabel='Loss',
                 series=[dict(key='masked', color='crimson'), dict(key='unmasked', color='steelblue')]),
            dict(title='Recon MSE: Window vs Patch\n(mse_level_0=window avg, mse_level_1=patch — see MeSAEFlat._recon_loss)',
                 ylabel='MSE', series=self.indexed_series('mse_level_', cmap_name='plasma', train_only=False)),
        ]

        stamp_health_panels = [
            dict(title='Stamp Aux-K Loss (dead-atom revival)\n[train only, 0 in eval by design]',
                 ylabel='Aux loss', series=[dict(key='aux', color='darkorange', train_only=True)]),
            dict(title='Dead Feature Rate', ylabel='Fraction',
                 series=[dict(key='dead_feature_rate', color='crimson')]),
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
            dict(title='Per-Block Contribution Norm\n(flat near-zero = block not used)',
                 ylabel='Mean |delta| per block', series=self.indexed_series('block_norm_', cmap_name='viridis')),
            dict(title='Per-Block Contribution Norm, Relative\n(delta / incoming x norm — isolates real '
                       'reshaping from norm_out gain artifacts)',
                 ylabel='block_norm / x_in norm', series=self.indexed_series('block_relnorm_', cmap_name='viridis')),
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
    finetune_cls=MeSAEFlatFinetune,
    trainer_cls=MeSAEFlatTrainer,
    checker_cls=MeSAEFlatChecker,
    plotter_cls=MeSAEFlatPlotter,
    codebook_checker_cls=MeSAEFlatCodebookChecker,
)
