"""MeSAE's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import os
import random

import numpy as np
import torch

from model.MeSAE.MeSAE import MeSAEPretrain, build_finetune
from model.MeSAE.MeSAE_modules import overlap_add_patches
from model.base_trainer import BaseTrainer
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from tools.viz.extract import extract_flat_stamp_psd_by_patch, extract_flat_stamp_gallery
from tools.viz.stamp_plots import plot_event_stamp_dynamics
from tools.viz.codebook import (plot_stamp_similarity, plot_patch_position_consistency,
                           plot_stamp_identity_consistency, plot_fingerprint_similarity,
                           plot_stamp_phase_consistency, plot_topography_distance)
from tools.analysis import event_onset_patch, lookup_event_onset_sample
from IO.preprocessing import slice_patches


def build_model(bp, num_channels):
    """bp: config['model_params']['MeSAE']['pretrain']. stamp_bank: n_stamps, hidden_width (a static
    checkpoint's build_config from before docs/adr/0022 names them n_shared_stamps /
    stamp_shared_hidden_width, with n_routed_stamps 0 -- still read)."""
    sb = bp.get('stamp_bank', {})
    moe_ffn = bp.get('moe_ffn', {})
    if sb.get('n_routed_stamps', 0):
        raise ValueError("routed stamps were removed (docs/adr/0022): use the `routed-stamps` branch, "
                         "or set stamp_bank to {n_stamps, hidden_width}")

    return MeSAEPretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=moe_ffn.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        blocks_per_stage=bp.get('blocks_per_stage', 2),
        num_channels=num_channels,
        spatial_embedding=bp.get('spatial_embedding', True),
        n_stamps=sb.get('n_stamps', sb.get('n_shared_stamps', 16)),
        stamp_hidden_width=sb.get('hidden_width', sb.get('stamp_shared_hidden_width', 16)),
        # Both None (the default) = fully continuous (a, b), exactly as before
        # quantization existed. See StampBank._quantize_amp_phase.
        stamp_amp_levels=sb.get('amp_levels'),
        stamp_phase_levels=sb.get('phase_levels'),
        stamp_amp_log2_range=tuple(sb.get('amp_log2_range', (-6.0, 3.0))),
        n_routed_ffn_experts=moe_ffn.get('n_routed_experts', 4),
        n_shared_ffn_experts=moe_ffn.get('n_shared_experts', 1),
        ffn_top_k=moe_ffn.get('top_k', 2),
        # patch_stride duplicates preprocess_params (same convention as patch_len
        # above) — the shared build_model(bp, num_channels) interface
        # (model/factory.py) doesn't pass preprocess_params through.
        patch_stride=bp.get('patch_stride'),
        # 'gated' UNet skips (default), 'finest' (only the finest skip) or 'none'; decoder_blocks per-channel temporal conv blocks after each upsample
        skip_mode=bp.get('skip_mode', 'gated'),
        decoder_blocks=bp.get('decoder_blocks', 0),
        skip_drop=bp.get('skip_drop', 0.0),   # per-sample skip drop-path p (training); a list = one per skip, finest first
        temporal_bias=bp.get('temporal_bias', False),   # RelativeTemporalBias on temporal attention
    )


class MeSAETrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, **hparams):
        removed = {'hierarchical_mse_weight': '0011', 'mse_trial_weight': '0021', 'stft_weight': '0019',
                   'stft_sizes': '0019', 'nested_sizes': '0018', 'nested_weights': '0018', 'aux_weight': '0022'}
        stale = sorted(k for k in hparams if k in removed and hparams[k])
        if stale:   # a removed loss term: fail loudly rather than train a silently different loss
            raise ValueError(f"loss keys {stale} were removed (docs/adr/{', '.join(sorted({removed[k] for k in stale}))}); "
                             "drop them from the config")
        ffn_lb_weight = hparams.get('ffn_lb_weight', 0.01)
        mp_weight = hparams.get('mp_weight', 0.0)
        return model.get_loss(x, out.recon, bool_masked_pos=mp,
                               mse_patch_weight=hparams.get('mse_patch_weight', 1.0),
                               unmasked_weight=hparams.get('unmasked_weight', 1.0),
                               ffn_lb_loss=out.ffn_lb_loss, ffn_lb_weight=ffn_lb_weight,
                               valid_channels=out.valid_channels,
                               mp_loss=out.mp_loss, mp_weight=mp_weight, mp_map=out.mp_map)

    def update_diagnostics(self, model, out):
        model.update_ffn_router_metrics(out.ffn_router_entropy, out.ffn_router_load_std, out.ffn_gate_entropy)

    def epoch_metrics(self, model, out):
        # mse_patch (and mse_mp) are accumulated per-batch and epoch-averaged in
        # train_pretrain.py (train_one_epoch/validate_one_epoch), not added here — this
        # function only ever sees the last batch's out, which would make them a
        # last-batch snapshot instead of an epoch average like every other loss stat.
        metrics = model.get_metrics()
        metrics['ffn_lb_loss'] = out.ffn_lb_loss.item() if hasattr(out.ffn_lb_loss, 'item') else float(out.ffn_lb_loss)
        return metrics


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
        """[N, n_stamps] usage, one row per PATCH POSITION: each stamp's post-rms amp magnitude
        (StampBank.forward's h; G = N for a B=1 trial)."""
        out = model(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
        return out.h.detach().cpu().numpy()

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
        return min(model.n_stamps, model.head_dim)

    def _render_fingerprint_similarity(self, viz_dir, model):
        plot_fingerprint_similarity(
            os.path.join(viz_dir, 'stamp_fingerprint_similarity.png'),
            self.decoder_fingerprint_matrix(model), unit_label=self.unit_label)

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
        z, _ = model.stage_features(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
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
        from tools.viz.codebook import plot_stamp_identity_consistency
        from tools.viz.iclabel import ICLABEL_CLASSES

        # Keyed by (dataset, stamp id): channel-validity differs per dataset (e.g. Nakanishi2015
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
            z, _ = model.stage_features(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
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
            unit_label=self.unit_label)

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

    def _render_patch_position_consistency(self, ds_trials, ds_name, viz_dir, model, device, seed, config=None):
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
            unit_label=self.unit_label, seed=seed, code_label='decoder output',
            event_patch=event_onset_patch(config, ds_name) if config else None)

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

        Event onset per dataset: this dataset's own metadata.json event_onset_sample (see
        tools.analysis.lookup_event_onset_sample); absent -> trajectory + heatmap only, no
        pre/post split. Same expensive-on-a-subsample tradeoff as
        _render_patch_position_consistency."""
        if not ds_trials:
            return
        from collections import defaultdict
        pp = config.get('preprocess_params', {})
        fs = pp.get('sample_freq')
        patch_len = pp.get('patch_length', 100)
        native_stride = pp.get('patch_stride', patch_len)

        onset = lookup_event_onset_sample(config, ds_name)
        # `is not None`, not truthiness -- an onset of literal 0 (event at trial start,
        # e.g. BCICIV1_Train/Inria_Train/PhysionetMI/BNCI2014001, all trigger-cut with no
        # pre-event buffer, see that dataset's own metadata.json event_onset_sample_note)
        # is a real, legitimate value. `onset and fs` treated 0 as falsy and silently fell
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
        # Grouped: loss/reconstruction -> FFN routing -> architecture diagnostics. Order is the only grouping lever
        # `render`'s flat ncols grid gives us — no row breaks/section labels, so panels of a
        # group may still straddle a row edge.
        recon = ('masked', 'crimson'), ('unmasked', 'steelblue'), ('mse_patch', 'darkorchid')
        loss_panels = [
            dict(title="Total Loss (the training objective)\n(weighted sum of this run's loss terms: not comparable "
                       "across loss configs)", ylabel='Loss', series=[dict(key='loss', color='b')]),
            # Val is the comparable number: every skip on, no drop-path. masked is a 1.0 placeholder in
            # the tokenizer phase; mse_patch mixes masked and visible.
            dict(title='Val Recon MSE: compare runs here\n(all skips on; masked = 1.0 placeholder in tokenizer phase)',
                 ylabel='MSE', series=[dict(key=k, color=c, val_only=True, style_val='-') for k, c in recon]),
            dict(title='Train Recon MSE\n(skip drop-path active, so above val by design)',
                 ylabel='MSE', series=[dict(key=k, color=c, train_only=True) for k, c in recon]),
        ]
        if self.has_signal('mse_mp'):   # the per-stamp anti-duplicate term, when trained
            loss_panels[1]['series'].append(dict(key='mse_mp', color='gray', val_only=True, style_val='-', label='mp_loss'))

        # FFN Router Health — the MoEFFN routers inside every TSABlock (averaged across blocks),
        # see MeSAE.update_ffn_router_metrics / docs/adr/0008-moe-ffn-for-mesae.md.
        ffn_router_series, ffn_twin_series = self.router_health_series(
            'ffn', entropy_label='Router entropy (load balance)')

        routing_panels = [
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

        panels = loss_panels + routing_panels + architecture_panels
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
    plotter_cls=MeSAEPlotter,
    codebook_checker_cls=MeSAECodebookChecker,
)
