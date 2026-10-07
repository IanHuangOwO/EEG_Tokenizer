"""Qtome's implementation of the shared model-plugin contract (model/base_trainer.py,
model/base_checker.py, model/base_plotter.py)."""

import os
import random

import numpy as np
import torch

from model.Qtome.Qtome import QtomePretrain, build_finetune
from model.Qtome.Qtome_modules import overlap_add_patches
from model.base_trainer import BaseTrainer
from model.base_codebook_checker import BaseCodebookChecker
from model.base_plotter import BasePlotter
from model.base_plugin import BasePlugin
from tools.viz.extract import extract_atom_psd_by_patch
from tools.viz.atom_plots import plot_event_atom_dynamics
from tools.viz.codebook import (plot_atom_identity_consistency, plot_fingerprint_similarity,
                           plot_atom_phase_consistency, plot_topography_distance)
from tools.analysis import lookup_event_onset_sample
from IO.preprocessing import slice_patches


def build_model(bp, num_channels):
    """bp: config['model_params']['Qtome']['pretrain']."""
    sb = bp.get('atom_bank', {})
    moe_ffn = bp.get('moe_ffn', {})

    return QtomePretrain(
        embed_dim=bp.get('embed_dim', 100),
        enc_depth=bp.get('enc_depth', 12),
        mlp_ratio=moe_ffn.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 20),
        spatial_heads=bp.get('spatial_heads', 8),
        dropout=bp.get('dropout', 0.0),
        blocks_per_stage=bp.get('blocks_per_stage', 2),
        num_channels=num_channels,
        spatial_embedding=bp.get('spatial_embedding', True),
        n_atoms=sb.get('n_atoms', 16),
        atom_hidden_width=sb.get('hidden_width', 16),
        n_routed_ffn_experts=moe_ffn.get('n_routed_experts', 4),
        n_shared_ffn_experts=moe_ffn.get('n_shared_experts', 1),
        ffn_top_k=moe_ffn.get('top_k', 2),
        # patch_stride duplicates preprocess_params (same convention as patch_len
        # above) — the shared build_model(bp, num_channels) interface
        # (model/factory.py) doesn't pass preprocess_params through.
        patch_stride=bp.get('patch_stride'),
        skip_drop=bp.get('skip_drop', 0.0),   # per-sample skip drop-path p (training); a list = one per skip, finest first
    )


class QtomeTrainer(BaseTrainer):
    def compute_loss(self, model, x, out, mp, **hparams):
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


class QtomeCodebookChecker(BaseCodebookChecker):
    unit_label = 'Atom'
    needs_raw_tensors = True  # identity consistency and event dynamics re-run the Q-atom bank on a
    # subsample of trials (see needs_raw_tensors' docstring on the base class)

    @torch.no_grad()
    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """[N, n_atoms] usage, one row per PATCH POSITION: each Q-atom's post-rms amp magnitude
        (AtomBank.forward's h; G = N for a B=1 trial)."""
        return model.encode_atoms(x_in, c_in, time_idx=t_in, valid_channels=vc_in).h.cpu().numpy()

    def decoder_fingerprint_matrix(self, model):
        """Pairwise cosine similarity of the unit templates D_s (content-free: D never depends on
        input) -- the check on template diversity that mp_loss is meant to keep."""
        D = model.atoms.templates()[0].detach().cpu().numpy()        # [n_atoms, patch_len], unit rows
        return D @ D.T

    def _render_fingerprint_similarity(self, viz_dir, model):
        plot_fingerprint_similarity(
            os.path.join(viz_dir, 'atom_fingerprint_similarity.png'),
            self.decoder_fingerprint_matrix(model), unit_label=self.unit_label)

    @torch.no_grad()
    def _render_unit_consistency(self, trial_records, viz_dir, model, device, seed):
        """Identity, phase and topography consistency of the Q-atoms on a subsample of trials."""
        rng = random.Random(seed)
        sample = trial_records if len(trial_records) <= 60 else rng.sample(trial_records, 60)
        self._render_identity_consistency(sample, viz_dir, model, device)

    @torch.no_grad()
    def _render_identity_consistency(self, trial_records, viz_dir, model, device):
        """Does one Q-atom id mean one thing across patches/trials? The waveform half is
        trivially yes (D_s is a fixed parameter), so this measures the TOPOGRAPHY: every
        occurrence's mixing column, compared within-id vs between-id. Nothing in the
        architecture ties a Q-atom's topography across patches, so this is a real open
        question, not a formality. See
        viz.codebook.plot_atom_identity_consistency for the metric's construction (and
        the two biases it has to avoid)."""
        from collections import defaultdict
        from tools.viz.codebook import plot_atom_identity_consistency

        # Keyed by (dataset, Q-atom id): channel-validity differs per dataset (e.g. Nakanishi2015
        # maps 8 of 64 channels, BETA_4s 58), so mixing columns from different datasets
        # have different lengths AND live in different channel subspaces — comparing
        # them would be meaningless even if the shapes matched. Statistics are computed
        # within each dataset and pooled.
        cols = defaultdict(list)
        # ab_cols: same (dataset, id) keying, but the raw signed (a, b) pair per valid
        # channel per occurrence (not just magnitude) — feeds _render_topography_distance's
        # coherent per-channel average below. phase_cols: dataset-agnostic (a scalar, not
        # a channel-shaped vector, so pooling across datasets is fine) — every occurrence's
        # OVERALL phase (channel-summed complex value's angle), feeds the phase
        # consistency panel.
        ab_cols, phase_cols = defaultdict(list), defaultdict(list)
        for t in trial_records:
            ds_name = t.get('dataset', '_')
            x_in, c_in, t_in, vc_in = (v.to(device) for v in t['raw'])
            o = model.encode_atoms(x_in, c_in, time_idx=t_in, valid_channels=vc_in)
            m = vc_in[0].bool()
            amp = o.amp[:, m].cpu()                                # [G, Cv, S, 2]
            mag = amp.pow(2).sum(-1).sqrt()                        # [G, Cv, S]
            ab_sum = amp.sum(dim=1)                                # [G, S, 2] channel-summed (a, b)
            phase = torch.atan2(ab_sum[..., 1], ab_sum[..., 0])    # [G, S] each occurrence's overall phase
            for g in range(mag.shape[0]):
                for sid in range(mag.shape[2]):
                    cols[(ds_name, sid)].append(mag[g, :, sid].numpy())
                    ab_cols[(ds_name, sid)].append(amp[g, :, sid].numpy())
                    phase_cols[sid].append(float(phase[g, sid]))

        n_atoms = model.n_atoms
        circ_var = np.full(n_atoms, np.nan)
        fire_count = np.zeros(n_atoms, dtype=np.int64)
        for sid, ph in phase_cols.items():
            ph = np.asarray(ph)
            fire_count[sid] = len(ph)
            circ_var[sid] = 1.0 - np.abs(np.exp(1j * ph).mean())
        plot_atom_phase_consistency(
            os.path.join(viz_dir, 'atom_phase_consistency.png'), circ_var, fire_count,
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
            print('  [codebook] identity consistency skipped (too few repeated atoms)')
            return
        P = {k: prep(np.stack(cols[k])) for k in keys}

        # Topography distance matrix, per dataset: same coherent per-occurrence (a, b) average
        # + reference-phase projection _atom_summary uses for one trial's amp_topo, here
        # pooled across every occurrence in this dataset's sampled trials instead of one trial's
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
            os.path.join(viz_dir, 'atom_topography_distance.png'), mats, unit_label=self.unit_label)

        rng = np.random.default_rng(0)
        within, between, per_atom, ids = [], [], [], []
        for ds_name, sids in by_ds.items():
            if len(sids) < 2:
                continue
            for sid in sids:
                U = P[(ds_name, sid)]; S = U @ U.T; n = len(U)
                per_atom.append(float((S.sum() - n) / (n * (n - 1)))); ids.append(sid)
            # within- and between-id pairs both drawn WITHIN this dataset as pairs of INDIVIDUAL
            # occurrences, same count, so the two histograms (and their means) are comparable
            for _ in range(4000 // max(1, len(by_ds))):
                U = P[(ds_name, sids[rng.integers(len(sids))])]
                i, j = rng.choice(len(U), 2, replace=False)
                within.append(float(U[i] @ U[j]))
                a, b = rng.choice(len(sids), 2, replace=False)
                Ua, Ub = P[(ds_name, sids[a])], P[(ds_name, sids[b])]
                between.append(float(Ua[rng.integers(len(Ua))] @ Ub[rng.integers(len(Ub))]))
        plot_atom_identity_consistency(
            os.path.join(viz_dir, 'atom_identity_consistency.png'),
            np.asarray(within), np.asarray(between), ids, per_atom,
            unit_label=self.unit_label)

    @torch.no_grad()
    def _render_event_atom_dynamics(self, ds_trials, ds_name, viz_dir, model, device, seed, config):
        """Event-locked Q-atom-strength / power trajectory -> event_atom_dynamics_<ds_name>.png.

        The tokenizer was trained at one patch stride; to read Q-atom strength/power on a finer
        time axis WITHOUT an out-of-distribution token spacing, this does a sliding-window
        eval: for each sub-stride offset it re-patchifies the raw trial at the NATIVE
        stride (every forward pass in-distribution), runs the Q-atom bank, and places each
        patch's result at its true sample time. Pooled over trials x offsets -> per-time-bin
        Q-atom strength h and power. Trials are onset-aligned (assemble_trials=False in the
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

        n_atoms = int(model.n_atoms)
        # One accumulator per time-bin center `c`, so the per-bin fields can't drift out of sync.
        bins = defaultdict(lambda: {'amp': np.zeros(n_atoms), 'obs': 0, 'pow': 0.0, 'pow_sq': 0.0})

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
                grid = extract_atom_psd_by_patch(
                    model, xps.unsqueeze(0), c_in, time_idx=tidx.unsqueeze(0).to(device),
                    valid_channels=vc_in, fs=fs, freq_resolution=None, patch_stride=1)
                hh = grid.h                                               # [P, n_atoms]
                re = (grid.recon_topo ** 2).mean(axis=1)                  # [P]
                for pi in range(hh.shape[0]):
                    b = bins[off + pi * native_stride + patch_len // 2]
                    b['obs'] += 1; b['pow'] += re[pi]; b['pow_sq'] += re[pi] ** 2; b['amp'] += hh[pi]

        centers = np.array(sorted(bins))
        if len(centers) == 0:
            return
        B = len(centers)
        am = np.zeros((n_atoms, B))
        pm, ps_ = np.zeros(B), np.zeros(B)
        for j, c in enumerate(centers):
            b = bins[c]
            o = b['obs']
            am[:, j] = b['amp'] / o
            pm[j] = b['pow'] / o
            ps_[j] = np.sqrt(max(b['pow_sq'] / o - pm[j] ** 2, 0.0))
        t_axis = centers / fs if fs else centers.astype(float)

        plot_event_atom_dynamics(
            os.path.join(viz_dir, f'event_atom_dynamics_{ds_name}.png'),
            t_axis, am, pm, ps_, onset_sec=onset_sec, unit_label=self.unit_label,
            title_suffix=f' — {ds_name} ({len(sample)} trials x {len(offsets)} offsets, '
                         f'step {fine} samp)')


class QtomePlotter(BasePlotter):
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
        if self.has_signal('mse_mp'):   # the per-atom anti-duplicate term, when trained
            loss_panels[1]['series'].append(dict(key='mse_mp', color='gray', val_only=True, style_val='-', label='mp_loss'))

        # FFN Router Health — the MoEFFN routers inside every TSABlock (averaged across blocks),
        # see Qtome.update_ffn_router_metrics.
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
        self.render(panels, filename, suptitle='Tokenizer (Atom) Training Dashboard', ncols=4)

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
    trainer_cls=QtomeTrainer,
    plotter_cls=QtomePlotter,
    codebook_checker_cls=QtomeCodebookChecker,
)
