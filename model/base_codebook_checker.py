"""BaseCodebookChecker: cross-dataset codebook/vocab diagnostics, triggered only from
analysis_pretrain.py (--analysis codebook), never from the training loop. Template method — owns
the corpus-sampling + panel-render flow, subclasses only implement the two model-specific
extraction hooks. Parallel to BaseEpochChecker (model/base_checker.py) but operates over
many trials across many datasets at once instead of one trial at a time, so it lives in
its own base class rather than growing BaseEpochChecker a second unrelated flow.
"""

import os
import random

import numpy as np
import torch

from tools.analysis import event_onset_patch
from tools.viz.codebook import (
    plot_usage_and_activity, plot_embedding_scatter_by_dataset, plot_embedding_scatter_by_target,
    plot_patch_position_consistency, plot_dataset_relation,
)


class BaseCodebookChecker:
    """unit_label: 'Expert' | 'Filter' | ... — used in panel titles/axis labels."""
    unit_label = 'Unit'

    # True for subclasses whose hooks need a fresh forward pass per trial (too expensive for every
    # sampled trial up front) — check_codebook stashes each trial's (cpu-side, cheap) input
    # tensors in trial_records only when this is set, so the default (usage-only) path pays nothing.
    needs_raw_tensors = False

    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """One trial -> np.ndarray [M, Q]: M patches, Q units, a nonnegative strength per unit."""
        raise NotImplementedError

    def decoder_fingerprint_matrix(self, model):
        """np.ndarray [Q, Q] cosine similarity between each unit's own decoder weights
        (structural redundancy, independent of any particular dataset's activations)."""
        raise NotImplementedError

    def _render_unit_consistency(self, trial_records, viz_dir, model, device, seed):
        """Default: no-op. Override for per-unit consistency panels that need a fresh forward pass
        on a subsample of trials (see needs_raw_tensors)."""

    def _render_patch_position_consistency(self, ds_trials, ds_name, viz_dir, model, device, seed, config=None):
        """Default: plot_patch_position_consistency off the already-computed (cheap,
        gating-strength) usage in ds_trials -> patch_position_consistency_<ds_name>.png.
        Override (e.g. QtomeCodebookChecker) to render a different question/basis
        instead -- model/device are passed through only for overrides that need a fresh
        forward pass per trial (see needs_raw_tensors), unused by this default."""
        plot_patch_position_consistency(
            os.path.join(viz_dir, f'patch_position_consistency_{ds_name}.png'), ds_trials,
            unit_label=self.unit_label, seed=seed,
            event_patch=event_onset_patch(config, ds_name) if config else None)

    def _render_event_atom_dynamics(self, ds_trials, ds_name, viz_dir, model, device, seed, config):
        """Default: no-op. Override (e.g. QtomeCodebookChecker) to render an event-locked
        unit-selection / power trajectory -> event_atom_dynamics_<ds_name>.png. Needs a
        fresh sliding-window forward pass per trial, so only subclasses with
        needs_raw_tensors can implement it. Event onset per dataset comes from that
        dataset's own metadata.json event_onset_sample (see
        tools.analysis.lookup_event_onset_sample); datasets whose metadata.json has none
        still get the trajectory + heatmap without the pre/post split."""

    def _render_fingerprint_similarity(self, viz_dir, model):
        """Default: no-op. Override to render decoder_fingerprint_matrix(model) (already
        required by that method's contract) as a heatmap -> <unit_label>_fingerprint_
        similarity.png. Purely structural/dataset-independent, so unlike every other
        panel here it needs none of check_codebook's sampled trials -- safe to call
        unconditionally before the per-dataset sampling loop."""

    @staticmethod
    def _trial_tensors(dataset, trial_idx, device):
        x_patches, coords, _mask, time_indices, label, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)
        return x_in, c_in, t_in, vc_in, int(label)

    @staticmethod
    def _subject_id(dataset, trial_idx):
        """PretrainDataset index -> that trial's subject id."""
        return int(dataset.base_dataset.subject_data[trial_idx].item())

    @torch.no_grad()
    def check_codebook(self, config, output_dir, model, datasets_by_name,
                        max_trials_per_dataset=200, max_scatter_points=3000, seed=0):
        device = next(model.parameters()).device
        model.eval()
        rng = random.Random(seed)

        usage_by_dataset, labels_by_dataset = {}, {}          # patch-level: [M_total, Q], [M_total]
        trial_usage_by_dataset, trial_labels_by_dataset = {}, {}  # trial-level: [n_trials, Q], [n_trials]
        trial_records = []  # one dict per trial: {usage: [M, Q], dataset, subject[, raw]}
        for ds_name, dataset in datasets_by_name.items():
            n = len(dataset)
            n_trials = min(max_trials_per_dataset, n)
            trial_idxs = rng.sample(range(n), n_trials) if n_trials < n else list(range(n))

            chunks, label_chunks, trial_chunks, trial_labels = [], [], [], []
            for t_idx in trial_idxs:
                x_in, c_in, t_in, vc_in, label = self._trial_tensors(dataset, t_idx, device)
                usage = self.extract_usage(model, x_in, c_in, t_in, vc_in)  # [M, Q]
                chunks.append(usage)
                label_chunks.append(np.full(usage.shape[0], label, dtype=np.int64))  # label per trial -> broadcast to all M patches in it
                trial_chunks.append(usage.mean(axis=0))  # [Q] -- one point per trial, patches averaged out
                trial_labels.append(label)
                record = dict(usage=usage, dataset=ds_name, subject=self._subject_id(dataset, t_idx))
                if self.needs_raw_tensors:
                    # cpu + no grad: cheap to keep ~max_trials_per_dataset of these around
                    # (small EEG patch tensors), unlike the dense content they'll later be
                    # used to recompute on demand for only a small subsample of trials.
                    record['raw'] = (x_in.cpu(), c_in.cpu(), t_in.cpu(), vc_in.cpu())
                trial_records.append(record)
            usage_by_dataset[ds_name] = np.concatenate(chunks, axis=0)  # [M_total, Q]
            labels_by_dataset[ds_name] = np.concatenate(label_chunks, axis=0)  # [M_total]
            trial_usage_by_dataset[ds_name] = np.stack(trial_chunks, axis=0)  # [n_trials, Q]
            trial_labels_by_dataset[ds_name] = np.array(trial_labels, dtype=np.int64)  # [n_trials]
            print(f"  [codebook] {ds_name}: sampled {n_trials}/{n} trials "
                  f"-> {usage_by_dataset[ds_name].shape[0]} patches")

        viz_dir = os.path.join(output_dir, 'codebook')
        os.makedirs(viz_dir, exist_ok=True)

        # Structural, dataset-independent -- doesn't need any of the sampling above.
        self._render_fingerprint_similarity(viz_dir, model)

        plot_embedding_scatter_by_dataset(
            os.path.join(viz_dir, 'patch_embedding_scatter_by_dataset.png'), usage_by_dataset,
            unit_label=self.unit_label, max_points=max_scatter_points, random_state=seed)
        plot_embedding_scatter_by_target(
            os.path.join(viz_dir, 'patch_embedding_scatter_by_target.png'),
            os.path.join(viz_dir, 'patch_embedding_scatter_per_dataset.png'),
            usage_by_dataset, labels_by_dataset,
            unit_label=self.unit_label, max_points=max_scatter_points, random_state=seed)

        # trial-level: same three views, one point per trial (patches mean-pooled) instead of per patch
        plot_embedding_scatter_by_dataset(
            os.path.join(viz_dir, 'trial_embedding_scatter_by_dataset.png'), trial_usage_by_dataset,
            unit_label=self.unit_label, max_points=max_scatter_points, random_state=seed)
        plot_embedding_scatter_by_target(
            os.path.join(viz_dir, 'trial_embedding_scatter_by_target.png'),
            os.path.join(viz_dir, 'trial_embedding_scatter_per_dataset.png'),
            trial_usage_by_dataset, trial_labels_by_dataset,
            unit_label=self.unit_label, max_points=max_scatter_points, random_state=seed)
        self._render_unit_consistency(trial_records, viz_dir, model, device, seed)
        plot_dataset_relation(
            os.path.join(viz_dir, 'dataset_relation.png'), usage_by_dataset, unit_label=self.unit_label)

        # Unit x Dataset specialization: mean per-unit strength by dataset.
        dataset_order = list(usage_by_dataset.keys())
        strength = np.concatenate([usage_by_dataset[d] for d in dataset_order], axis=0)          # [M_total, Q]
        combined_dataset = np.concatenate(
            [np.full(usage_by_dataset[d].shape[0], d) for d in dataset_order])                     # [M_total]
        plot_usage_and_activity(
            os.path.join(viz_dir, 'unit_activity_by_dataset.png'), strength, combined_dataset,
            category_order=dataset_order, unit_label=self.unit_label)

        # Cross-trial, patch-position-aligned consistency (per dataset -- patch position
        # only means the same timeline slot within one dataset's own trial length/patch_len).
        for ds_name in dataset_order:
            ds_trials = [t for t in trial_records if t['dataset'] == ds_name]
            self._render_patch_position_consistency(ds_trials, ds_name, viz_dir, model, device, seed, config=config)
            self._render_event_atom_dynamics(ds_trials, ds_name, viz_dir, model, device, seed, config)

        print(f"  [codebook] -> {viz_dir}")
