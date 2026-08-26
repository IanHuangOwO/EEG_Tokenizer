"""BaseCodebookChecker: cross-dataset codebook/vocab diagnostics, triggered only from
check_model.py (--mode codebook), never from the training loop. Template method — owns
the corpus-sampling + panel-render flow, subclasses only implement the two model-specific
extraction hooks. Parallel to BaseEpochChecker (model/base_checker.py) but operates over
many trials across many datasets at once instead of one trial at a time, so it lives in
its own base class rather than growing BaseEpochChecker a second unrelated flow.

See docs/adr/0004-model-plugin-base-classes.md.
"""

import os
import random

import numpy as np
import torch

from viz.codebook import (
    plot_usage_and_activity, plot_embedding_scatter_by_dataset, plot_embedding_scatter_by_target,
    plot_patch_similarity_hierarchy,
    plot_patch_position_consistency, plot_dataset_relation, plot_unit_freedom,
)


class BaseCodebookChecker:
    """unit_label: 'Expert' | 'Filter' | ... — used in panel titles/axis labels."""
    unit_label = 'Unit'

    # True only for subclasses whose _render_patch_similarity override needs a fresh
    # forward pass per trial (e.g. MeSAEFlatCodebookChecker's dense per-token decoder
    # content — too expensive to compute for every sampled trial up front, see
    # extract_stamp_content) — check_codebook stashes each trial's (cpu-side, cheap) input
    # tensors in trial_records only when this is set, so the default (usage-only) path
    # pays nothing for it.
    needs_raw_tensors = False

    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """One trial -> np.ndarray [M, Q, F]: M patches, Q units (Experts/Filters), F
        dictionary/discrete-code activations per unit. Subclass-specific: reads whatever
        the model's forward() returns for its own sparse-code representation."""
        raise NotImplementedError

    def decoder_fingerprint_matrix(self, model):
        """np.ndarray [Q, Q] cosine similarity between each unit's own decoder weights
        (structural redundancy, independent of any particular dataset's activations)."""
        raise NotImplementedError

    def rank_ceiling(self, model):
        """Architectural cap on plot_unit_freedom's effective-rank panel, e.g.
        min(sparse-code k, embed_dim) -- the code is generated from an embed_dim-dim
        bottleneck and/or top-k-sparse encoding, so its covariance rank can't usefully
        exceed that regardless of dictionary size F. Return None (default) to let
        plot_unit_freedom fall back to F, a much looser bound. Override per model."""
        return None

    def _render_patch_similarity(self, trial_records, viz_dir, model, device, seed):
        """Default: plot_patch_similarity_hierarchy off the already-computed (cheap,
        gating-strength) usage in trial_records -> patch_similarity_hierarchy.png.
        Override (e.g. MeSAEFlatCodebookChecker) to render a different question/basis
        instead -- model/device are passed through only for overrides that need a fresh
        forward pass per trial (see needs_raw_tensors), unused by this default."""
        plot_patch_similarity_hierarchy(
            os.path.join(viz_dir, 'patch_similarity_hierarchy.png'), trial_records,
            unit_label=self.unit_label, seed=seed)

    def _render_patch_position_consistency(self, ds_trials, ds_name, viz_dir, model, device, seed):
        """Default: plot_patch_position_consistency off the already-computed (cheap,
        gating-strength) usage in ds_trials -> patch_position_consistency_<ds_name>.png.
        Override (e.g. MeSAEFlatCodebookChecker) to render a different question/basis
        instead -- model/device are passed through only for overrides that need a fresh
        forward pass per trial (see needs_raw_tensors), unused by this default."""
        plot_patch_position_consistency(
            os.path.join(viz_dir, f'patch_position_consistency_{ds_name}.png'), ds_trials,
            unit_label=self.unit_label, seed=seed)

    @staticmethod
    def _trial_tensors(dataset, trial_idx, device):
        x_patches, coords, _mask, time_indices, label, _, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)
        return x_in, c_in, t_in, vc_in, int(label)

    @staticmethod
    def _subject_id(dataset, trial_idx):
        """PretrainDataset.__getitem__ index (0..len(base_dataset)*mask_multiplier-1)
        maps to a real trial via `index % len(base_dataset)` (see IO/masking.py resolve) --
        same modulo here to read the true trial's subject id off base_dataset.subject_data."""
        base_idx = trial_idx % len(dataset.base_dataset)
        return int(dataset.base_dataset.subject_data[base_idx].item())

    @torch.no_grad()
    def check_codebook(self, config, output_dir, model, datasets_by_name,
                        max_trials_per_dataset=200, max_scatter_points=3000, seed=0):
        device = next(model.parameters()).device
        model.eval()
        rng = random.Random(seed)

        usage_by_dataset, labels_by_dataset = {}, {}          # patch-level: [M_total, Q, F], [M_total]
        trial_usage_by_dataset, trial_labels_by_dataset = {}, {}  # trial-level: [n_trials, Q, F], [n_trials]
        trial_records = []  # one dict per trial: {usage: [M,Q,F], dataset, subject[, raw]} -- feeds _render_patch_similarity
        for ds_name, dataset in datasets_by_name.items():
            n = len(dataset)
            n_trials = min(max_trials_per_dataset, n)
            trial_idxs = rng.sample(range(n), n_trials) if n_trials < n else list(range(n))

            chunks, label_chunks, trial_chunks, trial_labels = [], [], [], []
            for t_idx in trial_idxs:
                x_in, c_in, t_in, vc_in, label = self._trial_tensors(dataset, t_idx, device)
                usage = self.extract_usage(model, x_in, c_in, t_in, vc_in)  # [M, Q, F]
                chunks.append(usage)
                label_chunks.append(np.full(usage.shape[0], label, dtype=np.int64))  # label per trial -> broadcast to all M patches in it
                trial_chunks.append(usage.mean(axis=0))  # [Q, F] -- one point per trial, patches averaged out
                trial_labels.append(label)
                record = dict(usage=usage, dataset=ds_name, subject=self._subject_id(dataset, t_idx))
                if self.needs_raw_tensors:
                    # cpu + no grad: cheap to keep ~max_trials_per_dataset of these around
                    # (small EEG patch tensors), unlike the dense content they'll later be
                    # used to recompute on demand for only a small subsample of trials.
                    record['raw'] = (x_in.cpu(), c_in.cpu(), t_in.cpu(), vc_in.cpu())
                trial_records.append(record)
            usage_by_dataset[ds_name] = np.concatenate(chunks, axis=0)  # [M_total, Q, F]
            labels_by_dataset[ds_name] = np.concatenate(label_chunks, axis=0)  # [M_total]
            trial_usage_by_dataset[ds_name] = np.stack(trial_chunks, axis=0)  # [n_trials, Q, F]
            trial_labels_by_dataset[ds_name] = np.array(trial_labels, dtype=np.int64)  # [n_trials]
            print(f"  [codebook] {ds_name}: sampled {n_trials}/{n} trials "
                  f"-> {usage_by_dataset[ds_name].shape[0]} patches")

        viz_dir = os.path.join(output_dir, 'codebook')
        os.makedirs(viz_dir, exist_ok=True)

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
        self._render_patch_similarity(trial_records, viz_dir, model, device, seed)
        plot_dataset_relation(
            os.path.join(viz_dir, 'dataset_relation.png'), usage_by_dataset, unit_label=self.unit_label)
        plot_unit_freedom(
            os.path.join(viz_dir, 'unit_freedom.png'), usage_by_dataset, unit_label=self.unit_label,
            rank_ceiling=self.rank_ceiling(model))

        # Filter x Dataset specialization: strength = per-patch, per-unit L1 sum of the
        # sparse code (how much that Filter contributed to this patch), paired side by
        # side with the atom-usage histogram (same units, same datasets, different axis).
        dataset_order = list(usage_by_dataset.keys())
        combined_usage   = np.concatenate([usage_by_dataset[d] for d in dataset_order], axis=0)   # [M_total, Q, F] or [M_total, Q]
        combined_dataset = np.concatenate(
            [np.full(usage_by_dataset[d].shape[0], d) for d in dataset_order])                     # [M_total]
        # Flat [M,Q] usage (e.g. StampBank) already IS the per-unit strength, no F axis to
        # sum over -- summing axis=-1 there would collapse the wrong (unit) axis instead.
        strength = combined_usage if combined_usage.ndim == 2 else combined_usage.sum(axis=-1)  # [M_total, Q]

        plot_usage_and_activity(
            os.path.join(viz_dir, 'filter_usage_and_activity.png'), usage_by_dataset, strength, combined_dataset,
            category_order=dataset_order, unit_label=self.unit_label)

        # Cross-trial, patch-position-aligned consistency (per dataset -- patch position
        # only means the same timeline slot within one dataset's own trial length/patch_len).
        for ds_name in dataset_order:
            ds_trials = [t for t in trial_records if t['dataset'] == ds_name]
            self._render_patch_position_consistency(ds_trials, ds_name, viz_dir, model, device, seed)

        print(f"  [codebook] -> {viz_dir}")
