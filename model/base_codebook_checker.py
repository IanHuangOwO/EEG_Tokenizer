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
    plot_usage_histogram, plot_embedding_scatter_by_dataset, plot_embedding_scatter_by_target,
    plot_filter_relation, plot_dataset_relation,
)


class BaseCodebookChecker:
    """unit_label: 'Expert' | 'Filter' | ... — used in panel titles/axis labels."""
    unit_label = 'Unit'

    def extract_usage(self, model, x_in, c_in, t_in, vc_in):
        """One trial -> np.ndarray [M, Q, F]: M patches, Q units (Experts/Filters), F
        dictionary/discrete-code activations per unit. Subclass-specific: reads whatever
        the model's forward() returns for its own sparse-code representation."""
        raise NotImplementedError

    def decoder_fingerprint_matrix(self, model):
        """np.ndarray [Q, Q] cosine similarity between each unit's own decoder weights
        (structural redundancy, independent of any particular dataset's activations)."""
        raise NotImplementedError

    @staticmethod
    def _trial_tensors(dataset, trial_idx, device):
        x_patches, coords, _mask, time_indices, label, _, valid_channels = dataset[trial_idx]
        x_in  = x_patches.unsqueeze(0).to(device)
        c_in  = coords.unsqueeze(0).to(device)
        t_in  = time_indices.unsqueeze(0).to(device)
        vc_in = valid_channels.unsqueeze(0).to(device)
        return x_in, c_in, t_in, vc_in, int(label)

    @torch.no_grad()
    def check_codebook(self, config, output_dir, model, datasets_by_name,
                        max_trials_per_dataset=200, max_scatter_points=3000, seed=0):
        device = next(model.parameters()).device
        model.eval()
        rng = random.Random(seed)

        usage_by_dataset, labels_by_dataset = {}, {}          # patch-level: [M_total, Q, F], [M_total]
        trial_usage_by_dataset, trial_labels_by_dataset = {}, {}  # trial-level: [n_trials, Q, F], [n_trials]
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
            usage_by_dataset[ds_name] = np.concatenate(chunks, axis=0)  # [M_total, Q, F]
            labels_by_dataset[ds_name] = np.concatenate(label_chunks, axis=0)  # [M_total]
            trial_usage_by_dataset[ds_name] = np.stack(trial_chunks, axis=0)  # [n_trials, Q, F]
            trial_labels_by_dataset[ds_name] = np.array(trial_labels, dtype=np.int64)  # [n_trials]
            print(f"  [codebook] {ds_name}: sampled {n_trials}/{n} trials "
                  f"-> {usage_by_dataset[ds_name].shape[0]} patches")

        fp_matrix = self.decoder_fingerprint_matrix(model)

        viz_dir = os.path.join(output_dir, 'codebook')
        os.makedirs(viz_dir, exist_ok=True)

        plot_usage_histogram(
            os.path.join(viz_dir, 'filter_usage_histogram.png'), usage_by_dataset, unit_label=self.unit_label)
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
        plot_filter_relation(
            os.path.join(viz_dir, 'filter_relation.png'), usage_by_dataset, fp_matrix, unit_label=self.unit_label)
        plot_dataset_relation(
            os.path.join(viz_dir, 'dataset_relation.png'), usage_by_dataset, unit_label=self.unit_label)

        print(f"  [codebook] -> {viz_dir}")
