"""Cross-dataset codebook/vocab panels reused by BaseCodebookChecker.check_codebook
(model/base_codebook_checker.py). Every function here takes usage_by_dataset: dict of
dataset_name -> np.ndarray [M, Q, F] (M patches sampled from that dataset, Q units
(Experts/Filters), F dictionary/discrete-code activations per unit) built by the
checker's per-model extract_usage() hook, so these panels are model-agnostic.
"""

import math

import numpy as np
import matplotlib.pyplot as plt


def _palette(n):
    """n distinct colors, n unbounded (unlike tab10/tab20) — hsv wraps smoothly so
    high-cardinality targets (e.g. BETA's 40 classes) don't collide."""
    return plt.cm.hsv(np.linspace(0, 1, max(n, 1), endpoint=False))


def _usage_freq(arr):
    """[M, Q, F] -> [Q, F] fraction of patches each (unit, feature) fired on."""
    return (arr > 0).mean(axis=0)


def plot_usage_histogram(out_path, usage_by_dataset, unit_label='Filter'):
    """Per-unit panel: usage frequency of every dictionary feature, sorted descending,
    one line per dataset, log-y — a feature used by every dataset stays high across the
    whole curve; a dataset-specific feature shows up as a bump only that dataset has."""
    dataset_names = list(usage_by_dataset.keys())
    Q = next(iter(usage_by_dataset.values())).shape[1]
    freqs = {ds: _usage_freq(arr) for ds, arr in usage_by_dataset.items()}

    n_cols = min(4, Q)
    n_rows = math.ceil(Q / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_names), 1)))

    for q in range(Q):
        ax = axes[q // n_cols, q % n_cols]
        for ds, color in zip(dataset_names, colors):
            sorted_freq = np.sort(freqs[ds][q])[::-1]
            ax.plot(sorted_freq, color=color, label=ds, linewidth=1.2)
        ax.set_yscale('log')
        ax.set_title(f'{unit_label} {q}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Feature rank', fontsize=7)
        ax.set_ylabel('Usage freq', fontsize=7)
    for q in range(Q, n_rows * n_cols):
        axes[q // n_cols, q % n_cols].axis('off')
    axes[0, 0].legend(fontsize=6, loc='upper right')

    fig.suptitle(f'Codebook Usage Frequency by Dataset (sorted per {unit_label})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_embedding_scatter_by_dataset(out_path, usage_by_dataset, unit_label='Filter',
                                       max_points=3000, random_state=0):
    """PCA + t-SNE of each patch's full sparse code (all units concatenated), colored by
    source dataset. t-SNE runs on a PCA-reduced pre-projection for speed, standard practice
    at this corpus size."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    dataset_names = list(usage_by_dataset.keys())
    rng = np.random.RandomState(random_state)
    per_ds_cap = max(1, max_points // max(len(dataset_names), 1))

    feats, labels = [], []
    for ds in dataset_names:
        arr = usage_by_dataset[ds].reshape(usage_by_dataset[ds].shape[0], -1)
        idx = rng.choice(arr.shape[0], size=min(per_ds_cap, arr.shape[0]), replace=False)
        feats.append(arr[idx])
        labels += [ds] * len(idx)
    X = np.concatenate(feats, axis=0)
    labels = np.array(labels)

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
    n_pre = min(50, X.shape[0] - 1, X.shape[1])
    pca_pre = PCA(n_components=n_pre, random_state=random_state).fit_transform(X)
    perplexity = min(30, max(5, X.shape[0] // 10))
    tsne2 = TSNE(n_components=2, init='pca', random_state=random_state,
                 perplexity=perplexity).fit_transform(pca_pre)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_names), 1)))
    for proj, ax, title in ((pca2, axes[0], 'PCA'), (tsne2, axes[1], 't-SNE')):
        for ds, color in zip(dataset_names, colors):
            mask = labels == ds
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6, color=color, label=ds)
        ax.set_title(f'{title} of {unit_label} sparse codes', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, markerscale=2, loc='best')

    fig.suptitle('Cross-Dataset Embedding Separation', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_embedding_scatter_by_target(out_path_combined, out_path_per_dataset, usage_by_dataset,
                                      labels_by_dataset, unit_label='Filter',
                                      max_points=3000, random_state=0):
    """Same patch-level sparse codes/sampling as plot_embedding_scatter, but colored by class
    too (labels_by_dataset: dataset_name -> np.ndarray [M] int class id, one per patch,
    broadcast from that patch's trial label). One PCA/t-SNE fit is reused for both views so
    inter-dataset (combined, colored dataset_classK) and intra-dataset (per-dataset grid,
    colored by class only) plots are directly comparable."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    dataset_names = list(usage_by_dataset.keys())
    rng = np.random.RandomState(random_state)
    per_ds_cap = max(1, max_points // max(len(dataset_names), 1))

    feats, ds_labels, class_labels = [], [], []
    for ds in dataset_names:
        arr = usage_by_dataset[ds].reshape(usage_by_dataset[ds].shape[0], -1)
        cls = labels_by_dataset[ds]
        idx = rng.choice(arr.shape[0], size=min(per_ds_cap, arr.shape[0]), replace=False)
        feats.append(arr[idx])
        ds_labels += [ds] * len(idx)
        class_labels.append(cls[idx])
    X = np.concatenate(feats, axis=0)
    ds_labels = np.array(ds_labels)
    class_labels = np.concatenate(class_labels, axis=0)
    combo_labels = np.array([f'{d}_class_{c}' for d, c in zip(ds_labels, class_labels)])

    pca2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
    n_pre = min(50, X.shape[0] - 1, X.shape[1])
    pca_pre = PCA(n_components=n_pre, random_state=random_state).fit_transform(X)
    perplexity = min(30, max(5, X.shape[0] // 10))
    tsne2 = TSNE(n_components=2, init='pca', random_state=random_state,
                 perplexity=perplexity).fit_transform(pca_pre)

    # combined view: one color per (dataset, class) -- every target of every dataset
    combo_names = sorted(set(combo_labels.tolist()))
    colors = _palette(len(combo_names))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for proj, ax, title in ((pca2, axes[0], 'PCA'), (tsne2, axes[1], 't-SNE')):
        for name, color in zip(combo_names, colors):
            mask = combo_labels == name
            ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6, color=color, label=name)
        ax.set_title(f'{title} of {unit_label} sparse codes', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=5, markerscale=1.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   ncol=max(1, len(combo_names) // 25 + 1))

    fig.suptitle('Embedding Separation by Dataset x Class (all targets)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path_combined, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path_combined}")

    # per-dataset view: one row per dataset, colored by that dataset's own targets only
    n_ds = len(dataset_names)
    fig, axes = plt.subplots(n_ds, 2, figsize=(13, 4.5 * n_ds), squeeze=False)
    for row, ds in enumerate(dataset_names):
        ds_mask = ds_labels == ds
        classes = sorted(set(class_labels[ds_mask].tolist()))
        class_colors = _palette(len(classes))
        for proj, col, title in ((pca2, 0, 'PCA'), (tsne2, 1, 't-SNE')):
            ax = axes[row, col]
            for cls, color in zip(classes, class_colors):
                mask = ds_mask & (class_labels == cls)
                ax.scatter(proj[mask, 0], proj[mask, 1], s=6, alpha=0.6, color=color, label=f'class_{cls}')
            ax.set_title(f'{ds} ({len(classes)} targets) — {title}', fontsize=10, fontweight='bold')
            if col == 1:
                ax.legend(fontsize=5, markerscale=1.5, loc='center left', bbox_to_anchor=(1.02, 0.5),
                          ncol=max(1, len(classes) // 25 + 1))

    fig.suptitle(f'Intra-Dataset Class Separation, All Targets ({unit_label} sparse codes)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path_per_dataset, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path_per_dataset}")


def plot_filter_relation(out_path, usage_by_dataset, fingerprint_matrix, unit_label='Filter'):
    """Q x Q relation between units, two views side by side: usage-frequency correlation
    (data-dependent: do two units fire on the same features across the corpus) and decoder
    weight cosine similarity (structural: do two units decode near-identical functions,
    independent of any dataset). High in either = redundant units."""
    combined = np.concatenate(list(usage_by_dataset.values()), axis=0)
    freq = _usage_freq(combined)  # [Q, F]
    coact = np.corrcoef(freq)

    Q = coact.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title in ((axes[0], coact, 'Usage-Frequency Correlation'),
                            (axes[1], fingerprint_matrix, 'Decoder Weight Cosine Sim')):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(Q)); ax.set_yticks(range(Q))
        ax.set_xticklabels([f'{unit_label[0]}{q}' for q in range(Q)], fontsize=7, rotation=90)
        ax.set_yticklabels([f'{unit_label[0]}{q}' for q in range(Q)], fontsize=7)
        ax.set_title(title, fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'{unit_label}-{unit_label} Relation (high = redundant)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def _js_divergence(p, q, eps=1e-12):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps) + eps))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def plot_dataset_relation(out_path, usage_by_dataset, unit_label='Filter'):
    """D x D Jensen-Shannon divergence between datasets' pooled dictionary usage
    distributions (all units flattened together) — low divergence = datasets lean on the
    same vocabulary, high = distinct."""
    dataset_names = list(usage_by_dataset.keys())
    dists = {ds: _usage_freq(arr).reshape(-1) for ds, arr in usage_by_dataset.items()}

    D = len(dataset_names)
    js = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            js[i, j] = _js_divergence(dists[dataset_names[i]], dists[dataset_names[j]])

    fig, ax = plt.subplots(figsize=(1.2 * D + 3, 1.0 * D + 3))
    im = ax.imshow(js, vmin=0, vmax=math.log(2), cmap='YlOrRd')
    ax.set_xticks(range(D)); ax.set_yticks(range(D))
    ax.set_xticklabels(dataset_names, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(dataset_names, fontsize=8)
    for i in range(D):
        for j in range(D):
            color = 'white' if js[i, j] > math.log(2) / 2 else 'black'
            ax.text(j, i, f'{js[i, j]:.3f}', ha='center', va='center', fontsize=7, color=color)

    ax.set_title(f'Dataset Vocab-Usage Divergence (JS, over {unit_label} dictionary)',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
