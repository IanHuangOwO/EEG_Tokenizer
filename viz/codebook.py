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


def plot_filter_by_category(out_path, strength, categories, category_order, unit_label='Filter',
                             category_axis_label='Category', title=None, normalize=True):
    """Filter x Category heatmap of mean per-filter activity ("does this Filter prefer
    this category"). Model-agnostic like every other panel here: strength [M, Q] is just
    a per-patch, per-unit scalar (e.g. L1 sum of that unit's sparse code, or its raw
    attention mass) and categories [M] is any per-patch label — dataset name, an
    activity-level bin, anything. Same reduction serves both "which paradigm does this
    Filter specialize in" (categories=dataset) and "is this Filter event-driven or
    always-on" (categories=quiet/mid/burst amplitude bin).

    normalize=True divides each row by that Filter's own mean activity across categories
    (ratio-to-own-mean, 1.0 = no preference) so a uniformly loud/quiet Filter doesn't
    just paint one solid color row — only relative preference across categories shows.
    False shows raw magnitude instead (use when comparing absolute scale matters more
    than shape, e.g. Filters that barely fire anywhere vs ones firing everywhere)."""
    Q = strength.shape[1]
    categories = np.asarray(categories)
    raw = np.full((Q, len(category_order)), np.nan)
    for ci, cat in enumerate(category_order):
        mask = categories == cat
        if mask.any():
            raw[:, ci] = strength[mask].mean(axis=0)

    if normalize:
        row_mean = np.nanmean(raw, axis=1, keepdims=True)
        mat = np.divide(raw, row_mean, out=np.zeros_like(raw), where=row_mean > 0)
        vmin, vmax, cmap = 0, 2.0, 'RdBu_r'
    else:
        mat = raw
        vmin, vmax, cmap = 0, np.nanmax(mat), 'YlOrRd'

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(category_order) + 2), max(6, 0.16 * Q)))
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(category_order))); ax.set_xticklabels(category_order, fontsize=8, rotation=45, ha='right')
    ax.set_yticks(range(Q)); ax.set_yticklabels([f'{unit_label[0]}{q}' for q in range(Q)], fontsize=6)
    ax.set_xlabel(category_axis_label, fontsize=9)
    ax.set_ylabel(unit_label, fontsize=9)
    suffix = ' (ratio to own mean; 1.0 = no preference)' if normalize else ' (raw mean activity)'
    ax.set_title((title or f'{unit_label} Activity by {category_axis_label}') + suffix, fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")

    # Specialization entropy per unit: entropy (nats) of that unit's raw activity
    # distribution across categories -- low entropy = concentrated on one category
    # (specialized), high = spread evenly (generic). Always computed off `raw` (never
    # the ratio-normalized `mat`) since entropy needs an actual probability distribution.
    p = raw / (np.nansum(raw, axis=1, keepdims=True) + 1e-12)
    ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    max_ent = math.log(len(category_order))
    order = np.argsort(ent)
    top_n = min(5, Q)
    print(f"    specialization entropy: mean={ent.mean():.3f} std={ent.std():.3f} "
          f"(max possible={max_ent:.3f}, 0=fully specialized)")
    print(f"    most specialized {unit_label}s: " + ', '.join(
        f"{unit_label[0]}{q}->{category_order[np.nanargmax(raw[q])]} (H={ent[q]:.3f})" for q in order[:top_n]))


def _usage_grid_shape(Q, n_rows=None):
    """Pick a grid (n_cols, n_rows) for Q per-unit panels. Default (n_rows=None): near-square,
    capped at 4 cols, so the standalone plot doesn't get needlessly wide for large Q. Callers
    embedding this grid next to another panel of known height (e.g. plot_usage_and_activity's
    heatmap) pass the target n_rows so the grid's total height lines up instead of using
    whatever height min(4, Q) cols happens to produce."""
    if n_rows is not None:
        n_rows = max(1, min(n_rows, Q))
        n_cols = math.ceil(Q / n_rows)
        return n_cols, n_rows
    n_cols = min(4, math.ceil(math.sqrt(Q))) if Q > 4 else Q
    n_rows = math.ceil(Q / n_cols)
    return n_cols, n_rows


def _draw_usage_histogram(fig, gs_cell, usage_by_dataset, unit_label, n_cols, n_rows):
    """Shared render body for the per-unit atom-usage-frequency grid — used standalone by
    plot_usage_histogram and embedded as the left panel of plot_usage_and_activity. Grid
    shape (n_cols, n_rows) is chosen by the caller via _usage_grid_shape.

    Curve is "which atoms (F) does this unit reuse" -- meaningless for flat [M, Q] usage
    (e.g. StampBank, no per-unit feature axis, see docs/adr/0009's Monitoring impact
    section): a single scalar per unit has no atom-identity distribution to sort/plot.
    Draws a placeholder note instead of crashing on that axis."""
    first = next(iter(usage_by_dataset.values()))
    if first.ndim == 2:
        ax = fig.add_subplot(gs_cell)
        ax.text(0.5, 0.5, 'no per-unit feature axis\n(usage is flat [M, Q])',
                ha='center', va='center', fontsize=10, color='gray')
        ax.axis('off')
        return
    dataset_names = list(usage_by_dataset.keys())
    Q = first.shape[1]
    freqs = {ds: _usage_freq(arr) for ds, arr in usage_by_dataset.items()}

    inner = gs_cell.subgridspec(n_rows, n_cols)
    axes = np.empty((n_rows, n_cols), dtype=object)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dataset_names), 1)))

    for q in range(Q):
        ax = fig.add_subplot(inner[q // n_cols, q % n_cols])
        axes[q // n_cols, q % n_cols] = ax
        for ds, color in zip(dataset_names, colors):
            sorted_freq = np.sort(freqs[ds][q])[::-1]
            ax.plot(sorted_freq, color=color, label=ds, linewidth=1.2)
        ax.set_yscale('log')
        ax.set_title(f'{unit_label} {q}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Feature rank', fontsize=7)
        ax.set_ylabel('Usage freq', fontsize=7)
    for q in range(Q, n_rows * n_cols):
        fig.add_subplot(inner[q // n_cols, q % n_cols]).axis('off')
    axes[0, 0].legend(fontsize=6, loc='upper right')


def plot_usage_histogram(out_path, usage_by_dataset, unit_label='Filter'):
    """Per-unit panel: usage frequency of every dictionary feature, sorted descending,
    one line per dataset, log-y — a feature used by every dataset stays high across the
    whole curve; a dataset-specific feature shows up as a bump only that dataset has."""
    Q = next(iter(usage_by_dataset.values())).shape[1]
    n_cols, n_rows = _usage_grid_shape(Q)
    fig = plt.figure(figsize=(4.5 * n_cols, 3.2 * n_rows))
    gs = fig.add_gridspec(1, 1)
    _draw_usage_histogram(fig, gs[0, 0], usage_by_dataset, unit_label, n_cols, n_rows)

    fig.suptitle(f'Codebook Usage Frequency by Dataset (sorted per {unit_label})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_usage_and_activity(out_path, usage_by_dataset, strength, categories, category_order,
                             unit_label='Filter', normalize=True, max_rows_per_subplot=100):
    """Q x D heatmap of total per-unit activity magnitude by dataset — HOW HARD a unit
    fires on each dataset overall, atom identity collapsed away. Q rows are chunked across
    multiple side-by-side subplots (max_rows_per_subplot each) instead of one column, so
    a large Q (e.g. MeSAE's n_stamps, in the hundreds) makes the image wide, not
    absurdly tall. Was paired with plot_usage_histogram's per-unit atom-identity curves
    (WHICH atoms a unit reuses across datasets) — dropped here since that panel is a
    no-op placeholder for any unit whose usage has no per-atom F axis (flat [M, Q], e.g.
    StampBank, see docs/adr/0009's Monitoring impact section); use plot_usage_histogram
    directly for units that do have one.
    """
    Q = strength.shape[1]
    categories = np.asarray(categories)
    raw = np.full((Q, len(category_order)), np.nan)
    for ci, cat in enumerate(category_order):
        mask = categories == cat
        if mask.any():
            raw[:, ci] = strength[mask].mean(axis=0)
    if normalize:
        row_mean = np.nanmean(raw, axis=1, keepdims=True)
        mat = np.divide(raw, row_mean, out=np.zeros_like(raw), where=row_mean > 0)
        vmin, vmax, cmap = 0, 2.0, 'RdBu_r'
    else:
        mat = raw
        vmin, vmax, cmap = 0, np.nanmax(mat), 'YlOrRd'

    n_chunks = max(1, math.ceil(Q / max_rows_per_subplot))
    chunk_h = max(6, 0.16 * min(Q, max_rows_per_subplot) + 2)
    chunk_w = max(4, 0.7 * len(category_order) + 1.5)
    fig, axes = plt.subplots(1, n_chunks, figsize=(chunk_w * n_chunks, chunk_h), squeeze=False)
    axes = axes[0]

    im = None
    for ci in range(n_chunks):
        q0, q1 = ci * max_rows_per_subplot, min(Q, (ci + 1) * max_rows_per_subplot)
        ax = axes[ci]
        im = ax.imshow(mat[q0:q1], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(category_order)))
        ax.set_xticklabels(category_order, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(q1 - q0))
        ax.set_yticklabels([f'{unit_label[0]}{q}' for q in range(q0, q1)], fontsize=6)
        ax.set_xlabel('Dataset', fontsize=9)
        if ci == 0:
            ax.set_ylabel(unit_label, fontsize=9)

    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    suffix = ' (ratio to own mean)' if normalize else ' (raw mean activity)'
    fig.suptitle(f'{unit_label} Total Activity by Dataset' + suffix, fontsize=12, fontweight='bold')
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


def _intra_patch_relation(usage_by_dataset):
    """Q x Q relation between units MEASURED PER PATCH then averaged (unlike
    plot_filter_relation, which averages usage into one [Q, F] freq vector per unit
    first and compares those corpus-level summaries). Two units can share a corpus-level
    freq profile while never actually agreeing on any single patch (real per-patch
    specialization) or can genuinely pick the same atoms patch by patch (real
    redundancy) -- averaging first can't tell those apart, this does.

    Returns (jaccard, cos_sim), both [Q, Q]:
    Atom-Usage Jaccard: per patch, fraction of each pair of units' active (fired) atoms
    that overlap, averaged over patches.
    Vector Cosine Sim: per patch, cosine similarity between each pair of units' full
    activation vectors [F] (magnitude-aware, unlike Jaccard which only sees on/off),
    averaged over patches.
    Both near 0 = units specialize per patch even when their corpus-level usage looks
    similar; both high = real redundancy, not a corpus-averaging artifact.

    Flat [M, Q] usage (e.g. StampBank, no per-unit feature axis -- see docs/adr/0009's
    Monitoring impact section) has no atom SET or activation VECTOR per unit to compare in
    the first place (both degenerate to a single scalar, making Jaccard trivially 0/1 by
    definition and cosine a meaningless sign product) -- returns None in that case, the
    caller skips the panel rather than rendering a degenerate result."""
    combined = np.concatenate(usage_by_dataset, axis=0) if isinstance(usage_by_dataset, list) \
        else np.concatenate(list(usage_by_dataset.values()), axis=0)  # [M, Q, F]
    if combined.ndim == 2:
        return None, None
    active = combined > 0
    norm = np.linalg.norm(combined, axis=-1, keepdims=True)  # [M, Q, 1]
    unit_vec = np.divide(combined, norm, out=np.zeros_like(combined), where=norm > 0)

    # Chunked over M (patches): casting the full [M, Q, F] active mask to float64 before
    # the einsum (as a single batch) allocates M*Q*F*8 bytes up front -- at realistic
    # corpus sizes (e.g. M~48600, Q=64, F=3200) that's ~79GB before any contraction even
    # runs. Streaming chunks and accumulating the per-patch jaccard/cosine sums is
    # mathematically identical to the original one-shot .mean(axis=0) (mean = sum/M),
    # just bounded to chunk_size*Q*F in peak memory instead of M*Q*F.
    Q = combined.shape[1]
    M = combined.shape[0]
    chunk_size = 2000
    jaccard_sum = np.zeros((Q, Q), dtype=np.float64)
    cos_sum = np.zeros((Q, Q), dtype=np.float64)
    for m0 in range(0, M, chunk_size):
        a = active[m0:m0 + chunk_size].astype(np.float32)  # [m, Q, F]
        v = unit_vec[m0:m0 + chunk_size].astype(np.float32)  # [m, Q, F]
        inter = np.einsum('mqf,mrf->mqr', a, a)  # [m, Q, Q]
        counts = a.sum(axis=-1)  # [m, Q]
        union = counts[:, :, None] + counts[:, None, :] - inter
        jaccard_sum += np.divide(inter, union, out=np.zeros_like(inter), where=union > 0).sum(axis=0)
        cos_sum += np.einsum('mqf,mrf->mqr', v, v).sum(axis=0)
    jaccard = jaccard_sum / M
    cos_sim = cos_sum / M
    return jaccard, cos_sim


def print_redundant_unit_pairs(usage_by_dataset, unit_label='Filter', top_k=10):
    """Console-only: top_k most-redundant unit pairs by _intra_patch_relation's combined
    (jaccard + cosine) score, off-diagonal only. Replaces the Q x Q heatmap that used to
    sit next to plot_patch_similarity_hierarchy's bar chart -- once Q is large the dense
    grid reads as uniform warm noise with no actionable structure; a ranked list stays
    readable and scales to any Q. High-ranked pairs are decode-redundant candidates (same
    atoms fire on the same patches) worth merging or pruning."""
    jaccard, cos_sim = _intra_patch_relation(usage_by_dataset)
    if jaccard is None:
        print(f"    top-{top_k} redundant {unit_label} pairs: skipped (usage has no per-unit feature axis)")
        return
    Q = jaccard.shape[0]
    score = jaccard + cos_sim
    np.fill_diagonal(score, -np.inf)
    flat_idx = np.argsort(score, axis=None)[::-1][:top_k]
    print(f"    top-{top_k} redundant {unit_label} pairs (intra-patch jaccard + cosine):")
    seen = set()
    for idx in flat_idx:
        i, j = np.unravel_index(idx, score.shape)
        if (j, i) in seen:
            continue
        seen.add((i, j))
        print(f"      {unit_label[0]}{i}-{unit_label[0]}{j}: jaccard={jaccard[i, j]:.3f} cosine={cos_sim[i, j]:.3f}")


def _pairwise_stats(vecs, max_n, rng):
    """[N, D] continuous activations, subsampled to max_n rows -> mean/std weighted
    Jaccard (Ruzicka similarity: sum(min(a,b)) / sum(max(a,b)) per pair) and cosine over
    every unique pair. Shared by all grouping levels in plot_patch_similarity_hierarchy/
    plot_stamp_similarity -- only what rows get handed in differs.

    Weighted, not the plain binary (>0 mask) Jaccard this used to be: a stamp picked once
    with weak strength and one picked repeatedly/strongly no longer count as equally
    "present" just because both are nonzero -- Ruzicka reduces to classic binary Jaccard
    exactly when vecs is itself binary, so this is a strict generalization, not a
    different metric."""
    n = vecs.shape[0]
    if n > max_n:
        vecs = vecs[rng.choice(n, max_n, replace=False)]
        n = max_n
    if n < 2:
        return None
    vecs = np.maximum(vecs, 0.0)  # Ruzicka needs nonnegative weights; guards float noise
    mins = np.minimum(vecs[:, None, :], vecs[None, :, :]).sum(axis=-1)
    maxs = np.maximum(vecs[:, None, :], vecs[None, :, :]).sum(axis=-1)
    jac = np.divide(mins, maxs, out=np.zeros_like(mins), where=maxs > 0)
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = np.divide(vecs, norm, out=np.zeros_like(vecs), where=norm > 0)
    cos = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return jac[iu], cos[iu]


def _pair_index_stats(trial_means, pairs, cap, rng):
    """trial_means: [T, D] one flattened code per trial. pairs: list of (i, j) index
    tuples (already restricted to one grouping, e.g. same-subject or diff-subject) --
    subsampled to `cap` pairs, then weighted Jaccard (Ruzicka, see _pairwise_stats) /
    cosine computed per pair (not all-pairs, since `pairs` is already the exact set this
    grouping cares about)."""
    if len(pairs) > cap:
        pairs = [pairs[k] for k in rng.choice(len(pairs), cap, replace=False)]
    if not pairs:
        return None
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    va, vb = np.maximum(trial_means[ia], 0.0), np.maximum(trial_means[ib], 0.0)
    mins = np.minimum(va, vb).sum(axis=1)
    maxs = np.maximum(va, vb).sum(axis=1)
    jac = np.divide(mins, maxs, out=np.zeros_like(mins), where=maxs > 0)
    na, nb = np.linalg.norm(va, axis=1), np.linalg.norm(vb, axis=1)
    denom = na * nb
    cos = np.divide((va * vb).sum(axis=1), denom, out=np.zeros(len(pairs)), where=denom > 0)
    return jac, cos


def plot_patch_similarity_hierarchy(out_path, trial_records, unit_label='Filter',
                                     max_patches_per_trial=30, max_trials_per_group=60,
                                     max_pairs=4000, seed=0):
    """Same Jaccard(atom sets)/cosine(activation vectors) test as plot_intra_patch_relation,
    but instead of asking 'do two Filters agree within one patch', asks 'does the same
    Filter+atom code get reused across patches at increasing scope':

    trial_records: list of {usage: [M, Q, F] np.ndarray, dataset: str, subject: hashable}
    (one entry per sampled trial, built by BaseCodebookChecker.check_codebook).

    Intra-Trial:            patches within the SAME trial (temporal continuity effect, if any).
    Inter-Trial/Intra-Subj: trial-mean codes, DIFFERENT trials, SAME (dataset, subject).
    Inter-Subj/Intra-Dataset: trial-mean codes, SAME dataset, DIFFERENT subject.
    Cross-dataset pairs are excluded entirely -- plot_dataset_relation already answers "do
    datasets differ" (via usage-frequency divergence); mixing it into this per-instance
    code-similarity hierarchy would conflate two different questions under one bar and
    was the bug in the earlier 3-bucket version (its "Inter-Subject" bucket silently
    lumped diff-subject-same-dataset together with diff-dataset pairs).

    High Intra-Trial that drops at Inter-Trial/Intra-Subj means the code tracks
    trial-specific content, not just subject identity; high Inter-Subj/Intra-Dataset means
    the codebook has collapsed onto something generic (or the same physiological state)
    shared by every subject in that dataset, regardless of who or what trial it's looking at.

    Also prints a top-k most-redundant unit-pair list (see print_redundant_unit_pairs) --
    the former plot_intra_patch_relation Q x Q heatmap, condensed to just the pairs worth
    acting on instead of a dense grid that reads as uniform noise once Q is large. Answers
    a different question than the bars above: the bars say HOW MUCH code repeats at each
    scope; this says WHICH units are responsible when it does."""
    rng = np.random.RandomState(seed)

    sample = trial_records if len(trial_records) <= max_trials_per_group else \
        [trial_records[i] for i in rng.choice(len(trial_records), max_trials_per_group, replace=False)]
    intra_jac, intra_cos = [], []
    for t in sample:
        flat = t['usage'].reshape(t['usage'].shape[0], -1)  # [M, Q*F]
        stats = _pairwise_stats(flat, max_patches_per_trial, rng)
        if stats is None:
            continue
        intra_jac.append(stats[0]); intra_cos.append(stats[1])
    intra_jac = np.concatenate(intra_jac) if intra_jac else np.array([np.nan])
    intra_cos = np.concatenate(intra_cos) if intra_cos else np.array([np.nan])

    trial_means = np.stack([t['usage'].mean(axis=0).reshape(-1) for t in trial_records])  # [T, Q*F]
    datasets = [t['dataset'] for t in trial_records]
    subjects = [t['subject'] for t in trial_records]
    T = len(trial_records)
    idx_pool = np.arange(T) if T <= max_trials_per_group * 4 else \
        rng.choice(T, max_trials_per_group * 4, replace=False)

    same_subj_pairs, diff_subj_pairs = [], []
    for a in range(len(idx_pool)):
        i = idx_pool[a]
        for j in idx_pool[a + 1:]:
            if datasets[i] != datasets[j]:
                continue  # cross-dataset pairs excluded -- see plot_dataset_relation instead
            (same_subj_pairs if subjects[i] == subjects[j] else diff_subj_pairs).append((i, j))

    inter_trial = _pair_index_stats(trial_means, same_subj_pairs, max_pairs, rng)
    inter_subj  = _pair_index_stats(trial_means, diff_subj_pairs, max_pairs, rng)

    groups = [
        ('Intra-Trial\n(patches, same trial)', intra_jac, intra_cos),
        ('Inter-Trial\n(same subject)', *(inter_trial if inter_trial else (np.array([np.nan]),) * 2)),
        ('Inter-Subject\n(same dataset)', *(inter_subj if inter_subj else (np.array([np.nan]),) * 2)),
    ]
    labels    = [g[0] for g in groups]
    jac_mean  = [np.nanmean(g[1]) for g in groups]
    jac_std   = [np.nanstd(g[1]) for g in groups]
    cos_mean  = [np.nanmean(g[2]) for g in groups]
    cos_std   = [np.nanstd(g[2]) for g in groups]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, jac_mean, w, yerr=jac_std, capsize=4, color='darkorange', label='Weighted Jaccard (Ruzicka)')
    ax.bar(x + w / 2, cos_mean, w, yerr=cos_std, capsize=4, color='steelblue', label='Cosine (activation vectors)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Similarity'); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f'{unit_label} Selection Similarity by Grouping\n(high = same code reused across that grouping)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
    for name, jm, js, cm, cs in zip(labels, jac_mean, jac_std, cos_mean, cos_std):
        print(f"    {name.splitlines()[0]}: jaccard={jm:.3f}+/-{js:.3f}  cosine={cm:.3f}+/-{cs:.3f}")

    print_redundant_unit_pairs([t['usage'] for t in trial_records], unit_label=unit_label)


def plot_stamp_similarity(out_path, trial_records, unit_label='Stamp',
                           max_patches_per_trial=30, max_trials_per_group=60,
                           max_pairs=4000, seed=0):
    """Flat-token StampBank analog of plot_patch_similarity_hierarchy — same
    Jaccard(atom support)/cosine(content) test, same 3 of its 4 groupings, but two
    differences the flat (channel, patch) token design makes possible/necessary:

    trial_records: list of {content: [C, N, n_stamps, patch_len] np.ndarray, dataset,
    subject} (one entry per sampled trial) — REAL DECODED CONTENT per (channel, patch,
    stamp), zero-filled at stamps that (channel, patch) token didn't select (see
    model/MeSAE/plugin.py's extract_stamp_content), not the scalar gating strength `h`
    plot_patch_similarity_hierarchy's `usage` used. Cosine here therefore measures whether
    the same stamp produces similar real signal when reused, not just whether it was
    selected with a similar confidence — h is bounded (softmax over top_k / a fixed
    constant for shared), which compresses cosine values toward each other regardless of
    how different the underlying content actually was.

    Intra-Patch (channels, same patch, same trial): NEW grouping this flat design enables
    — a pooled design has one token per patch (no channel axis to compare within a patch);
    this one has C independent tokens per patch, so "do neighboring channels pick similar
    stamps/content at the same instant" is now a real, answerable question. High here
    means stamp selection is spread near-uniformly across channels at that patch; low
    means it's spatially localized (see the amplitude/localization discussion this
    architecture was built around).

    Intra-Trial/Inter-Trial/Inter-Subject: same definitions as
    plot_patch_similarity_hierarchy, computed on CHANNEL-COLLAPSED (mean over C) content —
    kept at the same granularity as before so these three stay comparable to the pooled
    model's numbers; only Intra-Patch and the content-vs-gating cosine basis are new.
    """
    rng = np.random.RandomState(seed)

    sample = trial_records if len(trial_records) <= max_trials_per_group else \
        [trial_records[i] for i in rng.choice(len(trial_records), max_trials_per_group, replace=False)]

    intra_patch_jac, intra_patch_cos = [], []
    intra_trial_jac, intra_trial_cos = [], []
    for t in sample:
        content = t['content']  # [C, N, n_stamps, patch_len]
        C, N = content.shape[0], content.shape[1]

        patch_idxs = np.arange(N) if N <= max_patches_per_trial else \
            rng.choice(N, max_patches_per_trial, replace=False)
        for n in patch_idxs:
            stats = _pairwise_stats(content[:, n].reshape(C, -1), C, rng)  # channels @ this patch
            if stats is not None:
                intra_patch_jac.append(stats[0]); intra_patch_cos.append(stats[1])

        collapsed = content.mean(axis=0).reshape(N, -1)  # [N, n_stamps*patch_len] -- channel-collapsed
        stats = _pairwise_stats(collapsed, max_patches_per_trial, rng)
        if stats is not None:
            intra_trial_jac.append(stats[0]); intra_trial_cos.append(stats[1])

    intra_patch_jac = np.concatenate(intra_patch_jac) if intra_patch_jac else np.array([np.nan])
    intra_patch_cos = np.concatenate(intra_patch_cos) if intra_patch_cos else np.array([np.nan])
    intra_trial_jac = np.concatenate(intra_trial_jac) if intra_trial_jac else np.array([np.nan])
    intra_trial_cos = np.concatenate(intra_trial_cos) if intra_trial_cos else np.array([np.nan])

    trial_means = np.stack([t['content'].mean(axis=(0, 1)).reshape(-1) for t in trial_records])  # [T, n_stamps*patch_len]
    datasets = [t['dataset'] for t in trial_records]
    subjects = [t['subject'] for t in trial_records]
    T = len(trial_records)
    idx_pool = np.arange(T) if T <= max_trials_per_group * 4 else \
        rng.choice(T, max_trials_per_group * 4, replace=False)

    same_subj_pairs, diff_subj_pairs = [], []
    for a in range(len(idx_pool)):
        i = idx_pool[a]
        for j in idx_pool[a + 1:]:
            if datasets[i] != datasets[j]:
                continue  # cross-dataset pairs excluded -- see plot_dataset_relation instead
            (same_subj_pairs if subjects[i] == subjects[j] else diff_subj_pairs).append((i, j))

    inter_trial = _pair_index_stats(trial_means, same_subj_pairs, max_pairs, rng)
    inter_subj  = _pair_index_stats(trial_means, diff_subj_pairs, max_pairs, rng)

    groups = [
        ('Intra-Patch\n(channels, same patch)', intra_patch_jac, intra_patch_cos),
        ('Intra-Trial\n(patches, same trial)', intra_trial_jac, intra_trial_cos),
        ('Inter-Trial\n(same subject)', *(inter_trial if inter_trial else (np.array([np.nan]),) * 2)),
        ('Inter-Subject\n(same dataset)', *(inter_subj if inter_subj else (np.array([np.nan]),) * 2)),
    ]
    labels    = [g[0] for g in groups]
    jac_mean  = [np.nanmean(g[1]) for g in groups]
    jac_std   = [np.nanstd(g[1]) for g in groups]
    cos_mean  = [np.nanmean(g[2]) for g in groups]
    cos_std   = [np.nanstd(g[2]) for g in groups]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, jac_mean, w, yerr=jac_std, capsize=4, color='darkorange', label='Weighted Jaccard (Ruzicka)')
    ax.bar(x + w / 2, cos_mean, w, yerr=cos_std, capsize=4, color='steelblue', label='Cosine (decoder output)')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Similarity'); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f'{unit_label} Selection Similarity by Grouping\n(high = same code (and content) reused across that grouping)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
    for name, jm, js, cm, cs in zip(labels, jac_mean, jac_std, cos_mean, cos_std):
        print(f"    {name.splitlines()[0]}: jaccard={jm:.3f}+/-{js:.3f}  cosine={cm:.3f}+/-{cs:.3f}")


def _patch_position_consistency_grids(codes, subjects, max_trials, rng):
    """codes: [T, N, D] (T trials, N patch positions, D=Q*F flattened code) -> two
    [T, N] grids (weighted Jaccard, Cosine): cell (t, n) = mean similarity of trial t's
    code at patch position n against every OTHER trial's code at that same position n.
    Rows are sorted by subject (stable sort) so a caller can draw subject-block
    boundaries -- lets a viewer check whether a pooled-average dip actually holds up
    within every subject's own block instead of only appearing once everyone's trials are
    averaged together.

    Jaccard here is weighted (Ruzicka similarity: sum(min(a,b))/sum(max(a,b)), see
    _pairwise_stats) rather than a binary (>0 mask) set overlap -- how STRONGLY/how many
    times a stamp fired now matters, not just whether it fired at all."""
    T = codes.shape[0]
    if T > max_trials:
        idx = rng.choice(T, max_trials, replace=False)
        codes, subjects = codes[idx], subjects[idx]
        T = max_trials
    order = np.argsort(subjects, kind='stable')
    codes, subjects = codes[order], subjects[order]

    N = codes.shape[1]
    jac_grid = np.zeros((T, N))
    cos_grid = np.zeros((T, N))
    for n in range(N):
        v = np.maximum(codes[:, n, :], 0.0)  # [T, D] -- Ruzicka needs nonnegative weights
        mins = np.minimum(v[:, None, :], v[None, :, :]).sum(axis=-1)
        maxs = np.maximum(v[:, None, :], v[None, :, :]).sum(axis=-1)
        jac = np.divide(mins, maxs, out=np.zeros_like(mins), where=maxs > 0)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        unit = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
        cos = unit @ unit.T
        np.fill_diagonal(jac, np.nan); np.fill_diagonal(cos, np.nan)
        jac_grid[:, n] = np.nanmean(jac, axis=1)
        cos_grid[:, n] = np.nanmean(cos, axis=1)
    return jac_grid, cos_grid, subjects


def plot_patch_position_consistency(out_path, trial_records, unit_label='Filter',
                                     max_trials=90, seed=0, code_label='activation vector'):
    """trial_records: trials from ONE dataset (usage [N, Q, F] each, same N across trials
    -- patch position n therefore means the same thing, e.g. time-since-trial-onset, in
    every trial; mixing datasets here would compare unrelated timelines). `usage` is
    basis-agnostic (gating strength like extract_usage's, or real decoder content like
    MeSAECodebookChecker.extract_stamp_content's channel-collapsed version) -- pass a
    matching code_label ('activation vector' default, e.g. 'decoder output') so the Cosine
    panel's title says which. Two Trial x
    Patch grids side by side, cell (t, n) = trial t's code at patch n vs every OTHER
    trial's code at that same n (see _patch_position_consistency_grids):

    - Jaccard (weighted/Ruzicka): do trials agree on WHICH Filters+atoms fire, and how strongly, at this patch position.
    - Cosine: do trials agree on the actual activation VECTOR there.

    A patch column bright in both = a structurally consistent slot across trials
    (baseline/ITI, before or after whatever event this trial contains). A column bright
    in Jaccard but dark in Cosine = trials converge on the same Filter vocabulary at that
    position but with very different activation content -- the signature of a
    trial-informative event patch (same "which Filters", different "what they say").

    Rows are grouped by subject (horizontal boundary lines, subject id on the y-axis)
    instead of left in arbitrary sample order -- a dip that only shows up once every
    subject's trials are averaged together, but doesn't hold up within any single
    subject's own block, is a pooling artifact, not a real per-subject phenomenon.
    max_trials defaults higher than the other panels here (90, not ~30) specifically so
    a multi-subject dataset still gets enough rows per subject to judge individually.

    A final 'Avg' row (separated by a thicker line) appends the per-patch mean across
    every sampled trial/subject -- the same curve the console dip/cliff detector below
    already computes numerically, drawn here so the pooled trend is visible directly
    against the per-subject rows above it instead of only living in printed text. Use it
    to check whether a dip visible in the average genuinely holds up across most subject
    blocks (real, patch-position-linked effect) or is dragged down by just one or two
    outlier subjects while the rest stay flat (a pooling artifact the average alone would
    hide)."""
    rng = np.random.RandomState(seed)
    Ns = {t['usage'].shape[0] for t in trial_records}
    n_keep = min(Ns)
    codes = np.stack([t['usage'][:n_keep].reshape(n_keep, -1) for t in trial_records])  # [T, N, D]
    subjects = np.array([t['subject'] for t in trial_records])

    jac_grid, cos_grid, subjects = _patch_position_consistency_grids(codes, subjects, max_trials, rng)
    T, N = jac_grid.shape

    # Append the per-patch average (across all sampled trials/subjects) as one extra row
    # at the bottom of each grid, visually separated -- see docstring's 'Avg row' note.
    jac_disp = np.vstack([jac_grid, np.nanmean(jac_grid, axis=0, keepdims=True)])
    cos_disp = np.vstack([cos_grid, np.nanmean(cos_grid, axis=0, keepdims=True)])

    # Fixed [0,1] color scale wastes almost the whole colormap: these values live in a
    # narrow band (typically ~0.15-0.35), so a real dip of a few hundredths was
    # indistinguishable by eye. Scale each panel to its OWN data range instead -- Jaccard
    # and Cosine already get separate colorbars, so there's no shared-scale reason to keep
    # them both pinned to [0,1].
    boundaries = np.where(np.diff(subjects) != 0)[0] + 0.5  # row index right after each subject's last row
    # one y-tick per subject, centered on that subject's own row block
    block_starts = np.concatenate(([0], boundaries + 0.5))
    block_ends = np.concatenate((boundaries + 0.5, [T]))
    tick_pos = (block_starts + block_ends) / 2 - 0.5
    tick_labels = [subjects[int(p)] for p in tick_pos]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 0.3 * N * 2), max(5, 0.15 * (T + 1))))
    for ax, grid, title in ((axes[0], jac_disp, 'Cross-Trial Weighted Jaccard\n(same Filters+atoms, similar strength)'),
                             (axes[1], cos_disp, f'Cross-Trial Cosine\n(same {code_label})')):
        im = ax.imshow(grid, aspect='auto', vmin=float(np.nanmin(grid)), vmax=float(np.nanmax(grid)), cmap='YlOrRd')
        for b in boundaries:
            ax.axhline(b, color='black', linewidth=0.8, alpha=0.6)
        ax.axhline(T - 0.5, color='black', linewidth=1.6)  # separates per-subject rows from the Avg row
        ax.set_yticks(list(tick_pos) + [T])
        ax.set_yticklabels([f'S{s}' for s in tick_labels] + ['Avg'], fontsize=6)
        ax.get_yticklabels()[-1].set_fontweight('bold')
        ax.set_xlabel('Patch (time within trial)', fontsize=9)
        ax.set_ylabel('Trial (grouped by subject)', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f'{unit_label} Cross-Trial Consistency by Patch Position\n'
                 '(bright both = consistent baseline; bright Jaccard + dark Cosine = same Filters, different content)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")

    # Per-patch-position dip detector, collapsing the Trial axis: for each of the two
    # curves (Jaccard-by-patch, Cosine-by-patch) independently -- NOT their difference,
    # a real event patch may dip in one curve deeper than the other -- find where that
    # curve is lowest and how deep relative to its OWN noise floor (z = (mean-min)/std).
    # An aggregate std-across-patches alone can't tell "flat" from "one sharp localized
    # dip drowned in a mostly-flat baseline"; this reports the dip itself, not just
    # whether any variance exists. A run of several adjacent low-z patches (not just one)
    # is the "cliff" signature of a multi-patch event; a single low point is a one-patch dip.
    jac_by_patch = np.nanmean(jac_grid, axis=0)  # [N]
    cos_by_patch = np.nanmean(cos_grid, axis=0)  # [N]
    for name, arr in (('Jaccard', jac_by_patch), ('Cosine', cos_by_patch)):
        m, s = arr.mean(), arr.std()
        amin = int(np.argmin(arr))
        z = (m - arr[amin]) / (s + 1e-8)
        # Tightened: 1 full std below mean (was 0.5), and only counts genuine CONSECUTIVE
        # runs of length >=2 -- the previous version reported min(idx)-max(idx) over every
        # qualifying patch regardless of whether they were adjacent, which inflated a
        # scatter of unrelated low points into a fake-looking wide "run".
        below = np.where(arr < m - s)[0]
        longest = []
        if len(below):
            splits = np.where(np.diff(below) > 1)[0] + 1
            runs = [r for r in np.split(below, splits) if len(r) >= 2]
            if runs:
                longest = max(runs, key=len)
        cliff = f"patches {longest[0]}-{longest[-1]} ({len(longest)} consecutive)" if len(longest) else "none"
        print(f"    {name} per-patch profile: mean={m:.3f} std={s:.3f} | "
              f"deepest dip: patch {amin} (value={arr[amin]:.3f}, z={z:.2f}) | cliff (>=2 consecutive, <mean-1std): {cliff}")


def _js_divergence(p, q, eps=1e-12):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps) + eps))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def plot_unit_freedom(out_path, usage_by_dataset, unit_label='Filter', rank_ceiling=None):
    """Per-unit MEASURED freedom, two views side by side:

    Effective rank: participation ratio (Sum(lambda))^2 / Sum(lambda^2) of the sparse
    code's [M, F] covariance eigenvalues, per unit — how many independent directions the
    unit's code actually spans across the sampled corpus.

    Atom-usage entropy: entropy (nats) of the unit's firing frequency across the F
    dictionary features — low entropy means the unit always reuses the same handful of
    atoms regardless of patch content, a different failure mode from low effective rank
    (a unit can have several active directions that are still always the SAME few atoms
    across every dataset/trial, i.e. high rank locally, low entropy globally, or vice
    versa) — report both, they're not redundant.

    rank_ceiling: a SOFT reference only (e.g. MeSAE's min(sae_k, embed_dim)), passed in by
    the caller since this module has no model handle. NOT a hard bound -- sae_k only caps
    how many atoms are nonzero per single patch; the corpus-pooled covariance rank isn't
    bounded by it, since different patches can activate different, only-partially-
    overlapping k-subsets, so measured effective rank routinely sits ABOVE this line. Read
    it as "rough scale of what one patch's sparsity implies," not "the ceiling," and don't
    treat a bar above the line as an error. Pass None to fall back to F (dictionary size,
    an even looser reference: the formula's own mathematical max, reached only if the
    eigenvalue spectrum were perfectly flat).

    Both panels draw a dashed reference line (rank: rank_ceiling or F if None -- soft
    reference, see above; entropy: log(F) always, which IS the true max there, no separate
    architectural cap applies) — gives a rough visual sense of scale/headroom, not a
    pass/fail bound.
    Want: both high and roughly uniform across units => each unit exploiting its share of
    capacity, diverse atom use. Both low on the same unit => dead/collapsed unit, candidate
    for pruning or codebook-size reduction. High rank + low entropy => unit varies
    magnitude/combination of the same few atoms without diversifying WHICH atoms fire —
    check router/gating; may mean the atom pool is too small for that unit's patch
    diversity. Cross-check against plot_dataset_relation: if a unit's low entropy tracks a
    single dataset's private vocabulary, the unit may be overfit to that dataset rather than
    learning paradigm-general structure.
    """
    combined = np.concatenate(list(usage_by_dataset.values()), axis=0)  # [M, Q, F]
    if combined.ndim == 2:
        # Flat [M, Q] usage (e.g. StampBank -- a single scalar per unit per patch, no F
        # axis, see docs/adr/0009's Monitoring impact section). Both panels here measure
        # per-unit structure WITHIN a feature axis that doesn't exist for this kind of
        # unit -- nothing meaningful to compute, skip rather than render degenerate/NaN
        # plots from a fabricated F=1 axis.
        print(f"  [codebook] -> {out_path} skipped (usage has no per-unit feature axis)")
        return
    Q, F = combined.shape[1], combined.shape[2]
    freq = _usage_freq(combined)  # [Q, F]

    eff_rank = np.zeros(Q)
    entropy = np.zeros(Q)
    for q in range(Q):
        eigvals = np.linalg.eigvalsh(np.cov(combined[:, q, :], rowvar=False))
        eigvals = np.clip(eigvals, 0, None)
        s = eigvals.sum()
        eff_rank[q] = (s ** 2) / (np.square(eigvals).sum() + 1e-12) if s > 0 else 0.0

        p = freq[q]
        p = p / (p.sum() + 1e-12)
        nz = p > 0
        entropy[q] = -(p[nz] * np.log(p[nz])).sum() if nz.any() else 0.0

    # rank_ceiling is a soft reference (e.g. min(sae_k, embed_dim)) when the caller has
    # one, NOT a hard bound -- measured rank routinely exceeds it, see docstring. F alone
    # is an even looser reference (the formula's own mathematical max). entropy's max is
    # always log(F), a true bound there.
    eff_rank_max = float(rank_ceiling) if rank_ceiling is not None else float(F)
    rank_max_label = f'soft ref = {eff_rank_max:.0f}' if rank_ceiling is not None else f'ref (F) = {eff_rank_max:.0f}'
    entropy_max = math.log(F) if F > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.35 * Q), 4.5))
    axes[0].bar(range(Q), eff_rank, color='steelblue')
    axes[0].axhline(eff_rank_max, color='crimson', linestyle='--', linewidth=1, label=rank_max_label)
    axes[0].set_title('Effective Rank (participation ratio)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel(unit_label, fontsize=8); axes[0].set_ylabel('Effective dims', fontsize=8)
    axes[0].legend(fontsize=7, loc='upper right')

    axes[1].bar(range(Q), entropy, color='darkorange')
    axes[1].axhline(entropy_max, color='crimson', linestyle='--', linewidth=1,
                     label=f'theoretical max = log(F) = {entropy_max:.2f}')
    axes[1].set_title('Atom-Usage Entropy (nats)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel(unit_label, fontsize=8); axes[1].set_ylabel('Entropy', fontsize=8)
    axes[1].legend(fontsize=7, loc='upper right')

    fig.suptitle(f'{unit_label} Freedom (measured capacity utilization, shared dictionary ceiling)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_dataset_relation(out_path, usage_by_dataset, unit_label='Filter'):
    """D x D Jensen-Shannon divergence between datasets' pooled dictionary usage
    distributions (all units flattened together) — low divergence = datasets lean on the
    same vocabulary, high = distinct.

    Expected pattern: within-paradigm pairs (e.g. two SSVEP datasets, or two MI datasets)
    should be LOW JS — same generative process, codebook should reuse atoms across
    subjects/sessions. Cross-paradigm pairs (SSVEP vs MI) should be HIGHER JS — different
    spectral/spatial signatures (SSVEP = narrowband steady-state entrainment, MI = broadband
    event-related desync/sync) justify distinct atom usage. Sorting dataset_names by
    paradigm should reveal block structure: low JS within blocks, high JS between.

    Failure modes: all cells uniformly high (~log(2)) => codebook fragmented into
    per-dataset private vocab, no shared structure learned. All cells uniformly low (~0)
    => codebook collapsed onto a handful of atoms regardless of paradigm (cross-check
    against plot_unit_freedom's entropy panel — low entropy + low JS everywhere = collapse,
    not generalization)."""
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


def plot_stamp_identity_consistency(out_path, within, between, per_stamp_ids, per_stamp_within,
                                     label_agree=None, unit_label='Stamp'):
    """Does a stamp id mean the same thing every time it fires?

    The waveform half of that question is trivially yes — a stamp's template D_i is a
    fixed parameter, so its temporal shape is identical at every occurrence up to
    amplitude and phase. The open half is the TOPOGRAPHY: the mixing column is
    recomputed per (channel, patch) from the data, and nothing in the architecture ties
    one occurrence to the next (group selection binds channels WITHIN a patch, not
    across patches). If stamp identity carries topographic meaning, two occurrences of
    the same id should be more alike than two occurrences of different ids.

    within/between: 1-D arrays of cosine similarities between per-occurrence mixing
    columns — same id vs different ids. Both must be computed the same way (columns
    centered across channels, then unit-normalized, and BOTH sides comparing individual
    occurrences): raw magnitude columns are non-negative so their cosines are pushed
    toward 1 regardless of structure, and comparing individual columns against averaged
    ones makes the averaged side look artificially self-similar. The separation
    (within - between) is the real readout: ~0 means the id predicts nothing about
    topography, i.e. the stamp is a waveform type rather than a source.

    per_stamp_ids/per_stamp_within: per-id mean within-consistency, for the bar panel.
    label_agree: optional dict id -> modal-ICLabel-class agreement across trials.
    """
    import numpy as np
    ncol = 3 if label_agree else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.4 * ncol, 4.2), squeeze=False)
    ax = axes[0, 0]
    bins = np.linspace(-1, 1, 60)
    ax.hist(between, bins=bins, alpha=0.6, label=f'different {unit_label.lower()}s', color='gray', density=True)
    ax.hist(within, bins=bins, alpha=0.6, label=f'same {unit_label.lower()}', color='seagreen', density=True)
    sep = float(np.mean(within) - np.mean(between))
    ax.axvline(np.mean(between), color='gray', ls='--', lw=1)
    ax.axvline(np.mean(within), color='seagreen', ls='--', lw=1)
    ax.set_title(f'Mixing-column similarity\nwithin {np.mean(within):.3f} vs between '
                 f'{np.mean(between):.3f}  (sep {sep:+.3f})', fontsize=10, fontweight='bold')
    ax.set_xlabel('cosine (centered columns)'); ax.set_ylabel('density'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    order = np.argsort(-np.asarray(per_stamp_within))
    ax.bar(range(len(order)), np.asarray(per_stamp_within)[order], color='steelblue')
    ax.axhline(np.mean(between), color='gray', ls='--', lw=1, label='between-stamp mean')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(per_stamp_ids[i]) for i in order], fontsize=5, rotation=90)
    ax.set_title('Per-stamp topographic self-consistency\n(above dashed line = id carries '
                 'topographic meaning)', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{unit_label} id'); ax.set_ylabel('mean within-id cosine'); ax.legend(fontsize=8)

    if label_agree:
        ax = axes[0, 2]
        vals = np.asarray(list(label_agree.values()))
        ax.hist(vals, bins=np.linspace(0, 1, 21), color='darkorange')
        ax.axvline(vals.mean(), color='k', ls='--', lw=1)
        ax.set_title(f'ICLabel class stability per {unit_label.lower()}\n'
                     f'mean modal agreement {vals.mean():.2f}, always-same '
                     f'{np.mean(vals == 1.0):.2f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('modal-class agreement across trials'); ax.set_ylabel(f'# {unit_label.lower()}s')

    fig.suptitle(f'{unit_label} Identity Consistency — does one id mean one thing?',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
    print(f"    within-id {np.mean(within):.3f} | between-id {np.mean(between):.3f} | "
          f"separation {sep:+.3f}"
          + (f" | ICLabel modal agreement {np.mean(list(label_agree.values())):.3f}" if label_agree else ""))
