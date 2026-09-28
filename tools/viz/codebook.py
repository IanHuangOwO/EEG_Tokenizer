"""Cross-dataset stamp panels for BaseCodebookChecker.check_codebook (model/base_codebook_checker.py).
usage_by_dataset: dict of dataset_name -> np.ndarray [M, Q] (M patches sampled from that dataset, Q
units: each stamp's strength h at that patch) built by the checker's extract_usage() hook.
"""

import math

import numpy as np
import matplotlib.pyplot as plt


def _palette(n):
    """n distinct colors, n unbounded (unlike tab10/tab20) — hsv wraps smoothly so
    high-cardinality targets (e.g. BETA's 40 classes) don't collide."""
    return plt.cm.hsv(np.linspace(0, 1, max(n, 1), endpoint=False))


def plot_usage_and_activity(out_path, strength, categories, category_order,
                             unit_label='Stamp', normalize=True, max_rows_per_subplot=100):
    """Q x D heatmap of mean per-unit strength by dataset (strength [M, Q], categories [M]): how hard
    each unit works on each dataset. Rows are chunked across side-by-side subplots
    (max_rows_per_subplot each) so a large Q makes the image wide, not absurdly tall."""
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


def plot_embedding_scatter_by_dataset(out_path, usage_by_dataset, unit_label='Stamp',
                                       max_points=3000, random_state=0):
    """PCA + t-SNE of each patch's stamp strengths (all units concatenated), colored by
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
        ax.set_title(f'{title} of {unit_label} strengths', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8, markerscale=2, loc='best')

    fig.suptitle('Cross-Dataset Embedding Separation', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_embedding_scatter_by_target(out_path_combined, out_path_per_dataset, usage_by_dataset,
                                      labels_by_dataset, unit_label='Stamp',
                                      max_points=3000, random_state=0):
    """Same patch-level strengths/sampling as plot_embedding_scatter, but colored by class
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
        ax.set_title(f'{title} of {unit_label} strengths', fontsize=11, fontweight='bold')
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

    fig.suptitle(f'Intra-Dataset Class Separation, All Targets ({unit_label} strengths)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path_per_dataset, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path_per_dataset}")


def _patch_position_consistency_grid(codes, subjects, max_trials, rng):
    """codes: [T, N, Q] nonnegative per-unit strength (T trials, N patch positions) -> [T, N] grid of
    weighted Jaccard (Ruzicka: sum(min(a,b))/sum(max(a,b))) of trial t's code at position n against
    every OTHER trial's code at that same n. Rows sorted by subject (stable) so a pooled dip can be
    checked within every subject's own block."""
    T = codes.shape[0]
    if T > max_trials:
        idx = rng.choice(T, max_trials, replace=False)
        codes, subjects = codes[idx], subjects[idx]
        T = max_trials
    order = np.argsort(subjects, kind='stable')
    codes, subjects = codes[order], subjects[order]
    N = codes.shape[1]
    grid = np.zeros((T, N))
    for n in range(N):
        v = np.maximum(codes[:, n, :], 0.0)                                  # [T, Q]
        mins = np.minimum(v[:, None, :], v[None, :, :]).sum(axis=-1)
        maxs = np.maximum(v[:, None, :], v[None, :, :]).sum(axis=-1)
        w_jac = np.divide(mins, maxs, out=np.zeros_like(mins), where=maxs > 0)
        np.fill_diagonal(w_jac, np.nan)
        grid[:, n] = np.nanmean(w_jac, axis=1)
    return grid, subjects


def plot_patch_position_consistency(out_path, trial_records, unit_label='Stamp',
                                     max_trials=90, seed=0, event_patch=None):
    """trial_records: trials from ONE dataset (usage [N, Q] each, same N across trials, so patch
    position n means the same time within the trial everywhere). Cross-trial weighted Jaccard of the
    per-unit strength at each patch position (see _patch_position_consistency_grid), drawn as a
    mean +/- std trend over positions with thin per-subject lines (up to 12 subjects): a dip that
    holds across subjects marks a trial-informative (event) position; a flat high curve is a
    consistent baseline. Also prints the deepest dip and the longest run below mean - 1 std."""
    rng = np.random.RandomState(seed)
    n_keep = min(t['usage'].shape[0] for t in trial_records)
    codes = np.stack([t['usage'][:n_keep] for t in trial_records])                # [T, N, Q]
    subjects = np.array([t['subject'] for t in trial_records])
    grid, subjects = _patch_position_consistency_grid(codes, subjects, max_trials, rng)
    N = grid.shape[1]
    x = np.arange(N)
    unique_subjects = sorted(set(subjects.tolist()))
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * N), 5))
    mean, std = np.nanmean(grid, axis=0), np.nanstd(grid, axis=0)
    ax.fill_between(x, mean - std, mean + std, color='steelblue', alpha=0.2, label='+/-1 std (all trials)')
    ax.plot(x, mean, color='steelblue', linewidth=2.2, label='Mean (all trials)')
    if len(unique_subjects) <= 12:
        for i, s in enumerate(unique_subjects):
            ax.plot(x, np.nanmean(grid[subjects == s], axis=0), color=cmap(i % 10), linewidth=0.8, alpha=0.6,
                    label=f'S{s}')
    if event_patch is not None:  # tools.analysis.event_onset_patch
        ax.axvline(event_patch, color='k', ls='--', lw=1.0, alpha=0.8, label='event onset')
    ax.set_xlabel('Patch (time within trial)', fontsize=9)
    ax.set_ylabel('Weighted Jaccard', fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc='best')
    fig.suptitle(f'{unit_label} Cross-Trial Consistency by Patch Position (weighted Jaccard of strengths)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")

    m, s = mean.mean(), mean.std()
    amin = int(np.argmin(mean))
    below = np.where(mean < m - s)[0]
    longest = []
    if len(below):
        runs = [r for r in np.split(below, np.where(np.diff(below) > 1)[0] + 1) if len(r) >= 2]
        if runs:
            longest = max(runs, key=len)
    cliff = f"patches {longest[0]}-{longest[-1]} ({len(longest)} consecutive)" if len(longest) else "none"
    print(f"    per-patch profile: mean={m:.3f} std={s:.3f} | deepest dip: patch {amin} "
          f"(value={mean[amin]:.3f}, z={(m - mean[amin]) / (s + 1e-8):.2f}) | cliff (>=2 consecutive, <mean-1std): {cliff}")


def _js_divergence(p, q, eps=1e-12):
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + eps) + eps))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def plot_dataset_relation(out_path, usage_by_dataset, unit_label='Stamp'):
    """D x D Jensen-Shannon divergence between datasets' mean per-unit strength distributions
    (usage [M, Q] averaged over patches, normalized to sum 1) -- low = datasets lean on the units in
    the same proportions, high = distinct. Within-paradigm pairs (two MI or two SSVEP sets) should be
    low, cross-paradigm pairs higher; sorting datasets by paradigm shows block structure."""
    dataset_names = list(usage_by_dataset.keys())
    dists = {ds: np.maximum(arr, 0).mean(axis=0) for ds, arr in usage_by_dataset.items()}
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

    ax.set_title(f'Dataset {unit_label}-Strength Divergence (JS of mean strength per {unit_label.lower()})',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")


def plot_stamp_identity_consistency(out_path, within, between, per_stamp_ids, per_stamp_within,
                                     label_agree=None, unit_label='Stamp'):
    """Does a stamp id mean the same thing at every occurrence?

    The waveform half of that question is trivially yes — a stamp's template D_i is a
    fixed parameter, so its temporal shape is identical at every occurrence up to
    amplitude and phase. The open half is the TOPOGRAPHY: the mixing column is
    recomputed per (channel, patch) from the data, and nothing in the architecture ties
    one occurrence to the next. If stamp identity carries topographic meaning, two occurrences of
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


def plot_fingerprint_similarity(out_path, matrix, unit_label='Stamp'):
    """Q x Q cosine similarity between each unit's own raw decoder template (content-free,
    no data dependence — see BaseCodebookChecker.decoder_fingerprint_matrix) — the direct
    structural redundancy check: two units with near-1 similarity here learned the same
    shape regardless of when/where they fire."""
    Q = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.08 * Q + 3), max(5, 0.08 * Q + 3)))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_title(f'{unit_label} Fingerprint Similarity (raw template cosine)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{unit_label} id'); ax.set_ylabel(f'{unit_label} id')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    off_diag = matrix[~np.eye(Q, dtype=bool)]
    print(f"  [codebook] -> {out_path}")
    print(f"    off-diagonal cosine: mean {off_diag.mean():.3f} | max {off_diag.max():.3f} "
          f"(near-1 = two atoms learned the same shape, mp_loss should keep this low)")


def plot_stamp_phase_consistency(out_path, circ_var, fire_count, unit_label='Stamp',
                                  min_fires=5):
    """Per-unit circular variance (1 - |mean(exp(i*phase))|, over all of that unit's occurrences'
    channel-summed phase atan2(b,a)) of the quadrature phase every occurrence carries but
    nothing in the loss ever reads directly (see CONTEXT.md's Stamp definition —
    contribution is a*D + b*H, amplitude sqrt(a^2+b^2), phase atan2(b,a)). Low variance
    (near 0) = phase-locked -- plausible for a genuine oscillatory source (e.g. line
    noise) that really does arrive at a consistent phase relative to the patch window.
    High variance (near 1) = effectively random phase -- broadband/transient content
    where phase carries no real information, just whatever the encoder's continuous
    quadrature gain happened to produce. Units with fewer than min_fires occurrences are
    dropped (a circular variance from 1-2 samples is meaningless)."""
    n_stamps = len(circ_var)
    keep = fire_count >= min_fires
    idx = np.where(keep)[0]
    if len(idx) == 0:
        print(f"  [codebook] phase consistency skipped (no unit fired >= {min_fires} times)")
        return
    order = idx[np.argsort(circ_var[idx])]

    fig, ax = plt.subplots(figsize=(max(8, 0.08 * len(order) + 4), 5))
    ax.bar(range(len(order)), circ_var[order], color='steelblue', width=1.0)
    ax.set_ylim(0, 1)
    ax.set_title(f'{unit_label} Phase Consistency, sorted (0 = phase-locked, 1 = random phase) — '
                 f'min {min_fires} occurrences',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel(f'{unit_label} rank by circular variance'); ax.set_ylabel('circular variance')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
    print(f"    phase circular variance ({len(order)}/{n_stamps} units with >= {min_fires} occurrences): "
          f"mean {circ_var[order].mean():.3f} | {int((circ_var[order] < 0.3).sum())} phase-locked (<0.3)")


def plot_topography_distance(out_path, matrices, unit_label='Stamp'):
    """One heatmap per dataset of pairwise cosine distance (1 - cosine similarity) between
    units' MEAN mixing column (the signed, coherently-averaged per-channel topography of
    every unit that fired at least a few times in that dataset -- see
    MeSAECodebookChecker._render_identity_consistency, which already computes the
    coherent per-occurrence (a,b) average this reuses). Complements
    plot_fingerprint_similarity (raw waveform shape, dataset-independent): two units can
    have very different D_i templates yet project to a similar scalp pattern, or vice
    versa -- this is the only panel that shows that pairing directly. Computed per
    dataset only (not pooled): different datasets map to different channel subsets, so a
    unit's topography vector isn't even the same length across datasets.

    matrices: dict dataset_name -> (dist [n,n] np.ndarray, unit_ids [n] list)."""
    names = list(matrices.keys())
    D = len(names)
    if D == 0:
        print('  [codebook] topography distance skipped (no dataset with enough repeated units)')
        return
    n_cols = min(3, D)
    n_rows = math.ceil(D / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows), squeeze=False)
    for i, ds_name in enumerate(names):
        dist, uids = matrices[ds_name]
        ax = axes[i // n_cols, i % n_cols]
        im = ax.imshow(dist, vmin=0, vmax=2, cmap='YlOrRd')
        n = len(uids)
        if n <= 30:
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(uids, fontsize=6, rotation=90)
            ax.set_yticklabels(uids, fontsize=6)
        else:
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{ds_name} (n={n})', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(D, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')
    fig.suptitle(f'{unit_label} Topography (mixing column) Cosine Distance, per dataset',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  [codebook] -> {out_path}")
