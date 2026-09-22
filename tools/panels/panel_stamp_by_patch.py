"""stamp_by_patch panel: the real per-patch stamp selection grid for one trial, driven by
tools/viz/extract.py's extract_flat_stamp_psd_by_patch. Reads ctx.bundle and ctx.cmap --
the caller builds the bundle via build_pretrain_bundle/build_finetune_bundle before
selecting this panel. plot_stamp_by_patch used to live in tools/viz/panels.py (now
tools/viz/stamp_plots.py) but had exactly one caller (this file), so it moved here
directly -- see that module's docstring."""
import os

import numpy as np
import matplotlib.pyplot as plt

from tools.viz.extract import extract_flat_stamp_psd_by_patch
from tools.viz.topomap import draw_topomap, build_triangulation, project_coords_2d

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = True
NEEDS_DATASET = True


def _log_pow(x):
    """log1p of a nonnegative power/PSD quantity (topo=L2 norm, psd=real^2+imag^2 — both
    always >=0, np.maximum guards float rounding noise below 0). log1p(0)=0 exactly, so
    the zero-fill "unused" floor (see _cell's vmin=0 anchoring) survives the log
    transform unchanged, while still compressing the large dynamic range a handful of
    high-power channels/bins would otherwise dominate on a linear scale."""
    return np.log1p(np.maximum(x, 0.0))


def _log_signed(x):
    """Signed counterpart of _log_pow for quantities where polarity matters (a stamp's
    per-channel amp — its mixing/topomap column): sign(x)*log1p(|x|). Odd function, 0
    maps to exactly 0, magnitudes compress the same way _log_pow's do, sign survives —
    so a symmetric diverging color scale centered on 0 stays meaningful after the
    transform."""
    return np.sign(x) * np.log1p(np.abs(x))


def plot_stamp_by_patch(out_path, pos2d, grid, cmap='YlOrRd', subject_id=None,
                            trial_idx=None, epoch_tag='', unit_label='Stamp',
                            n_routed=None, shared_color='crimson',
                            raw_power=None, recon_power=None, psd_raw=None, psd_recon=None,
                            signed_stamps=False, onset_col=None):
    """
    One column PER SAMPLED PATCH, topo+PSD side by side within a column (one column pair
    of subplot-columns). Row 0: each patch's own real raw input (grid.raw_topo/raw_psd).
    Row 1: that same patch's own real full reconstruction (grid.recon_topo/recon_psd).
    Rows 2..2+K-1: that patch's own top_k+n_shared slots, one row per slot — real
    per-patch selection and decoded content, not trial-averaged (see
    viz/extract.extract_flat_stamp_psd_by_patch/PatchGridResult): a stamp firing on many
    sampled patches shows up once per patch column it fired at, with that column's own
    real content, instead of being blurred into one dedup'd trial-averaged row the way
    plot_stamp_gallery's per-stamp rows are.

    raw_power/recon_power/psd_raw/psd_recon: optional whole-trial (not per-patch) Raw/
    Full-Recon topo+PSD, rendered as an extra header row above the per-patch grid when all
    four are given — omit (None, the default) to skip it, e.g. when that whole-trial view
    is rendered separately instead (see plot_stamp_gallery, MeSAE's split panel).

    Every topo/PSD cell is log1p-scaled (see _log_pow) before display — power/PSD values
    span orders of magnitude (a few strongly-selected channels/bins next to many
    near-zero ones), and a linear color scale mostly just shows the loudest cell; log1p
    compresses that range while keeping "exactly 0" (unselected) mapped to exactly 0, so
    the zero-anchored vmin below still means the same thing.

    grid: PatchGridResult. n_routed: global stamp id threshold — ids >= n_routed are
    shared stamps (same routed-then-shared layout StampBank.forward's idx uses, see
    PatchGridResult docstring), titled in shared_color instead of black. None disables
    the shared/routed title-color split.

    signed_stamps: True when grid.topo carries SIGNED per-channel amps (MeSAE's
    grouped StampBank — the mixing/topomap column, see _cell's signed branch); False
    (default) for unsigned norm-based topos (a pooled model's per-unit PSD extractor).

    onset_col: displayed-column index (0..P) where a real trial event occurs — columns
    before it are pre-event, columns from onset_col on are post-event (see
    build_pretrain_bundle's _lookup_event_onset for how this is derived from
    grid.patch_ids and the trial's own event_onset_sec). None (default, most calls — an
    assembled continuous window has no single event) draws nothing. Columns are laid out
    one-per-sampled-patch, evenly spaced regardless of real elapsed time (this is a
    subplot grid, not a shared time axis), so the boundary is drawn as a bold left border
    on every row's onset_col column rather than a positioned vertical line.
    """
    patch_ids, stamp_ids, topo, psd, h, freqs = (
        grid.patch_ids, grid.stamp_ids, grid.topo, grid.psd, grid.h, grid.freqs)
    P, K = stamp_ids.shape
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)
    has_header = raw_power is not None

    n_col_pairs = max(2, P)
    n_rows = (1 if has_header else 0) + 2 + K  # [header], per-patch raw, per-patch full recon, K stamp slots

    fig, axes = plt.subplots(n_rows, n_col_pairs * 2, figsize=(6 * n_col_pairs, 3.0 * n_rows),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f"Per-Patch {unit_label} Topo + PSD (every {P and patch_ids[1]-patch_ids[0] or 1}th "
                 f"patch's own top-k+shared, real content) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    triang = build_triangulation(pos2d)

    def _cell(row, col_pair, power, psd_cf, label, color, signed=False):
        # Unsigned cells (raw/recon power): vmin pinned to 0 — power is nonnegative by
        # construction, sequential cmap, per-cell vmax so each cell's own peak uses the
        # full range.
        # Signed cells (stamp topo = the stamp's per-channel amp, its mixing/topomap
        # column — see viz.extract.extract_flat_stamp_psd_by_patch): diverging RdBu_r
        # with SYMMETRIC limits so 0 = white and polarity reads directly — a dipolar
        # source's positive and negative lobes are the whole point of the plot; the old
        # abs+sequential rendering made a dipole look like two disconnected same-color
        # blobs, indistinguishable from two co-located sources. PSD stays unsigned
        # either way (real^2+imag^2).
        psd_cf = _log_pow(psd_cf)
        ax_topo, ax_psd = axes[row, col_pair * 2], axes[row, col_pair * 2 + 1]
        if signed:
            v = _log_signed(power)
            vlim = max(np.abs(v).max(), 1e-12)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap='RdBu_r',
                                 vmin=-vlim, vmax=vlim, triang=triang)
        else:
            v = _log_pow(power)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap=cmap,
                                 vmin=0.0, vmax=max(v.max(), 1e-12), triang=triang)
        ax_topo.set_title(f'{label} Topo', fontsize=8, fontweight='bold', color=color)
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd.imshow(psd_cf[::-1], aspect='auto', cmap=cmap, origin='lower',
                      vmin=0.0, vmax=max(psd_cf.max(), 1e-12),
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD', fontsize=8, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=6)
        ax_psd.set_yticks([])

    def _blank_rest(row, start_col):
        for c in range(start_col, n_col_pairs):
            axes[row, c * 2].axis('off')
            axes[row, c * 2 + 1].axis('off')

    row = 0
    if has_header:
        _cell(0, 0, raw_power, psd_raw, 'Raw', 'black')
        _cell(0, 1, recon_power, psd_recon, 'Full Recon', 'black')
        _blank_rest(0, 2)
        row = 1

    raw_row, recon_row = row, row + 1
    for pi in range(P):
        _cell(raw_row, pi, grid.raw_topo[pi], grid.raw_psd[pi], f'P{patch_ids[pi]} Raw', 'black')
    _blank_rest(raw_row, P)

    for pi in range(P):
        _cell(recon_row, pi, grid.recon_topo[pi], grid.recon_psd[pi], f'P{patch_ids[pi]} Full Recon', 'black')
    _blank_rest(recon_row, P)

    # Each patch sorts its own K slots by h (descending) independently — row k is "that
    # patch's k-th strongest slot", not a fixed stamp-selection-order slot index (slot 0
    # is routed top-1 vs slot K-1 always shared, say, would misalign across patches once
    # sorted by score anyway).
    order = np.argsort(-h, axis=1)  # [P, K]
    for k in range(K):
        krow = recon_row + 1 + k
        for pi in range(P):
            ki = order[pi, k]
            sid = int(stamp_ids[pi, ki])
            if sid < 0:
                # Pad sentinel (see viz.extract.extract_flat_stamp_psd_by_patch): this
                # patch's real union of stamps was smaller than k_display, no k-th slot
                # here — blank rather than plot a fake "-1" stamp at a degenerate
                # all-zero (vmin==vmax) scale.
                axes[krow, pi * 2].axis('off')
                axes[krow, pi * 2 + 1].axis('off')
                continue
            color = shared_color if n_routed is not None and sid >= n_routed else 'black'
            label = f'P{patch_ids[pi]} {unit_label[0]}{sid} (h={h[pi, ki]:.2f})'
            _cell(krow, pi, topo[pi, ki], psd[pi, ki], label, color, signed=signed_stamps)
        _blank_rest(krow, P)

    if onset_col is not None and 0 < onset_col < P:
        # Columns are a uniform subplot grid, not a shared time axis (each patch gets an
        # equal-width slot regardless of real elapsed time), so the boundary is a fixed
        # vertical line at the seam between column onset_col-1 and onset_col, not a
        # positioned one -- drawn in figure coordinates (constrained_layout has to
        # resolve final axes positions first, hence the draw() call) so it spans every
        # row cleanly, including rows where the boundary column itself is blanked (a
        # pad-sentinel patch, see the sid<0 branch above) and would otherwise carry no
        # visible marker at all.
        fig.canvas.draw()
        left_ax  = axes[0, (onset_col - 1) * 2]
        right_ax = axes[0, onset_col * 2]
        x_fig = (left_ax.get_position().x1 + right_ax.get_position().x0) / 2
        line = plt.Line2D([x_fig, x_fig], [0.02, 0.98], transform=fig.transFigure,
                           color='black', linewidth=2.5, linestyle='--')
        fig.add_artist(line)
        fig.text(x_fig, 0.995, 'event', ha='center', va='top', fontsize=9, fontweight='bold')

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Iz -> Fp1) — log1p scale', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run(ctx):
    bundle = ctx.bundle
    config = ctx.config
    model = bundle.psd_model
    viz_dir = os.path.join(ctx.output_dir, 'recon')
    os.makedirs(viz_dir, exist_ok=True)

    pos2d = project_coords_2d(bundle.coords)
    epoch_tag = (f'_ep{bundle.epoch:04d}' if bundle.epoch is not None else '') + bundle.filename_tag
    tagged_epoch_tag = f'{epoch_tag}{bundle.title_suffix}'

    pp = config.get('preprocess_params', {})
    fs = pp.get('sample_freq')
    bandpass = pp.get('bandpass_filter', {})
    l_freq, h_freq = bandpass.get('l_freq'), bandpass.get('h_freq')
    viz_cfg = config.get('training_params', {}).get('visualize_params', {})
    fft_resolution = viz_cfg.get('fft_resolution', 0.2)
    psd_range = viz_cfg.get('psd_freq_range')
    psd_l_freq, psd_h_freq = tuple(psd_range) if psd_range else (l_freq, h_freq)

    grid = extract_flat_stamp_psd_by_patch(
        model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
        fs=fs, freq_resolution=fft_resolution, patch_stride=5)

    if psd_l_freq is not None and psd_h_freq is not None:
        band = (grid.freqs >= psd_l_freq) & (grid.freqs <= psd_h_freq)
        grid.freqs = grid.freqs[band]
        grid.psd = grid.psd[:, :, :, band]
        grid.recon_psd = grid.recon_psd[:, :, band]
        grid.raw_psd = grid.raw_psd[:, :, band]

    # Real-trial event marker: convert the onset from seconds to a displayed-COLUMN
    # index. grid.patch_ids holds the raw patch-n each displayed column represents;
    # patch n's own start time is n * model.patch_stride / fs -- searchsorted finds the
    # first displayed column at or after the onset. None (an assembled continuous
    # window has no single event) draws nothing, see plot_stamp_by_patch's onset_col doc.
    onset_col = None
    if bundle.event_onset_sec is not None and fs:
        onset_patch_n = bundle.event_onset_sec * fs / model.patch_stride
        onset_col = int(np.searchsorted(grid.patch_ids, onset_patch_n))

    out_path = os.path.join(
        viz_dir, f"sub{bundle.subject_id}_trial{bundle.trial_idx}{epoch_tag}_stamp_by_patch.png")
    plot_stamp_by_patch(
        out_path, pos2d, grid, cmap=ctx.cmap,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', n_routed=model.n_routed_stamps,
        signed_stamps=True,  # grid.topo is signed amp (mixing columns)
        onset_col=onset_col,
    )
    print(f"  [panel] -> {out_path}")
