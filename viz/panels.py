"""
Shared epoch-snapshot panels (topo_psd_filter, attn_topo) reused by
BaseEpochChecker._render_snapshot (MeFSQ Experts or MeSAE stamps — viz/extract.py's
extract_head_*/extract_filter_* already return the shared PsdResult/SpectraResult
dataclasses). Keeps the two training-phase methods producing the exact same panel format
instead of drifting — see model/base_checker.py and docs/adr/0004-model-plugin-base-classes.md.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from viz.topomap import draw_topomap, build_triangulation


def plot_topo_psd_filter(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                          psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                          subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                          l_freq=None, h_freq=None, unit_colors=None):
    """
    Raw / Full-recon / per-unit (Expert or Filter) topo + PSD, side by side, one row each,
    sorted by `importance` below the first two fixed rows (Raw, Full Recon). Raw/recon PSD
    must already be computed with the same n_fft/freq axis as psd_x so every row is directly
    comparable, not just visually similar (see BaseEpochChecker.check_pretrain for the FFT call).

    psd_ch_x: [C, Q] per-unit per-channel decoded activation. psd_x: [Q, C, F] per-unit PSD.
    freqs/psd_x/psd_raw/psd_recon are expected already band-cropped by the caller if desired
    (l_freq/h_freq here are only used for the x-axis label/ticks, not re-cropping).
    unit_colors: optional [Q] list of title colors (e.g. MeSAE routed-gating: red = shared
    Filter, orange = routed Filter selected for this trial) — None means no color override.
    """
    Q = psd_ch_x.shape[1]
    sorted_ord = np.argsort(importance)[::-1]
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    entries = [('Raw', raw_power, psd_raw, 'black'), ('Full Recon', recon_power, psd_recon, 'black')]
    entries += [(f'{unit_label[0]}{q} ({importance[q]:.3f})', psd_ch_x[:, q], psd_x[q],
                 unit_colors[q] if unit_colors else 'black') for q in sorted_ord]
    n_items = len(entries)
    # (topo, psd) pairs per row — was fixed at 2 (4 total columns), bumped to 3 to cut
    # canvas height (and therefore savefig render/encode time) by ~1/3 for large Q.
    n_col_pairs = 3
    n_rows = math.ceil(n_items / n_col_pairs)

    fig, axes = plt.subplots(n_rows, n_col_pairs * 2, figsize=(10 * n_col_pairs, 3.0 * n_rows),
                              squeeze=False, constrained_layout=True)
    fig.suptitle(f"Raw / Recon / {unit_label} Topo + PSD ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    # Every row's topomap shares the same electrode layout (pos2d) — triangulate once
    # instead of once per row (n_items rows, e.g. 66 for 64 Filters + Raw/Recon).
    triang = build_triangulation(pos2d)

    def _psd_row(ax_topo, ax_psd, power, psd_cf, label, color):
        im_t = draw_topomap(ax_topo, pos2d, power, cmap=cmap, vmin=power.min(), vmax=power.max(), triang=triang)
        ax_topo.set_title(f'{label} Topo (power)', fontsize=9, fontweight='bold', color=color)
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd.imshow(psd_cf[::-1], aspect='auto', cmap=cmap, origin='lower',
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD (channel x freq)', fontsize=9, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=7)
        ax_psd.set_yticks([])

    for i, (label, power, psd_cf, color) in enumerate(entries):
        row, col_pair = divmod(i, n_col_pairs)
        _psd_row(axes[row, col_pair * 2], axes[row, col_pair * 2 + 1], power, psd_cf, label, color)
    for i in range(n_items, n_rows * n_col_pairs):
        row, col_pair = divmod(i, n_col_pairs)
        axes[row, col_pair * 2].axis('off')
        axes[row, col_pair * 2 + 1].axis('off')

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Iz -> Fp1)', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _log_pow(x):
    """log1p of a nonnegative power/PSD quantity (topo=L2 norm, psd=real^2+imag^2 — both
    always >=0, np.maximum guards float rounding noise below 0). log1p(0)=0 exactly, so
    the zero-fill "unused" floor (see plot_topo_psd_by_patch/_cell's vmin=0 anchoring)
    survives the log transform unchanged, while still compressing the large dynamic range
    a handful of high-power channels/bins would otherwise dominate on a linear scale."""
    return np.log1p(np.maximum(x, 0.0))


def _log_signed(x):
    """Signed counterpart of _log_pow for quantities where polarity matters (a stamp's
    per-channel amp — its mixing/topomap column): sign(x)*log1p(|x|). Odd function, 0
    maps to exactly 0, magnitudes compress the same way _log_pow's do, sign survives —
    so a symmetric diverging color scale centered on 0 stays meaningful after the
    transform."""
    return np.sign(x) * np.log1p(np.abs(x))


def plot_topo_psd_by_patch(out_path, pos2d, grid, cmap='YlOrRd', subject_id=None,
                            trial_idx=None, epoch_tag='', unit_label='Stamp',
                            n_routed=None, shared_color='crimson',
                            raw_power=None, recon_power=None, psd_raw=None, psd_recon=None,
                            signed_stamps=False):
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

    fig.text(0.5, 0.005, 'PSD y-axis: Channel (Iz -> Fp1) — log1p scale', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_stamp_gallery(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                        psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                        phase_ch_x=None, waveforms=None,
                        subject_id=None, trial_idx=None, epoch_tag='',
                        unit_label='Stamp', unit_colors=None, unit_ids=None, n_routed=None,
                        shared_color='crimson', n_per_row=5, iclabel_probs=None):
    """
    Standalone whole-trial panel — split out of plot_topo_psd_by_patch's old header row
    (see that function's docstring) so the trial-wide Raw/Full-Recon view and every stamp
    actually used SOMEWHERE in this trial (not per-patch) get their own dedicated figure,
    same layout family as plot_stamp_panel but without an attn column (flat-token
    StampBank has no cross-channel pool to read a channel-attention map from — see
    MeSAE_modules.StampBank class docstring).

    Header row: Raw and Full-Recon (whole trial) topo + PSD, one block each. Below: a
    grid of `n_per_row` blocks per row, one block per USED stamp (see
    viz.extract.extract_flat_stamp_gallery's trial-wide dedup, same _used_flat_stamps
    helper extract_flat_stamp_psd uses), sorted by accumulated importance descending,
    reading left-to-right then top-to-bottom. Each block: topo on the LEFT (spanning
    the block's full height), PSD (channel x freq, from psd_x) top-right, and — when
    phase_ch_x is given — a per-channel PHASE bar chart bottom-right, directly under
    the PSD and sharing its channel-axis orientation (see _cell) so a phase shift
    across channels reads paired against that same channel's PSD row. A single
    vertical bar chart at the right edge spans just the stamp grid (not the Raw/Recon
    header), one horizontal bar per stamp in the SAME sorted order (top = highest
    importance) — paired 1:1 with the gallery (by rank, not by grid position, since
    the grid wraps n_per_row-wide) so "how much did this stamp's real content matter"
    reads directly alongside "what did it actually look like".

    psd_ch_x: [C, Q]. psd_x: [Q, C, F]. phase_ch_x: [C, Q] radians or None (Raw/Recon
    header cells never show phase — there's no stamp/quadrature structure to a raw
    signal — and passing None here entirely skips the phase row for stamp cells too).
    importance: [Q]. unit_ids: optional [Q] real global stamp ids for row labels/color
    (see plot_stamp_panel's unit_ids doc) — falls back to row position if None. All
    topo/PSD cells are log1p-scaled (see _log_pow), same reasoning as
    plot_topo_psd_by_patch.

    waveforms: optional list of Q 1-D arrays (VARIABLE length per stamp — see
    viz.extract.extract_flat_stamp_gallery, real decoded content at that stamp's own
    strongest channel, concatenated over only the patches it fired on), rendered as an
    extra full-block-width row right above the ICLabel row (or the last row if
    iclabel_probs is None) — the actual time-domain signal ICLabel's call was computed
    from, placed so the two read as cause and effect: "here's the real waveform, here's
    what ICLabel decided it is."

    iclabel_probs: optional [Q, 7] ICLabel class distribution per stamp
    (viz.iclabel.ICLABEL_CLASSES order) — when given, each stamp block grows a row
    (spanning the full block width) under its PSD/phase pair (and under the waveform
    row, if that's also given): a small bar chart of the 7 class probabilities, best
    class named in the bar row's title (see viz/iclabel.py, including the caveat that
    these are interpretability hints, not calibrated probabilities). Both None keeps
    the old two-row layout.
    """
    Q = psd_ch_x.shape[1]
    display_ids = np.arange(Q) if unit_ids is None else np.asarray(unit_ids)
    order = np.argsort(-importance)
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    triang = build_triangulation(pos2d)

    # Each block is 2 grid-columns wide (topo | psd-over-phase) instead of 1 — topo
    # spans both of the block's rows on the left, PSD/phase stack on the right, so
    # topo_w/right_w need to roughly sum-match (psd_h+phase_h) for topo to land near
    # square (draw_topomap forces equal aspect, same col_w-matching rationale as
    # plot_stamp_panel).
    topo_w, right_w, bar_w = 2.6, 2.6, 3.5
    psd_h, phase_h = 1.4, 1.3
    wave_h = 0.8   # waveform row height (only present when waveforms given)
    icl_h = 0.9    # ICLabel bar row height (only present when iclabel_probs given)
    has_wave = waveforms is not None
    has_icl = iclabel_probs is not None
    rows_per_block = 2 + (1 if has_wave else 0) + (1 if has_icl else 0)
    block_h = psd_h + phase_h + (wave_h if has_wave else 0) + (icl_h if has_icl else 0)
    n_per_row = max(2, n_per_row)  # header needs 2 blocks (Raw, Full Recon) side by side
    n_stamp_rows = math.ceil(Q / n_per_row) if Q else 0
    n_cols = n_per_row * 2
    header_rows = 2  # psd-height row, phase-height row (topo spans both)
    total_rows = header_rows + n_stamp_rows * rows_per_block
    suptitle_in, margin_in = 1.6, 0.15
    fig_w = (topo_w + right_w) * n_per_row + bar_w
    fig_h = (psd_h + phase_h) + block_h * n_stamp_rows + suptitle_in + margin_in
    block_ratios = ([psd_h, phase_h] + ([wave_h] if has_wave else [])
                     + ([icl_h] if has_icl else []))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(total_rows, n_cols + 1,
                           height_ratios=[psd_h, phase_h] + block_ratios * n_stamp_rows,
                           width_ratios=[topo_w, right_w] * n_per_row + [bar_w],
                           left=0.02, right=0.98, top=1 - suptitle_in / fig_h, bottom=margin_in / fig_h,
                           hspace=0.6, wspace=0.35)
    fig.suptitle(f"Whole-Trial Raw / Recon / Used-{unit_label} Gallery "
                 f"({unit_label}s sorted by accumulated importance) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=13, fontweight='bold')

    def _cell(topo_row, block_col, power, psd_cf, phase_c, label, color, signed=False):
        # signed: stamp topos are the SIGNED trial-mean per-channel amp (the mixing/
        # topomap column, see extract_flat_stamp_gallery) — diverging RdBu_r, symmetric
        # limits, 0 = white, so dipole polarity reads directly (same rationale as
        # plot_topo_psd_by_patch's stamp cells). Raw/Recon header stays unsigned power.
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        psd_cf = _log_pow(psd_cf)
        ax_topo = fig.add_subplot(gs[topo_row:topo_row + 2, topo_col])
        if signed:
            v = _log_signed(power)
            vlim = max(np.abs(v).max(), 1e-12)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap='RdBu_r',
                                 vmin=-vlim, vmax=vlim, triang=triang)
        else:
            v = _log_pow(power)
            im_t = draw_topomap(ax_topo, pos2d, v, cmap=cmap,
                                 vmin=0.0, vmax=max(v.max(), 1e-12), triang=triang)
        ax_topo.set_title(f'{label}\nTopo', fontsize=8, fontweight='bold', color=color)
        fig.colorbar(im_t, ax=ax_topo, fraction=0.05, pad=0.02)

        ax_psd = fig.add_subplot(gs[topo_row, right_col])
        ax_psd.imshow(psd_cf[::-1], aspect='auto', cmap=cmap, origin='lower',
                      vmin=0.0, vmax=max(psd_cf.max(), 1e-12),
                      extent=[freqs[0], freqs[-1], 0, psd_cf.shape[0]])
        ax_psd.set_title(f'{label} PSD', fontsize=8, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=6)
        ax_psd.set_yticks([])

        ax_phase = fig.add_subplot(gs[topo_row + 1, right_col])
        if phase_c is None:
            ax_phase.axis('off')
        else:
            # Cyclic colormap (twilight) so a bar's color itself carries the phase,
            # not just its length/direction. invert_yaxis matches the PSD imshow
            # above it (channel 0 at TOP there too — imshow flips rows + origin=
            # 'lower', see ax_psd), so a given row means the same channel in both
            # panels — pairs 1:1, letting a phase shift across channels be read
            # directly against that channel's PSD row.
            Cc = len(phase_c)
            bar_colors = plt.get_cmap('twilight')((np.asarray(phase_c) + np.pi) / (2 * np.pi))
            ax_phase.barh(range(Cc), phase_c, color=bar_colors)
            ax_phase.axvline(0, color='gray', lw=0.6)
            ax_phase.set_xlim(-np.pi, np.pi)
            ax_phase.invert_yaxis()
            ax_phase.set_yticks([])
            ax_phase.set_xticks([-np.pi, 0, np.pi])
            ax_phase.set_xticklabels(['-π', '0', 'π'], fontsize=6)
            ax_phase.set_xlabel('Phase (rad)', fontsize=6)
            ax_phase.set_title(f'{label} Phase', fontsize=8, fontweight='bold', color=color)

    _cell(0, 0, raw_power, psd_raw, None, 'Raw', 'black')
    _cell(0, 1, recon_power, psd_recon, None, 'Full Recon', 'black')
    for block_col in range(2, n_per_row):
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        fig.add_subplot(gs[0:2, topo_col]).axis('off')
        fig.add_subplot(gs[0, right_col]).axis('off')
        fig.add_subplot(gs[1, right_col]).axis('off')

    def _stamp_color(q):
        if unit_colors:
            return unit_colors[q]
        return shared_color if n_routed is not None and int(display_ids[q]) >= n_routed else 'black'

    def _waveform_cell(row, block_col, sig, color, label):
        # Real decoded time-domain content at this stamp's own strongest channel,
        # concatenated over only the patches it fired on (see
        # viz.extract.extract_flat_stamp_gallery) — the same signal ICLabel's row right
        # below is computed from. Spans the block's full width, same as the ICLabel row.
        topo_col = block_col * 2
        ax = fig.add_subplot(gs[row, topo_col:topo_col + 2])
        if sig is None or len(sig) == 0:
            ax.axis('off')
            ax.set_title('Waveform: n/a (never fired)', fontsize=7, color='gray')
            return
        ax.plot(np.arange(len(sig)), sig, color=color, linewidth=0.6)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xlim(0, len(sig) - 1)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=5)
        ax.set_title(f'{label} Waveform (real content, concatenated over firing patches)',
                     fontsize=7, fontweight='bold', color=color)

    def _iclabel_cell(row, block_col, probs_q):
        # 7-class ICLabel distribution bar (see viz/iclabel.py, incl. its caveat) —
        # best class named in the title, its bar highlighted. Spans the block's full
        # width (both the topo and psd/phase sub-columns).
        from viz.iclabel import ICLABEL_CLASSES
        topo_col = block_col * 2
        ax = fig.add_subplot(gs[row, topo_col:topo_col + 2])
        if probs_q is None or not np.all(np.isfinite(probs_q)):
            # a stamp selected on very few patches has a near-all-zero stitched
            # activity — autocorr/psd features degenerate to NaN there; label it
            # honestly instead of argmaxing garbage.
            ax.axis('off')
            ax.set_title('ICLabel: n/a (too sparse)', fontsize=7, color='gray')
            return
        best = int(np.argmax(probs_q))
        colors = ['dimgray'] * len(ICLABEL_CLASSES)
        colors[best] = 'seagreen' if ICLABEL_CLASSES[best] == 'Brain' else 'darkorange'
        ax.bar(range(len(ICLABEL_CLASSES)), probs_q, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(ICLABEL_CLASSES)))
        ax.set_xticklabels(ICLABEL_CLASSES, fontsize=5, rotation=45)
        ax.set_yticks([0, 1])
        ax.tick_params(axis='y', labelsize=5)
        ax.set_title(f'ICLabel: {ICLABEL_CLASSES[best]} {probs_q[best]:.2f}',
                     fontsize=7, fontweight='bold')

    for i, q in enumerate(order):
        block_row, block_col = divmod(i, n_per_row)
        topo_row = header_rows + block_row * rows_per_block
        sid = int(display_ids[q])
        color = _stamp_color(q)
        label = f'{unit_label} {sid} ({importance[q]:.3f})'
        phase_c = phase_ch_x[:, q] if phase_ch_x is not None else None
        _cell(topo_row, block_col, psd_ch_x[:, q], psd_x[q], phase_c, label, color, signed=True)
        next_row = topo_row + 2
        if has_wave:
            _waveform_cell(next_row, block_col, waveforms[q], color, label)
            next_row += 1
        if has_icl:
            _iclabel_cell(next_row, block_col, iclabel_probs[q])

    for i in range(Q, n_stamp_rows * n_per_row):
        block_row, block_col = divmod(i, n_per_row)
        topo_row = header_rows + block_row * rows_per_block
        topo_col, right_col = block_col * 2, block_col * 2 + 1
        fig.add_subplot(gs[topo_row:topo_row + 2, topo_col]).axis('off')
        fig.add_subplot(gs[topo_row, right_col]).axis('off')
        fig.add_subplot(gs[topo_row + 1, right_col]).axis('off')
        next_row = topo_row + 2
        if has_wave:
            fig.add_subplot(gs[next_row, topo_col:topo_col + 2]).axis('off')
            next_row += 1
        if has_icl:
            fig.add_subplot(gs[next_row, topo_col:topo_col + 2]).axis('off')

    if Q > 0:
        ax_bar = fig.add_subplot(gs[header_rows:, n_cols])
        bar_labels = [f'{unit_label[0]}{int(display_ids[q])}' for q in order]
        # Bars default steelblue (routed) / shared_color (shared) — same n_routed split as
        # the grid titles above, so a shared stamp reads the same way in both places
        # instead of only being distinguishable in the topo/PSD grid.
        bar_colors = [(shared_color if (n_routed is not None and int(display_ids[q]) >= n_routed)
                       else 'steelblue') for q in order]
        bars = ax_bar.barh(range(Q), importance[order], color=bar_colors)
        ax_bar.set_yticks(range(Q))
        ax_bar.set_yticklabels(bar_labels, fontsize=6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Importance', fontsize=9)
        ax_bar.set_title(f'{unit_label} Importance (crimson = shared)', fontsize=10, fontweight='bold')
        for tick, q in zip(ax_bar.get_yticklabels(), order):
            tick.set_color(_stamp_color(q))

    fig.text(0.5, 0.002, 'PSD y-axis: Channel (Iz -> Fp1) — log1p scale', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_stamp_panel(out_path, pos2d, psd_ch_x, psd_x, freqs, attn, importance,
                      cmap='YlOrRd', subject_id=None, trial_idx=None, epoch_tag='',
                      unit_label='Filter', valid_channels=None, unit_colors=None,
                      unit_ids=None, n_per_row=6):
    """
    One block per unit (Expert/Filter/Stamp), sorted by `importance`: recon topo + attn
    topo side by side on top, that unit's own PSD (channel x freq) spanning both columns
    below. `n_per_row` blocks per grid row.

    psd_ch_x: [C, Q] per-unit decoded channel power (recon topo). psd_x: [Q, C, F]
    per-unit PSD. attn: [Q, C] per-unit channel attention (attn topo) — same convention as
    plot_attn_topo. valid_channels: [C] bool or None, hides padded channels on the topos.
    unit_ids: optional [Q] real global unit ids for the title, when attn/psd_ch_x/psd_x's
    row order is a caller-side compacted subset (see plot_attn_topo's unit_ids doc).
    """
    Q = psd_ch_x.shape[1]
    display_ids = np.arange(Q) if unit_ids is None else np.asarray(unit_ids)
    sorted_ord = np.argsort(importance)[::-1]
    freq_label = 'Hz' if freqs is not None and len(freqs) else 'cyc/patch'
    freq_ticks = np.linspace(freqs[0], freqs[-1], 5)

    valid = valid_channels if valid_channels is not None else np.ones(pos2d.shape[0], dtype=bool)
    pos2d_v = pos2d[valid]
    triang = build_triangulation(pos2d_v)

    n_rows = math.ceil(Q / n_per_row)
    # Topo cells use ax.set_aspect('equal') (draw_topomap) — under constrained_layout a
    # cell whose height_ratio doesn't roughly match its allotted width leaves the rest of
    # its row blank to keep that square aspect, which is where the big white gaps came
    # from. col_w keeps each topo cell ~square (topo row height == 1 col-width unit) so
    # there's nothing left over to pad with; psd_h stays a flatter rectangle.
    col_w, psd_h, bar_w = 2.6, 1.7, 3.5
    n_cols = n_per_row * 2
    suptitle_in, margin_in = 1.0, 0.15  # fixed inches, not a fraction — stays small on tall figures
    fig_w = col_w * n_cols + bar_w
    fig_h = (col_w + psd_h) * n_rows + suptitle_in + margin_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows * 2, n_cols + 1,
                           height_ratios=[col_w, psd_h] * n_rows,
                           width_ratios=[col_w] * n_cols + [bar_w],
                           left=0.02, right=0.98, top=1 - suptitle_in / fig_h, bottom=margin_in / fig_h,
                           hspace=0.6, wspace=0.35)
    fig.suptitle(f"Per-{unit_label} Recon Topo / Attn Topo / PSD ({unit_label}s sorted by "
                 f"contribution) — Sub {subject_id}, Trial {trial_idx}{epoch_tag}",
                 fontsize=13, fontweight='bold')

    for i, q in enumerate(sorted_ord):
        block_row, block_col = divmod(i, n_per_row)
        topo_row, psd_row = block_row * 2, block_row * 2 + 1
        col_recon, col_attn = block_col * 2, block_col * 2 + 1
        color = unit_colors[q] if unit_colors else 'black'
        label = f'{unit_label} {display_ids[q]} ({importance[q]:.3f})'

        ax_recon = fig.add_subplot(gs[topo_row, col_recon])
        recon_vals = psd_ch_x[:, q][valid]
        im_r = draw_topomap(ax_recon, pos2d_v, recon_vals, cmap=cmap,
                             vmin=recon_vals.min(), vmax=recon_vals.max(), triang=triang)
        ax_recon.set_title(f'{label}\nRecon Topo', fontsize=8, fontweight='bold', color=color)
        fig.colorbar(im_r, ax=ax_recon, fraction=0.05, pad=0.02)

        ax_attn = fig.add_subplot(gs[topo_row, col_attn])
        attn_vals = attn[q][valid]
        im_a = draw_topomap(ax_attn, pos2d_v, attn_vals, cmap='viridis',
                             vmin=attn_vals.min(), vmax=attn_vals.max(), triang=triang)
        ax_attn.set_title(f'{label}\nAttn Topo', fontsize=8, fontweight='bold', color=color)
        fig.colorbar(im_a, ax=ax_attn, fraction=0.05, pad=0.02)

        ax_psd = fig.add_subplot(gs[psd_row, col_recon:col_attn + 1])
        ax_psd.imshow(psd_x[q][::-1], aspect='auto', cmap=cmap, origin='lower',
                      extent=[freqs[0], freqs[-1], 0, psd_x[q].shape[0]])
        ax_psd.set_title(f'{label} PSD (channel x freq)', fontsize=8, fontweight='bold', color=color)
        ax_psd.set_xticks(freq_ticks)
        ax_psd.set_xticklabels([f'{f:.0f}' for f in freq_ticks], fontsize=6)
        ax_psd.set_xlabel(freq_label, fontsize=7)
        ax_psd.set_yticks([])

    for i in range(Q, n_rows * n_per_row):
        block_row, block_col = divmod(i, n_per_row)
        topo_row, psd_row = block_row * 2, block_row * 2 + 1
        col_recon, col_attn = block_col * 2, block_col * 2 + 1
        fig.add_subplot(gs[topo_row, col_recon]).axis('off')
        fig.add_subplot(gs[topo_row, col_attn]).axis('off')
        fig.add_subplot(gs[psd_row, col_recon:col_attn + 1]).axis('off')

    ax_bar = fig.add_subplot(gs[:, n_cols])
    bar_labels = [f'{unit_label[0]}{display_ids[q]}' for q in sorted_ord]
    bars = ax_bar.barh(range(Q), importance[sorted_ord], color='steelblue')
    ax_bar.set_yticks(range(Q))
    ax_bar.set_yticklabels(bar_labels, fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Importance', fontsize=9)
    ax_bar.set_title(f'{unit_label} Importance', fontsize=10, fontweight='bold')
    if unit_colors:
        for tick, bar, q in zip(ax_bar.get_yticklabels(), bars, sorted_ord):
            tick.set_color(unit_colors[q])
            bar.set_color(unit_colors[q])

    fig.text(0.5, 0.002, 'PSD y-axis: Channel (Iz -> Fp1)', ha='center', fontsize=8)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_attn_topo(out_path, pos2d, attn, importance, channel_names, valid_channels=None,
                    subject_id=None, trial_idx=None, epoch_tag='', unit_label='Filter',
                    unit_colors=None, unit_ids=None, topo_attn=None, heatmap_attn=None, heatmap_ylabels=None,
                    heatmap_ylabel='Channel', heatmap_title=None, heatmap_transpose=True,
                    bar_vertical=None):
    """
    Per-unit (Expert/Filter/Head) channel topography, one topomap per unit (sorted by
    `importance`), plus one large heatmap spanning all rows.
    attn: [Q, C] — each unit's own channel attention weight (rows sum to 1). Used for both
    the topomaps and the big heatmap unless overridden below.
    unit_ids: optional [Q] real unit ids for tick-label text, when `attn`'s row order is a
    caller-side compacted/filtered subset (e.g. MeSAE's used_stamp_ids) whose row position
    no longer matches the model's own global unit numbering. None (default) labels units by
    their row position in `attn` — correct only when `attn` already covers every unit in
    its natural order (e.g. MeFSQ's fixed Expert pool, never filtered).
    topo_attn: optional [Q, C] override for the topomap values only — e.g. channel
    attention scaled by each unit's overall importance, so units are visually comparable
    (raw per-unit attention rows each sum to 1 and aren't).
    heatmap_attn: optional [Q, R] override for the big heatmap (R rows need not be C) —
    e.g. a finetune head's own Patch x Filter temporal attention instead of channel
    attention. heatmap_ylabels/heatmap_ylabel/heatmap_title describe that override; ignored
    when heatmap_attn is None.
    heatmap_transpose: True (default, channel-attn convention) puts the heatmap_attn's own
    R axis on Y and units on X. False puts units on Y and R on X — e.g. the finetune Patch
    x Filter panel wants patch on X, filter on Y.
    bar_vertical: orientation of the importance bar chart. None (default) matches
    heatmap_transpose (vertical when units are on the heatmap's X, horizontal when on Y).
    Pass explicitly to decouple — e.g. the pretrain Channel x Filter panel puts filter on Y
    (heatmap_transpose=False) but still wants a vertical bar chart (bar_vertical=True).
    valid_channels: [C] bool or None — padded/invalid channels hidden from the topomaps only.
    unit_colors: optional [Q] list of title/tick-label colors (e.g. MeSAE routed-gating:
    red = shared stamp, orange = routed stamp that actually fired for this trial) — None
    means no color override (default black).
    """
    Q, C = attn.shape
    display_ids = np.arange(Q) if unit_ids is None else np.asarray(unit_ids)
    sorted_ord = np.argsort(importance)[::-1]
    topo_src = attn if topo_attn is None else topo_attn
    attn_masked = topo_src if valid_channels is None else np.where(valid_channels[None, :], topo_src, np.nan)

    hm_src = attn if heatmap_attn is None else heatmap_attn  # [Q, R]
    hm_ylabels = channel_names if heatmap_ylabels is None else heatmap_ylabels
    hm_ylabel = 'Channel' if heatmap_attn is None else heatmap_ylabel
    hm_title = (f'Channel x {unit_label} Attention' if heatmap_title is None and heatmap_attn is None
                else heatmap_title if heatmap_title is not None else f'{unit_label} Attention')
    unit_tick_labels = [f'{unit_label[0]}{display_ids[q]}\n{importance[q]:.3f}' for q in sorted_ord]
    img = hm_src[sorted_ord].T if heatmap_transpose else hm_src[sorted_ord]

    # Topomap columns per row — was fixed at 2, bumped to 4 to roughly halve the number
    # of rows (and therefore canvas height / savefig render+encode time) for large Q.
    # Width scaled to keep each topomap's and the heatmap's on-canvas size unchanged.
    n_topo_cols = 4
    heatmap_ratio = 3.5
    bar_ratio = 1.3
    old_width, old_ratio_sum = 40.0, 2 * 1.0 + heatmap_ratio
    per_ratio_width = old_width / old_ratio_sum
    fig_width = per_ratio_width * (n_topo_cols * 1.0 + heatmap_ratio + bar_ratio)

    n_topo_rows = math.ceil(Q / n_topo_cols)
    fig = plt.figure(figsize=(fig_width, 3.0 * n_topo_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_topo_rows, n_topo_cols + 2,
                           width_ratios=[1.0] * n_topo_cols + [heatmap_ratio, bar_ratio], wspace=0.3)

    # valid_channels masking is the same for every unit (only depends on the channel, not
    # q), so the NaN pattern — and therefore pos2d[valid] — is identical across all Q
    # topomaps; triangulate once instead of once per unit.
    valid = ~np.isnan(attn_masked[sorted_ord[0]]) if Q else None
    triang = build_triangulation(pos2d[valid]) if valid is not None else None

    for i, q_orig in enumerate(sorted_ord):
        row, col = divmod(i, n_topo_cols)
        ax = fig.add_subplot(gs[row, col])
        vals = attn_masked[q_orig][valid]
        vmin_q, vmax_q = float(np.nanmin(vals)), float(np.nanmax(vals))
        topo_im = draw_topomap(ax, pos2d[valid], vals, cmap='viridis', triang=triang,
                                vmin=vmin_q, vmax=vmax_q)
        color = unit_colors[q_orig] if unit_colors else 'black'
        ax.set_title(f'{unit_label} {q_orig}  ({importance[q_orig]:.3f})', fontsize=9, fontweight='bold', color=color)
        fig.colorbar(topo_im, ax=ax, fraction=0.046, pad=0.02)
    for i in range(Q, n_topo_rows * n_topo_cols):
        row, col = divmod(i, n_topo_cols)
        fig.add_subplot(gs[row, col]).axis('off')

    ax_big = fig.add_subplot(gs[:, n_topo_cols])
    if heatmap_transpose:
        y_labels, y_axis_label = hm_ylabels, hm_ylabel
        x_labels, x_axis_label = unit_tick_labels, f'{unit_label} (sorted by contribution)'
        unit_ticklabels_getter = ax_big.get_xticklabels
    else:
        y_labels, y_axis_label = unit_tick_labels, f'{unit_label} (sorted by contribution)'
        x_labels, x_axis_label = hm_ylabels, hm_ylabel
        unit_ticklabels_getter = ax_big.get_yticklabels

    im_big = ax_big.imshow(img, aspect='auto', cmap='viridis')
    ax_big.set_yticks(range(img.shape[0]))
    ax_big.set_yticklabels(y_labels, fontsize=5)
    ax_big.set_ylabel(y_axis_label, fontsize=9)
    ax_big.set_xticks(range(img.shape[1]))
    ax_big.set_xticklabels(x_labels, fontsize=6, rotation=90)
    ax_big.set_xlabel(x_axis_label, fontsize=9)
    if unit_colors:
        for tick, q in zip(unit_ticklabels_getter(), sorted_ord):
            tick.set_color(unit_colors[q])
    ax_big.set_title(hm_title, fontsize=10, fontweight='bold')
    fig.colorbar(im_big, ax=ax_big, fraction=0.03, pad=0.02).set_label('Attention weight', fontsize=8)

    # Importance bar chart, beside the heatmap. By default oriented to align row-for-row
    # (or column-for-column) with wherever the heatmap put its unit axis, so a tall bar
    # lines up visually with that unit's row/column in the heatmap above — pass
    # bar_vertical explicitly to decouple the two (e.g. a vertical bar chart is more
    # readable for many units even when the heatmap itself has units on Y).
    ax_bar = fig.add_subplot(gs[:, n_topo_cols + 1])
    bar_values = importance[sorted_ord]
    if heatmap_transpose if bar_vertical is None else bar_vertical:
        # vertical bars, one per unit, in the same sorted order as the heatmap/topomaps
        bars = ax_bar.bar(range(Q), bar_values, color='steelblue')
        ax_bar.set_xticks(range(Q))
        ax_bar.set_xticklabels(unit_tick_labels, fontsize=6, rotation=90)
        ax_bar.set_ylabel('Importance', fontsize=9)
        ax_bar.set_xlabel(f'{unit_label} (sorted by contribution)', fontsize=9)
        bar_ticklabels = ax_bar.get_xticklabels()
    else:
        # horizontal bars, one per unit, same top-to-bottom order as imshow's default origin
        bars = ax_bar.barh(range(Q), bar_values, color='steelblue')
        ax_bar.set_yticks(range(Q))
        ax_bar.set_yticklabels(unit_tick_labels, fontsize=6)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel('Importance', fontsize=9)
        bar_ticklabels = ax_bar.get_yticklabels()
    if unit_colors:
        for tick, bar, q in zip(bar_ticklabels, bars, sorted_ord):
            tick.set_color(unit_colors[q])
            bar.set_color(unit_colors[q])
    ax_bar.set_title(f'{unit_label} Importance', fontsize=10, fontweight='bold')

    topo_label = 'Channel Attention' if topo_attn is None else 'Channel Attention (scaled by contribution)'
    fig.suptitle(f"Per-{unit_label} {topo_label} ({unit_label}s sorted by contribution) — "
                 f"Sub {subject_id}, Trial {trial_idx}{epoch_tag}", fontsize=12, fontweight='bold')
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_event_stamp_dynamics(out_path, t_sec, sel_rate, amp_mean, pow_mean, pow_std,
                               h_mean, onset_sec=None, unit_label='Stamp', title_suffix='',
                               top=40, n_movers=15):
    """Event-locked unit-selection / power trajectory for an epoch-structured dataset.

    Trials are onset-aligned (one epoch each), so a fixed time offset within the trial
    means the same thing across trials. The caller has already done a sliding-window
    evaluation (re-patchify each raw trial at several sub-stride offsets, native patch
    stride per forward pass so the encoder stays in-distribution, place each patch's
    result at its true time) and pooled over trials x offsets. This just draws it.

    t_sec:    [B] time-bin centres (seconds)
    sel_rate: [Q, B] fraction of (trial, offset, patch) observations in each bin that
              selected unit q
    amp_mean: [Q, B] mean selection strength of unit q where it was selected (0 elsewhere)
    pow_mean/pow_std/h_mean: [B] per-bin full-recon energy and mean slot amp
    onset_sec: event onset (s) -> vertical marker + pre/post split; None -> trajectory only

    Panels: (1) selection-rate heatmap for the top-`top` units, (2) power vs time,
    (3) top |post-pre| selection-rate movers, (4) per-unit pre-vs-post selection rate.
    """
    B = len(t_sec)
    Q = sel_rate.shape[0]
    onset_bin = None
    if onset_sec is not None and t_sec[0] <= onset_sec <= t_sec[-1]:
        onset_bin = int(np.searchsorted(t_sec, onset_sec))

    tot = sel_rate.sum(axis=1)
    top_ids = np.argsort(-tot)[:top]

    split = onset_bin is not None and 0 < onset_bin < B
    if split:
        fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    else:
        fig, axrow = plt.subplots(1, 2, figsize=(16, 5))
        ax = np.array([axrow, [None, None]], dtype=object)

    a = ax[0, 0]
    im = a.imshow(sel_rate[top_ids], aspect='auto', cmap='magma', vmin=0, vmax=1,
                  extent=[t_sec[0], t_sec[-1], len(top_ids), 0], interpolation='nearest')
    a.set_yticks(np.arange(len(top_ids)) + 0.5)
    a.set_yticklabels(top_ids, fontsize=6)
    a.set_title(f'{unit_label} selection rate vs time')
    a.set_xlabel('time (s)'); a.set_ylabel(f'{unit_label} id')
    fig.colorbar(im, ax=a, fraction=0.04)

    a = ax[0, 1]
    a.plot(t_sec, pow_mean, '-', color='C2', label='recon energy')
    a.fill_between(t_sec, pow_mean - pow_std, pow_mean + pow_std, alpha=0.2, color='C2')
    ab = a.twinx()
    ab.plot(t_sec, h_mean, '-', color='C0', label='mean slot amp')
    a.set_title('power vs time'); a.set_xlabel('time (s)')
    a.set_ylabel('recon energy', color='C2'); ab.set_ylabel('mean slot amp', color='C0')

    if split:
        a2, a3 = ax[1, 0], ax[1, 1]
        pre = np.arange(B) < onset_bin
        ever = tot > 0
        d_rate = sel_rate[:, ~pre].mean(1) - sel_rate[:, pre].mean(1)
        order = [s for s in np.argsort(-np.abs(d_rate)) if ever[s]][:n_movers]

        d = d_rate[order]
        a2.barh(range(len(order)), d, color=['C3' if v > 0 else 'C0' for v in d])
        a2.set_yticks(range(len(order))); a2.set_yticklabels(order, fontsize=7)
        a2.invert_yaxis(); a2.axvline(0, color='k', lw=0.8)
        a2.set_title(f'top |post-pre| selection-rate movers (onset {onset_sec:.2f}s)')
        a2.set_xlabel('post_rate - pre_rate')

        pr, po = sel_rate[:, pre].mean(1), sel_rate[:, ~pre].mean(1)
        a3.scatter(pr, po, s=8, alpha=0.4)
        for s in order:
            a3.annotate(str(s), (pr[s], po[s]), fontsize=6)
        lim = max(pr.max(), po.max(), 1e-3) * 1.05
        a3.plot([0, lim], [0, lim], 'k--', lw=0.8)
        a3.set_xlim(0, lim); a3.set_ylim(0, lim)
        a3.set_title(f'per-{unit_label} selection rate: pre vs post')
        a3.set_xlabel('pre'); a3.set_ylabel('post')
        for aa, cvline in ((ax[0, 0], 'w'), (ax[0, 1], 'k')):
            aa.axvline(onset_sec, color=cvline, ls='--', lw=1.5)

    suffix = title_suffix if split else f'{title_suffix}  [no event onset configured]'
    fig.suptitle(f'Event-locked {unit_label} dynamics{suffix}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
