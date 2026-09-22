"""
Plotting functions for the stamp_by_patch/stamp_gallery panels (tools/panels/) and the
event_stamp_dynamics codebook diagnostic (model/MeSAE/plugin.py's MeSAECodebookChecker).
Pure rendering only -- calculation lives in tools/analysis/ and tools/viz/extract.py.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from tools.viz.topomap import draw_topomap, build_triangulation


def _log_pow(x):
    """log1p of a nonnegative power/PSD quantity (topo=L2 norm, psd=real^2+imag^2 — both
    always >=0, np.maximum guards float rounding noise below 0). log1p(0)=0 exactly, so
    the zero-fill "unused" floor (see plot_stamp_by_patch/_cell's vmin=0 anchoring)
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
    BaseEpochChecker._lookup_event_onset / MeSAEChecker._render_topo_psd for how this is
    derived from grid.patch_ids and the trial's own event_onset_sec). None (default, most
    calls — an assembled continuous window has no single event) draws nothing. Columns
    are laid out one-per-sampled-patch, evenly spaced regardless of real elapsed time
    (this is a subplot grid, not a shared time axis), so the boundary is drawn as a bold
    left border on every row's onset_col column rather than a positioned vertical line.
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


def plot_stamp_gallery(out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
                        psd_ch_x, psd_x, freqs, importance, cmap='YlOrRd',
                        phase_ch_x=None, waveforms=None,
                        subject_id=None, trial_idx=None, epoch_tag='',
                        unit_label='Stamp', unit_colors=None, unit_ids=None, n_routed=None,
                        shared_color='crimson', n_per_row=5, iclabel_probs=None):
    """
    Standalone whole-trial panel — split out of plot_stamp_by_patch's old header row
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
    plot_stamp_by_patch.

    waveforms: optional list of Q 1-D arrays, all SAME length (the real trial length —
    see viz.extract.extract_flat_stamp_gallery): real decoded content at ONE pinned
    channel, placed at each fired patch's true position, NaN where the stamp never
    fired. Rendered as an extra full-block-width row right above the ICLabel row (or
    the last row if iclabel_probs is None) — NOT the same signal ICLabel's call was
    computed from (that one is a separate gap-free concatenation, see the function's
    docstring) but drawn from the same firings, so the two still read together: "here's
    where/how it actually appeared, here's what ICLabel decided it is."

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
        # plot_stamp_by_patch's stamp cells). Raw/Recon header stays unsigned power.
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
        # Real decoded time-domain content at ONE pinned channel (the channel with the
        # most total energy across this stamp's firings, see
        # viz.extract.extract_flat_stamp_gallery), placed at each fired patch's true
        # position in the trial — NOT this stamp's ICLabel row (that reads a separate
        # gap-free concatenation; see that function's docstring for why the two differ).
        # NaN gaps (never-fired stretches) break the plotted line automatically.
        topo_col = block_col * 2
        ax = fig.add_subplot(gs[row, topo_col:topo_col + 2])
        if sig is None or not np.isfinite(sig).any():
            ax.axis('off')
            ax.set_title('Waveform: n/a (never fired)', fontsize=7, color='gray')
            return
        ax.plot(np.arange(len(sig)), sig, color=color, linewidth=0.6)
        ax.axhline(0, color='gray', lw=0.4)
        ax.set_xlim(0, len(sig) - 1)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=5)
        ax.set_title(f'{label} Waveform (real content, true trial position, gaps = never fired)',
                     fontsize=7, fontweight='bold', color=color)

    def _iclabel_cell(row, block_col, probs_q):
        # 7-class ICLabel distribution bar (see viz/iclabel.py, incl. its caveat) —
        # best class named in the title, its bar highlighted. Spans the block's full
        # width (both the topo and psd/phase sub-columns).
        from tools.viz.iclabel import ICLABEL_CLASSES
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
