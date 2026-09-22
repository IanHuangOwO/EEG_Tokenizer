"""stamp_gallery panel: whole-trial Raw/Full-Recon view plus every stamp used somewhere
in the trial, driven by tools/viz/extract.py's extract_flat_stamp_gallery. Reads
ctx.bundle and ctx.cmap -- the caller builds the bundle via
build_pretrain_bundle/build_finetune_bundle before selecting this panel.
plot_stamp_gallery used to live in tools/viz/panels.py (now tools/viz/stamp_plots.py)
but had exactly one caller (this file), so it moved here directly -- see that module's
docstring."""
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from tools.viz.extract import extract_flat_stamp_gallery
from tools.viz.topomap import draw_topomap, build_triangulation, project_coords_2d

STAGES = frozenset({'pretrain', 'finetune'})
NEEDS_CHECKPOINT = True
NEEDS_DATASET = True


def _log_pow(x):
    """log1p of a nonnegative power/PSD quantity (topo=L2 norm, psd=real^2+imag^2 — both
    always >=0, np.maximum guards float rounding noise below 0). log1p(0)=0 exactly, so
    the zero-fill "unused" floor survives the log transform unchanged, while still
    compressing the large dynamic range a handful of high-power channels/bins would
    otherwise dominate on a linear scale."""
    return np.log1p(np.maximum(x, 0.0))


def _log_signed(x):
    """Signed counterpart of _log_pow for quantities where polarity matters (a stamp's
    per-channel amp — its mixing/topomap column): sign(x)*log1p(|x|). Odd function, 0
    maps to exactly 0, magnitudes compress the same way _log_pow's do, sign survives —
    so a symmetric diverging color scale centered on 0 stays meaningful after the
    transform."""
    return np.sign(x) * np.log1p(np.abs(x))


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
    helper), sorted by accumulated importance descending, reading left-to-right then
    top-to-bottom. Each block: topo on the LEFT (spanning the block's full height), PSD
    (channel x freq, from psd_x) top-right, and — when phase_ch_x is given — a
    per-channel PHASE bar chart bottom-right, directly under the PSD and sharing its
    channel-axis orientation (see _cell) so a phase shift across channels reads paired
    against that same channel's PSD row. A single vertical bar chart at the right edge
    spans just the stamp grid (not the Raw/Recon header), one horizontal bar per stamp in
    the SAME sorted order (top = highest importance) — paired 1:1 with the gallery (by
    rank, not by grid position, since the grid wraps n_per_row-wide) so "how much did
    this stamp's real content matter" reads directly alongside "what did it actually
    look like".

    psd_ch_x: [C, Q]. psd_x: [Q, C, F]. phase_ch_x: [C, Q] radians or None (Raw/Recon
    header cells never show phase — there's no stamp/quadrature structure to a raw
    signal — and passing None here entirely skips the phase row for stamp cells too).
    importance: [Q]. unit_ids: optional [Q] real global stamp ids for row labels/color —
    falls back to row position if None. All topo/PSD cells are log1p-scaled (see
    _log_pow), same reasoning as plot_stamp_by_patch.

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
    # square (draw_topomap forces equal aspect).
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
        # topomap column) — diverging RdBu_r, symmetric limits, 0 = white, so dipole
        # polarity reads directly (same rationale as plot_stamp_by_patch's stamp cells).
        # Raw/Recon header stays unsigned power.
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
        # most total energy across this stamp's firings), placed at each fired patch's
        # true position in the trial — NOT this stamp's ICLabel row (that reads a
        # separate gap-free concatenation; see that function's docstring for why the two
        # differ). NaN gaps (never-fired stretches) break the plotted line automatically.
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

    # Whole-trial raw/recon PSD -- same FFT settings as the gallery's own per-stamp PSD
    # (freq_resolution=fft_resolution drives both), so the header row is directly
    # comparable to the stamp rows below it. n_fft must match extract_flat_stamp_
    # gallery's own n_fft (round(fs/fft_resolution)); rfft's n= transparently
    # zero-pads a short trial or truncates a long one to match.
    raw_t   = bundle.raw_t[0].numpy()
    recon_t = bundle.recon_t[0].numpy()
    T = raw_t.shape[-1]
    n_fft = int(round(fs / fft_resolution)) if fs else T

    def _demean_hann_rfft_np(x):
        x = x - x.mean(axis=-1, keepdims=True)
        win = np.hanning(x.shape[-1])
        return np.fft.rfft(x * win, n=n_fft, axis=-1)

    fft_raw   = _demean_hann_rfft_np(raw_t)
    fft_recon = _demean_hann_rfft_np(recon_t)
    psd_raw   = fft_raw.real**2   + fft_raw.imag**2
    psd_recon = fft_recon.real**2 + fft_recon.imag**2

    raw_power   = (bundle.raw_cnl   ** 2).mean(axis=(1, 2))
    recon_power = (bundle.recon_cnl ** 2).mean(axis=(1, 2))

    (used_ids, gal_importance, psd_ch_x_g, psd_x_g, gal_freqs, phase_ch_x_g,
     waveforms_g, iclabel_probs) = extract_flat_stamp_gallery(
        model, bundle.x_in, bundle.c_in, time_idx=bundle.t_in, valid_channels=bundle.vc_in,
        fs=fs, freq_resolution=fft_resolution)

    if psd_l_freq is not None and psd_h_freq is not None:
        band = (gal_freqs >= psd_l_freq) & (gal_freqs <= psd_h_freq)
        gal_freqs = gal_freqs[band]
        psd_x_g = psd_x_g[:, :, band]
        psd_raw, psd_recon = psd_raw[:, band], psd_recon[:, band]

    out_path = os.path.join(
        viz_dir, f"sub{bundle.subject_id}_trial{bundle.trial_idx}{epoch_tag}_stamp_gallery.png")
    plot_stamp_gallery(
        out_path, pos2d, raw_power, recon_power, psd_raw, psd_recon,
        psd_ch_x_g, psd_x_g, gal_freqs, gal_importance, cmap=ctx.cmap,
        phase_ch_x=phase_ch_x_g, waveforms=waveforms_g,
        subject_id=bundle.subject_id, trial_idx=bundle.trial_idx, epoch_tag=tagged_epoch_tag,
        unit_label='Stamp', unit_ids=used_ids, n_routed=model.n_routed_stamps,
        iclabel_probs=iclabel_probs,
    )
    print(f"  [panel] -> {out_path}")
