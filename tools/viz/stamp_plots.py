"""
Plotting for the event_stamp_dynamics codebook diagnostic
(model/MeSAE/plugin.py's MeSAECodebookChecker). Pure rendering only -- calculation lives
in tools/analysis/ and tools/viz/extract.py.

stamp_by_patch/stamp_gallery used to live here too; each had exactly one caller (its
matching tools/panels/panel_*.py file), so both moved into their panel file directly --
see tools/panels/panel_stamp_by_patch.py, panel_stamp_gallery.py.
"""

import numpy as np
import matplotlib.pyplot as plt


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
