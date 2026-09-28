"""
Per-stamp feature extraction for the snapshot and codebook panels: each stamp's decoded content,
signed topography, phase, PSD and whole-trial waveform on one trial. Model-coupled (runs the frozen
backbone through MeSAEPretrain.encode_stamps), unlike viz/topomap.py. Every stamp is active at every
patch (static dictionary, docs/adr/0022), so nothing here tracks selection.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PatchGridResult:
    """Per-patch stamp content, no cross-patch averaging (see extract_stamp_psd_by_patch).
    patch_ids: [P] sampled patch indices. S = n_stamps; column s is stamp s.
    topo: [P, S, C] signed per-channel amp (the mixing column) of stamp s at that patch.
    psd: [P, S, C, F] per-channel power spectrum of stamp s's decoded content at that patch.
    h: [P, S] stamp strength at that patch.
    recon_topo / recon_psd: [P, C] / [P, C, F] norm and spectrum of the patch's full
      reconstruction (computed from the summed signal: norm and FFT are not linear).
    raw_topo / raw_psd: [P, C] / [P, C, F] the same for the raw input patch.
    freqs: [F].
    """
    patch_ids: np.ndarray
    topo: np.ndarray
    psd: np.ndarray
    h: np.ndarray
    recon_topo: np.ndarray
    recon_psd: np.ndarray
    raw_topo: np.ndarray
    raw_psd: np.ndarray
    freqs: np.ndarray


def _demean_hann_rfft(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Demean + Hann-taper x along its last dim, THEN zero-pad to n_fft and rfft.
    Every PSD in this file is built from a single short patch_len window, far too short to resolve
    real Delta/Theta content, and a bare rfft on an un-tapered snippet is an implicit rectangular
    window whose mainlobe (~2*fs/L Hz wide) smears any DC offset or patch-boundary discontinuity
    across 0-20 Hz. Demeaning kills the DC spike; the Hann taper suppresses the sidelobes.
    Zero-padding (n_fft > L) only interpolates the spectrum for display -- no real resolution below
    ~1/(L/fs) Hz."""
    x = x - x.mean(dim=-1, keepdim=True)
    win = torch.hann_window(x.shape[-1], periodic=False, device=x.device, dtype=x.dtype)
    return torch.fft.rfft(x * win, n=n_fft, dim=-1)


def _n_fft(patch_len, fs, freq_resolution):
    return max(patch_len, int(round(fs / freq_resolution))) if fs and freq_resolution else patch_len


def _signed_topo(ab):
    """ab [..., C, 2] (a, b) pairs -> [..., C] signed scalar: each channel's pair projected onto the
    channel-mean phase direction, A_c * cos(phi_c - phi_ref). Zero-lag sources keep their magnitude
    with the dipole's sign structure; out-of-phase (travelling-wave) components drop out."""
    ref = ab.mean(dim=-2, keepdim=True)
    ref = ref / (ref.norm(dim=-1, keepdim=True) + 1e-8)
    return (ab * ref).sum(dim=-1)


@torch.no_grad()
def _stamp_summary(model, x, coords, time_idx=None, valid_channels=None):
    """One trial (B=1) -> (importance [S] summed strength h over patches, fp [S, C, patch_len] mean
    decoded content, amp_topo [S, C] signed trial-mean topography, phase_topo [S, C] raw per-channel
    phase of the trial-mean (a, b), out: the StampBank output). The trial-mean (a, b) is a coherent
    average: a source arriving at random phase per patch partially cancels."""
    out = model.encode_stamps(x, coords, time_idx=time_idx, valid_channels=valid_channels)
    importance = out.h.sum(dim=0).cpu().numpy()                           # [S]
    fp = model.stamps.decode(out.amp).mean(dim=0).transpose(0, 1)         # [S, C, L]
    ab = out.amp.mean(dim=0).transpose(0, 1)                              # [S, C, 2]
    return importance, fp, _signed_topo(ab), torch.atan2(ab[..., 1], ab[..., 0]), out


@torch.no_grad()
def extract_stamp_gallery(model, x: torch.Tensor, coords: torch.Tensor,
                          time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                          fs: float = None, freq_resolution: float = None):
    """
    Everything the whole-trial stamp gallery (tools/viz/stamp_plots.plot_stamp_gallery) needs, from
    one _stamp_summary call. Returns (ids [S], importance [S], psd_ch_x [C, S] SIGNED trial-mean amp
    per channel (the mixing column, rendered as a diverging topo), psd_x [S, C, F], freqs [F],
    phase_ch_x [C, S] raw per-channel phase (radians), and waveforms: S arrays of the real trial length
    T = (N-1)*patch_stride + patch_len -- stamp s's decoded content at ONE pinned channel (the channel
    with the most total energy for that stamp), overlapping patches averaged.
    """
    importance, fp, amp_topo, phase_topo, out = _stamp_summary(
        model, x, coords, time_idx=time_idx, valid_channels=valid_channels)
    S, C, L = fp.shape
    n_fft = _n_fft(L, fs, freq_resolution)
    fft_c = _demean_hann_rfft(fp.float(), n_fft)                          # [S, C, F]
    psd_x = (fft_c.real.pow(2) + fft_c.imag.pow(2)).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    # --- whole-trial waveform per stamp ---
    D, H = model.stamps.templates()                                       # [S, L]
    N = out.amp.shape[0]
    stride = getattr(model, 'patch_stride', None) or L
    T = (N - 1) * stride + L
    vc = valid_channels[0].bool() if valid_channels is not None else torch.ones(C, dtype=torch.bool, device=x.device)
    energy = out.amp.pow(2).sum(-1).masked_fill(~vc.view(1, C, 1), 0.0).sum(0)   # [C, S]
    waveforms = []
    for s in range(S):
        c = int(energy[:, s].argmax())    # ONE pinned channel for the whole waveform
        seg = (out.amp[:, c, s, 0, None] * D[s] + out.amp[:, c, s, 1, None] * H[s]).cpu().numpy()   # [N, L]
        acc, wsum = np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
        for n in range(N):
            acc[n * stride:n * stride + L] += seg[n]
            wsum[n * stride:n * stride + L] += 1.0
        waveforms.append(acc / np.maximum(wsum, 1.0))

    return (np.arange(S), importance, amp_topo.transpose(0, 1).cpu().numpy(), psd_x, freqs,
            phase_topo.transpose(0, 1).cpu().numpy(), waveforms)


@torch.no_grad()
def extract_stamp_psd_by_patch(model, x: torch.Tensor, coords: torch.Tensor,
                               time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                               fs: float = None, freq_resolution: float = None,
                               patch_stride: int = 1) -> PatchGridResult:
    """Every patch_stride-th patch of one trial (B=1): each stamp's signed topography, strength h and
    decoded-content spectrum at that patch, plus the patch's raw and full-reconstruction spectra."""
    out = model.encode_stamps(x, coords, time_idx=time_idx, valid_channels=valid_channels)
    contribution = model.stamps.decode(out.amp)                           # [N, C, S, L]
    N, _, _, L = contribution.shape
    sel = list(range(0, N, patch_stride))
    n_fft = _n_fft(L, fs, freq_resolution)

    def spectrum(t):
        f = _demean_hann_rfft(t.float(), n_fft)
        return (f.real.pow(2) + f.imag.pow(2)).cpu().numpy()

    recon = contribution[sel].sum(dim=2)                                  # [P, C, L]
    raw = x[0, :, sel, :].permute(1, 0, 2)                                # [P, C, L]
    return PatchGridResult(
        patch_ids=np.array(sel),
        topo=_signed_topo(out.amp[sel].transpose(1, 2)).cpu().numpy(),  # [P, S, C]
        psd=spectrum(contribution[sel].transpose(1, 2)),                  # [P, S, C, F]
        h=out.h[sel].cpu().numpy(),
        recon_topo=recon.norm(dim=-1).cpu().numpy(), recon_psd=spectrum(recon),
        raw_topo=raw.norm(dim=-1).cpu().numpy(), raw_psd=spectrum(raw),
        freqs=np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0),
    )
