"""
Per-unit (MeFSQ Expert / MeSAE Filter) feature extraction: activation norms, affinity,
routing/importance scores, and PSD — read by each model's plugin.py and fed into
viz/panels.py. Model-coupled (runs a partial/full forward pass), unlike viz/topomap.py.

extract_head_*/extract_filter_* return the shared PsdResult/SpectraResult dataclasses —
same shape for MeFSQ Experts and MeSAE Filters (the "Unit" abstraction model/base_checker.py
and model/base_plotter.py already use for exactly this reason), so BaseEpochChecker never
needs to know which model it's plotting.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PsdResult:
    """psd_ch_x: [C, Q] mean per-channel decoded activation norm per Unit.
    norms: [Q]. affinity: [Q, Q] cosine similarity. importance: [Q] ranking score
    (routing fraction for MeFSQ Experts, decoded-contribution magnitude for MeSAE Filters)."""
    psd_ch_x: np.ndarray
    norms: np.ndarray
    affinity: np.ndarray
    importance: np.ndarray


@dataclass
class SpectraResult:
    """psd: [Q, C, F] per-Unit per-channel power spectrum. freqs: [F].
    importance: [Q], same ranking score as PsdResult.importance."""
    psd: np.ndarray
    freqs: np.ndarray
    importance: np.ndarray


@torch.no_grad()
def extract_head_psd(model, x: torch.Tensor, coords: torch.Tensor,
                     time_idx: torch.Tensor = None) -> PsdResult:
    """
    Per-head per-channel activation norm, combining BOTH MoE pools (shared experts first,
    at indices [0, n_shared_experts), then routed) — same order as encode_pre_vq, so this
    lines up with the finetune head's attn_h/attn_n head axis.

    Computed in up-projected embed_dim space (z_per_head = v_q @ vq_proj), not raw v_q —
    routed/shared pools can have different r (quantizer vocab width), so their raw v_q
    can't be concatenated/compared directly, but both always up-project to the same
    embed_dim, which is also the space the decoder actually reads.

    psd_ch_h — [C, H_total] mean per-channel decoded activation norm per head (fused
      per-patch VQ has no per-channel axis pre-decode, so this decodes each pool through
      its own decoder to recover a per-channel magnitude, rather than the pre-decode
      embedding norm used before the channel fusion).
    importance — fraction of patches selecting each head (model.shared_weight for shared,
      always active but down-weighted).
    """
    B, C, N, L = x.shape
    out = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)

    z_shared = torch.einsum('mhr,hdr->mhd', out.v_q_shared, model.vq_proj_shared)  # [M, Hs, D]
    z_routed = torch.einsum('mhr,hdr->mhd', out.v_q_routed, model.vq_proj_routed)  # [M, Hr, D]
    z_all    = torch.cat([z_shared, z_routed], dim=1)                              # [M, H_total, D]
    H = z_all.shape[1]

    head_norms    = z_all.norm(dim=-1).mean(dim=0).cpu().numpy()  # [H_total]
    v_mean_h      = z_all.mean(dim=0)
    v_mean_n      = F.normalize(v_mean_h, dim=-1)
    head_affinity = (v_mean_n @ v_mean_n.T).cpu().numpy()

    recon_shared = model.decoder_shared(z_shared)               # [M, Hs, C*patch_len]
    recon_routed = model.decoder_routed(z_routed)               # [M, Hr, C*patch_len]
    recon_all    = torch.cat([recon_shared, recon_routed], dim=1)  # [M, H_total, C*patch_len]
    patch_len    = recon_all.shape[-1] // C
    recon_all    = recon_all.reshape(B, N, H, C, patch_len)
    psd_ch_h     = recon_all.norm(dim=-1).mean(dim=1)[0].permute(1, 0).cpu().numpy()  # [C, H_total]

    # routing importance: shared experts are always active, scaled by their fixed contribution
    # weight to recon (model.shared_weight) rather than a flat 1.0 — otherwise they'd always
    # rank as "most important" even though their recon contribution is deliberately down-weighted;
    # routed experts by fraction of patches that selected them
    routing_shared = torch.full((model.n_shared_experts,), model.shared_weight, device=z_all.device)
    routing_routed = (out.gate_mask_routed.detach() > 0).float().mean(dim=0)
    routing_score  = torch.cat([routing_shared, routing_routed]).cpu().numpy()  # [H_total]

    return PsdResult(psd_ch_h, head_norms, head_affinity, routing_score)


@torch.no_grad()
def extract_head_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                          time_idx: torch.Tensor = None, fs: float = None,
                          freq_resolution: float = None) -> SpectraResult:
    """
    Per-head, per-channel power spectrum of that head's OWN decoded reconstruction
    (v_q -> vq_proj -> decoder, per head, un-gated so every routed head shows what it would
    reconstruct if selected — shared heads have no gating to begin with) — not the
    shared-embedding VQ activation norm extract_head_psd reports, an actual frequency-domain
    view of each head's specialization. Combines both MoE pools (shared experts first, at
    indices [0, n_shared_experts), then routed) — same order as encode_pre_vq.
    fs: sample rate in Hz for the freq axis; if None, freqs are cycles/patch (bin index).
    freq_resolution: Hz per bin via zero-padded FFT (n_fft = fs / freq_resolution) — the
    patch itself (L samples) is far shorter than what a fine resolution needs, so this is
    padding for display resolution, not real added information beyond the L-sample window.
    """
    B, C, N, L = x.shape
    out = model(x, coords=coords, time_idx=time_idx, bool_masked_pos=None)

    z_shared = torch.einsum('mhr,hdr->mhd', out.v_q_shared,     model.vq_proj_shared)  # [M, Hs, D]
    z_routed = torch.einsum('mhr,hdr->mhd', out.v_q_routed_raw, model.vq_proj_routed)  # [M, Hr, D], un-gated
    recon_shared = model.decoder_shared(z_shared)  # [M, Hs, C*L] — fused per-patch decode covers all channels jointly
    recon_routed = model.decoder_routed(z_routed)  # [M, Hr, C*L]
    recon_all = torch.cat([recon_shared, recon_routed], dim=1)  # [M, H_total, C*L]
    H = recon_all.shape[1]
    recon_all = recon_all.reshape(B, N, H, C, L)[0].permute(2, 0, 1, 3)   # [C, N, H, L] (single trial)

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = torch.fft.rfft(recon_all.float(), n=n_fft, dim=-1)
    psd   = fft_c.real.pow(2) + fft_c.imag.pow(2)                      # [C, N, H, F]
    psd   = psd.mean(dim=1)                                            # [C, H, F] — average over patches
    psd   = psd.permute(1, 0, 2).cpu().numpy()                         # [H, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    routing_shared = torch.full((model.n_shared_experts,), model.shared_weight, device=recon_routed.device)
    routing_routed = (out.gate_mask_routed.detach() > 0).float().mean(dim=0)
    routing_score  = torch.cat([routing_shared, routing_routed]).cpu().numpy()  # [H_total]

    return SpectraResult(psd, freqs, routing_score)


@torch.no_grad()
def extract_filter_psd(model, x: torch.Tensor, coords: torch.Tensor,
                       time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None) -> PsdResult:
    """
    MeSAEPretrain analog of extract_head_psd: per-filter per-channel decoded activation
    norm. No Router/MoE here, so there's no gate score to rank by — importance is
    the mean magnitude of each filter's own decoded contribution to the reconstruction
    instead (how much of the final signal that filter is actually responsible for).

    psd_ch_x — [C, Q] mean per-channel decoded activation norm per filter.
    norms — [Q] norm of the SAE-decoded vector the decoder reads.
    affinity — [Q, Q] cosine similarity between filters' mean SAE-decoded vectors.
    importance — [Q] mean decoded-contribution magnitude per filter.
    """
    B, C, N, L = x.shape
    z = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)
    pooled, _ = model.filter_pool(z_bnc, valid_mask)
    sae_out, _, _ = model.sae(pooled)  # [M, Q, D] — the space the decoder actually reads
    Q = sae_out.shape[1]

    filter_norms = sae_out.norm(dim=-1).mean(dim=0).cpu().numpy()  # [Q]
    v_mean_n     = F.normalize(sae_out.mean(dim=0), dim=-1)
    filter_affinity = (v_mean_n @ v_mean_n.T).cpu().numpy()

    recon_per_filter = model.decoder(sae_out)  # [M, Q, C*patch_len]
    patch_len = recon_per_filter.shape[-1] // C
    recon_per_filter = recon_per_filter.reshape(B, N, Q, C, patch_len)
    psd_ch_q = recon_per_filter.norm(dim=-1).mean(dim=1)[0].permute(1, 0).cpu().numpy()  # [C, Q]

    filter_importance = recon_per_filter.norm(dim=-1).mean(dim=(0, 1, 3)).cpu().numpy()  # [Q]

    return PsdResult(psd_ch_q, filter_norms, filter_affinity, filter_importance)


@torch.no_grad()
def extract_filter_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                           time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                           fs: float = None, freq_resolution: float = None) -> SpectraResult:
    """
    MeSAEPretrain analog of extract_head_spectra: per-filter, per-channel power spectrum
    of that filter's own decoded contribution. No gating to worry about — every filter is
    always active — so this is just each filter's own contribution, un-gated by construction.
    """
    B, C, N, L = x.shape
    z = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)
    pooled, _ = model.filter_pool(z_bnc, valid_mask)
    sae_out, _, _ = model.sae(pooled)  # [M, Q, D]
    Q = sae_out.shape[1]

    recon_per_filter = model.decoder(sae_out)  # [M, Q, C*L]
    recon_per_filter = recon_per_filter.reshape(B, N, Q, C, L)[0].permute(2, 0, 1, 3)  # [C, N, Q, L]

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = torch.fft.rfft(recon_per_filter.float(), n=n_fft, dim=-1)
    psd   = fft_c.real.pow(2) + fft_c.imag.pow(2)          # [C, N, Q, F]
    psd   = psd.mean(dim=1)                                 # [C, Q, F] — average over patches
    psd   = psd.permute(1, 0, 2).cpu().numpy()              # [Q, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    filter_importance = recon_per_filter.norm(dim=-1).mean(dim=(0, 1)).cpu().numpy()  # [Q]

    return SpectraResult(psd, freqs, filter_importance)
