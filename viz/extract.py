"""
Per-unit (MeFSQ Expert / MeSAE Filter) feature extraction: activation norms, affinity,
routing/importance scores, and PSD — read by each model's plugin.py and fed into
viz/panels.py. Model-coupled (runs a partial/full forward pass), unlike viz/topomap.py.

extract_head_*/extract_filter_* return the shared PsdResult/SpectraResult dataclasses —
same shape for MeFSQ Experts and MeSAE Filters (the "Unit" abstraction model/base_checker.py
and model/base_plotter.py already use for exactly this reason), so BaseEpochChecker never
needs to know which model it's plotting. The four functions share their FFT/norm/affinity
math (_psd_per_channel/_spectra_per_channel/_cosine_affinity below) — only where each
model's per-Unit decoded vector comes from, and how importance is scored, differs.
_split_channel_major is duplicated here (not imported from a shared module) — kept local
to each of this file, model/MeFSQ/MeFSQ.py, and model/MeSAE/MeSAE.py on purpose, same
per-model-ownership rationale as docs/adr/0006.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _split_channel_major(flat, C, P):
    """[..., C*P] channel-major (C outer, P inner) -> [..., C, P]. Every decoder this
    file reads from (MeFSQ's MultiHeadDecoder, MeSAE's TopKSAE decoder) is C-major.
    See docs/agents/reshape-pitfalls.md."""
    assert flat.shape[-1] == C * P, (
        f"_split_channel_major: last dim {flat.shape[-1]} != C*P ({C}*{P}={C * P})")
    return flat.reshape(*flat.shape[:-1], C, P)


@dataclass
class PsdResult:
    """psd_ch_x: [C, Q] mean per-channel decoded activation norm per Unit.
    norms: [Q]. affinity: [Q, Q] cosine similarity. importance: [Q] ranking score — routing
    fraction for MeFSQ Experts, accumulated gate score (sum over patches) for MeSAE
    Filters; either way, constant for always-on shared units, patch-selection-dependent for
    routed ones."""
    psd_ch_x: np.ndarray
    norms: np.ndarray
    affinity: np.ndarray
    importance: np.ndarray


@dataclass
class PatchGridResult:
    """Real per-patch stamp selection/content — no cross-patch averaging or trial-wide
    dedup (unlike PsdResult/_used_stamps): a stamp firing on many sampled patches shows up
    once per patch it fired at, with that patch's own real decoded content, instead of
    being blurred into one trial-averaged row. See extract_filter_psd_by_patch.
    patch_ids: [P] sampled patch indices (every patch_stride-th patch).
    stamp_ids: [P, K] GLOBAL stamp id selected at each sampled patch's each of K=top_k+
      n_shared slots (same layout StampBank.forward's `idx` uses, routed then shared).
    topo: [P, K, C] per-channel real-response norm (topomap power) for that (patch, slot).
    psd: [P, K, C, F] per-channel power spectrum of that (patch, slot)'s real decoded content.
    h: [P, K] that slot's real selection strength at that patch.
    recon_topo: [P, C] per-channel norm of that sampled patch's REAL full reconstruction
      (sum of all K slots' contributions — the same sum StampBank.forward's `recon` is,
      just for this one patch, not the whole trial) — topo of a sum is not the sum of
      topos (norm isn't linear), so this is computed from the summed signal, not derived
      from `topo` above.
    recon_psd: [P, C, F] power spectrum of that same per-patch full reconstruction.
    raw_topo: [P, C] per-channel norm of that sampled patch's REAL raw input (the model's
      own `x` at that patch, before any encoding/decoding) — same FFT settings as
      recon_topo/psd so it's directly comparable, not derived from recon.
    raw_psd: [P, C, F] power spectrum of that same per-patch raw input.
    freqs: [F].
    """
    patch_ids: np.ndarray
    stamp_ids: np.ndarray
    topo: np.ndarray
    psd: np.ndarray
    h: np.ndarray
    recon_topo: np.ndarray
    recon_psd: np.ndarray
    raw_topo: np.ndarray
    raw_psd: np.ndarray
    freqs: np.ndarray


@dataclass
class SpectraResult:
    """psd: [Q, C, F] per-Unit per-channel power spectrum. freqs: [F].
    importance: [Q], same ranking score as PsdResult.importance."""
    psd: np.ndarray
    freqs: np.ndarray
    importance: np.ndarray


def _demean_hann_rfft(x: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Demean + Hann-taper x along its last dim, THEN zero-pad to n_fft and rfft.
    Every PSD in this file is built from a single short patch_len window (~100ms), far too
    short to resolve real Delta/Theta content and, worse, a bare rfft on an un-tapered
    snippet is an implicit rectangular window whose mainlobe (~2*fs/L Hz wide) smears any
    DC offset or patch-boundary discontinuity across 0-20Hz — that leakage is what was
    showing up as suspiciously uniform "high power" at 0-10Hz across every dataset/model,
    not real signal. Demeaning kills the DC spike; the Hann taper suppresses the
    sidelobes carrying the rest of the leakage. Zero-padding (n_fft > L) still only
    interpolates this cleaned-up spectrum for display — it doesn't add real resolution
    below ~1/(L/fs) Hz, that ceiling is unavoidable at patch_len scale."""
    x = x - x.mean(dim=-1, keepdim=True)
    win = torch.hann_window(x.shape[-1], periodic=False, device=x.device, dtype=x.dtype)
    return torch.fft.rfft(x * win, n=n_fft, dim=-1)


def _cosine_affinity(v_mean: torch.Tensor) -> np.ndarray:
    """v_mean: [Q, D] mean decoded/embedded vector per Unit -> [Q, Q] cosine similarity."""
    v_mean_n = F.normalize(v_mean, dim=-1)
    return (v_mean_n @ v_mean_n.T).cpu().numpy()


def _psd_per_channel(recon_flat: torch.Tensor, B: int, N: int, C: int, P: int) -> np.ndarray:
    """recon_flat: [M, Q, C*P] per-Unit decoded reconstruction, channel-major.
    Returns [C, Q] mean per-channel activation norm (norm over patch length P, mean over
    patches N, first trial in the batch)."""
    recon = _split_channel_major(recon_flat, C, P).reshape(B, N, -1, C, P)
    return recon.norm(dim=-1).mean(dim=1)[0].permute(1, 0).cpu().numpy()


def _spectra_per_channel(recon_flat: torch.Tensor, B: int, N: int, C: int, L: int,
                          fs: float = None, freq_resolution: float = None):
    """recon_flat: [M, Q, C*L] per-Unit decoded reconstruction, channel-major.
    Zero-padded FFT power spectrum, averaged over patches, first trial in the batch.
    Returns (psd [Q, C, F], freqs [F])."""
    Q = recon_flat.shape[1]
    recon = _split_channel_major(recon_flat, C, L).reshape(B, N, Q, C, L)[0].permute(2, 0, 1, 3)  # [C, N, Q, L]

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = _demean_hann_rfft(recon.float(), n_fft)
    psd = fft_c.real.pow(2) + fft_c.imag.pow(2)   # [C, N, Q, F]
    psd = psd.mean(dim=1)                          # [C, Q, F] — average over patches
    psd = psd.permute(1, 0, 2).cpu().numpy()       # [Q, C, F]

    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)
    return psd, freqs


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

    head_norms    = z_all.norm(dim=-1).mean(dim=0).cpu().numpy()  # [H_total]
    head_affinity = _cosine_affinity(z_all.mean(dim=0))

    recon_shared = model.decoder_shared(z_shared)               # [M, Hs, C*patch_len]
    recon_routed = model.decoder_routed(z_routed)               # [M, Hr, C*patch_len]
    recon_all    = torch.cat([recon_shared, recon_routed], dim=1)  # [M, H_total, C*patch_len]
    patch_len    = recon_all.shape[-1] // C
    psd_ch_h     = _psd_per_channel(recon_all, B, N, C, patch_len)  # [C, H_total]

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

    psd, freqs = _spectra_per_channel(recon_all, B, N, C, L, fs=fs, freq_resolution=freq_resolution)

    routing_shared = torch.full((model.n_shared_experts,), model.shared_weight, device=recon_routed.device)
    routing_routed = (out.gate_mask_routed.detach() > 0).float().mean(dim=0)
    routing_score  = torch.cat([routing_shared, routing_routed]).cpu().numpy()  # [H_total]

    return SpectraResult(psd, freqs, routing_score)


def _used_stamps(model, x, coords, time_idx=None, valid_channels=None, max_stamps=100):
    """(used_ids [Qu] LongTensor, importance [Qu] np.ndarray, fp [Qu, C, patch_len]) —
    stamps actually selected somewhere in this trial (union across all patches), capped at
    max_stamps, ranked by accumulated selection strength (see MeSAEPretrain.used_stamp_ids
    — shared stamps always included first, remaining budget filled by highest-usage routed
    stamps). Replaces a dense all-n_stamps view: at n_stamps=800 with hard top-k selection,
    most stamps never fire in a given trial at all, and showing all 800 regardless swamps
    the panels.

    fp is built from StampBank.dense_probe — each stamp's REAL pooled_i (its actual
    channel-weighted view of these real patches), not fingerprint()'s fabricated zero
    vector. The zero probe collapses the tied bottleneck's `pre = probe@W_down + b_down`
    down to just `b_down` (zero-init, slow to grow), so it mostly reflects shared
    near-zero bias behavior rather than each atom's actual (possibly already quite
    different) weight-driven response to content — systematically understating diversity,
    especially early in training. fp here is dense_probe's [C, patch_len] contribution
    averaged over patches (dense_probe decodes straight to [C, patch_len] now — no outer
    product needed here, W_out dropped the rank-1 factorization, see StampBank.__init__'s
    W_out comment). Shared helper for extract_filter_psd/extract_filter_spectra below."""
    z, _ = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)
    out = model.stamps(z_bnc, valid_mask=valid_mask)  # eval-mode call, aux/dead-atom path never runs
    used_ids = model.used_stamp_ids(out, max_stamps=max_stamps)  # [Qu]

    shared = out.dense_routed.new_full((out.dense_routed.shape[0], model.n_shared_stamps), model.shared_weight)
    dense_full = torch.cat([out.dense_routed, shared], dim=-1)  # [M, n_stamps]
    importance = dense_full[:, used_ids].sum(dim=0).cpu().numpy()  # [Qu]

    contribution, _attn = model.stamps.dense_probe(z_bnc, valid_mask=valid_mask)  # [M,n_stamps,C,patch_len]
    fp = contribution[:, used_ids, :, :].mean(dim=0)  # [Qu, C, patch_len]

    return used_ids, importance, fp


@torch.no_grad()
def extract_filter_psd(model, x: torch.Tensor, coords: torch.Tensor,
                       time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None) -> PsdResult:
    """
    MeSAEPretrain analog of extract_head_psd, adapted for StampBank (see
    docs/adr/0009-spatiotemporal-stamp-dictionary-for-mesae.md's Monitoring impact
    section). Unlike the retired per-Filter decode, a stamp's rank-1 pattern here is built
    from its REAL response to this trial's actual patches (StampBank.dense_probe, mean
    over patches — see `_used_stamps`), not a content-independent fingerprint.

    psd_ch_x — [C, Qu] per-stamp per-channel real-response norm, restricted to stamps used
      this trial (Qu = len(used_ids) <= 100).
    norms — [Qu] norm of each used stamp's flattened real-response pattern.
    affinity — [Qu, Qu] cosine similarity between used stamps' real-response patterns —
      same matrix MeSAECodebookChecker.decoder_fingerprint_matrix computes (that one still
      uses the zero-probe fingerprint, dictionary-wide with no trial data available),
      duplicated/restricted here rather than cross-imported (same per-file-ownership
      convention as this module's docstring already states for MeFSQ/MeSAE).
    importance — [Qu] accumulated selection strength (sum over patches) per used stamp.
    """
    used_ids, stamp_importance, fp = _used_stamps(model, x, coords, time_idx=time_idx,
                                                   valid_channels=valid_channels, max_stamps=100)
    flat = fp.reshape(fp.shape[0], -1)

    stamp_norms = flat.norm(dim=-1).cpu().numpy()  # [Qu]
    stamp_affinity = _cosine_affinity(flat)

    psd_ch_q = fp.norm(dim=-1).permute(1, 0).cpu().numpy()  # [C, Qu]

    return PsdResult(psd_ch_q, stamp_norms, stamp_affinity, stamp_importance)


@torch.no_grad()
def extract_filter_psd_by_patch(model, x: torch.Tensor, coords: torch.Tensor,
                                 time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                                 fs: float = None, freq_resolution: float = None,
                                 patch_stride: int = 3) -> PatchGridResult:
    """
    Per-patch counterpart to extract_filter_psd/_used_stamps: instead of deduplicating a
    stamp across the whole trial and averaging its content over every patch it fired at
    (hiding whether it fired once or on every patch), this keeps every sampled patch's own
    real top_k+n_shared selection and decodes each slot's content AT that specific patch
    only (StampBank.decode_selected, the same per-slot contribution forward() sums into
    `recon` — nothing new computed, just not summed away here). patch_stride subsamples
    patches (not config-exposed — a caller-side rendering-cost knob, not a modeling
    choice) since a full (top_k+n_shared)*N_patches grid is impractically large to render
    (e.g. 20 slots * 40 patches = 800 topo+PSD columns).
    """
    z, _ = model.stage_features(x, coords, time_idx=time_idx)
    z_bnc, valid_mask = model._pool_channels(z, valid_channels)  # [M, C, D], M = N (B=1)
    out = model.stamps(z_bnc, valid_mask=valid_mask)  # eval-mode call, aux/dead-atom path never runs

    contribution, _z_h = model.stamps.decode_selected(out.idx, out.h, out.pooled)  # [M, K, C, L]

    M = contribution.shape[0]
    sel = list(range(0, M, patch_stride))
    contribution_sel = contribution[sel]  # [P, K, C, L]
    L = contribution_sel.shape[-1]

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))
    fft_c = _demean_hann_rfft(contribution_sel.float(), n_fft)  # [P, K, C, F]
    psd = (fft_c.real.pow(2) + fft_c.imag.pow(2)).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    topo = contribution_sel.norm(dim=-1).cpu().numpy()  # [P, K, C]

    # Real per-patch full reconstruction (sum of all K slots' contributions, same sum
    # StampBank.forward's `recon` computes) — its own topo/psd, not derived from the
    # per-slot ones above (norm/FFT aren't linear, see PatchGridResult docstring).
    recon_sel = contribution_sel.sum(dim=1)  # [P, C, L]
    fft_recon = _demean_hann_rfft(recon_sel.float(), n_fft)  # [P, C, F]
    recon_psd = (fft_recon.real.pow(2) + fft_recon.imag.pow(2)).cpu().numpy()
    recon_topo = recon_sel.norm(dim=-1).cpu().numpy()  # [P, C]

    # Real per-patch raw input (model's own x at that patch, B=1) — same n_fft as recon
    # above so raw/recon/slot rows share one freq axis.
    raw_sel = x[0, :, sel, :].permute(1, 0, 2)  # [P, C, L]
    fft_raw = _demean_hann_rfft(raw_sel.float(), n_fft)  # [P, C, F]
    raw_psd = (fft_raw.real.pow(2) + fft_raw.imag.pow(2)).cpu().numpy()
    raw_topo = raw_sel.norm(dim=-1).cpu().numpy()  # [P, C]

    return PatchGridResult(
        patch_ids=np.array(sel), stamp_ids=out.idx[sel].cpu().numpy(),
        topo=topo, psd=psd, h=out.h[sel].cpu().numpy(),
        recon_topo=recon_topo, recon_psd=recon_psd,
        raw_topo=raw_topo, raw_psd=raw_psd, freqs=freqs,
    )


# ==========================================
# MeSAEFlat (flat per-(channel,patch) StampBank) — no channel-attention pool exists to
# read a "real response" from (see MeSAEFlat_modules.StampBank class docstring), so these
# decode every (channel, patch) token directly and ZERO-FILL (channel, {patch|stamp})
# combinations that were never actually selected, instead of reading a pooled view —
# separate functions from extract_filter_psd/extract_filter_psd_by_patch above (which stay
# untouched, still used by regular pooled-channel MeSAE) since the two architectures need
# genuinely different extraction logic, not a shared code path.
# ==========================================

@torch.no_grad()
def _used_flat_stamps(model, x, coords, time_idx=None, valid_channels=None, max_stamps=100):
    """(used_ids [Qu], importance [Qu], fp [Qu, C, patch_len]) — grouped-StampBank
    analog of _used_stamps. Selection is per PATCH POSITION now (shared by all C
    channels, see MeSAEFlat_modules.StampBank class docstring), so importance is the
    accumulated selection confidence h over the patches that picked each used stamp,
    and fp is each used stamp's per-channel contribution (amp_c * rms_c * D_hat, the
    stamp's real mixing/topomap content) averaged over ALL N patches — a patch that
    didn't select the stamp contributes an explicit 0 (dilution toward 0, same
    convention as before), but WITHIN a selected patch every channel now has a real
    dense amp value: the per-channel zero-holes of the old per-token selection (a
    channel that lost the top-k race showing 0 despite genuinely containing the
    source) are gone by construction."""
    z, _ = model.stage_features(x, coords, time_idx=time_idx)  # [1, C, N, D]
    B, C, N, D = z.shape
    z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, D)           # [G=N, C, D] (B=1)
    x_g = x.permute(0, 2, 1, 3).reshape(B * N, C, -1)
    # rms must match the training path (see MeSAEFlatPretrain.forward) — contribution
    # is amp*rms*D_hat, so without rms every decoded panel is mis-scaled.
    rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()
    vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C) \
        if valid_channels is not None else None

    out = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vc_g)  # eval-mode call
    idx, h = out.idx, out.h  # [N, K]
    K = idx.shape[1]
    n_stamps = model.n_stamps

    dense_imp = h.new_zeros(n_stamps)
    dense_imp.scatter_add_(0, idx.reshape(-1), h.reshape(-1))
    used_ids = model.used_stamp_ids(out, max_stamps=max_stamps)
    importance = dense_imp[used_ids].cpu().numpy()

    contribution = model.stamps.decode_selected(idx, out.amp)  # [N, C, K, patch_len]
    patch_len = contribution.shape[-1]

    # Accumulate per stamp over patches: group-major/slot-minor flatten of idx matches
    # the same flatten of contribution's (N, K) axes.
    buf = contribution.new_zeros(n_stamps, C, patch_len)
    buf.index_add_(0, idx.reshape(-1), contribution.permute(0, 2, 1, 3).reshape(N * K, C, patch_len))
    fp_full = buf / N  # unselected patches dilute toward 0

    fp = fp_full[used_ids]  # [Qu, C, patch_len]
    return used_ids, importance, fp


@torch.no_grad()
def extract_flat_stamp_psd(model, x: torch.Tensor, coords: torch.Tensor,
                            time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None) -> PsdResult:
    """
    MeSAEFlatPretrain analog of extract_filter_psd — see the module-section docstring
    above for why this is a separate function rather than a shared one. valid_channels
    feeds the group selection score (see StampBank.forward), matching training.

    psd_ch_x — [C, Qu] per-stamp per-channel trial-averaged real-response norm (zero for
      channels that stamp never fired at), restricted to stamps used this trial.
    norms/affinity — same formulas as extract_filter_psd, off the same fp.
    importance — [Qu] accumulated selection strength (sum over channels AND patches).
    """
    used_ids, stamp_importance, fp = _used_flat_stamps(
        model, x, coords, time_idx=time_idx, valid_channels=valid_channels, max_stamps=100)
    flat = fp.reshape(fp.shape[0], -1)

    stamp_norms = flat.norm(dim=-1).cpu().numpy()
    stamp_affinity = _cosine_affinity(flat)
    psd_ch_q = fp.norm(dim=-1).permute(1, 0).cpu().numpy()  # [C, Qu]

    return PsdResult(psd_ch_q, stamp_norms, stamp_affinity, stamp_importance)


@torch.no_grad()
def extract_flat_stamp_gallery(model, x: torch.Tensor, coords: torch.Tensor,
                                time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                                fs: float = None, freq_resolution: float = None, max_stamps: int = 100):
    """
    Everything MeSAEFlatChecker's standalone whole-trial stamp gallery panel
    (viz.panels.plot_stamp_gallery) needs, off ONE _used_flat_stamps call — avoids calling
    extract_flat_stamp_psd AND a separate spectra extraction, each independently
    re-running decode_selected/StampBank.forward for the same trial, when both just read
    different views (norm vs FFT) of the same fp. Not part of the shared PsdResult/
    SpectraResult "Unit" contract (model/base_checker.py's extract_psd/extract_spectra) —
    this is a bespoke one-off panel, not the generic per-model dispatch path.

    Returns (used_ids [Qu] np.ndarray, importance [Qu] np.ndarray, psd_ch_x [C, Qu]
    np.ndarray, psd_x [Qu, C, F] np.ndarray, freqs [F] np.ndarray).
    """
    used_ids, importance, fp = _used_flat_stamps(
        model, x, coords, time_idx=time_idx, valid_channels=valid_channels, max_stamps=max_stamps)
    psd_ch_x = fp.norm(dim=-1).permute(1, 0).cpu().numpy()  # [C, Qu]

    patch_len = fp.shape[-1]
    n_fft = patch_len
    if fs and freq_resolution:
        n_fft = max(patch_len, int(round(fs / freq_resolution)))
    fft_c = _demean_hann_rfft(fp.float(), n_fft)  # [Qu, C, F]
    psd_x = (fft_c.real.pow(2) + fft_c.imag.pow(2)).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    return used_ids.cpu().numpy(), importance, psd_ch_x, psd_x, freqs


@torch.no_grad()
def extract_flat_stamp_psd_by_patch(model, x: torch.Tensor, coords: torch.Tensor,
                                     time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                                     fs: float = None, freq_resolution: float = None,
                                     patch_stride: int = 1, k_display: int = None) -> PatchGridResult:
    """
    MeSAEFlatPretrain analog of extract_filter_psd_by_patch. StampBank selects ONE
    top_k+n_shared stamp set per patch position, shared by all C channels (group
    selection, see MeSAEFlat_modules.StampBank class docstring) — so the per-patch grid
    is direct: slot k of patch n is a real global stamp id for every channel, and its
    [C] row of decoded content is that stamp's actual dense per-channel contribution
    (amp_c * rms_c * D_hat — the stamp's mixing/topomap column at that patch time, no
    zero-holes). The old per-token union/zero-fill machinery (union of C independent
    selections, importance-ranked, -1-padded) is gone with the architecture that
    needed it.

    k_display, if given, keeps only the first k_display slots (routed slots come
    score-ordered first, shared last — so a small k_display drops shared first).
    h is the group's selection confidence per slot (softmax over routed group scores;
    constant shared_weight at shared slots).
    """
    z, _ = model.stage_features(x, coords, time_idx=time_idx)  # [1, C, N, D]
    B, C, N, D = z.shape
    z_g = z.permute(0, 2, 1, 3).reshape(B * N, C, D)           # [G=N, C, D] (B=1)
    x_g = x.permute(0, 2, 1, 3).reshape(B * N, C, -1)
    # rms must match the training path (see MeSAEFlatPretrain.forward) — contribution
    # is amp*rms*D_hat, so without rms every decoded panel is mis-scaled.
    rms = x_g.pow(2).mean(dim=-1, keepdim=True).sqrt()
    vc_g = valid_channels.unsqueeze(1).expand(B, N, C).reshape(B * N, C) \
        if valid_channels is not None else None

    out = model.stamps(z_g, x_target=None, rms=rms, valid_channels=vc_g)  # eval-mode call
    idx, h = out.idx, out.h  # [N, K]
    K = idx.shape[1]
    Kd = min(k_display, K) if k_display is not None else K

    contribution = model.stamps.decode_selected(idx, out.amp)  # [N, C, K, patch_len]
    patch_len = contribution.shape[-1]

    sel = list(range(0, N, patch_stride))

    grid_ids = idx[sel, :Kd]                                        # [P, Kd]
    grid_h = h[sel, :Kd]                                            # [P, Kd]
    grid_contrib = contribution[sel][:, :, :Kd].permute(0, 2, 1, 3)  # [P, Kd, C, patch_len]

    n_fft = patch_len
    if fs and freq_resolution:
        n_fft = max(patch_len, int(round(fs / freq_resolution)))
    fft_c = _demean_hann_rfft(grid_contrib.float(), n_fft)  # [P, Kd, C, F]
    psd = (fft_c.real.pow(2) + fft_c.imag.pow(2)).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)
    topo = grid_contrib.norm(dim=-1).cpu().numpy()  # [P, Kd, C] — |amp*rms| per channel

    # Real per-patch, per-channel full reconstruction — ALL K slots summed (StampBank.
    # forward's recon for this patch), not just the Kd displayed ones.
    recon_cn = contribution[sel].sum(dim=2)           # [P, C, patch_len]
    fft_recon = _demean_hann_rfft(recon_cn.float(), n_fft)
    recon_psd = (fft_recon.real.pow(2) + fft_recon.imag.pow(2)).cpu().numpy()
    recon_topo = recon_cn.norm(dim=-1).cpu().numpy()

    raw_sel = x[0, :, sel, :].permute(1, 0, 2)  # [P, C, L]
    fft_raw = _demean_hann_rfft(raw_sel.float(), n_fft)
    raw_psd = (fft_raw.real.pow(2) + fft_raw.imag.pow(2)).cpu().numpy()
    raw_topo = raw_sel.norm(dim=-1).cpu().numpy()

    return PatchGridResult(
        patch_ids=np.array(sel), stamp_ids=grid_ids.cpu().numpy(),
        topo=topo, psd=psd, h=grid_h.cpu().numpy(),
        recon_topo=recon_topo, recon_psd=recon_psd,
        raw_topo=raw_topo, raw_psd=raw_psd, freqs=freqs,
    )


@torch.no_grad()
def extract_filter_spectra(model, x: torch.Tensor, coords: torch.Tensor,
                           time_idx: torch.Tensor = None, valid_channels: torch.Tensor = None,
                           fs: float = None, freq_resolution: float = None) -> SpectraResult:
    """
    MeSAEPretrain analog of extract_head_spectra, adapted for StampBank. Every USED stamp's
    (see extract_filter_psd, capped at 100) real mean-over-patches response FFT'd per
    channel — real content, not a fabricated zero probe (see `_used_stamps`).
    """
    used_ids, stamp_importance, fp = _used_stamps(model, x, coords, time_idx=time_idx,
                                                   valid_channels=valid_channels, max_stamps=100)
    L = fp.shape[-1]

    n_fft = L
    if fs and freq_resolution:
        n_fft = max(L, int(round(fs / freq_resolution)))

    fft_c = _demean_hann_rfft(fp.float(), n_fft)
    psd = (fft_c.real.pow(2) + fft_c.imag.pow(2)).cpu().numpy()  # [Qu, C, F]
    freqs = np.fft.rfftfreq(n_fft, d=(1.0 / fs) if fs else 1.0)

    return SpectraResult(psd, freqs, stamp_importance)
