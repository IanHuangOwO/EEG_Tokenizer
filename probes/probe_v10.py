"""v10 (fused run) probe: same BNCI2014001 (formerly BCICIV2a) trials, same features, same per-subject LDA as
pretrain_probe.py, compared paired against the cached v9 tokenizer/pretrain columns
(pretrain_probe_feats.npz). Usage: python probe_v10.py <run_name> [ckpt=last.pth]

v10 checkpoints restore their own phase flags on load (MeSAE _restore_phase), so no
enable_* calls here. Alive-stamp rule and feature definitions are copied verbatim.
"""
import json, os, sys
import numpy as np
import torch
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from IO.dataset import build_dataset_from_config, resolve_canonical_channels
from IO.preprocessing import slice_patches
from model.factory import build_pretrain_from_config
from model.MeSAE.MeSAE_modules import overlap_add_patches
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUN = sys.argv[1]
# Feature caches are large (tens of MB) and run-specific: they live beside the run, not in
# the repo. output/ is gitignored.
OUT = os.path.join('output', RUN, 'probes')
os.makedirs(OUT, exist_ok=True)
CKPT = f'output/{RUN}/checkpoint/{sys.argv[2] if len(sys.argv) > 2 else "last.pth"}'
DS = 'BNCI2014001'  # renamed from BCICIV2a 2026-09-22
KEYS = ('head_z', 'chan_mag', 'z_mean', 'z_chan', 'recon_bandpow', 'stamp_bandpow')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BANDS = ((8, 13), (13, 30))

cfg = json.load(open(f'output/{RUN}/artifacts/config.json'))
FS, PL, PS = (cfg['preprocess_params'][k] for k in ('sample_freq', 'patch_length', 'patch_stride'))
cfg['dataset_params']['finetune'] = {DS: {
    'dataset_path': f'datas/finetune/{DS}', 'subject_to_use': ['all'], 'channels_to_use': ['all']}}
ds = build_dataset_from_config(cfg, mode='finetune')
subs = ds.base_dataset.subject_data.numpy()

m = build_pretrain_from_config(cfg, mode='pretrain')
m.load_state_dict(torch.load(CKPT, map_location='cpu')['model_state_dict'])
m.to(DEV).eval()
assert bool(m.masked_phase), 'expected a masked-phase checkpoint'
alive = (m.stamps.fire_ema >= 0.1 * (m.stamps.top_k / m.stamps.n_routed)).nonzero().flatten()
hz_idx = torch.cat([alive, torch.arange(m.stamps.n_routed, m.n_stamps, device=alive.device)])
# stamp_bandpow: H_i is D_i's quadrature partner, so the a*b cross term vanishes in every
# positive-frequency bin and a*D_i + b*H_i carries a^2*|FFT D_i|^2 + b^2*|FFT H_i|^2 there,
# exactly (H is renormalized after zeroing DC/Nyquist, hence its own table). E_D/E_H[i, band]
# = band energy of each template. Same dense amps as chan_mag, spectrally weighted; the sum
# over stamps ignores cross-stamp terms. At patch_len 50 / 200 Hz the bins are 4 Hz.
with torch.no_grad():
    D_tab, H_tab = (t[hz_idx] for t in m.stamps._template_tables())          # [S, L] each
    frD = np.fft.rfftfreq(D_tab.shape[-1], 1.0 / FS)

    def band_energy(T):
        sp = torch.fft.rfft(T, dim=-1).abs().pow(2)
        return torch.stack([sp[:, torch.from_numpy((frD >= lo) & (frD < hi)).to(sp.device)].sum(-1) for lo, hi in BANDS], -1)
    E_D, E_H = band_energy(D_tab), band_energy(H_tab)                      # [S, 2]
print(f'{RUN}: {len(alive)} alive routed stamps of {m.stamps.n_routed}  ({CKPT})')

F = {k: [] for k in KEYS}
AMP, SEL = [], []
ys, gs = [], []
with torch.no_grad():
    for i in range(len(ds)):
        x, co, lab, vc, _vl = ds[i]
        xp, _ = slice_patches(x.unsqueeze(0), PL, PS)
        tix = torch.arange(xp.shape[2]).unsqueeze(0)
        xp = xp.to(DEV)
        cob, vcb, vcn = co.unsqueeze(0).to(DEV), vc.unsqueeze(0).to(DEV), vc.numpy()
        vct = torch.from_numpy(vcn).to(DEV)
        hz = m.encode_post_stamp_expert(xp, cob, time_idx=tix, valid_channels=vcb)
        F['head_z'].append(hz[0][:, hz_idx].mean(0).flatten().cpu().numpy())
        zf, _ = m.stage_features(xp, cob, time_idx=tix)
        B_, C_, N_, D_ = zf.shape
        vt = vct.view(1, C_, 1, 1).float()
        F['z_mean'].append(((zf * vt).sum((1, 2)) / max(vcn.sum() * N_, 1))[0].cpu().numpy())
        # z_chan: per-channel time-mean encoder z on valid channels [C_valid * D] -- the
        # conventional FM input (encoder features), without the cross-channel mean that
        # makes z_mean unable to express a C3-C4 contrast. LDA sees every channel.
        F['z_chan'].append(zf[0, vct].mean(1).flatten().cpu().numpy())
        zg = zf.permute(0, 2, 1, 3).reshape(B_ * N_, C_, D_)
        xg = xp.permute(0, 2, 1, 3).reshape(B_ * N_, C_, -1)
        amp = m.stamps.dense_amp(zg, rms=xg.pow(2).mean(-1, keepdim=True).sqrt())[:, :, alive, :]
        amp_s = m.stamps.dense_amp(zg, rms=xg.pow(2).mean(-1, keepdim=True).sqrt())[:, :, hz_idx, :]
        pw = (torch.einsum('gcs,sb->cb', amp_s[..., 0].pow(2), E_D)
              + torch.einsum('gcs,sb->cb', amp_s[..., 1].pow(2), E_H)) / zg.shape[0]   # [C, 2]
        F['stamp_bandpow'].append((torch.log(pw + 1e-12) * vct.view(C_, 1).float()).flatten().cpu().numpy())
        # per-stamp dump for stamp_relevance.py: (a, b) per patch x valid channel x stamp
        AMP.append(amp_s[:, vct].half().cpu().numpy())                    # [N, Cv, S, 2]
        # approx selection (topk mode): pre-rms group score over valid channels, routed only
        score = m.stamps.dense_amp(zg)[:, vct][:, :, :m.stamps.n_routed].pow(2).sum(-1).mean(1)  # [N, R]
        SEL.append(torch.zeros_like(score).scatter_(1, score.topk(m.stamps.top_k, dim=1).indices, 1.0)
                   [:, alive].bool().cpu().numpy())                       # [N, S_alive]
        mg = amp.pow(2).sum(-1).sqrt() * vct.view(1, C_, 1).float()
        F['chan_mag'].append(mg.mean(0).flatten().cpu().numpy())
        out = m(xp, cob, time_idx=tix, bool_masked_pos=None, valid_channels=vcb)
        rec = overlap_add_patches(out.recon[0], PS).cpu().numpy() * vcn[:, None]
        sp = np.abs(np.fft.rfft(rec, axis=-1)) ** 2
        fr = np.fft.rfftfreq(rec.shape[-1], 1.0 / FS)
        F['recon_bandpow'].append(np.nan_to_num(np.concatenate(
            [np.log(sp[:, (fr >= lo) & (fr < hi)].sum(-1) + 1e-12) for lo, hi in ((8, 13), (13, 30))])))
        ys.append(int(lab)); gs.append(int(subs[i]))
y, g = np.array(ys), np.array(gs)
X = {k: np.nan_to_num(np.stack(v)) for k, v in F.items()}
np.savez_compressed(os.path.join(OUT, f'probe_feats_{RUN}.npz'), y=y, g=g, **X)
np.savez(os.path.join(OUT, f'stamp_dump_{RUN}.npz'), y=y, g=g, amp=np.stack(AMP), sel=np.stack(SEL),
         stamp_ids=hz_idx.cpu().numpy(), n_alive=len(alive), E_D=E_D.cpu().numpy(), E_H=E_H.cpu().numpy(),
         patch_stride=PS, patch_len=PL, fs=FS, valid=vcn,
         chan_names=np.array(resolve_canonical_channels(cfg['preprocess_params']['canonical_channels'])),
         D_peak_hz=frD[torch.fft.rfft(D_tab, dim=-1).abs().argmax(-1).cpu().numpy()])
print('per-stamp dump saved', flush=True)

# Optional: the v9 tokenizer/pretrain feature cache written by the older pretrain_probe.py.
# Without it the v9 comparison columns are skipped.
V9 = os.path.join('output', 'pretrain', 'mesae_pretrain_v9', 'probes', 'pretrain_probe_feats.npz')
old = np.load(V9) if os.path.exists(V9) else None
if old is None:
    print(f'(no v9 cache at {V9}: v9 comparison columns skipped)')
else:
    assert (old['y'] == y).all() and (old['g'] == g).all(), 'trial order differs from the v9 cache'


def per_subject(Xk, pca=64):
    """Per-subject accuracies (array), identical pipeline to pretrain_probe.py."""
    accs = []
    for s in np.unique(g):
        mk = g == s
        steps = [StandardScaler()]
        if Xk.shape[1] > pca:
            steps += [PCA(n_components=pca, random_state=0), StandardScaler()]
        steps.append(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        accs.append(cross_val_score(make_pipeline(*steps), Xk[mk], y[mk],
                                    cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())
    return np.array(accs)


print(f'\n{DS} per-subject LDA(shrinkage), PCA64, n={len(y)}, chance 0.250. '
      f'Paired t vs v9 columns ({len(np.unique(g))} subjects).')
print(f'{"feature":14s} {"width":>5s} {"v9 tok":>7s} {"v9 pre":>7s} {RUN[:12]:>12s} '
      f'{"d vs pre":>8s} {"p":>6s} {"wins":>5s}')
for k in KEYS:
    a_new = per_subject(X[k])
    if old is None or f'pretrain|{k}' not in old.files:   # no v9 cache, or a new feature
        print(f'{k:14s} {X[k].shape[1]:5d} {"-":>7s} {"-":>7s} {a_new.mean():12.3f}', flush=True)
        continue
    a_tok = per_subject(old[f'tokenizer|{k}'])
    a_pre = per_subject(old[f'pretrain|{k}'])
    d = a_new - a_pre
    p = stats.ttest_rel(a_new, a_pre).pvalue
    print(f'{k:14s} {X[k].shape[1]:5d} {a_tok.mean():7.3f} {a_pre.mean():7.3f} {a_new.mean():12.3f} '
          f'{d.mean():+8.3f} {p:6.3f} {int((d > 0).sum()):>3d}/{len(d)}', flush=True)
