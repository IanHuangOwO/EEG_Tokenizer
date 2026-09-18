"""SSVEP phase probe on BETA_4s (ADR 0014 experiment C prerequisite).

Question: does the stamp code's phase advance consistently from patch to patch at the
stimulus frequency? If so, a complex linear filter over patches can recover frequency
far finer than a stamp template's 4 Hz bins.

Each patch's phase is measured from that patch's own start, so a sinusoid at f
advances by dphi(f) = 2*pi*f*stride between consecutive patches (stride 25 samples at
200 Hz = 0.125 s). Over 8.0-15.8 Hz that is 1.0-1.975 cycles: the 40 classes land on 40
distinct angles 9 degrees apart.

Classifiers (per trial; stimulus window 0.64-3.5 s; occipital channels):
  stamp_adv  per-stamp z_s = sum_{n,ch} c[n+1] conj(c[n]), c = a + i b. Stamps whose
             template peaks in 8-16 Hz vote for f, those in 16-32 Hz vote for 2f;
             score(f) = sum_s Re(z_s e^{-i dphi}), argmax over the 40 frequencies.
             Both orientation conventions are scored (the quadrature partner is -90 deg,
             which may flip the direction); the better one is reported and named.
  raw_psda   standard power-spectral peak picking on raw EEG (stimulus window, Hann):
             power at f + power at 2f, argmax.
Result on mesae_v10_small_uw01: the '-' orientation is the right one (the quadrature
partner's -90 deg convention reverses the advance). ANG is collected but not yet scored.

Usage: python phase_probe_beta.py <run_name>
"""
import json, os, sys
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from IO.dataset import build_dataset_from_config, resolve_canonical_channels
from IO.preprocessing import slice_patches
from model.factory import build_pretrain_from_config

RUN = sys.argv[1]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJECTS = [str(s) for s in range(19, 31)]          # not in v10 small pretraining (16-18)
OCC = ('O1', 'Oz', 'O2', 'PO3', 'POz', 'PO4', 'PO7', 'PO8')

cfg = json.load(open(f'output/{RUN}/artifacts/config.json'))
FS, PL, PS = (cfg['preprocess_params'][k] for k in ('sample_freq', 'patch_length', 'patch_stride'))
cfg['dataset_params']['finetune'] = {'BETA_4s': {
    'dataset_path': 'datas/BETA_4s', 'subject_to_use': SUBJECTS, 'channels_to_use': ['all']}}
ds = build_dataset_from_config(cfg, mode='finetune')
subs = ds.base_dataset.subject_data.numpy()
meta = json.load(open('datas/BETA_4s/metadata.json'))['data_metadata']['targets']
FREQ = np.array([meta[str(k)]['stimulus_frequency_hz'] for k in range(40)])
names = np.array(resolve_canonical_channels(cfg['preprocess_params']['canonical_channels']))

m = build_pretrain_from_config(cfg, mode='pretrain')
m.load_state_dict(torch.load(f'output/{RUN}/checkpoint/last.pth', map_location='cpu')['model_state_dict'])
m.to(DEV).eval()
st = m.stamps
alive = (st.fire_ema >= st.dead_threshold).nonzero().flatten()
keep = torch.cat([alive, torch.arange(st.n_routed, st.n_stamps, device=alive.device)])
with torch.no_grad():
    D = st._template_tables()[0][keep]
    peak = np.fft.rfftfreq(D.shape[-1], 1.0 / FS)[torch.fft.rfft(D, dim=-1).abs().argmax(-1).cpu().numpy()]
fund = (peak >= 8) & (peak <= 16)
harm = (peak > 16) & (peak <= 32)
print(f'{RUN}: {len(keep)} stamps; fundamental-band stamps {int(fund.sum())} (peaks {sorted(set(peak[fund]))}), '
      f'harmonic-band {int(harm.sum())} (peaks {sorted(set(peak[harm]))})')

dt = PS / FS
starts = np.arange(64) * PS  # enough patches; trimmed below
Z, ANG, RAWP, Y, G = [], [], [], [], []
with torch.no_grad():
    for i in range(len(ds)):
        x, co, lab, vc, _ = ds[i]
        vcn = vc.numpy().astype(bool)
        occ = np.array([vcn[k] and names[k] in OCC for k in range(len(names))])
        if i == 0:
            print('occipital channels used:', list(names[occ]))
        xp, _ = slice_patches(x.unsqueeze(0), PL, PS)
        N = xp.shape[2]
        s_ = np.arange(N) * PS
        win = (s_ >= int(0.64 * FS)) & (s_ + PL <= int(3.5 * FS))
        xp_d = xp.to(DEV)
        z, _ = m.stage_features(xp_d, co.unsqueeze(0).to(DEV), time_idx=torch.arange(N, device=DEV)[None])
        zg = z.permute(0, 2, 1, 3).reshape(N, z.shape[1], -1)
        xg = xp_d.permute(0, 2, 1, 3).reshape(N, z.shape[1], PL)
        amp = st.dense_amp(zg, rms=xg.pow(2).mean(-1, keepdim=True).sqrt())[:, :, keep]   # [N, C, S, 2]
        c = torch.complex(amp[..., 0], amp[..., 1])[torch.from_numpy(win).to(DEV)][:, torch.from_numpy(occ).to(DEV)]
        Z.append((c[1:] * c[:-1].conj()).sum((0, 1)).cpu().numpy())                       # [S]
        ANG.append(torch.angle(c).mean(1).cpu().numpy())                                   # [Nw, S] (circular-ish summary)
        sig = x.numpy()[occ][:, int(0.64 * FS):int(3.5 * FS)]
        sp = np.abs(np.fft.rfft(sig * np.hanning(sig.shape[-1]), n=8 * FS, axis=-1)) ** 2   # 0.125 Hz grid
        fr = np.fft.rfftfreq(8 * FS, 1.0 / FS)
        RAWP.append(np.array([sp[:, np.argmin(np.abs(fr - f))].mean() + sp[:, np.argmin(np.abs(fr - 2 * f))].mean()
                              for f in FREQ]))
        Y.append(int(lab)); G.append(int(subs[i]))
Z, RAWP, Y, G = np.stack(Z), np.stack(RAWP), np.array(Y), np.array(G)


def acc_by_subject(pred):
    return np.array([(pred[G == g] == Y[G == g]).mean() for g in np.unique(G)])


dphi = 2 * np.pi * FREQ * dt
res = {}
for sign, tag in ((1, 'advance +'), (-1, 'advance -')):
    rot_f = np.exp(-1j * sign * dphi)            # [40]
    rot_h = np.exp(-1j * sign * 2 * dphi)
    score = (Z[:, fund, None] * rot_f[None, None]).real.sum(1) + (Z[:, harm, None] * rot_h[None, None]).real.sum(1)
    res[tag] = acc_by_subject(score.argmax(1))
    fo = (Z[:, fund, None] * rot_f[None, None]).real.sum(1).argmax(1)
    res[tag + ' (fund only)'] = acc_by_subject(fo)
res['raw_psda'] = acc_by_subject(RAWP.argmax(1))
print(f'\nBETA_4s subjects {SUBJECTS[0]}-{SUBJECTS[-1]}, {len(Y)} trials, 40 classes, chance 0.025')
for k, a in res.items():
    print(f'  {k:22s} mean {a.mean():.3f}   per-subject {np.round(a, 2)}')

# observed vs expected advance, per class, fundamental-band stamps pooled
obs = np.angle(np.array([Z[Y == k][:, fund].sum() for k in range(40)]))
exp_ = np.angle(np.exp(1j * dphi))
order = np.argsort(FREQ)
print('\nclass-mean observed advance vs expected (deg), sorted by frequency:')
for k in order[::4]:
    print(f'  {FREQ[k]:5.1f} Hz  expected {np.degrees(exp_[k]):7.1f}  observed {np.degrees(obs[k]):7.1f}')
circ = np.abs(np.mean(np.exp(1j * (obs - exp_))))
circ_neg = np.abs(np.mean(np.exp(1j * (obs + exp_))))
print(f'circular agreement |mean e^(i(obs-exp))| = {circ:.3f}  (vs sign-flipped {circ_neg:.3f}; 1 = perfect, ~0.16 = random for 40)')
