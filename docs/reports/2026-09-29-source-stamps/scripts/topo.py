"""Render every source topography A_s[:, k] of a spatial_rank backbone on the 10-10 montage -> topographies.png."""
import json, sys, math, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model.factory import build_from_checkpoint
from model.MeSAE.MeSAE_modules import fourier_features
from IO.loader import get_standard_coords
bb = sys.argv[1]
st = build_from_checkpoint(torch.load(f'output/{bb}/pretrain/checkpoint/last.pth', map_location='cpu', weights_only=False)).stamps.eval()
names = json.load(open('configs/montages.json'))['10-10']
names = [c['label'] for c in names['channels']]
xyz = torch.tensor([get_standard_coords(n) for n in names if get_standard_coords(n) is not None], dtype=torch.float32)
S, K = st.n_stamps, st.spatial_rank
with torch.no_grad():
    A = st.topo(fourier_features(xyz)).view(len(xyz), 2, S, K)[:, 1]            # [C, S, K]
D, _ = st.templates()
f = torch.fft.rfftfreq(D.shape[-1], 1 / 200); peak = f[torch.fft.rfft(D, dim=-1).abs().argmax(-1)]
fig, ax = plt.subplots(S, K, figsize=(1.6 * K, 1.5 * S))
for s in range(S):
    for k in range(K):
        a = A[:, s, k]; v = float(a.abs().max())
        ax[s, k].scatter(xyz[:, 0], xyz[:, 1], c=a, cmap='RdBu_r', vmin=-v, vmax=v, s=18)
        ax[s, k].set_xticks([]); ax[s, k].set_yticks([]); ax[s, k].set_aspect('equal')
    ax[s, 0].set_ylabel(f's{s} {peak[s]:.0f}Hz', fontsize=7)
plt.tight_layout(); plt.savefig(f'output/{bb}/pretrain/analysis/topographies.png', dpi=110)
print('saved', f'output/{bb}/pretrain/analysis/topographies.png')
