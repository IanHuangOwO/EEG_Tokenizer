"""atom_duplicates: are Q-atoms copies of each other up to phase (tools.analysis.atom_dist.
atom_similarity)? Prints pairs above --dup-threshold (default 0.9) and each Q-atom's nearest neighbour
-> analysis/atom_duplicates.png."""
import os

import numpy as np

from tools.analysis.atom_dist import atom_similarity
from tools.viz.atom_plots import plot_similarity_matrix

STAGES = frozenset({'pretrain'})


def run(ctx):
    sim, label = atom_similarity(ctx.model)
    thr = ctx.args.dup_threshold
    pairs = sorted(((sim[i, j], i, j) for i, j in zip(*np.triu_indices(len(label), 1)) if sim[i, j] >= thr),
                   reverse=True)
    print(f"{len(label)} alive atoms; {len(pairs)} pair(s) with similarity >= {thr}")
    for s, i, j in pairs:
        print(f"  {s:.3f}  #{label[i]} <-> #{label[j]}")
    nn = np.nanargmax(sim, axis=1)
    print("nearest neighbour: " + ', '.join(f"#{label[k]}->#{label[j]} {sim[k, j]:.2f}" for k, j in enumerate(nn)))
    out = os.path.join(ctx.out_dir, 'atom_duplicates.png')
    plot_similarity_matrix(out, sim, label, f'Phase-invariant atom similarity sqrt(<Di,Dj>^2+<Di,Hj>^2)\n'
                                            f'{len(pairs)} pair(s) >= {thr}')
    print(f"  -> {out}")
