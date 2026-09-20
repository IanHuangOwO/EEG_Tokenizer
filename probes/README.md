# probes/

One-off diagnostics for ADR 0014 (finetune head) and ADR 0012. Not part of training;
each is run by hand and prints a table. Run them from anywhere — every script `chdir`s to
the repo root itself.

Feature caches are tens to hundreds of MB and belong to one run, so they are written to
`output/<run>/probes/`, which is gitignored.

| script | what it answers |
|---|---|
| `probe_v10.py <run> [ckpt=last.pth]` | Which representation carries the task information? Per-subject shrinkage LDA on BCICIV2a for `head_z`, `chan_mag`, `z_mean`, `z_chan`, `recon_bandpow`, `stamp_bandpow` (ADR 0014 experiment A). Writes `probe_feats_<run>.npz` and the per-stamp `stamp_dump_<run>.npz`. Compares against `output/mesae_pretrain_v9/probes/pretrain_probe_feats.npz` when that cache is present. |
| `stamp_relevance.py <run> [onset=200] [post_end_s=4.0]` | Which stamp, and when, carries task information? Per-stamp decodability, ERD/ERS, C3/C4 lateralization and inter-trial phase coherence, BH-FDR corrected. Reads the dump above; writes a CSV next to it. |
| `phase_probe_beta.py <run>` | Does stamp phase advance track the SSVEP stimulus frequency? 40-class BETA_4s, stamp phase advance vs the raw power-peak baseline (ADR 0014 experiment C prerequisite). |
| `ft_summary.py <log> [<log> ...]` | Per-subject `balanced_acc` from `train_finetune.py` logs: best epoch, last epoch, and the mean of the last 10 epochs, plus a paired t-test between runs. The last-10 mean is the number ADR 0014 reports; best-val is optimistic on 60-trial validation sets. |
| `select_eval_subsets.py [--datasets eegmmidb beta4s]` | Seeded (42) seen/unseen train/eval subject subsets for `subject_group_runs`; EEGMMIdb eval stratified by a band-power LDA difficulty proxy, BETA_4s random. Writes `config/subject_groups/*.json`. |
| `group_summary.py <group_eval.json> [...]` | Stats over `train_finetune.py`'s `subject_groups` output: per-subject last-10-epoch `tail` per group (kfold `heldout` folds pooled), paired head-minus-reference (first file) per group at subject level, and seen-vs-unseen Welch per head. |

Read the results and the caveats in `docs/adr/0014-finetune-head-test-plan.md` before
reusing any of these numbers. In particular: stamp power's ERD *sign* disagrees with a
model-free check on the raw signal, so read which stamp and when, not the direction.
