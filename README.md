# EEG Tokenizer

An EEG tokenizer that turns multi-channel EEG into sparse, interpretable per-patch features.
The model is **MeSAE**. A temporal-spatial attention (TSA) encoder feeds a sparse **stamp
dictionary** (StampBank), and the patch is reconstructed as a linear sum of a few learned
templates. The model is pretrained with masked reconstruction on a multi-dataset corpus, then
frozen. Small heads are trained on its stamp features for downstream EEG benchmarks.

> The earlier discrete tokenizer, MeFSQ (FSQ/VQ), has been removed (`docs/adr/0013`).
> Its terms are kept in `CONTEXT.md` only as a glossary for reading older ADRs.

## How it works

```
raw dataset files
  → datas/<split>/<Name>/loader.py     cut Trials (event trials: event −1 s … +4 s)
  → cache_dataset.py                   bandpass 0.5–100 Hz, resample to 200 Hz, bake .npz cache
  → IO/dataset.py                      channel-map / pad, z-score, Trial → Windows → Patches
  → train_pretrain.py                  tokenizer phase (unmasked) → masked phase, one run
  → cache_feature.py / train_finetune.py   frozen backbone → stamp features → head
```

- **Patches.** 5 s windows (1000 samples at 200 Hz) are cut into 39 patches of 50 samples,
  with a stride of 25 (50 % overlap). The model input is `[B, C=64, N=39, L=50]`.
- **Stamps.** A stamp is a unit-norm temporal template `D` plus a derived Hilbert
  quadrature partner `H`. Each active stamp adds `a·D + b·H` to a channel, which gives an
  amplitude and a phase per channel. One top-k stamp set is picked per patch position and
  shared across channels, so each stamp's per-channel gains form an ICA-style mixing
  vector (a topography).
- **Routed and shared pools.** Routed stamps compete, and only the `stamp_top_k` highest
  scores decode. Shared stamps fire on every patch.
- **Sparsity budget (a hard limit).** Keep `2·(stamp_top_k + n_shared_stamps) < patch_len`
  with some margin. Past that line the active slots can fit any patch exactly, and the
  model stops doing sparse coding (`docs/adr/0011`). The default budget is 2·(12+4) = 32,
  against a `patch_len` of 50.
- **Two-phase pretraining** (`docs/adr/0013`). In the *tokenizer phase*, only the shallow
  encoder blocks run, with temporal mixing only and no masking. In the *masked phase*,
  every block runs, spatial attention and the 3-D coordinate embedding are switched on,
  and the masking curriculum starts.
- **Loss.** The loss is patch MSE + overlap-added trial MSE + matching-pursuit residual
  ordering (`mp`) + dead-atom rescue (`aux`) + MoE load balance (`ffn_lb`).

The default model has about 2.29M parameters. The StampBank is roughly 2 % of them.
`CONTEXT.md` ("Current MeSAE defaults") has the full snapshot, and `docs/adr/` has the
reasoning behind each choice.

## Setup

```bash
pip install -r requirements.txt   # plus a CUDA build of PyTorch (CUDA 11.8)
```

Use an environment that has `mne` installed. Without it, `IO/loader.py` silently falls back
to flat polar channel coordinates from `metadata.json` instead of MNE 3-D positions, and
finetune numbers are then not comparable.

## Usage

```bash
# 1. Compile raw datasets into bandpassed/resampled per-subject caches (verification runs automatically)
python cache_dataset.py --config configs/compile.json

# 2. Pretrain a backbone
mkdir -p configs/runs/<model_name>
cp configs/pretrain.template.json configs/runs/<model_name>/pretrain.json
python train_pretrain.py --config configs/runs/<model_name>/pretrain.json

# 3. Finetune a head on the frozen backbone (the feature cache is built automatically)
cp configs/finetune.template.json configs/runs/<model_name>/finetune/<head>/<dataset>_<mode>.json
#    set base_config, pretrained_checkpoint, output_path, split in the copy
python train_finetune.py --config configs/runs/<model_name>/finetune/<head>/<dataset>_<mode>.json

# Analysis / diagnostics
python analysis_pretrain.py --panel profile [--train]            # param counts + timing, no data needed
python analysis_pretrain.py --config configs/analysis_pretrain.template.json --checkpoint <path>
python analysis_finetune.py --config configs/analysis_finetune.template.json \
    --base-config <run>/artifacts/config.json --checkpoint <head.pth>
```

The files in `configs/*.template.json` are starting points and are never run directly. Real
run configs go in `configs/runs/`, which is git-ignored because every run saves a snapshot of
its config into `output/`. See `configs/README.md` for the full convention.

**Finetune splits.** `training_params.finetune.split` supports `intra_subject` (k-fold within
each subject) and `inter_subject` (subject k-fold, LOSO, or explicit/auto eval subjects).
Read `docs/finetune-caveats.md` before you report or compare finetune numbers.

## Outputs

```
output/<backbone>/
  pretrain/
    checkpoint/{best,last}.pth    # prefer last.pth: best.pth locks onto easy curriculum epochs
    artifacts/config.json
    visualization/                # loss curves, topomap reconstructions
    feature_cache/                # regenerable stamp-feature cache for finetuning
  finetune/<head>/<dataset>_<mode>/
    finetune/run_<name>/head.pth
    artifacts/group_eval.json     # per-subject balanced accuracy
```

## Repository layout

| Path | What |
|---|---|
| `train_pretrain.py`, `train_finetune.py` | Training entry points |
| `cache_dataset.py`, `cache_feature.py` | Dataset compilation and frozen-backbone feature caching |
| `analysis_pretrain.py`, `analysis_finetune.py` | Post-training diagnostics |
| `model/MeSAE/` | `MeSAE_modules.py` (embeddings, `TSABlock`, `TSAEncoder`, `StampBank`, finetune `FeatureHead`), `MeSAE.py` (`MeSAEPretrain`, `FinetuneModel`), `plugin.py` |
| `model/` | Plugin base classes and `factory.py` (`MODEL_REGISTRY`) |
| `IO/` | Loading, preprocessing (windowing, patching, normalization), masking |
| `datas/pretrain/`, `datas/finetune/` | One folder per dataset: `loader.py`, `metadata.json`, git-ignored `raw/` and `cache/` |
| `tools/` | `analysis/` (calculation), `viz/` (rendering), `panels/` (CLI panels), `misc/` (one-off scripts) |
| `configs/` | Templates, `compile.json`, `montages.json`, eval splits |
| `docs/` | ADRs, agent protocols, finetune caveats |

## Datasets

`datas/DATASETS.md` lists every dataset with its paradigm, benchmark membership (EEG-FM-Bench /
EEG-FM-Compass), subject count, event position, compiled hours and status. It currently
covers 29 pretrain datasets and 19 finetune datasets. The file is generated, so regenerate it
with `python -m tools.misc.dataset_inventory` after you add or compile a dataset.
Pretraining splits train/val by subject, so no subject's data appears in both.

## Further reading

- `CONTEXT.md`: canonical terms and current model defaults
- `docs/adr/`: architecture decisions (stamp dictionary 0009, residual loss 0011,
  two-phase run 0013, finetune heads 0016, finetune restart 0017)
- `docs/agents/`: how-tos for adding a model, dataset, montage or tool, plus reshape pitfalls
- `CLAUDE.md`: detailed developer and agent reference
