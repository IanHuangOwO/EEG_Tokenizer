# EEG Tokenizer

An EEG foundation model that turns multi-channel EEG into interpretable per-patch features.
The model is **MeSAE**: a temporal-spatial attention (TSA) encoder feeds a **stamp dictionary**
(StampBank), and each patch is reconstructed as a linear sum of learned waveform templates. It is
pretrained with masked reconstruction on a multi-dataset corpus, then frozen. Downstream tasks read
either the encoder output directly (a linear probe) or the stamp codes (a structured head).

> The earlier discrete tokenizer, MeFSQ (FSQ/VQ), has been removed (`docs/adr/0013`).
> Its terms are kept in `CONTEXT.md` only as a glossary for reading older ADRs.

## How it works

```
raw dataset files
  → datas/<split>/<Name>/loader.py     read recordings; cut event trials (event −1 s … +4 s) or fixed windows
  → cache_dataset.py                   bandpass 0.5–100 Hz + resample to 200 Hz per continuous recording,
                                       then cut; drop flat-line windows; bake per-subject .npz caches
  → IO/dataset.py                      map onto the canonical montage (missing → padding), z-score,
                                       Trial → Windows → Patches; one-montage batches
  → train_pretrain.py                  tokenizer phase (unmasked) → masked phase, one run
  → cache_feature.py / train_finetune.py   frozen backbone → cached features → small head
```

- **Patches.** 5 s windows (1000 samples at 200 Hz) are cut into 39 patches of 50 samples with a
  stride of 25 (50 % overlap). The model input is `[B, C=64, N=39, L=50]`.
- **Encoder.** 8 TSA blocks (temporal attention → spatial attention → MoE FFN, with LayerScale) in
  4 stages of 2. The patch axis is pooled 39 → 20 → 10 → 5 between stages and linearly upsampled
  back, with gated skips from each stage. Electrode positions enter through a Fourier coordinate
  embedding and a relative spatial bias in every block; an optional relative temporal bias does the
  same for time lags. During training each skip is dropped per sample (graded drop-path, strongest
  on the finest skip), so the deeper stages have to carry patch detail too.
- **Stamps.** A stamp is a unit-norm temporal template `D` plus its quadrature partner `H`. Each
  stamp adds `a·D + b·H` to a channel, which gives an amplitude and a phase per channel; the
  per-channel gains of one stamp form a topography. The current recipe uses 16 shared (always-on)
  stamps and no routed ones.
- **Sparsity budget (a hard limit).** Keep `2·(stamp_top_k + n_shared_stamps) < patch_len` with
  margin. Past that line the active stamps can fit any patch exactly and the model stops doing
  sparse coding (`docs/adr/0011`). The current budget is 2·16 = 32 against a `patch_len` of 50.
- **Two-phase pretraining** (`docs/adr/0013`). In the *tokenizer phase* every block runs with
  temporal attention only and no masking. In the *masked phase* spatial attention and the
  coordinate embedding switch on and the masking curriculum starts (channel clusters, random
  channels, time blocks, plus channel subsampling to sparse montages).
- **Loss.** Patch MSE (visible patches count at 0.1 of masked ones) + a per-stamp matching-pursuit
  term (`mp`: stamps ranked per patch by strength, each trained on what the stronger ones left, so a
  duplicate earns nothing) + MoE load balance. Tested and removed: an overlap-added trial MSE
  (`docs/adr/0021`: without it neighbouring patches agree better), a masked STFT loss (`0019`: it
  restores masked band power but did not reach the tasks), and a merged nested loss (`0018`).
- **Evaluation.** Each backbone change is judged by its own mechanism metric on held-out windows
  (`docs/adr/0020`: coordinate / time / skip ablations, seam disagreement, stamp usage, masked band
  power, a closed-form ridge probe), not by single-seed finetuning, which runs only when a recipe is
  frozen.

The default model has about 2.26M parameters; the StampBank is under 1 % of them. `CONTEXT.md`
has the canonical terms and `docs/adr/` the reasoning behind each choice.

## Setup

```bash
pip install -r requirements.txt   # plus a CUDA build of PyTorch
```

Use an environment that has `mne` (and `moabb`) installed. Without `mne`, `IO/loader.py` silently
falls back to flat polar channel coordinates from `metadata.json` instead of MNE 3-D positions,
and finetune numbers are then not comparable.

## Usage

```bash
# 1. Compile raw datasets into per-subject caches (verification runs automatically)
python cache_dataset.py --config configs/compile.json            # --dataset NAME, --verify-only

# 2. Pretrain a backbone
mkdir -p configs/runs/<backbone>
cp configs/pretrain_tiny.template.json configs/runs/<backbone>/pretrain.json   # set model_name, output_path
python train_pretrain.py --config configs/runs/<backbone>/pretrain.json

# 3. Finetune a head on the frozen backbone (the feature cache is built on first use)
cp configs/finetune.template.json configs/runs/<backbone>/finetune/<head>/<dataset>_<split>.json
#    set base_config, pretrained_checkpoint, output_path, protocol / split in the copy
python train_finetune.py --config configs/runs/<backbone>/finetune/<head>/<dataset>_<split>.json

# Any config value can be overridden on the command line (dotted path, JSON value)
python train_finetune.py --config <cfg> --set model_params.MeSAE.finetune.spatial_k=8

# Experiments: a sweep file -> queue plan; run_queue runs it resumably, a few jobs at a time
python -m tools.misc.sweep configs/sweeps/<sweep>.json > output/queue/<plan>.plan
python -m tools.misc.run_queue output/queue/<plan>.plan --max-parallel 2 --threads 8

# Analysis
python analysis_pretrain.py --run <backbone> [--preset quick] [--panel <name> ...]
python analysis_pretrain.py --panel profile [--train]            # parameter counts + timing
python analysis_finetune.py --group base=<backbone> --group X=<backbone> --ref base
python -m tools.analysis.summarize_runs 'output/<backbone>/finetune/<head>/*'
```

The files in `configs/*.template.json` are starting points and are never run directly. Real run
configs go in `configs/runs/`, which is git-ignored because every run saves its effective config
in `output/.../artifacts/config.json`. See `configs/README.md`.

**Finetune.** A head is a list of feature entries (`model_params.MeSAE.finetune.features`), each
with its own spatial filter: stamp-code entries (`stamp_power`, `signed_ab`, ...), latent entries
on the encoder output z (`latent_signed` is the linear probe), and raw-signal baselines. Splits
(`training_params.finetune.split.type`): `loso`, `subject_kfold`, `eval_subjects`, `kfold`,
`blocked_kfold`, `fewshot`. Frozen protocols (`configs/finetune_protocols.json`: `mi_loso`,
`mi_fewshot`, `p300_loso`, `p300_fewshot`) were tuned on development sets only. Read
`docs/finetune-caveats.md` before you report or compare finetune numbers.

## Outputs

```
output/<backbone>/
  pretrain/
    checkpoint/{best,last}.pth    # prefer last.pth: best.pth locks onto easy curriculum epochs
    artifacts/config.json
    visualization/                # training dashboard, reconstruction snapshots
    analysis/                     # analysis_pretrain.py panels
    feature_cache/                # regenerable features for finetuning
  finetune/<head>/<dataset>_<split>/
    artifacts/group_eval.json     # per-subject balanced accuracy (tail = mean of the last 10 epochs)
output/reports/                   # cross-backbone comparison reports
output/archive/                   # superseded experiments
```

## Repository layout

| Path | What |
|---|---|
| `train_pretrain.py`, `train_finetune.py` | Training entry points |
| `cache_dataset.py`, `cache_feature.py` | Dataset compilation and frozen-backbone feature caching |
| `analysis_pretrain.py`, `analysis_finetune.py` | Analysis entry points (panels in `tools/panels/`) |
| `model/MeSAE/` | `MeSAE_modules.py` (embeddings, `TSABlock`, `TSAEncoder`, `StampBank`, finetune `FeatureHead`), `MeSAE.py` (`MeSAEPretrain`, losses, `FinetuneModel`), `plugin.py` |
| `model/` | Plugin base classes and `factory.py` (`MODEL_REGISTRY`, `build_from_checkpoint`) |
| `IO/` | Loading, preprocessing (windowing, patching, normalization), masking |
| `datas/pretrain/`, `datas/finetune/` | One folder per dataset: `loader.py`, `metadata.json`, git-ignored `raw/` and `cache/` |
| `tools/` | `analysis/` (calculation), `viz/` (rendering), `panels/` (analysis units), `misc/` (sweeps, queue, corpus and inventory scripts) |
| `configs/` | Templates, `compile.json`, `montages.json`, `finetune_protocols.json`, `sweeps/` |
| `docs/` | ADRs, agent how-tos, finetune caveats |

## Datasets

`datas/DATASETS.md` lists every dataset with its paradigm, benchmark membership (EEG-FM-Bench /
EEG-FM-Compass), subject count, event position, compiled hours and status: currently 24 pretrain
datasets (438 h compiled) and 19 finetune datasets, plus 5 archived. The file is generated, so
regenerate it with `python -m tools.misc.dataset_inventory` after you add or compile a dataset.
Pretraining splits train/val by person, per cohort, so no subject's data appears in both.

## Further reading

- `CONTEXT.md`: canonical terms
- `docs/adr/`: architecture decisions (stamp dictionary 0009, matching-pursuit loss 0011,
  two-phase run 0013, loss trimming 0015, finetune heads 0016, nested loss 0018)
- `docs/agents/`: how-tos for adding a model, dataset, montage or tool, plus reshape pitfalls
- `CLAUDE.md`: detailed developer and agent reference
