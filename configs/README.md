# configs/

| File / folder | What |
|---|---|
| `pretrain.template.json` | Starting point for a backbone's pretrain config (pretrain keys only). |
| `finetune.template.json` | Starting point for a finetune overlay (finetune keys + `base_config`). |
| `analysis_pretrain.template.json` / `analysis_finetune.template.json` | Overlays for `analysis_pretrain.py` / `analysis_finetune.py`. |
| `compile.json` | Dataset compile settings (`cache_dataset.py`). |
| `montages.json` | Named channel montages (`preprocess_params.canonical_channels`). |
| `finetune_eval_splits/` | Seeded train/eval subject splits (`split.eval_subjects: "auto"`). |
| `runs/` | Real run configs, one folder per model -- **git-ignored**. |

## runs/

A real run's config, never the templates directly. `runs/` is git-ignored: each run
snapshots its config into `output/<run>/artifacts/config.json` (plus a timestamped copy),
so those snapshots are the record of what actually ran. Every model version gets its own
folder from the start — `configs/runs/<model_name>/` — even before any finetune run
exists for it (mirrors `output/`, where pretrain always lands in
`output/<model_name>/pretrain/`, see CLAUDE.md's "Outputs" section).

Pretrain and finetune configs are split: a finetune overlay sets `"base_config"` to its
backbone's pretrain file and `load_config` (`tools/analysis/__init__.py`) deep-merges the
two — only the finetune keys need to be in the overlay (`dataset_params.finetune`,
`model_params.MeSAE.finetune`, `training_params.finetune`,
`training_params.visualize_params.finetune`).

```bash
cp configs/pretrain.template.json configs/runs/<model_name>/pretrain.json
cp configs/finetune.template.json configs/runs/<model_name>/finetune/<head>/<dataset>_<mode>.json
# then in the overlay: "base_config": "configs/runs/<model_name>/pretrain.json",
# pretrained_checkpoint / output_path / model_name / split for this run
```

A run's actual write location under `output/` is `training_params.<mode>.output_path`,
not `model_name` -- `model_name` stays a clean identity string (what gets logged at
startup), `output_path` is the field allowed to carry a path segment (when unset it
defaults to `"<model_name>/pretrain"` for pretrain, plain `model_name` for finetune, see
CLAUDE.md's "Outputs" section). e.g. `configs/runs/mesae_v12_small/pretrain.json` sets only
`model_name: "mesae_v12_small"` and lands in `output/mesae_v12_small/pretrain/`;
`configs/runs/mesae_v10_small/finetune/learned/BNCI2014001_intra.json` sets
`model_name: "learned_BNCI2014001_intra"`,
`output_path: "mesae_v10_small/finetune/learned/BNCI2014001_intra"`, producing
`output/mesae_v10_small/finetune/learned/BNCI2014001_intra/`. Edit the copy, not the
template — see CLAUDE.md's "`configs/` layout" section.

If a backbone gets re-pretrained (a second pretrain run under the same `model_name`),
name that new config `pretrain_<timestamp>.json` instead of overwriting `pretrain.json` --
mirrors `artifacts/config_<timestamp>.json`'s non-destructive run snapshot, so an older
pretrain config a live finetune overlay still points at (`base_config`) isn't clobbered.

## Analysis overlays (`analysis_pretrain.py`/`analysis_finetune.py --config`)

`configs/analysis_pretrain.template.json`/`configs/analysis_finetune.template.json` are
also never pointed at directly for a REAL analysis session on a specific model -- their
`dataset_params` lists whatever datasets happened to be in the template, not necessarily
what that model was trained/finetuned on. When you'll re-run `--analysis`/`--panel`
against one model repeatedly, copy the fitting template into that model's own
`configs/runs/<model_name>/` folder (same reasoning as `pretrain.json`/finetune overlays
above -- a real run's config, never the template directly) and edit `dataset_params`/
`check` to match that model:

```bash
cp configs/analysis_pretrain.template.json configs/runs/<model_name>/analysis_pretrain.json
cp configs/analysis_finetune.template.json configs/runs/<model_name>/analysis_finetune.json
```

Then `python analysis_pretrain.py --config configs/runs/<model_name>/analysis_pretrain.json
--checkpoint <path>` (finetune: same, plus its always-required `--base-config`). Not
automated/enforced anywhere -- a one-off check can still pass `--config
configs/analysis_pretrain.template.json` (or a scratch overlay) directly; this is just
where to put it once it's a repeated thing for one model.
