# configs/

| File / folder | What |
|---|---|
| `pretrain.template.json` | Starting point for a backbone's pretrain config: the full corpus and the default recipe. |
| `pretrain_tiny.template.json` | The same with `window_fraction` 0.05 (the tiny corpus). |
| `finetune.template.json` | Starting point for a finetune overlay (finetune keys + `base_config`). |
| `analysis_pretrain.template.json` / `analysis_finetune.template.json` | Overlays for `analysis_pretrain.py` / `analysis_finetune.py`. |
| `finetune_protocols.json` | Frozen finetune protocols (split + head + hyperparameters), selected by `training_params.finetune.protocol`. |
| `sweeps/` | Sweep files for `tools/misc/sweep.py` (e.g. `frozen_protocols.json`: every protocol x reported dataset x backbone). |
| `compile.json` | Compile settings and the dataset list (`cache_dataset.py`). |
| `montages.json` | Named channel montages: the canonical one and the sub-montages channel subsampling uses. |
| `runs/` | Real run configs, one folder per backbone -- **git-ignored**. |

## runs/

A real run never points at a template. `runs/` is git-ignored because every run snapshots its effective
config (overrides applied) into `output/<run>/artifacts/config.json`; those snapshots are the record.

```bash
cp configs/pretrain_tiny.template.json configs/runs/<backbone>/pretrain.json
cp configs/finetune.template.json configs/runs/<backbone>/finetune/<head>/<cell>.json
# overlay: "base_config": "configs/runs/<backbone>/pretrain.json", pretrained_checkpoint, output_path, protocol
```

A finetune overlay holds only finetune keys; `load_config` (`tools/analysis/__init__.py`) deep-merges it onto
its `base_config`. `model_name` is an identity string; `output_path` is where the run writes under `output/`
(pretrain default `<model_name>/pretrain`; finetune e.g. `<backbone>/finetune/<head>/<cell>`). A trained
backbone is always rebuilt from its checkpoint's own `build_config`, so editing `pretrain.json` afterwards
cannot change what a finetune loads -- but keep it as the record of how the backbone was trained anyway.

For repeated analysis of one backbone, copy the analysis template into `configs/runs/<backbone>/` and set its
`dataset_params` to that backbone's datasets.
