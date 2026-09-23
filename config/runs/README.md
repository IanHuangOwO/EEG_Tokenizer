# config/runs/

A real run's config, never the templates directly. Every model version gets its own
folder from the start — `config/runs/<model_name>/` — even before any finetune run
exists for it (this is stricter than `output/`'s own rule, see CLAUDE.md's "Outputs"
section, which stays flat until a backbone has finetune runs; `config/runs/` always
nests, config edits happen far more often than a backbone gains its first finetune run).

Pretrain and finetune configs are split: a finetune overlay sets `"base_config"` to its
backbone's pretrain file and `load_config` (`tools/analysis/__init__.py`) deep-merges the
two — only the finetune keys need to be in the overlay (`dataset_params.finetune`,
`model_params.MeSAE.finetune`, `training_params.finetune`,
`training_params.visualize_params.finetune`).

```bash
cp config/config.template.json config/runs/<model_name>/pretrain.json
# finetune overlay: dataset_params.finetune / model_params.MeSAE.finetune /
# training_params.finetune / training_params.visualize_params.finetune only,
# plus "base_config": "config/runs/<model_name>/pretrain.json"
```

A run's actual write location under `output/` is `training_params.<mode>.output_path`,
not `model_name` -- `model_name` stays a clean identity string (what gets logged at
startup), `output_path` is the field allowed to carry a path segment (falls back to
`model_name` when unset, see CLAUDE.md's "Outputs" section). e.g.
`config/runs/mesae_v10_small/pretrain.json` sets `model_name: "mesae_v10_small"`,
`output_path: "mesae_v10_small/pretrain"`, producing `output/mesae_v10_small/pretrain/`;
`config/runs/mesae_v10_small/finetune/learned/BNCI2014001_intra.json` sets
`model_name: "learned_BNCI2014001_intra"`,
`output_path: "mesae_v10_small/finetune/learned/BNCI2014001_intra"`, producing
`output/mesae_v10_small/finetune/learned/BNCI2014001_intra/`. Edit the copy, not the
template — see CLAUDE.md's "`config/` layout" section.

If a backbone gets re-pretrained (a second pretrain run under the same `model_name`),
name that new config `pretrain_<timestamp>.json` instead of overwriting `pretrain.json` --
mirrors `artifacts/config_<timestamp>.json`'s non-destructive run snapshot, so an older
pretrain config a live finetune overlay still points at (`base_config`) isn't clobbered.

## Analysis overlays (`analysis_pretrain.py`/`analysis_finetune.py --config`)

`config/analysis_pretrain.template.json`/`config/analysis_finetune.template.json` are
also never pointed at directly for a REAL analysis session on a specific model -- their
`dataset_params` lists whatever datasets happened to be in the template, not necessarily
what that model was trained/finetuned on. When you'll re-run `--analysis`/`--panel`
against one model repeatedly, copy the fitting template into that model's own
`config/runs/<model_name>/` folder (same reasoning as `pretrain.json`/finetune overlays
above -- a real run's config, never the template directly) and edit `dataset_params`/
`check` to match that model:

```bash
cp config/analysis_pretrain.template.json config/runs/<model_name>/analysis_pretrain.json
cp config/analysis_finetune.template.json config/runs/<model_name>/analysis_finetune.json
```

Then `python analysis_pretrain.py --config config/runs/<model_name>/analysis_pretrain.json
--checkpoint <path>` (finetune: same, plus its always-required `--base-config`). Not
automated/enforced anywhere -- a one-off check can still pass `--config
config/analysis_pretrain.template.json` (or a scratch overlay) directly; this is just
where to put it once it's a repeated thing for one model.
