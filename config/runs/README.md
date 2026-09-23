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

e.g. `config/runs/mesae_v10_small/pretrain.json` produces
`output/mesae_v10_small/pretrain/`, and
`config/runs/mesae_v10_small/finetune/learned/BNCI2014001.json` produces
`output/mesae_v10_small/finetune/learned/BNCI2014001_intra/`. Edit the copy, not the
template — see CLAUDE.md's "`config/` layout" section.

If a backbone gets re-pretrained (a second pretrain run under the same `model_name`),
name that new config `pretrain_<timestamp>.json` instead of overwriting `pretrain.json` --
mirrors `artifacts/config_<timestamp>.json`'s non-destructive run snapshot, so an older
pretrain config a live finetune overlay still points at (`base_config`) isn't clobbered.
