# config/runs/

A real run's config, never the templates directly. Pretrain and finetune configs are
split: a finetune overlay sets `"base_config"` to its backbone's pretrain file and
`load_config` (`tools/analysis/__init__.py`) deep-merges the two — only the finetune
keys need to be in the overlay (`dataset_params.finetune`, `model_params.MeSAE.finetune`,
`training_params.finetune`, `training_params.visualize_params.finetune`).

```bash
cp config/config.template.json config/runs/<model_name>.json               # pretrain, no finetune runs yet
cp config/config.template.json config/runs/<model_name>/pretrain.json      # pretrain, once finetune runs exist too
# finetune overlay: dataset_params.finetune / model_params.MeSAE.finetune /
# training_params.finetune / training_params.visualize_params.finetune only,
# plus "base_config" pointing at the backbone's pretrain file above
```

Path mirrors `output/` exactly (see CLAUDE.md's "Outputs" section for the same rule
applied there): a backbone with no finetune runs keeps its pretrain config flat at
`config/runs/<model_name>.json`. Once it has finetune runs nested under it too, its own
pretrain config moves to `config/runs/<model_name>/pretrain.json` so the two stay visually
separate — e.g. `config/runs/mesae_v10_small/pretrain.json` produces
`output/mesae_v10_small/pretrain/`, and `config/runs/mesae_v10_small/finetune/learned/
BNCI2014001.json` produces `output/mesae_v10_small/finetune/learned/BNCI2014001_intra/`.
Edit the copy, not the template — see CLAUDE.md's "`config/` layout" section.

If a backbone gets re-pretrained (a second pretrain run under the same `model_name`),
name that new config `pretrain_<timestamp>.json` instead of overwriting `pretrain.json` --
mirrors `artifacts/config_<timestamp>.json`'s non-destructive run snapshot, so an older
pretrain config a live finetune overlay still points at (`base_config`) isn't clobbered.
