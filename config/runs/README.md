# config/runs/

A real run's config, never the templates directly. To start a new run:

```bash
cp config/config.template.json config/runs/<model_name>.json          # pretrain
cp config/config.template.json config/runs/<backbone>/finetune/<head>.json  # finetune
```

Path mirrors `output/`: `config/runs/mesae_v10_small.json` produces
`output/mesae_v10_small/`; `config/runs/mesae_v10_small/finetune/learned.json` produces
`output/mesae_v10_small/finetune/learned/...`. Edit the copy, not the template — see
CLAUDE.md's "`config/` layout" section.
