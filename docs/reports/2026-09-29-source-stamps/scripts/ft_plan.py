"""Finetune plan for the source-stamps card: per backbone and head label (patch_probe = z probe, cw_stamp = stamp
head), BNCI2014004 / 001 / 008 x loso / few-shot, finetune seeds 1-3. Writes cell configs, prints run_queue lines."""
import json, os, sys

TMPL = json.load(open('configs/finetune.template.json'))
CELLS = [('BNCI2014004', 'loso', 'mi_loso'), ('BNCI2014004', 'fewshot', 'mi_fewshot'),
         ('BNCI2014001', 'loso', 'mi_loso'), ('BNCI2014001', 'fewshot', 'mi_fewshot'),
         ('BNCI2014008', 'loso', 'p300_loso'), ('BNCI2014008', 'fewshot', 'p300_fewshot')]
HEADS = {'patch_probe': ('probe', '[{"type":"latent_signed","time_rank":2,"latent_proj":"pca"}]'),
         'cw_stamp': ('stamp', '[{"type":"stamp_power","time_pool":"learned","time_rank":2}]')}
bb, head = sys.argv[1], sys.argv[2]
tag, feats = HEADS[head]
for ds, split, proto in CELLS:
    for seed in (1, 2, 3):
        cell = f'{ds}_{split}_seed{seed}'
        c = {'base_config': f'configs/runs/{bb}/pretrain.json',
             'dataset_params': {'finetune': {ds: {'dataset_path': f'datas/finetune/{ds}',
                                                 'subject_to_use': ['all'], 'channels_to_use': ['all']}}},
             'training_params': {'finetune': dict(TMPL['training_params']['finetune'],
                model_name=f'{tag}_{cell}', protocol=proto, num_threads=6, seed=seed,
                pretrained_checkpoint=f'output/{bb}/pretrain/checkpoint/last.pth',
                output_path=f'{bb}/finetune/{head}/{cell}'),
                'visualize_params': TMPL['training_params']['visualize_params']}}
        d = f'configs/runs/{bb}/finetune/{head}'
        os.makedirs(d, exist_ok=True)
        json.dump(c, open(f'{d}/{cell}.json', 'w'), indent=2)
        print(f"job {cell} :: python train_finetune.py --config {d}/{cell}.json "
              f"--set 'model_params.MeSAE.finetune.features={feats}' --set model_params.MeSAE.finetune.spatial_k=8 "
              f"--set model_params.MeSAE.finetune.dropout=0.3")
