"""Finetune plan for the scale-up card: per backbone, pre-stamp probe (latent_signed, pca, spatial_k 8)
and the protocol stamp head (spatial_k 8), on BNCI2014004 / 001 / 008, loso and few-shot (Compass protocols incl.
_dataset_split: 004 session 3, 008 few-shot 5%), finetune seeds 1-3. Writes cell configs, prints plan lines.
Usage: FT_LABEL=scale python ft_plan.py <backbone> [<backbone> ...]  (FT_LABEL names the head folders)"""
import json
import os
import sys

LABEL = os.environ.get('FT_LABEL', 'scale')

TMPL = json.load(open('configs/finetune.template.json'))
CELLS = [('BNCI2014004', 'loso', 'mi_loso'), ('BNCI2014004', 'fewshot', 'mi_fewshot'),
         ('BNCI2014001', 'loso', 'mi_loso'), ('BNCI2014001', 'fewshot', 'mi_fewshot'),
         ('BNCI2014008', 'loso', 'p300_loso'), ('BNCI2014008', 'fewshot', 'p300_fewshot')]
PROBE = ("--set 'model_params.MeSAE.finetune.features=[{\"type\":\"latent_signed\",\"time_rank\":2,"
         "\"latent_proj\":\"pca\"}]' --set model_params.MeSAE.finetune.spatial_k=8")
STAMP = "--set model_params.MeSAE.finetune.spatial_k=8"   # the protocol's head, spatial_k 8 instead of 2 (user, 2026-09-28)
for bb in sys.argv[1:]:
    for ds, split, proto in CELLS:
        for head in ('probe', 'stamp'):
            for seed in (1, 2, 3):
                cell = f'{ds}_{split}_seed{seed}'
                c = {'base_config': f'configs/runs/{bb}/pretrain.json',
                     'dataset_params': {'finetune': {ds: {'dataset_path': f'datas/finetune/{ds}',
                                                         'subject_to_use': ['all'], 'channels_to_use': ['all']}}},
                     'training_params': {'finetune': dict(TMPL['training_params']['finetune'],
                        model_name=f'{head}_{cell}', protocol=proto, num_threads=8, seed=seed,
                        pretrained_checkpoint=f'output/{bb}/pretrain/checkpoint/last.pth',
                        output_path=f'{bb}/finetune/{LABEL}_{head}/{cell}'),
                        'visualize_params': TMPL['training_params']['visualize_params']}}
                d = f'configs/runs/{bb}/finetune/{LABEL}_{head}'
                os.makedirs(d, exist_ok=True)
                p = f'{d}/{cell}.json'
                json.dump(c, open(p, 'w'), indent=2)
                cmd = f'python train_finetune.py --config {p}' + (f' {PROBE}' if head == 'probe' else f' {STAMP}')
                print(f'job ft/{bb}/{head}/{cell} :: {cmd}')
