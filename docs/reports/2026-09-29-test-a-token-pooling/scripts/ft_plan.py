"""z-probe finetune plan for the patch-length card: per backbone, latent_signed (pca, spatial_k 8) on
BNCI2014004 / 001 / 008 x loso / few-shot (Compass protocols), finetune seeds 1-3. Writes cell configs under
configs/runs/<bb>/finetune/patch_probe_pool2/, prints run_queue plan lines."""
import json, os, sys

TMPL = json.load(open('configs/finetune.template.json'))
CELLS = [('BNCI2014004', 'loso', 'mi_loso'), ('BNCI2014004', 'fewshot', 'mi_fewshot'),
         ('BNCI2014001', 'loso', 'mi_loso'), ('BNCI2014001', 'fewshot', 'mi_fewshot'),
         ('BNCI2014008', 'loso', 'p300_loso'), ('BNCI2014008', 'fewshot', 'p300_fewshot')]
PROBE = ("--set 'model_params.MeSAE.finetune.features=[{\"type\":\"latent_signed\",\"time_rank\":2,"
         "\"latent_proj\":\"pca\"}]' --set model_params.MeSAE.finetune.spatial_k=8")
bb = sys.argv[1]
for ds, split, proto in CELLS:
    for seed in (1, 2, 3):
        cell = f'{ds}_{split}_seed{seed}'
        c = {'base_config': f'configs/runs/{bb}/pretrain.json',
             'dataset_params': {'finetune': {ds: {'dataset_path': f'datas/finetune/{ds}',
                                                 'subject_to_use': ['all'], 'channels_to_use': ['all']}}},
             'training_params': {'finetune': dict(TMPL['training_params']['finetune'],
                model_name=f'probe_{cell}', protocol=proto, num_threads=6, seed=seed, latent_pool=2,
                pretrained_checkpoint=f'output/{bb}/pretrain/checkpoint/last.pth',
                output_path=f'{bb}/finetune/patch_probe_pool2/{cell}'),
                'visualize_params': TMPL['training_params']['visualize_params']}}
        d = f'configs/runs/{bb}/finetune/patch_probe_pool2'
        os.makedirs(d, exist_ok=True)
        json.dump(c, open(f'{d}/{cell}.json', 'w'), indent=2)
        print(f'job {cell} :: python train_finetune.py --config {d}/{cell}.json {PROBE}')
