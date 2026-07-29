import torch
from model.MeFSQ.plugin import PLUGIN as MEFSQ_PLUGIN
from model.MeSAE.plugin import PLUGIN as MESAE_PLUGIN

# Adding a model = implement model/<Name>/plugin.py (Trainer/Checker/Plotter + build_model,
# bundled into a BasePlugin) and register the PLUGIN instance here. No other shared file
# needs editing — see docs/adr/0004-model-plugin-base-classes.md.
MODEL_REGISTRY = {
    'MeFSQ': MEFSQ_PLUGIN,
    'MeSAE': MESAE_PLUGIN,
}


def build_pretrain_from_config(config, mode='pretrain'):
    train_params = config['training_params'][mode]
    model_type   = train_params.get('model_type', 'MeFSQ')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    plugin = MODEL_REGISTRY[model_type]

    canonical_channels = config.get('preprocess_params', {}).get('canonical_channels')
    if not canonical_channels:
        raise ValueError(
            "preprocess_params.canonical_channels must be set — each Expert's/filter's "
            "decoder reconstructs all channels jointly (C*patch_len output) and needs a "
            "fixed channel count at model-construction time."
        )
    num_channels = len(canonical_channels)

    bp = config['model_params'][model_type]['pretrain']
    model = plugin.build(bp, num_channels)

    return model


def build_finetune_from_config(config, num_classes, mode='finetune'):
    """
    Builds the pretrained backbone (via build_pretrain_from_config, same config section),
    loads its checkpoint, and wraps it in the classification head (dispatched via
    MODEL_REGISTRY the same way build_pretrain_from_config dispatches its backbone).
    num_classes is dataset-dependent (label set size) so it can't be read from config —
    pass it in.
    """
    train_params = config['training_params'][mode]
    model_type   = train_params.get('model_type', 'MeFSQ')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    plugin = MODEL_REGISTRY[model_type]

    backbone = build_pretrain_from_config(config, mode=mode)
    ckpt_path = train_params['pretrained_checkpoint']
    state = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(state['model_state_dict'])
    plugin.trainer_cls().on_tokenizer_start(backbone)

    canonical_channels = config['preprocess_params']['canonical_channels']
    ft_params = config['model_params'][model_type].get('finetune', {})
    return plugin.finetune_cls(
        backbone, len(canonical_channels), num_classes,
        hidden=ft_params.get('hidden', 128),
        freeze_backbone=ft_params.get('freeze_backbone', False),
        dropout=ft_params.get('dropout', 0.1),
    )
