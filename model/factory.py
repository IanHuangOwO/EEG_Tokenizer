import torch
from model.MeFSQ.plugin import PLUGIN as MEFSQ_PLUGIN
from model.MeSAE.plugin import PLUGIN as MESAE_PLUGIN
from IO.dataset import resolve_canonical_channels

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
    canonical_channels = resolve_canonical_channels(canonical_channels)
    num_channels = len(canonical_channels)

    # mode='tokenizer' uses its own architecture block when the model config declares one
    # (lets the Tokenizer stage run a shallower/differently-pooled encoder than Pretrain,
    # see docs/adr/0003-mesae-two-stage-masked-training.md) — falls back to 'pretrain' for
    # any model_type that hasn't split one out yet. mode='finetune' always uses 'pretrain':
    # model_params.<type>.finetune is a different schema entirely (head params like
    # hidden/freeze_backbone, not an architecture block) and build_finetune_from_config
    # calls through here with mode='finetune' to build the backbone, not the head.
    if mode == 'tokenizer':
        bp = config['model_params'][model_type].get('tokenizer') or config['model_params'][model_type]['pretrain']
    else:
        bp = config['model_params'][model_type]['pretrain']
    model = plugin.build(bp, num_channels)

    return model


def optimizer_param_groups(model, weight_decay):
    """Splits model.parameters() into decay/no-decay groups for AdamW — ndim<=1 params
    (LayerNorm/RMSNorm weights, every bias) get weight_decay=0.0, everything else (linear/
    conv/embedding matrices, StampBank's W_down/W_out/u, ...) gets the configured decay.
    Standard BERT/ViT-style recipe: decaying a norm's gain or a bias toward zero fights the
    norm's job and has no overfitting-prevention upside (biases have no capacity to
    memorize on their own), so excluding them is a strict improvement, not a tunable
    tradeoff — unlike the decay *value* itself, which does trade fit against generalization."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


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

    canonical_channels = resolve_canonical_channels(config['preprocess_params']['canonical_channels'])
    ft_params = config['model_params'][model_type].get('finetune', {})
    return plugin.finetune_cls(
        backbone, len(canonical_channels), num_classes,
        hidden=ft_params.get('hidden', 128),
        freeze_backbone=ft_params.get('freeze_backbone', False),
        dropout=ft_params.get('dropout', 0.1),
    )
