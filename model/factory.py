import torch
from model.Qtome.plugin import PLUGIN as QTOME_PLUGIN
from IO.dataset import resolve_canonical_channels

# Adding a model = implement model/<Name>/plugin.py (Trainer/Checker/Plotter + build_model,
# bundled into a BasePlugin) and register the PLUGIN instance here. No other shared file
# needs editing.
MODEL_REGISTRY = {
    'Qtome': QTOME_PLUGIN,
}


def checkpoint_build_config(config, mode='pretrain'):
    """Everything that decides the backbone's architecture, saved inside every pretrain checkpoint:
    a trained backbone is always rebuilt from its own checkpoint, never from a config file that
    may have been edited since it was trained."""
    model_type = config['training_params'][mode].get('model_type', 'Qtome')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    canonical_channels = config.get('preprocess_params', {}).get('canonical_channels')
    if not canonical_channels:
        raise ValueError("preprocess_params.canonical_channels must be set (it fixes the channel count)")
    return {'model_type': model_type, 'num_channels': len(resolve_canonical_channels(canonical_channels)),
            'model_params': config['model_params'][model_type]['pretrain']}


def build_pretrain_from_config(config, mode='pretrain'):
    """A fresh, untrained backbone -- training starts here; a trained one comes from build_from_checkpoint."""
    bc = checkpoint_build_config(config, mode)
    return MODEL_REGISTRY[bc['model_type']].build(bc['model_params'], bc['num_channels'])


def build_from_checkpoint(ckpt):
    """The backbone exactly as trained: built from ckpt['build_config'], weights loaded (the load
    restores its phase flags, Qtome _restore_phase)."""
    if 'build_config' not in ckpt:
        raise ValueError("checkpoint has no build_config: trained before the 2026-09-26 encoder redesign, "
                         "the current code cannot rebuild it")
    bc = ckpt['build_config']
    model = MODEL_REGISTRY[bc['model_type']].build(bc['model_params'], bc['num_channels'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


def optimizer_param_groups(model, weight_decay):
    """Splits model.parameters() into decay/no-decay groups for AdamW — ndim<=1 params
    (LayerNorm/RMSNorm weights, every bias) get weight_decay=0.0, everything else (linear/
    conv/embedding matrices, AtomBank's W_down/W_out/u, ...) gets the configured decay.
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


def load_backbone(config, checkpoint_path=None, mode='finetune'):
    """Frozen-backbone loader shared by the finetune builders and the feature cache."""
    path = checkpoint_path or config['training_params'][mode]['pretrained_checkpoint']
    return build_from_checkpoint(torch.load(path, map_location='cpu', weights_only=False))


def build_finetune_from_config(config, num_classes, mode='finetune', num_patches=None, channel_idx=None):
    """
    Builds the pretrained backbone (via build_pretrain_from_config, same config section),
    loads its checkpoint, and wraps it in the classification head (dispatched via
    MODEL_REGISTRY the same way build_pretrain_from_config dispatches its backbone).
    num_classes is dataset-dependent (label set size) so it can't be read from config —
    pass it in. num_patches is likewise dataset-dependent (trial length varies by dataset,
    independent of preprocess_params.window_length, which is a pretrain-only concept) --
    only required by heads whose parameter shapes depend on the patch axis length (e.g.
    the head's time_pool='learned'/'none'); every other head ignores it.
    channel_idx: indices of the real (non-padded) channels the head's spatial filter acts on.
    """
    train_params = config['training_params'][mode]
    model_type   = train_params.get('model_type', 'Qtome')
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    plugin = MODEL_REGISTRY[model_type]

    backbone = load_backbone(config, mode=mode)

    canonical_channels = resolve_canonical_channels(config['preprocess_params']['canonical_channels'])
    ft_params = config['model_params'][model_type].get('finetune', {})
    # The whole finetune block is passed through; the plugin picks what its head takes.
    return plugin.finetune_cls(
        backbone, len(canonical_channels), num_classes,
        sample_freq=config['preprocess_params']['sample_freq'], num_patches=num_patches,
        channel_idx=channel_idx, **ft_params,
    )


def load_finetune_checkpoint(config, path, device):
    """Rebuild a finetune model from a head checkpoint: backbone from ckpt['backbone_checkpoint'],
    head from ckpt['head_config'] (no shape inference)."""
    from model.Qtome.Qtome import FinetuneModel
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    backbone = load_backbone(config, ckpt['backbone_checkpoint'])
    return FinetuneModel.from_checkpoint(backbone, ckpt).to(device).eval()
