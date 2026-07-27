import sys
import os
import shutil
import torch
from model.MeFSQ.MeFSQ import MeFSQPretrain, MeFSQFinetune
from model.MeSAE.MeSAE import MeSAEPretrain, MeSAEFinetune


def build_pretrain_from_config(config, src_output_dir=None, mode='pretrain'):
    train_params = config['training_params'][mode]
    model_params = config['model_params']
    model_type   = train_params.get('model_type', 'MeFSQ')

    canonical_channels = config.get('preprocess_params', {}).get('canonical_channels')
    if not canonical_channels:
        raise ValueError(
            "preprocess_params.canonical_channels must be set — each Expert's/filter's "
            "decoder reconstructs all channels jointly (C*patch_len output) and needs a "
            "fixed channel count at model-construction time."
        )
    num_channels = len(canonical_channels)

    if model_type == 'MeFSQ':
        bp  = model_params['MeFSQ']['pretrain']
        moe = bp.get('moe', {})
        routed = moe.get('routed_expert', {})
        shared = moe.get('shared_expert', {})

        model = MeFSQPretrain(
            embed_dim=bp.get('embed_dim', 128),
            enc_depth=bp.get('enc_depth', 8),
            mlp_ratio=bp.get('mlp_ratio', 4.0),
            patch_len=bp.get('patch_len', 100),
            spatial_heads=bp.get('spatial_heads', 8),
            dropout=bp.get('dropout', 0.0),
            pool_after_blocks=bp.get('pool_after_blocks', []),
            upsample_residual_add=bp.get('upsample_residual_add', True),
            n_routed_experts=moe.get('n_routed_experts', 64),
            top_k=moe.get('top_k', 4),
            n_shared_experts=moe.get('n_shared_experts', 2),
            routed_r=routed.get('r', 10),
            routed_num_discrete=routed.get('num_discrete', 3),
            routed_decoder_hidden=routed.get('decoder_hidden'),
            shared_r=shared.get('r', 16),
            shared_num_discrete=shared.get('num_discrete', 5),
            shared_decoder_hidden=shared.get('decoder_hidden'),
            num_channels=num_channels,
        )
        module_filename = 'MeFSQ.py'

    elif model_type == 'MeSAE':
        bp  = model_params['MeSAE']['pretrain']
        sae = bp.get('sae', {})

        model = MeSAEPretrain(
            embed_dim=bp.get('embed_dim', 100),
            enc_depth=bp.get('enc_depth', 12),
            mlp_ratio=bp.get('mlp_ratio', 4.0),
            patch_len=bp.get('patch_len', 20),
            spatial_heads=bp.get('spatial_heads', 8),
            dropout=bp.get('dropout', 0.0),
            pool_after_blocks=bp.get('pool_after_blocks', []),
            upsample_residual_add=bp.get('upsample_residual_add', True),
            num_channels=num_channels,
            n_filters=sae.get('n_filters', 8),
            pool_hidden=sae.get('pool_hidden', 32),
            pool_temperature=sae.get('pool_temperature', 1.0),
            sae_expansion=sae.get('sae_expansion', 8),
            sae_k=sae.get('sae_k', 32),
            decoder_hidden=sae.get('decoder_hidden'),
        )
        module_filename = 'MeSAE.py'

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        shutil.copy(sys.modules[model.__module__].__file__, os.path.join(src_output_dir, module_filename))

    return model


def build_finetune_from_config(config, num_classes, src_output_dir=None, mode='finetune'):
    """
    Builds the pretrained backbone (via build_pretrain_from_config, same config section),
    loads its checkpoint, and wraps it in the classification head (MeFSQFinetune or
    MeSAEFinetune, dispatched the same way build_pretrain_from_config dispatches its
    backbone). num_classes is dataset-dependent (label set size) so it can't be read from
    config — pass it in.
    """
    train_params = config['training_params'][mode]
    model_type   = train_params.get('model_type', 'MeFSQ')
    if model_type not in ('MeFSQ', 'MeSAE'):
        raise ValueError(f"Unknown model type: {model_type}")

    backbone = build_pretrain_from_config(config, src_output_dir=src_output_dir, mode=mode)
    ckpt_path = train_params['pretrained_checkpoint']
    state = torch.load(ckpt_path, map_location='cpu')
    backbone.load_state_dict(state['model_state_dict'])
    backbone.enable_spatial()
    if hasattr(backbone, 'enable_temporal'):
        backbone.enable_temporal()

    canonical_channels = config['preprocess_params']['canonical_channels']
    ft_params = config['model_params'][model_type].get('finetune', {})
    finetune_cls = MeFSQFinetune if model_type == 'MeFSQ' else MeSAEFinetune
    return finetune_cls(
        backbone, len(canonical_channels), num_classes,
        hidden=ft_params.get('hidden', 128),
        freeze_backbone=ft_params.get('freeze_backbone', False),
        dropout=ft_params.get('dropout', 0.1),
    )
