import sys
import os
import shutil
from model.MeFSQ.MeFSQ import MeFSQPretrain


def build_model_from_config(config, src_output_dir=None, mode='pretrain'):
    train_params = config['training_params'][mode]
    model_params = config['model_params']
    model_type   = train_params.get('model_type', 'MeFSQ')

    if model_type != 'MeFSQ':
        raise ValueError(f"Unknown model type: {model_type}")

    bp  = model_params['MeFSQ']['pretrain']
    moe = bp.get('moe', {})
    routed = moe.get('routed_expert', {})
    shared = moe.get('shared_expert', {})

    canonical_channels = config.get('preprocess_params', {}).get('canonical_channels')
    if not canonical_channels:
        raise ValueError(
            "preprocess_params.canonical_channels must be set — the fused per-patch VQ "
            "(model/MeFSQ/MeFSQ.py) concatenates all channels into one token and needs a "
            "fixed channel count at model-construction time."
        )
    num_channels = len(canonical_channels)

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

    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        shutil.copy(sys.modules[model.__module__].__file__, os.path.join(src_output_dir, 'MeFSQ.py'))

    return model
