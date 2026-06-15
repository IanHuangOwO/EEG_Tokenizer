import sys
import os
import shutil
from model.MeFSQ.MeFSQ import MeFSQPretrain


def build_model_from_config(config, src_output_dir=None):
    train_params = config['training_params']
    model_params = config['model_params']
    model_type   = train_params.get('model_type', 'MeFSQ')

    if model_type != 'MeFSQ':
        raise ValueError(f"Unknown model type: {model_type}")

    bp = model_params['MeFSQ']['pretrain']

    model = MeFSQPretrain(
        embed_dim=bp.get('embed_dim', 128),
        enc_depth=bp.get('enc_depth', 8),
        enc_heads=bp.get('enc_heads', 8),
        mlp_ratio=bp.get('mlp_ratio', 4.0),
        patch_len=bp.get('patch_len', 100),
        vq_head_num=bp.get('vq_head_num', 64),
        vq_head_vocab_size=bp.get('vq_head_vocab_size', 16),
        vq_num_discrete=bp.get('vq_num_discrete', 5),
        spatial_heads=bp.get('spatial_heads', 8),
        stage_indices=bp.get('stage_indices', None),
    )

    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        shutil.copy(sys.modules[model.__module__].__file__, os.path.join(src_output_dir, 'MeFSQ.py'))

    return model

