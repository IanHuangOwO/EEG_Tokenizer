import sys
import os
import shutil
from model.AttnVQ.modeling_tokenizer import AttnVQTokenizer
from model.AttnVQ.modeling_backbone import AttnVQBackbone

from model.AttnVQ.preprocessing import AttnVQProcessing

def build_model_from_config(config, n_fft_trial=None, src_output_dir=None):
    """
    Builds the tokenizer model based on the provided configuration dictionary.
    n_fft_trial: trial-level FFT size (num_patches * patch_len), computed from the dataset.
    Optionally copies the model source code to src_output_dir for reproducibility.
    """
    train_params = config['training_params']
    model_params = config['model_params']
    model_type = train_params.get('model_type', 'AttnVQ')

    preprocess_params = model_params[model_type].get('preprocess', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })

    if model_type == "AttnVQ":
        params = model_params['AttnVQ']['tokenizer']
        model = AttnVQTokenizer(
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            enc_heads=params['enc_heads'],
            enc_mlp_ratio=params.get('enc_mlp_ratio', 4.0),
            patch_len=params.get('patch_len', 25),
            n_fft_trial=n_fft_trial,
            fs=preprocess_params['target_freq'],
            decoder_heads_config=params.get('decoder_heads_config', None),
            use_spatial_embedding=params.get('use_spatial_embedding', False)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save source code for reproducibility
    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        model_src_path = sys.modules[model.__module__].__file__
        shutil.copy(model_src_path, os.path.join(src_output_dir, 'modeling_tokenizer.py'))
            
    return model

def build_backbone_from_config(config, src_output_dir=None):
    """
    Builds the backbone model based on the provided configuration dictionary.
    """
    train_params = config['training_params']
    model_params = config['model_params']
    
    model_type = train_params.get('model_type', 'AttnVQ')
    
    if model_type == "AttnVQ":
        tokenizer_params = model_params['AttnVQ']['tokenizer']
        backbone_params = model_params['AttnVQ']['backbone']
        
        model = AttnVQBackbone(
            embed_dim=backbone_params.get('embed_dim', tokenizer_params['embed_dim']),
            enc_depth=backbone_params.get('enc_depth', 12),
            enc_heads=backbone_params.get('enc_heads', 8),
            mlp_ratio=backbone_params.get('mlp_ratio', 4.0),
            patch_len=backbone_params.get('patch_len', tokenizer_params.get('patch_len', 3)),
            vq_head_num=backbone_params.get('vq_head_num', tokenizer_params.get('vq_head_num', 16)),
            vq_head_vocab_size=backbone_params.get('vq_head_vocab_size', tokenizer_params.get('vq_head_vocab_size', 8)),
            vq_num_discrete=backbone_params.get('vq_num_discrete', tokenizer_params.get('vq_num_discrete', 5)),
            spatial_heads=backbone_params.get('spatial_heads', 8)
            )
    else:
        raise ValueError(f"Backbone not implemented for model type: {model_type}")

    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        model_src_path = sys.modules[model.__module__].__file__
        shutil.copy(model_src_path, os.path.join(src_output_dir, 'modeling_backbone.py'))

    return model

def build_preprocessing_from_config(config, fs_orig=None):
    """
    Factory function to build the appropriate preprocessing transform.
    If fs_orig is not provided, it tries to pull from config['data_metadata'].
    """
    train_params = config['training_params']
    model_params = config['model_params']
    model_type = train_params.get('model_type', 'AttnVQ')
    
    preprocess_params = model_params[model_type].get('preprocess', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })
    
    if fs_orig is None:
        # Fallback to metadata in config
        if 'data_metadata' not in config or 'Sample_Frequency' not in config['data_metadata']:
            # Instead of raising error, we can return None or a generic transform if needed, 
            # but raising is safer for now.
            raise ValueError("Config must contain 'data_metadata' with 'Sample_Frequency' to build preprocessing if fs_orig is not provided.")
        fs_orig = config['data_metadata']['Sample_Frequency']
        
    # Preprocessing Arguments
    p_args = {
        'original_freq': fs_orig, 
        'target_freq': preprocess_params['target_freq'], 
        'l_freq': preprocess_params['l_freq'], 
        'h_freq': preprocess_params['h_freq'], 
        'normalization_type': preprocess_params['normalization_type']
    }

    if model_type == 'AttnVQ':
        return AttnVQProcessing(**p_args)
