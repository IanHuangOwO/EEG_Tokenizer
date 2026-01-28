import sys
import os
import shutil
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer
from model.RecurrentVQ.modeling_tokenizer import RecurrentVQTokenizer
from model.RecurrentFSQ.modeling_tokenizer import RecurrentFSQTokenizer
from model.LaBraM.modeling_tokenizer import LaBraMTokenizer

from model.NeuroRVQ.preprocessing import NeuroRVQProcessing
from model.RecurrentVQ.preprocessing import RecurrentVQProcessing
from model.LaBraM.preprocessing import LaBraMProcessing

def build_model_from_config(config, src_output_dir=None):
    """
    Builds the tokenizer model based on the provided configuration dictionary.
    Optionally copies the model source code to src_output_dir for reproducibility.
    """
    train_params = config['training_params']
    model_params = config['model_params']
    preprocess_params = config.get('preprocess_params', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })
    
    model_type = train_params.get('model_type', 'NeuroRVQ')
    
    if model_type == "NeuroRVQ":
        params = model_params['NeuroRVQ']
        model = NeuroRVQTokenizer(
            in_chans=params.get('in_chans', 1),
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            enc_heads=params['enc_heads'],
            enc_mlp_ratio=params.get('enc_mlp_ratio', 4.0),
            dec_depth=params['dec_depth'],
            dec_heads=params.get('dec_heads', params['enc_heads']),
            dec_mlp_ratio=params.get('dec_mlp_ratio', 4.0),
            num_scales=params.get('num_scales', 4),
            vocab_size=params['vocab_size'],
            n_codebooks=params['num_codebooks'],
            freq_resolution=params.get('freq_resolution', 1.0),
            min_freq=params.get('min_freq', 0.0),
            max_freq=params.get('max_freq', 100.0),
            fs=preprocess_params['target_freq']
        )
    elif model_type == "RecurrentVQ":
        params = model_params['RecurrentVQ']
        model = RecurrentVQTokenizer(
            in_chans=params.get('in_chans', 1),
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            enc_heads=params['enc_heads'],
            enc_mlp_ratio=params.get('enc_mlp_ratio', 4.0),
            dec_depth=params['dec_depth'],
            dec_heads=params.get('dec_heads', params['enc_heads']),
            dec_mlp_ratio=params.get('dec_mlp_ratio', 4.0),
            num_scales=params.get('num_scales', 4),
            vocab_size=params['vocab_size'],
            num_recurrent_steps=params['num_recurrent_steps'],
            freq_resolution=params.get('freq_resolution', 1.0),
            min_freq=params.get('min_freq', 0.0),
            max_freq=params.get('max_freq', 100.0),
            fs=preprocess_params['target_freq']
        )
    elif model_type == "RecurrentFSQ":
        params = model_params['RecurrentFSQ']
        model = RecurrentFSQTokenizer(
            in_chans=params.get('in_chans', 1),
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            enc_heads=params['enc_heads'],
            enc_mlp_ratio=params.get('enc_mlp_ratio', 4.0),
            dec_depth=params['dec_depth'],
            dec_heads=params.get('dec_heads', params['enc_heads']),
            dec_mlp_ratio=params.get('dec_mlp_ratio', 4.0),
            num_scales=params.get('num_scales', 4),
            num_recurrent_steps=params['num_recurrent_steps'],
            fsq_levels=params.get('fsq_levels', [8, 5, 5, 5]),
            freq_resolution=params.get('freq_resolution', 1.0),
            min_freq=params.get('min_freq', 0.0),
            max_freq=params.get('max_freq', 100.0),
            fs=preprocess_params['target_freq']
        )
    elif model_type == "LaBraM":
        params = model_params['LaBraM']
        model = LaBraMTokenizer(
            in_chans=1,
            embed_dim=params['embed_dim'],
            enc_depth=params['enc_depth'],
            dec_depth=params['dec_depth'],
            n_code=params['vocab_size']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save source code for reproducibility
    if src_output_dir is not None:
        os.makedirs(src_output_dir, exist_ok=True)
        model_src_path = sys.modules[model.__module__].__file__
        shutil.copy(model_src_path, os.path.join(src_output_dir, 'modeling_tokenizer.py'))
            
    return model

def build_preprocessing_from_config(config):
    """
    Factory function to build the appropriate preprocessing transform.
    """
    train_params = config['training_params']
    preprocess_params = config.get('preprocess_params', {
        'target_freq': 200, 'l_freq': 0.1, 'h_freq': 80.0, 'normalization_type': 'zscore'
    })
    
    # Ensure metadata is present
    if 'data_metadata' not in config or 'Sample_Frequency' not in config['data_metadata']:
        raise ValueError("Config must contain 'data_metadata' with 'Sample_Frequency' to build preprocessing.")
        
    fs_orig = config['data_metadata']['Sample_Frequency']
    model_type = train_params.get('model_type', 'NeuroRVQ')
    
    if model_type == 'RecurrentVQ' or model_type == 'RecurrentFSQ':
        return RecurrentVQProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )
    elif model_type == 'LaBraM':
        return LaBraMProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )
    else: # Default NeuroRVQ
        return NeuroRVQProcessing(
            original_freq=fs_orig, 
            target_freq=preprocess_params['target_freq'], 
            l_freq=preprocess_params['l_freq'], 
            h_freq=preprocess_params['h_freq'], 
            normalization_type=preprocess_params['normalization_type']
        )