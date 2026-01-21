import torch
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer
from model.RecurrentVQ.modeling_tokenizer import RecurrentVQTokenizer
from model.LaBraM.modeling_tokenizer import LaBraMTokenizer

def build_model_from_config(config):
    """
    Builds the tokenizer model based on the provided configuration dictionary.
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
        
    return model