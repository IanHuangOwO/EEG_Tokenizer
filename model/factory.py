import torch
from model.LaBraM.modeling_tokenizer import LaBraMTokenizer
from model.LaBraM.modeling_backbone import LaBraMBackbone
from model.NeuroRVQ.modeling_tokenizer import NeuroRVQTokenizer
from model.NeuroRVQ.modeling_backbone import NeuroRVQBackbone

def build_model_from_config(config, mode='backbone'):
    """
    Factory function to build a model (Backbone or Tokenizer) from configuration.
    
    Args:
        config (dict): The configuration dictionary.
        mode (str): 'backbone' or 'tokenizer'.
    """
    model_name = config.get('training_params', {}).get('model_name', 'NeuroRVQ')
    params = config.get('model_params', {}).get(model_name, {})
    
    # Common defaults
    embed_dim = params.get('embed_dim', 200)
    vocab_size = params.get('vocab_size', 8192)
    
    if model_name == 'LaBraM':
        if mode == 'tokenizer':
            return LaBraMTokenizer(
                embed_dim=embed_dim,
                enc_depth=params.get('enc_depth', 6), # Tokenizer usually shallower
                dec_depth=params.get('dec_depth', 6),
                n_code=vocab_size,
                code_dim=params.get('code_dim', 32)
            )
        else:
            return LaBraMBackbone(
                embed_dim=embed_dim,
                enc_depth=params.get('enc_depth', 12),
                enc_heads=params.get('enc_heads', 10),
                vocab_size=vocab_size,
                dropout=params.get('dropout', 0.1)
            )
            
    elif model_name == 'NeuroRVQ':
        if mode == 'tokenizer':
            return NeuroRVQTokenizer(
                embed_dim=embed_dim,
                enc_depth=params.get('enc_depth', 12),
                enc_heads=params.get('enc_heads', 10),
                dec_depth=params.get('dec_depth', 3),
                n_codebooks=params.get('num_codebooks', 8),
                vocab_size=vocab_size
            )
        else:
            return NeuroRVQBackbone(
                embed_dim=embed_dim,
                enc_depth=params.get('enc_depth', 12),
                enc_heads=params.get('enc_heads', 10),
                dec_depth=params.get('dec_depth', 4),
                vocab_size=vocab_size,
                num_codebooks=params.get('num_codebooks', 8),
                dropout=params.get('dropout', 0.1)
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
