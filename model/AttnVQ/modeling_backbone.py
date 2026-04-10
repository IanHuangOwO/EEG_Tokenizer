import torch
import torch.nn as nn
from model.AttnVQ.modeling_tokenizer import SpatialTemporalEncoder, Encoder

# --- Backbone Model (Encoder-Only) ---

class AttnVQBackbone(nn.Module):
    """
    The Global Foundation Model (Backbone).
    Reuses the SpatialTemporalEncoder from the Tokenizer as the patch embedder.
    Learns global trial context through a deep Transformer stack.
    """
    def __init__(
        self,
        embed_dim=256,
        enc_depth=12,
        enc_heads=8,
        mlp_ratio=4.,
        in_chans=1,
        in_scales=3,
        num_heads=16,
        vq_head_vocab_size=8,
        num_discrete=3,
        **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.vq_head_vocab_size = vq_head_vocab_size
        self.num_discrete = num_discrete
        
        # Total subspace dimensions to predict
        self.total_sub_dim = num_heads * vq_head_vocab_size
        
        # 1. Unified Spatial-Temporal encoding (Patch Embedder)
        self.spatial_temporal_encoder = SpatialTemporalEncoder(
            in_chans=in_chans, 
            in_scales=in_scales,
            base_filters=16, 
            embed_dim=embed_dim
        )
        
        # 2. Foundation Model Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Global Transformer Stack (The "Brain")
        self.encoder = Encoder(
            embed_dim=embed_dim, 
            depth=enc_depth, 
            heads=enc_heads, 
            mlp_ratio=mlp_ratio
        )
        
        # 4. Classification Head (Multi-Head categorical prediction)
        self.head = nn.Linear(embed_dim, self.total_sub_dim * num_discrete)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, coords, time_indices=None, bool_masked_pos=None):
        """
        Input x: (Batch, Channels, Patches, Time)
        Input coords: (Batch, Channels, 3)
        Input time_indices: (Batch, Patches) or (Patches,)
        Returns: Logits (Batch, Tokens, total_sub_dim, num_discrete)
        """
        B, C, P, T = x.shape
        
        # 1. Local Patch Embedding (Gated Multi-Scale Fusion)
        # Returns z_projected: (Batch * Patches, Channels, Dim)
        z = self.spatial_temporal_encoder(
            x, coords, time_idx=time_indices, 
            mask=bool_masked_pos, 
            mask_token=self.mask_token
        )
        
        # 2. Local Context Pass (Across Channels only, per patch)
        # Sequence Length = Channels (C)
        z = self.encoder(z) # (B*P, C, D)
        
        # 3. Prediction Head: (B*P, C, D) -> (B*P, C, total_sub_dim * N)
        logits = self.head(z)
        
        # 4. Reshape to (B*P, C, total_sub_dim, N)
        logits = logits.reshape(B * P, C, self.total_sub_dim, self.num_discrete)
        
        # 5. Reshape to Trial structure: (Batch, Channels, Patches, total_sub_dim, N)
        # (B*P, C, ...) -> (B, P, C, ...)
        logits = logits.reshape(B, P, C, self.total_sub_dim, self.num_discrete)
        
        # Permute to (B, C, P, ...) then flatten to (Batch, Tokens, ...)
        # where Tokens = Channels * Patches
        logits = logits.permute(0, 2, 1, 3, 4).reshape(B, C * P, self.total_sub_dim, self.num_discrete)
        
        return logits.contiguous()
