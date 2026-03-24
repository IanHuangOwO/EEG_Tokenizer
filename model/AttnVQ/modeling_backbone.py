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
        **kwargs
    ):
        super().__init__()
        
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
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, coords, time_indices=None, bool_masked_pos=None):
        """
        Input x: (Batch, Channels, Patches, Time)
        Input coords: (Batch, Channels, 3)
        Input time_indices: (Batch, Patches) or (Patches,)
        Returns: Contextualized latent features z (Batch, Tokens, Dim)
        """
        B, C, P, T = x.shape
        
        # 1. Local Patch Embedding (Gated Multi-Scale Fusion)
        # SpatialTemporalEncoder returns (B*P, C, D)
        z_projected = self.spatial_temporal_encoder(x, coords, time_idx=time_indices)
        
        # 2. Reshape to (Batch, Tokens, Dim) where Tokens = C * P
        # Split (B*P) back to (B, P) -> (B, P, C, D)
        z = z_projected.reshape(B, P, C, -1)
        # Permute to (B, C, P, D) then flatten to (B, C*P, D)
        z = z.permute(0, 2, 1, 3).reshape(B, C * P, -1)
        
        # 3. Masking
        if bool_masked_pos is not None:
            m = bool_masked_pos.unsqueeze(-1)
            z = z.masked_fill(m, 0.0) + self.mask_token * m.type_as(z)
            
        # 4. Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        z = torch.cat((cls_tokens, z), dim=1) 
        
        # 5. Global Context Pass
        z = self.encoder(z)
        z = self.norm(z)
        
        # 6. Output Tokens: (B, C*P, D) - Excluding CLS token
        return z[:, 1:, :]
