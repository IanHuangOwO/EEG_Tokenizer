import torch
import torch.nn as nn
from model.AttnVQ.modeling_tokenizer import SpatialTemporalEncoder, TransformerLayer

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
        **kwargs
    ):
        super().__init__()
        
        # 1. Shared Front-end (The "Tokenizer's Encoder")
        self.patch_embed = SpatialTemporalEncoder(
            in_chans=in_chans, 
            base_filters=16, 
            embed_dim=embed_dim
        )
        
        # 2. Foundation Model Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Scale Parameters (Buffers - copied from Tokenizer)
        # These are required to project z into the Tokenizer's distribution space
        self.register_buffer('logit_scale', torch.ones(in_scales, 1, 1, num_heads, 1))
        self.register_buffer('head_weights', torch.zeros(in_scales, 1, 1, num_heads, 1))
        
        # 4. Global Transformer Stack (The "Brain")
        self.transformer = nn.ModuleList([
            TransformerLayer(embed_dim, enc_heads, mlp_ratio, dropout=0.0) 
            for _ in range(enc_depth)
        ])
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
        
        # 1. Local Patch Embedding
        ms_features = self.patch_embed(x, coords, time_idx=time_indices)
        
        # Sum multi-scale features
        x_feat = sum(ms_features) # (B*P, C, D)
        
        # Reshape to (Batch, Tokens, Dim) where Tokens = C * P
        # Instead of rearrange(x_feat, '(b p) c d -> b (c p) d', b=B, p=P)
        # 1. Split (B*P) back to (B, P) -> (B, P, C, D)
        x = x_feat.reshape(B, P, C, -1)
        # 2. Permute to (B, C, P, D) then flatten to (B, C*P, D)
        x = x.permute(0, 2, 1, 3).reshape(B, C * P, -1)
        
        # 2. Masking
        if bool_masked_pos is not None:
            m = bool_masked_pos.unsqueeze(-1)
            x = x.masked_fill(m, 0.0) + self.mask_token * m.type_as(x)
            
        # 3. Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        # 4. Global Context Pass
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        
        # 5. Output: (B, Tokens, D)
        z_tokens = x[:, 1:, :]
        
        return z_tokens
