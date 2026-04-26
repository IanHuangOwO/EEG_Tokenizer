import torch
import torch.nn as nn

# Import the updated modules from your tokenizer file
from model.AttnVQ.modeling_tokenizer import SpatialTemporalEmbeddings, HierarchicalTSAEncoder

# --- Backbone Model (Encoder-Only Foundation Model) ---

class AttnVQBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        enc_depth=12,
        enc_heads=8,
        mlp_ratio=4.0,
        in_chans=1,
        in_scales=200, # patch_len
        vq_head_num=8,
        vq_head_vocab_size=64,
        vq_num_discrete=5,
        merge_factors=[1, 2, 2],
        spatial_heads=8
    ):
        super().__init__()
        self.patch_len = in_scales
        self.vq_head_num, self.vq_head_vocab_size, self.num_discrete = vq_head_num, vq_head_vocab_size, vq_num_discrete
        self.total_sub_dim = vq_head_num * vq_head_vocab_size
        
        # 1. Pixel-Wise Embedding (Replaces Patch-Wise)
        self.embed = SpatialTemporalEmbeddings(embed_dim)
        
        # 2. Hierarchical Encoder
        self.encoder = HierarchicalTSAEncoder(
            embed_dim, depth=enc_depth, num_heads=enc_heads, mlp_ratio=mlp_ratio,
            merge_factors=merge_factors, apply_cross_dim=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        
        # 3. SPATIAL STAGE
        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, spatial_heads, batch_first=True)
        
        # 4. Prediction Head (Predicts discrete codes for each coarse pixel-token)
        self.head = nn.Linear(embed_dim, self.total_sub_dim * self.num_discrete)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, coords, bool_masked_pos=None):
        """
        x:               [B, C, N, L]
        coords:          [B, C, 3]
        bool_masked_pos: [B, C, T] (Now at pixel level!)
        """
        B, C, N, L = x.shape
        T = N * L
        
        # 1. Pixel-Wise Embedding -> [B, C, T, D]
        z = self.embed(x)
        
        # 2. Apply Masking (at pixel level)
        if bool_masked_pos is not None:
            # bool_masked_pos expected shape: [B, C, T]
            mask = bool_masked_pos.unsqueeze(-1).type_as(z)
            z = z * (1.0 - mask) + self.mask_token * mask
            
        # 3. Hierarchical Encode -> [B, C, T_coarse, D]
        z_temp = self.encoder(z)
        T_coarse = z_temp.shape[2]
        
        # 4. SPATIAL STAGE (Contextualize across physical channels)
        z_spatial = z_temp
        
        # Inject Spatial Coordinates
        s_emb = self.spatial_mlp(coords).unsqueeze(2) # [B, C, 1, D]
        z_spatial = z_spatial + s_emb
        
        # Attend over Channels -> [Batch * T_coarse, Channels, Dim]
        z_spatial = z_spatial.permute(0, 2, 1, 3).reshape(B * T_coarse, C, -1)
        
        norm_spatial = self.spatial_norm(z_spatial)
        attn_out, _ = self.spatial_attn(norm_spatial, norm_spatial, norm_spatial)
        z_spatial = z_spatial + attn_out
        
        # Back to [B, C, T_coarse, D]
        z_final = z_spatial.reshape(B, T_coarse, C, -1).permute(0, 2, 1, 3)
        
        # 5. Prediction -> [B, C, T_coarse, H, r, N_discrete]
        logits = self.head(z_final).reshape(
            B, C, T_coarse, self.vq_head_num, self.vq_head_vocab_size, self.num_discrete
        )
        return logits.contiguous()

    @torch.no_grad()
    def load_tokenizer_weights(self, tokenizer_state_dict):
        """ Loads pretrained Temporal and Embedding weights dynamically. """
        missing_keys, unexpected_keys = self.load_state_dict(tokenizer_state_dict, strict=False)
        print(f"Loaded Tokenizer Weights. Missing {len(missing_keys)} keys (Expected for Spatial Layer/Head)")
        return missing_keys
