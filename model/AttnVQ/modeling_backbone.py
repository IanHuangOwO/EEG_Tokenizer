import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.AttnVQ.modeling_tokenizer import SpatialTemporalEncoder, TransformerLayer

# --- Separate Block Classes for Clarity ---

class EncoderBlock(nn.Module):
    """
    Detailed Transformer Encoder Block.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.block = TransformerLayer(embed_dim, num_heads, mlp_ratio)

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    """
    Detailed Transformer Decoder Block. 
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
        super().__init__()
        self.block = TransformerLayer(embed_dim, num_heads, mlp_ratio)

    def forward(self, x):
        return self.block(x)

# --- Patch Embedding ---

class AttnVQPatchEmbed(nn.Module):
    """
    Converts raw EEG signal into patches using the new SpatialTemporalEncoder.
    """
    def __init__(self, in_chans=1, embed_dim=256, max_temporal_patches=10):
        super().__init__()
        self.encoder = SpatialTemporalEncoder(in_chans, base_filters=16, embed_dim=embed_dim, max_temporal_patches=max_temporal_patches)
        
    def forward(self, x, coords):
        # x: (Batch, Channels, Patches, Time=200)
        B, C, P, T = x.shape
        
        # Flatten patches into the batch dimension for the convolutional encoder
        x_reshaped = rearrange(x, 'b c p t -> (b p) c t')
        # Coords also need to be repeated for each patch
        coords_reshaped = coords.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, C, 3)
        
        # Extract multiscale features: List of [(B*P), C, embed_dim]
        ms_features = self.encoder(x_reshaped, coords_reshaped)
        
        # Sum multi-scale features for the foundation model input
        x_feat = sum(ms_features) # ((B*P), C, embed_dim)
        
        # Reshape back to (Batch, Total_Tokens, Dim)
        # Tokens = Channels * Patches
        x_feat = rearrange(x_feat, '(b p) c d -> b (c p) d', b=B, p=P)
        
        return x_feat

# --- Main Foundation Model ---

class AttnVQBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        enc_depth=12,
        enc_heads=8,
        dec_depth=4,
        dec_heads=8,
        vq_head_vocab_size=64, # Matches tokenizer r
        in_scales=3,
        vq_head_num=8,
        dropout=0.1,
        in_chans=1,
        max_temporal_patches=10
    ):
        super().__init__()
        self.in_scales, self.vq_head_num, self.vq_head_vocab_size = in_scales, vq_head_num, vq_head_vocab_size
        
        # 1. Patch Embedding (SpatialTemporal)
        self.patch_embed = AttnVQPatchEmbed(in_chans=in_chans, embed_dim=embed_dim, max_temporal_patches=max_temporal_patches)
        
        # 2. Special Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Positional Encoding
        # Since SpatialTemporalEncoder already adds spatial/temporal embeddings inside patch_embed,
        # the backbone's learnable embeddings are used for additional context if needed.
        self.pos_embed = nn.Parameter(torch.zeros(1, 128 * max_temporal_patches + 1, embed_dim))
        
        # 4. Encoder Stack
        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, enc_heads, dropout=dropout) 
            for _ in range(enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(embed_dim)
        
        # 5. Decoder Stack
        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, dec_heads, dropout=dropout)
            for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(embed_dim)
        
        # 6. Prediction Head: Predicting weights for all heads and scales
        # Vectorized for efficiency
        self.head = nn.Linear(embed_dim, in_scales * vq_head_num * vq_head_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def forward(self, x, coords, bool_masked_pos=None):
        """
        x: (Batch, Channels, Patches, Time)
        coords: (Batch, Channels, 3)
        bool_masked_pos: (Batch, Channels * Patches) - True for masked
        """
        B, C, P, T = x.shape
        
        # 1. Embed Patches (Includes Spatial/Temporal internal embeddings)
        x = self.patch_embed(x, coords) # (B, C*P, D)
        
        # 2. Add CLS and Positional Encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, C*P + 1, D)
        
        # Add position embeddings (matching current sequence length)
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # 3. Masking (Apply to tokens, not CLS)
        if bool_masked_pos is not None:
            # Shift masked pos because of CLS token
            m = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), bool_masked_pos], dim=1).unsqueeze(-1)
            x = x.masked_fill(m, 0.0) + self.mask_token * m.type_as(x)
            
        # 4. Encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        
        # 5. Decoder
        for blk in self.decoder:
            x = blk(x)
        x = self.dec_norm(x)
        
        # Remove CLS token from predictions
        x_tokens = x[:, 1:, :] # (B, C*P, D)
        
        # 6. Prediction Head (Vectorized)
        # logits: (B, C*P, S*H*r)
        logits = self.head(x_tokens) 
        
        # Reshape and Permute to (S, H, B, Tokens, r)
        # Tokens = C * P
        logits = logits.view(B, -1, self.in_scales, self.vq_head_num, self.vq_head_vocab_size)
        logits = logits.permute(2, 3, 0, 1, 4)
        
        return logits
