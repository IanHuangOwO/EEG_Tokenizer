import torch
import torch.nn as nn
from einops import rearrange
from model.LaBraM.modeling_tokenizer import LaBraMTemporalConv

# --- Standard Blocks (Can be shared or duplicated for independence) ---

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class LaBraMBackbone(nn.Module):
    """
    Clean implementation of LaBraM Masked Autoencoder Backbone.
    """
    def __init__(
        self,
        embed_dim=200,
        enc_depth=12,
        enc_heads=10,
        vocab_size=8192,
        patch_size=200,
        dropout=0.1
    ):
        super().__init__()
        
        # 1. Patch Embedding
        self.patch_embed = LaBraMTemporalConv(in_chans=1, out_chans=8)
        
        # 2. Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Max 128 channels, Max 16 seconds
        self.spatial_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        
        # 3. Encoder (LaBraM typically uses only an Encoder for MAE prediction)
        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, enc_heads, dropout=dropout)
            for _ in range(enc_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Prediction Head
        # LaBraM predicts token IDs directly from the encoder output
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.spatial_embed, std=.02)
        torch.nn.init.normal_(self.temporal_embed, std=.02)

    def forward(self, x, channel_indices, bool_masked_pos=None):
        """
        x: (Batch, Channels, Patches, 200)
        """
        B, C, P, T = x.shape
        
        # 1. Patch Embedding
        # Treat each patch as a group of channels: (Batch * Patches, Channels, Time)
        x_patch = rearrange(x, 'b c p t -> (b p) c t')
        
        # Embed: (BP, C, D)
        x_emb = self.patch_embed(x_patch) 
        
        # Reshape to (Batch, Total_Tokens, Dim)
        # Total_Tokens = Channels * Patches
        x_emb = rearrange(x_emb, '(b p) c d -> b (c p) d', b=B, p=P)
        
        # 2. Add Positional Embeddings
        # Spatial (B, C, D)
        s_emb = self.spatial_embed[0, channel_indices, :] 
        # Expand spatial over patches: (B, C, 1, D) -> (B, C, P, D) -> (B, CP, D)
        s_emb = s_emb.unsqueeze(2).expand(-1, -1, P, -1).flatten(1, 2)
        
        # Temporal (1, P, D)
        t_emb = self.temporal_embed[:, :P, :].unsqueeze(1).expand(B, C, -1, -1).flatten(1, 2)
        
        x = x_emb + s_emb + t_emb
        
        # Masking
        if bool_masked_pos is not None:
            m = bool_masked_pos.unsqueeze(-1).type_as(x)
            x = x * (1 - m) + self.mask_token * m
            
        # Add CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        
        # Predict
        # Remove CLS
        logits = self.head(x[:, 1:, :])
        
        return logits

def labram_base_patch200():
    return LaBraMBackbone(
        embed_dim=200,
        enc_depth=12,
        enc_heads=10,
        vocab_size=8192
    )
