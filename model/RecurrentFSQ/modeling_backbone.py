import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.RecurrentFSQ.modeling_tokenizer import MultiScaleTemporalEncoder, TransformerEncoder

# --- Separate Block Classes for Clarity ---

class EncoderBlock(nn.Module):
    """
    Detailed Transformer Encoder Block.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
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
        # x: (Batch, SeqLen, Dim)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x

class DecoderBlock(nn.Module):
    """
    Detailed Transformer Decoder Block. 
    In MAE, this is similar to Encoder but often used in a shallower stack.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4., dropout=0.1):
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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x

# --- Patch Embedding ---

class RecurrentFSQPatchEmbed(nn.Module):
    """
    Converts raw EEG signal into patches using the RecurrentFSQ Multi-Scale approach.
    """
    def __init__(self, in_chans=1, embed_dim=200):
        super().__init__()
        self.temporal_encoder = MultiScaleTemporalEncoder(in_chans, embed_dim)
        
    def forward(self, x):
        # x: (Batch, Channels, Patches, Time=200) or 5D
        if x.dim() == 5:
            B, C, P, F, T = x.shape
            x = rearrange(x, 'b c p f t -> (b p) c f t')
        else:
            B, C, P, T = x.shape
            x = rearrange(x, 'b c p t -> (b p) c t')
        
        # Extract features (List of [BP, C, D])
        ms_features = self.temporal_encoder(x)
        
        # Sum multi-scale features (standard RecurrentFSQ approach)
        x_feat = sum(ms_features) # (BP, C, D)
        
        # Reshape back to (Batch, Total_Tokens, Dim)
        x_feat = rearrange(x_feat, '(b p) c d -> b (c p) d', b=B, p=P)
        
        return x_feat

# --- Main Foundation Model ---

class RecurrentFSQBackbone(nn.Module):
    def __init__(
        self,
        embed_dim=200,
        enc_depth=12,
        enc_heads=10,
        dec_depth=4,
        dec_heads=10,
        vocab_size=1000, # Approx for [8,5,5,5]
        in_scales=4,
        num_recurrent_steps=8,
        dropout=0.1,
        in_chans=1
    ):
        super().__init__()
        
        # 1. Patch Embedding
        self.patch_embed = RecurrentFSQPatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        
        # 2. Special Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Position & Channel Embeddings
        self.spatial_embed = nn.Parameter(torch.zeros(1, 128, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        
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
        
        self.total_codes = in_scales * num_recurrent_steps
        
        # 6. Prediction Heads
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size) for _ in range(self.total_codes)
        ])
        
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.spatial_embed, std=.02)
        torch.nn.init.normal_(self.temporal_embed, std=.02)

    def forward(self, x, channel_indices, bool_masked_pos=None):
        """
        x: (Batch, Channels, Patches, 200)
        channel_indices: (Batch, Channels)
        bool_masked_pos: (Batch, Channels * Patches) - True for masked
        """
        if x.dim() == 5:
            B, C, P, F, T = x.shape
        else:
            B, C, P, T = x.shape
        
        # 1. Embed Patches
        x = self.patch_embed(x) # (B, C*P, D)
        
        # 2. Add Positional/Spatial Information
        s_emb = self.spatial_embed[0, channel_indices, :] 
        s_emb = s_emb.unsqueeze(2).expand(-1, -1, P, -1).flatten(1, 2)
        
        t_emb = self.temporal_embed[:, :P, :].unsqueeze(1).expand(B, C, -1, -1).flatten(1, 2)
        
        x = x + s_emb + t_emb
        
        # 3. Masking
        if bool_masked_pos is not None:
            m = bool_masked_pos.unsqueeze(-1).type_as(x)
            x = x * (1 - m) + self.mask_token * m
            
        # 4. Encoder
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        
        # 5. Decoder
        for blk in self.decoder:
            x = blk(x)
        x = self.dec_norm(x)
        
        # Remove CLS token from predictions
        x_tokens = x[:, 1:, :] # (B, C*P, D)
        
        # 6. Prediction Heads
        logits = [head(x_tokens) for head in self.heads]
        
        return logits

def recurrent_fsq_base_patch200():
    return RecurrentFSQBackbone(
        embed_dim=200,
        enc_depth=12,
        enc_heads=10,
        dec_depth=4,
        vocab_size=1000,
        num_recurrent_steps=8
    )
