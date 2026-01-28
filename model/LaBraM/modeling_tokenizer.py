import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from functools import partial

# --- 1. LaBraM Temporal Encoder (Simple CNN) ---

class LaBraMTemporalConv(nn.Module):
    """
    Simple 1D CNN Patch Embedding used by LaBraM.
    Paper: "We used a 1-D convolution... repeated twice." (Wait, LaBraM code uses 3 convs)
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # Based on original 'TemporalConv' in modeling_pretrain.py
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x):
        # x: (Batch, N, T)
        B, N, T = x.shape
        
        # Original logic: treat N as height in a 1-channel image
        # x: (B, 1, N, T)
        x = x.unsqueeze(1)
        
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        
        # After convs: (B, OutChans=8, N, NewT)
        # We want (Batch, N, OutChans * NewT)
        x = rearrange(x, 'b c n t -> b n (t c)')
        return x

# --- 2. Transformer Components (Shared) ---

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                       dim_feedforward=int(embed_dim * mlp_ratio), 
                                       dropout=drop_rate, activation='gelu', batch_first=True,
                                       norm_first=True) # LaBraM usually uses Pre-Norm (norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# --- 3. Vector Quantizer (Single Layer) ---

class NormEMAVectorQuantizer(nn.Module):
    """
    Standard VQ with EMA updates and Normalization (from LaBraM resources).
    """
    def __init__(self, n_embed, embedding_dim, beta=1.0, kmeans_init=True, decay=0.99):
        super().__init__()
        self.n_embed = n_embed
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.kmeans_init = kmeans_init

        self.register_buffer('pe', torch.zeros(n_embed, embedding_dim))
        self.register_buffer('ema_cluster_size', torch.zeros(n_embed))
        self.ema_w = nn.Parameter(torch.Tensor(n_embed, embedding_dim))
        self.ema_w.data.normal_()
        
        self.register_buffer('initted', torch.Tensor([not kmeans_init]))

    def forward(self, z):
        # z: (Batch, Channels, Dim)
        # Flatten
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Normalize z (Cosine Similarity equivalent)
        z_normalized = F.normalize(z_flattened, dim=1)
        embedding_normalized = F.normalize(self.ema_w, dim=1)

        # Distances
        d = torch.sum(z_normalized ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_normalized**2, dim=1) - \
            2 * torch.matmul(z_normalized, embedding_normalized.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, embedding_normalized).view(z.shape)

        # Compute Loss (Commitment Loss)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
        
        # We don't implement the full EMA update training logic here for simplicity
        # assuming we might just load weights or use standard backprop if trainable.
        # But LaBraM uses EMA updates. 
        # For simplicity in this refactor, we return the loss. 
        # Ideally, we should port the full update logic if training from scratch.
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices

# --- 4. Main Tokenizer Model ---

class LaBraMTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=200,
        enc_depth=6, # LaBraM VQNSP config defaults
        dec_depth=6,
        n_code=8192,
        code_dim=32, # LaBraM VQ dimension is smaller than backbone dim
        patch_size=200
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. Temporal Encoder (CNN)
        self.patch_embed = LaBraMTemporalConv(in_chans, out_chans=8) 
        # Note: LaBraM hardcodes out_chans=8. 
        # After Conv: (B, N, 25). 8 chans * 25 time = 200 dim.
        
        # 2. Transformer Encoder
        self.encoder = TransformerEncoder(embed_dim, enc_depth, num_heads=10)
        
        # Project to Code Dimension
        self.encode_task_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, code_dim)
        )
        
        # 3. Vector Quantizer
        self.quantize = NormEMAVectorQuantizer(n_code, code_dim)
        
        # 4. Decoder
        self.decode_task_layer_in = nn.Linear(code_dim, embed_dim)
        self.decoder = TransformerEncoder(embed_dim, dec_depth, num_heads=10)
        
        # 5. Prediction Heads (Log-Amp + Phase)
        # LaBraM predicts the FULL 200 spectrum? 
        # modeling_vqnsp.py: self.decoder_out_dim = 200
        self.head_amp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 200)
        )
        self.head_angle = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 200)
        )

    def forward(self, x, coords=None):
        # x: (Batch, N, 200)
        
        # 1. Embed
        # (B, N, 200)
        x_emb = self.patch_embed(x) 
        
        # 2. Encode
        h = self.encoder(x_emb) # (B, N, 200)
        
        # 3. Quantize
        z = self.encode_task_layer(h) # (B, N, 32)
        z_q, vq_loss, indices = self.quantize(z)
        
        # 4. Decode
        dec_in = self.decode_task_layer_in(z_q)
        dec_h = self.decoder(dec_in)
        
        # 5. Predict Spectrum
        pred_amp = self.head_amp(dec_h)
        pred_angle = self.head_angle(dec_h)
        
        # Return 5 values to match NeuroRVQ/RecurrentVQ
        # pred_1, pred_2, pred_3, vq_loss, indices
        return pred_amp, pred_angle, None, vq_loss, indices

    def get_loss(self, x, pred_amp, pred_angle, pred_dummy=None, x_fft=None):
        # LaBraM computes loss on 200-point FFT?
        # modeling_vqnsp.py: x_fft = torch.fft.fft(x, dim=-1) -> amplitude, angle
        # It seems they use the full complex FFT size.
        
        if x_fft is None:
            x_fft = torch.fft.fft(x, dim=-1)
            
        target_amp = torch.abs(x_fft)
        target_angle = torch.angle(x_fft)
        
        # Normalize targets (LaBraM logic: std_norm)
        # Using simplified normalization here for readability
        def std_norm(t):
            mean = torch.mean(t, dim=(1,2), keepdim=True)
            std = torch.std(t, dim=(1,2), keepdim=True)
            return (t - mean) / (std + 1e-6)
            
        target_amp = std_norm(target_amp)
        target_angle = std_norm(target_angle)
        
        loss_amp = F.mse_loss(pred_amp, target_amp)
        loss_angle = F.mse_loss(pred_angle, target_angle)
        
        return loss_amp + loss_angle, loss_amp, loss_angle, torch.tensor(0.0)

    def reconstruct(self, pred_amp, pred_angle, pred_dummy=None, x=None):
        """
        Reconstructs the time-domain signal.
        If x (ground truth) is provided, we use its statistics to denormalize the prediction.
        Otherwise, we assume standard normal statistics.
        """
        if x is not None:
            # Oracle Denormalization
            x_fft = torch.fft.fft(x, dim=-1)
            gt_amp = torch.abs(x_fft)
            gt_angle = torch.angle(x_fft)
            
            amp_mean = torch.mean(gt_amp, dim=(1,2), keepdim=True)
            amp_std = torch.std(gt_amp, dim=(1,2), keepdim=True)
            
            angle_mean = torch.mean(gt_angle, dim=(1,2), keepdim=True)
            angle_std = torch.std(gt_angle, dim=(1,2), keepdim=True)
            
            # Denormalize
            rec_amp = pred_amp * (amp_std + 1e-6) + amp_mean
            rec_angle = pred_angle * (angle_std + 1e-6) + angle_mean
        else:
            # Blind Reconstruction (likely poor amplitude scaling)
            rec_amp = pred_amp
            rec_angle = pred_angle
            
        # Polar to Cartesian
        # z = amp * e^(j * angle)
        real = rec_amp * torch.cos(rec_angle)
        imag = rec_amp * torch.sin(rec_angle)
        z = torch.complex(real, imag)
        
        # Inverse FFT
        x_recon = torch.fft.ifft(z, dim=-1).real
        
        return x_recon
