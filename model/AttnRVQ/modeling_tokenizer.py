import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Multi-Scale Temporal Encoder ---

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 11], stride=1):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # Calculate channels per branch
        num_branches = len(kernel_sizes) + 1 # +1 for pooling branch
        branch_channels = out_channels // num_branches
        last_channel = out_channels - (num_branches - 1) * branch_channels
        
        bottleneck_dim = max(in_channels // num_branches, 1) if in_channels > num_branches else in_channels
        
        # Kernel Branches
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_dim, 1, bias=False),
                nn.BatchNorm1d(bottleneck_dim),
                nn.GELU(),
                nn.Conv1d(bottleneck_dim, branch_channels, k, stride=stride, padding=k//2, bias=False),
                nn.BatchNorm1d(branch_channels),
                nn.GELU()
            ))
            
        # Pooling Branch
        self.branches.append(nn.Sequential(
            nn.MaxPool1d(3, stride=stride, padding=1),
            nn.Conv1d(in_channels, last_channel, 1, bias=False),
            nn.BatchNorm1d(last_channel),
            nn.GELU()
        ))
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1) + self.shortcut(x)

class MultiScaleTemporalEncoder(nn.Module):
    """
    Extracts features at multiple time-scales and fuses them via a Top-Down Bridge.
    """
    def __init__(self, in_chans=1, embed_dim=200, base_filters=8, num_scales=4, input_length=200, kernel_sizes=[3, 5, 11]):
        super().__init__()
        self.num_scales = num_scales
        self.in_chans = in_chans
        
        # --- 1. The Bottom-Up Pathway ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_chans, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU()
        )
        
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        current_filters = base_filters
        current_t = input_length 
        
        for i in range(num_scales):
            stride = 1 if i == 0 else 2
            out_filters = current_filters if i == 0 else current_filters * 2
            
            self.layers.append(InceptionBlock(current_filters, out_filters, kernel_sizes=kernel_sizes, stride=stride))
            
            current_filters = out_filters
            current_t = (current_t - 1) // stride + 1
            
            self.projections.append(nn.Sequential(
                nn.Linear(current_filters * current_t, embed_dim),
                nn.LayerNorm(embed_dim)
            ))

    def forward(self, x):
        B, N, T = x.shape
        x = x.view(B * N, 1, T)
        x = self.stem(x)
        
        raw_features = []
        feat = x
        for layer, proj in zip(self.layers, self.projections):
            feat = layer(feat)
            raw_features.append(proj(feat.flatten(1)).view(B, N, -1))
        
        return raw_features

# --- 2. Transformer Decoder ---

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                       dim_feedforward=int(embed_dim * mlp_ratio), 
                                       dropout=drop_rate, activation='gelu', batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

# --- 3. AttnVQ Multi-Head Deep Quantization ---

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AttnRVQ(nn.Module):
    """
    Multi-Head Sparse Attention VQ with Independent Codebooks per Head.
    """
    def __init__(self, num_scales, num_heads, n_e, e_dim, top_k=8, temperature=5):
        super().__init__()
        assert e_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.n_e = n_e
        self.e_dim = e_dim
        self.head_dim = e_dim // num_heads
        self.top_k = top_k
        
        # Temperature per scale and head
        self.temperature = nn.Parameter(torch.ones(num_scales, num_heads, 1, 1) * temperature)

        # Keys (Selection): (S, H, N_E, D_head)
        self.key_embeddings = nn.Parameter(torch.empty(num_scales, num_heads, n_e, self.head_dim))
        nn.init.uniform_(self.key_embeddings, -1.0 / n_e, 1.0 / n_e)
        
        # Values (Content): (S, H, N_E, D_head)
        self.value_embeddings = nn.Parameter(torch.empty(num_scales, num_heads, n_e, self.head_dim))
        nn.init.normal_(self.value_embeddings, std=0.02)

    def forward(self, z):
        # Input z: (S*B, N, D)
        S, H, V, D_h = self.num_scales, self.num_heads, self.n_e, self.head_dim
        SB, N, D = z.shape
        B = SB // S
        
        # Reshape to (S, B, N, H, D_h)
        z_reshaped = z.view(S, B, N, H, D_h)
        
        # --- 1. Similarity (Dot Product) ---
        # z: (S, B, N, H, D_h)
        # keys: (S, H, V, D_h)
        # Output: (S, B, N, H, V)
        logits = torch.einsum('sbnhd,shvd->sbnhv', z_reshaped, self.key_embeddings)
        
        # --- 2. Top-K Selection ---
        top_vals, indices = torch.topk(logits, k=self.top_k, dim=-1) # (S, B, N, H, K)
        
        # --- 3. Weighted Mixing ---
        # temperature is (S, H, 1, 1). We need it to be (S, 1, 1, H, 1) to broadcast with (S, B, N, H, K)
        temp_broadcast = self.temperature.view(self.num_scales, 1, 1, self.num_heads, 1)
        weights = F.softmax(top_vals / temp_broadcast.clamp(min=1e-3), dim=-1) # (S, B, N, H, K)
        
        # --- 4. Reconstruction ---
        # Flattened Index calculation for Multi-Head independent gathering
        s_idx = torch.arange(S, device=z.device).view(S, 1, 1, 1, 1)
        h_idx = torch.arange(H, device=z.device).view(1, 1, 1, H, 1)
        offset = s_idx * (H * V) + h_idx * V 
        
        global_indices = (indices + offset).view(-1, self.top_k)
        flat_values = self.value_embeddings.view(-1, D_h)
        
        # Gathered: (S*B*N*H, K, D_h)
        selected_vectors = F.embedding(global_indices, flat_values)
        selected_vectors = selected_vectors.view(S, B, N, H, self.top_k, D_h)
        
        # Weighted Sum: (S, B, N, H, D_h)
        z_q = torch.einsum('sbnhk,sbnhkd->sbnhd', weights, selected_vectors)
        
        # Merge Heads: (S, B, N, D)
        z_q = z_q.reshape(SB, N, D)
        
        # --- 5. Losses ---
        # Commitment Loss on the merged representation
        loss_commit = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        loss_ortho = self.orthogonal_loss()
        
        return z_q, loss_commit + loss_ortho, indices

    def orthogonal_loss(self):
        # Vectorized implementation
        # key_embeddings: (S, H, V, D_h) -> (S*H, V, D_h)
        vectors = self.key_embeddings.view(-1, self.n_e, self.head_dim)
        
        # Normalize: (S*H, V, D_h)
        vectors_norm = F.normalize(vectors, p=2, dim=-1)
        
        # Gram Matrix: (S*H, V, V)
        gram = torch.bmm(vectors_norm, vectors_norm.transpose(1, 2))
        
        # Identity Mask: (1, V, V) - broadcasts to (S*H, V, V)
        identity = torch.eye(self.n_e, device=gram.device).unsqueeze(0)
        
        # Off-diagonal elements
        off_diag = gram * (1 - identity)
        
        # Loss
        threshold = 0.1
        loss = F.relu(off_diag.abs() - threshold).pow(2).mean()
        
        return loss

class AttnRVQEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_scales, n_e, top_k, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attnrvq_vq = AttnRVQ(num_scales, num_heads, n_e, embed_dim, top_k)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # 1. Quantize (Attention substitute)
        z_q, loss, indices = self.attnrvq_vq(self.norm1(x))
        
        # 2. Update stream (Residual)
        x = x + z_q
        
        # 3. Transition (Non-residual warp)
        x = self.mlp(self.norm2(x))
        
        return x, z_q, loss, indices

class AttnRVQTransformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, num_scales, n_e, top_k, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            AttnRVQEncoderLayer(embed_dim, num_heads, num_scales, n_e, top_k, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])

    def forward(self, x):
        total_z_q = 0
        total_loss = 0
        all_indices = []
        for layer in self.layers:
            x, z_q, loss, indices = layer(x)
            
            # Sum up quantized vectors
            total_z_q = total_z_q + z_q
            total_loss += loss
            all_indices.append(indices)
            
        return total_z_q, total_loss, all_indices

class TopDownFPN(nn.Module):
    """
    Fuses features from Coarse (High Index) to Fine (Low Index) scales.
    """
    def __init__(self, num_scales, embed_dim):
        super().__init__()
        self.num_scales = num_scales
        self.adapters = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_scales - 1)
        ])
        self.gates = nn.Parameter(torch.zeros(num_scales - 1))

    def forward(self, features):
        # features: (S, B, N, D)
        scales = list(features.unbind(0))
        context = scales[-1]
        for i in range(self.num_scales - 2, -1, -1):
            proj_ctx = self.adapters[i](context)
            gate = torch.sigmoid(self.gates[i])
            scales[i] = scales[i] + gate * proj_ctx
            context = scales[i]
        return torch.stack(scales, dim=0)

# --- 4. Main Tokenizer Model (AttnRVQTokenizer) ---

class AttnRVQTokenizer(nn.Module):
    def __init__(
        self,
        in_chans=1,
        embed_dim=200,
        enc_depth=4,
        enc_heads=8, 
        enc_mlp_ratio=4.,
        dec_depth=3, 
        dec_heads=10,
        dec_mlp_ratio=4.,
        num_scales=4,
        top_k=8,
        vocab_size=512,
        freq_resolution=1.0,
        min_freq=0.0,
        max_freq=100.0,
        fs=200.0,
        input_length=200,
        kernel_sizes=[3, 5, 11]
    ):
        super().__init__()
        self.num_scales = num_scales
        self.embed_dim = embed_dim
        self.fs = fs
        self.fft_dim = int(round((max_freq - min_freq) / freq_resolution)) + 1 
        self.n_fft = int(self.fs / freq_resolution)
        
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.fs)
        mask = (freqs >= min_freq - 1e-5) & (freqs <= max_freq + 1e-5)
        self.register_buffer('freq_mask', mask)
        self.register_buffer('freq_indices', torch.where(mask)[0])
        
        # 1. Multi-Scale Feature Extraction
        self.temporal_encoder = MultiScaleTemporalEncoder(
            in_chans, embed_dim, num_scales=num_scales, input_length=input_length, kernel_sizes=kernel_sizes
        )
        
        self.spatial_mlp = nn.Sequential(nn.Linear(3, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        nn.init.zeros_(self.spatial_mlp[-1].weight)
        nn.init.zeros_(self.spatial_mlp[-1].bias)
        
        # Use AttnRVQTransformer as the encoder
        self.attnrvq_encoder = AttnRVQTransformer(embed_dim, enc_depth, enc_heads, num_scales, vocab_size, top_k, enc_mlp_ratio)
        
        self.transformer_decoder = TransformerDecoder(embed_dim, dec_depth, dec_heads, mlp_ratio=dec_mlp_ratio) 
        
        # 5. FPN
        self.fpn = TopDownFPN(num_scales, embed_dim)
        
        self.head_amp = nn.Linear(embed_dim, self.fft_dim)
        self.head_sin = nn.Linear(embed_dim, self.fft_dim)
        self.head_cos = nn.Linear(embed_dim, self.fft_dim)

    def forward(self, x, coords):
        B, N, T = x.shape
        S = self.num_scales
        
        # Feature Extraction
        ms_features = self.temporal_encoder(x) 
        spatial_emb = self.spatial_mlp(coords) 
        h_all = torch.stack(ms_features, dim=0) + spatial_emb.unsqueeze(0) 
        h_all = h_all.view(S * B, N, -1)
        
        # Deep Quantization
        z_fused_flat, vq_loss, top_k_indices = self.attnrvq_encoder(h_all)
        
        # 4. Independent Decoding (Batch-Folded)
        # z_fused_flat: (S*B, N, D)
        dec_h = self.transformer_decoder(z_fused_flat)
        
        # 5. FPN Fusion
        dec_h_scales = dec_h.view(S, B, N, -1)
        dec_h_fused = self.fpn(dec_h_scales)
        dec_h_flat = dec_h_fused.view(S * B, N, -1)
        
        # 6. Prediction Heads
        # Output: (S*B, N, Freqs)
        raw_amp = self.head_amp(dec_h_flat)
        raw_sin = self.head_sin(dec_h_flat)
        raw_cos = self.head_cos(dec_h_flat)
        
        # 7. Late Fusion (Sum Predictions)
        # Reshape to (S, B, N, Freqs) and Sum across S
        pred_amp = torch.sum(raw_amp.view(S, B, N, -1), dim=0)
        pred_sin = torch.sum(raw_sin.view(S, B, N, -1), dim=0)
        pred_cos = torch.sum(raw_cos.view(S, B, N, -1), dim=0)
        
        return pred_amp, pred_sin, pred_cos, vq_loss, torch.stack(top_k_indices, dim=0)

    def get_loss(self, x, pred_amp, pred_sin, pred_cos, x_fft=None):
        if x_fft is None:
            x_fft = torch.fft.rfft(x, n=self.n_fft, dim=-1)
        target_fft = x_fft[..., self.freq_mask]
        
        if target_fft.shape[-1] != pred_amp.shape[-1]:
             min_len = min(target_fft.shape[-1], pred_amp.shape[-1])
             target_fft, pred_amp, pred_sin, pred_cos = target_fft[..., :min_len], pred_amp[..., :min_len], pred_sin[..., :min_len], pred_cos[..., :min_len]
        
        gt_amp, gt_phase = torch.abs(target_fft), torch.angle(target_fft)
        loss_amp = F.mse_loss(pred_amp, torch.log1p(gt_amp))
        loss_phase = F.mse_loss(pred_sin, torch.sin(gt_phase)) + F.mse_loss(pred_cos, torch.cos(gt_phase))
        loss_temp = F.l1_loss(self.reconstruct(pred_amp, pred_sin, pred_cos, n_samples=x.shape[-1]), x)
        
        return loss_amp + loss_phase + loss_temp, loss_amp, loss_phase, loss_temp

    def reconstruct(self, pred_amp, pred_sin, pred_cos, n_samples=200, x=None):
        amp = torch.clamp(torch.exp(pred_amp) - 1, min=0)
        pred_norm = torch.sqrt(pred_cos**2 + pred_sin**2 + 1e-8)
        real, imag = amp * (pred_cos / pred_norm), amp * (pred_sin / pred_norm)
        z_pred = torch.complex(real, imag)
        full_z = torch.zeros((z_pred.shape[0], z_pred.shape[1], self.n_fft // 2 + 1), dtype=z_pred.dtype, device=z_pred.device)
        indices = self.freq_indices
        count = min(len(indices), z_pred.shape[-1])
        full_z[..., indices[:count]] = z_pred[..., :count]
        return torch.fft.irfft(full_z, n=self.n_fft, dim=-1)[..., :n_samples]

    # --- Analysis Helpers ---

    def get_codebooks(self):
        """
        Returns a list of (tensor, name) tuples for analysis.
        Extracts Key embeddings from ALL layers, Scales, and Heads.
        Name format: "L{layer}_S{scale}_H{head}"
        """
        codebooks = []
        encoder = self.attnrvq_encoder
        
        for l_idx, layer in enumerate(encoder.layers):
            keys = layer.attnrvq_vq.key_embeddings # (S, H, V, D_h)
            S, H, V, D = keys.shape
            
            for s in range(S):
                for h in range(H):
                    cb = keys[s, h].detach().cpu()
                    name = f"L{l_idx}_S{s}_H{h}"
                    codebooks.append((cb, name))
                    
        return codebooks

    def get_indices(self, x, coords):
        """
        Runs forward pass and returns indices for analysis.
        Returns: (Batch*N, Depth, Scales, Heads, Top-K) flat tensor
        """
        # Run forward pass (ignoring gradients)
        with torch.no_grad():
            _, _, _, _, indices = self.forward(x, coords)
        
        # indices is (Depth, Scales, Batch, N, Heads, Top-K)
        # Permute to (Batch, N, Depth, Scales, Heads, Top-K)
        indices = indices.permute(2, 3, 0, 1, 4, 5)
        
        # Flatten Batch and N -> (Batch*N, Depth, Scales, Heads, Top-K)
        indices_flat = indices.reshape(-1, *indices.shape[2:])
        
        return indices_flat