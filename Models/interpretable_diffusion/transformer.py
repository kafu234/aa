import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, RMSNorm, GELU2, series_decomp
import os


# ============================================================
#  SEED 62-channel electrode coordinates (10-20 system)
#  2D azimuthal equidistant projection, normalized to [-1, 1]
# ============================================================

SEED_62_CHANNELS = [
    'FP1','FPZ','FP2','AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'CB1','O1','OZ','O2','CB2',
]

def _get_seed_62_coords():
    """
    SEED 62-channel 3D electrode coordinates (meters).
    Source: MNE-Python standard_1020 montage (mne.channels.make_standard_montage).
    CB1/CB2 (cerebellar): interpolated from O1/O2 + PO7/PO8.
    Coordinate system: x(left-/right+), y(posterior-/anterior+), z(inferior-/superior+).
    Returns: (62, 3) tensor
    """
    # fmt: off
    coords = torch.tensor([
        [-0.029437,  0.083917, -0.006990],  # FP1
        [ 0.000112,  0.088247, -0.001713],  # FPZ
        [ 0.029872,  0.084896, -0.007080],  # FP2
        [-0.033701,  0.076837,  0.021227],  # AF3
        [ 0.035712,  0.077726,  0.021956],  # AF4
        [-0.070263,  0.042474, -0.011420],  # F7
        [-0.064466,  0.048035,  0.016921],  # F5
        [-0.050244,  0.053111,  0.042192],  # F3
        [-0.027496,  0.056931,  0.060342],  # F1
        [ 0.000312,  0.058512,  0.066462],  # FZ
        [ 0.029514,  0.057602,  0.059540],  # F2
        [ 0.051836,  0.054305,  0.040814],  # F4
        [ 0.067914,  0.049830,  0.016367],  # F6
        [ 0.073043,  0.044422, -0.012000],  # F8
        [-0.080775,  0.014120, -0.011135],  # FT7
        [-0.077215,  0.018643,  0.024460],  # FC5
        [-0.060182,  0.022716,  0.055544],  # FC3
        [-0.034062,  0.026011,  0.079987],  # FC1
        [ 0.000376,  0.027390,  0.088668],  # FCZ
        [ 0.034784,  0.026438,  0.078808],  # FC2
        [ 0.062293,  0.023723,  0.055630],  # FC4
        [ 0.079534,  0.019936,  0.024438],  # FC6
        [ 0.081815,  0.015417, -0.011330],  # FT8
        [-0.084161, -0.016019, -0.009346],  # T7
        [-0.080280, -0.013760,  0.029160],  # C5
        [-0.065358, -0.011632,  0.064358],  # C3
        [-0.036158, -0.009984,  0.089752],  # C1
        [ 0.000401, -0.009167,  0.100244],  # CZ
        [ 0.037672, -0.009624,  0.088412],  # C2
        [ 0.067118, -0.010900,  0.063580],  # C4
        [ 0.083456, -0.012776,  0.029208],  # C6
        [ 0.085080, -0.015020, -0.009490],  # T8
        [-0.084830, -0.046022, -0.007056],  # TP7
        [-0.079592, -0.046551,  0.030949],  # CP5
        [-0.063556, -0.047009,  0.065624],  # CP3
        [-0.035513, -0.047292,  0.091315],  # CP1
        [ 0.000386, -0.047318,  0.099432],  # CPZ
        [ 0.038384, -0.047073,  0.090695],  # CP2
        [ 0.066612, -0.046637,  0.065580],  # CP4
        [ 0.083322, -0.046101,  0.031206],  # CP6
        [ 0.085549, -0.045545, -0.007130],  # TP8
        [-0.072434, -0.073453, -0.002487],  # P7
        [-0.067272, -0.076291,  0.028382],  # P5
        [-0.053007, -0.078788,  0.055940],  # P3
        [-0.028620, -0.080525,  0.075436],  # P1
        [ 0.000325, -0.081115,  0.082615],  # PZ
        [ 0.031920, -0.080487,  0.076716],  # P2
        [ 0.055667, -0.078560,  0.056561],  # P4
        [ 0.067888, -0.075904,  0.028091],  # P6
        [ 0.073056, -0.073068, -0.002540],  # P8
        [-0.054840, -0.097528,  0.002792],  # PO7
        [-0.048424, -0.099341,  0.021599],  # PO5
        [-0.036511, -0.100853,  0.037167],  # PO3
        [ 0.000216, -0.102178,  0.050608],  # POZ
        [ 0.036782, -0.100849,  0.036397],  # PO4
        [ 0.049820, -0.099446,  0.021727],  # PO6
        [ 0.055667, -0.097625,  0.002730],  # PO8
        [-0.042127, -0.120449,  0.000815],  # CB1 (interpolated: midpoint O1+PO7)
        [-0.029413, -0.112449,  0.008839],  # O1
        [ 0.000108, -0.114892,  0.014657],  # OZ
        [ 0.029843, -0.112156,  0.008800],  # O2
        [ 0.042755, -0.120156,  0.000765],  # CB2 (interpolated: midpoint O2+PO8)
    ], dtype=torch.float32)
    # fmt: on
    return coords  # (62, 3)


# ============================================================
#  Spatial modules
# ============================================================

class SpatialPositionalEncoding(nn.Module):
    """
    Inject electrode 3D coordinates as positional encoding.
    Added after token embedding, before encoder/decoder input.
    """
    def __init__(self, n_channels=62, d_model=256):
        super().__init__()
        coords = _get_seed_62_coords()  # (62, 3)
        self.register_buffer('coords', coords)
        self.proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Scale factor: start small so it doesn't destabilize early training
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """x: (batch, n_channels, d_model)"""
        spatial_emb = self.proj(self.coords)  # (62, d_model)
        return x + self.scale * spatial_emb.unsqueeze(0)


class SpatialAttentionBias(nn.Module):
    """
    Per-head learnable soft spatial bias for attention.
    
    Each head learns (alpha, beta):
        bias[h, i, j] = alpha_h * dist(i, j) + beta_h
    
    - alpha < 0  →  head focuses on nearby electrodes (local)
    - alpha ≈ 0  →  head ignores distance (global, like vanilla attention)
    - alpha > 0  →  head prefers distant electrodes (cross-region connectivity)
    
    This replaces hard graph masking with a fully differentiable, 
    data-driven spatial prior that each head learns independently.
    """
    def __init__(self, n_channels, n_head):
        super().__init__()
        coords = _get_seed_62_coords()
        # Pairwise distance matrix, normalized to [0, 1]
        dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        dist = dist / dist.max()
        self.register_buffer('dist_matrix', dist)  # (C, C)

        # Per-head learnable parameters
        # Initialize alpha slightly negative → mild local preference as starting point
        self.head_alpha = nn.Parameter(torch.randn(n_head) * 0.1 - 0.3)
        self.head_beta  = nn.Parameter(torch.zeros(n_head))

    def forward(self):
        """Returns (1, n_head, C, C) bias to add to attention logits."""
        alpha = self.head_alpha.view(1, -1, 1, 1)  # (1, nh, 1, 1)
        beta  = self.head_beta.view(1, -1, 1, 1)
        dist  = self.dist_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, C, C)
        return alpha * dist + beta


# ============================================================
#  Original blocks (kept as-is)
# ============================================================

class TrendBlock(nn.Module):
    """[DEPRECATED] Kept for compatibility. Not used in EEG mode."""
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )
        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


class EEGBandBlock(nn.Module):
    """
    EEG 频带分解模块，替代 TrendBlock + FourierLayer。
    """
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50),
    }

    def __init__(self, n_channel, n_embd, n_feat, sfreq=200):
        super().__init__()
        self.n_feat = n_feat
        self.sfreq = sfreq
        self.n_bands = len(self.BANDS)

        self.to_time = nn.Linear(n_embd, n_feat)
        self.band_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_feat, n_feat),
                nn.GELU(),
                nn.Linear(n_feat, n_feat),
            )
            for _ in range(self.n_bands)
        ])

        self.band_attention = nn.Sequential(
            nn.Linear(n_feat * self.n_bands, self.n_bands),
            nn.Softmax(dim=-1),
        )

        n_freq = n_feat // 2 + 1
        freqs = torch.fft.rfftfreq(n_feat, d=1.0 / sfreq)
        masks = []
        for band_name, (low, high) in self.BANDS.items():
            mask = ((freqs >= low) & (freqs < high)).float()
            masks.append(mask)
        self.register_buffer('band_masks', torch.stack(masks, dim=0))

    def forward(self, x):
        x_time = self.to_time(x)
        x_freq = torch.fft.rfft(x_time, dim=2)

        band_outputs = []
        for i in range(self.n_bands):
            mask = self.band_masks[i]
            x_band_freq = x_freq * mask.unsqueeze(0).unsqueeze(0)
            x_band = torch.fft.irfft(x_band_freq, n=self.n_feat, dim=2)
            x_band = self.band_nets[i](x_band)
            band_outputs.append(x_band)

        concat = torch.cat(band_outputs, dim=2)
        weights = self.band_attention(concat)
        stacked = torch.stack(band_outputs, dim=-1)
        weights = weights.unsqueeze(2)
        out = (stacked * weights).sum(dim=-1)
        return out


class DEBandBlock(nn.Module):
    """
    EEGBandBlock 的 DE 模式替代品.
    当 n_feat=5 (DE 特征) 时, 5 个特征本身就是频带, 不需要 FFT 分解.
    """
    def __init__(self, n_channel, n_embd, n_feat=5):
        super().__init__()
        self.to_feat = nn.Linear(n_embd, n_feat)
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, n_feat * 4),
            nn.GELU(),
            nn.Linear(n_feat * 4, n_feat),
        )

    def forward(self, x):
        x_feat = self.to_feat(x)  # (B, 62, n_feat)
        return self.mlp(x_feat)   # (B, 62, n_feat)


class MovingBlock(nn.Module):
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)
        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]
        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)
        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


# ============================================================
#  RoPE helpers (unchanged)
# ============================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
#  Attention with spatial bias  [MODIFIED]
# ============================================================

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 n_head,
                 n_channel=62,         # ← NEW
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 max_len=None
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            max_len * 4,
            50000,
        )

        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))

        # ---- Spatial attention bias (NEW) ----
        self.spatial_bias = SpatialAttentionBias(n_channel, n_head)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = self.k_norm(k) + 0.1*k
        q = self.q_norm(q) + 0.1*q

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if int(os.environ.get('hucfg_attention_rope_use', '-1')) == 1: 
            freqs_cis = self.freqs_cis.cuda()[0 : T]
            q, k = apply_rotary_emb(q.permute(0,2,1,3), k.permute(0,2,1,3), freqs_cis=freqs_cis)
            q, k = q.permute(0,2,1,3), k.permute(0,2,1,3)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # ---- Add spatial bias before softmax (NEW) ----
        # spatial_bias: (1, n_head, C, C),  att: (B, n_head, T, T)
        # Only apply when T matches n_channel (skip if registers change seq len)
        sp_bias = self.spatial_bias()  # (1, nh, C, C)
        if T == sp_bias.shape[-1]:
            att = att + sp_bias

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd,
                 condition_embd,
                 n_head,
                 n_channel=62,         # ← NEW
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 max_len=None
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            max_len * 4,
            50000,
        )

        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))
        self.register_2 = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))

        # ---- Spatial attention bias (NEW) ----
        # Cross-attention: Q=decoder channels, K=encoder channels → same spatial layout
        self.spatial_bias = SpatialAttentionBias(n_channel, n_head)

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()

        k = self.key(encoder_output)
        q = self.query(x)

        k = self.k_norm(k) + 0.1*k
        q = self.q_norm(q) + 0.1*q

        k = k.view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        freqs_cis = self.freqs_cis.cuda()[0 : T]
        q, k = apply_rotary_emb(q.permute(0,2,1,3), k.permute(0,2,1,3), freqs_cis=freqs_cis)
        q, k = q.permute(0,2,1,3), k.permute(0,2,1,3)

        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # ---- Add spatial bias (NEW) ----
        sp_bias = self.spatial_bias()
        if T == sp_bias.shape[-1] and T_E == sp_bias.shape[-1]:
            att = att + sp_bias

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        att = att.mean(dim=1, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, att
        

# ============================================================
#  Encoder / Decoder blocks — pass n_channel through  [MODIFIED]
# ============================================================

class EncoderBlock(nn.Module):
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 n_channel=62,         # ← NEW
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 max_len=None,
                 electrode_coords=None,   # ← NEW
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd, electrode_coords=electrode_coords)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                n_channel=n_channel,   # ← NEW
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len=max_len
            )
        
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        
    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        n_channel=62,              # ← NEW
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=None,
        electrode_coords=None,     # ← NEW
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                n_channel=n_channel,   # ← NEW
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                max_len=max_len,
                electrode_coords=electrode_coords,  # ← NEW
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 max_len=None,
                 electrode_coords=None,   # ← NEW
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd, electrode_coords=electrode_coords)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                n_channel=n_channel,   # ← NEW
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                max_len=max_len
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                n_channel=n_channel,   # ← NEW
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len=max_len
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd, electrode_coords=electrode_coords)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.band_block = DEBandBlock(n_channel, n_embd, n_feat) if n_feat <= 10 \
            else EEGBandBlock(n_channel, n_embd, n_feat, sfreq=200)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a

        # Cross-attention 也接收情绪条件 (原来没传 label_emb)
        a, att = self.attn2(self.ln1_1(x, timestep, label_emb), encoder_output, mask=mask)
        x = x + a

        band_out = self.band_block(x)

        x = x + self.mlp(self.ln2(x))

        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), band_out
    

class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512,
        max_len=None,
        electrode_coords=None,     # ← NEW
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
                max_len=max_len,
                electrode_coords=electrode_coords,  # ← NEW
        ) for _ in range(n_layer)])
      
    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        mean = []
        band = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_band = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            band += residual_band
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, band


# ============================================================
#  Main Transformer — with SpatialPositionalEncoding  [MODIFIED]
# ============================================================

class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)
        self.max_len = max_len

        # ---- Spatial positional encoding (NEW) ----
        self.spatial_pe = SpatialPositionalEncoding(n_channel, n_embd)

        # ---- Electrode coordinates for channel-aware conditioning ----
        electrode_coords = _get_seed_62_coords()  # (62, 3)

        # ---- Pass n_channel + electrode_coords to encoder (NEW) ----
        self.encoder = Encoder(
            n_layer_enc, n_embd, n_heads,
            n_channel=n_channel,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            block_activate=block_activate, max_len=self.max_len,
            electrode_coords=electrode_coords,
        )

        self.decoder = Decoder(
            n_channel, n_feat, n_embd, n_heads, n_layer_dec,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            block_activate=block_activate, condition_dim=n_embd,
            max_len=self.max_len,
            electrode_coords=electrode_coords,
        )

    def forward(self, input, t, padding_masks=None, return_res=False, label_emb=None):
        emb = self.emb(input)

        # ---- Inject spatial structure (NEW) ----
        emb = self.spatial_pe(emb)

        inp_enc = emb
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks, label_emb=label_emb)

        inp_dec = emb
        output, mean, band = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks, label_emb=label_emb)

        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        mean_out = self.combine_m(mean) + res_m
        out = mean_out + (res - res_m) + band

        return out


if __name__ == '__main__':
    pass