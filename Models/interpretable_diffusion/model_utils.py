import math
import scipy
import torch
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, reduce
from scipy.fftpack import next_fast_len


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


# normalization functions

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# learnable positional embeds

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean 


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)
    

class Conv_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, resid_pdrop=0.):
        super().__init__()
        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            # nn.Conv1d(in_dim, out_dim, 3, stride=1, padding=1),
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),  ## hucfg925
            # nn.Conv1d(in_dim, out_dim, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)
    

class Transformer_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=1, padding=0),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, padding=1),
            act,
            nn.Conv1d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd,  kernel_size=3, padding=1),
            nn.Dropout(p=resid_pdrop),
        )

    def forward(self, x):
        return self.sequential(x)
    

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class SpatialEmotionConditioner(nn.Module):
    """
    通道感知的情绪条件嵌入.

    为每个情绪类别学习 n_modes 个空间激活模式,
    每个模式 = 空间权重 (哪些脑区) × 特征方向 (怎么调制).
    多模式组合可以表达复杂的脑区×频带交互:
      - 模式 1: 左前额 alpha 降低 (正面情绪)
      - 模式 2: 右前额 alpha 升高 (负面情绪)
      - 模式 3: 枕区 gamma 变化 (唤醒度)

    Input:  labels (B,) LongTensor
    Output: (B, n_channel, d_model) 通道特异性情绪嵌入
    """
    def __init__(self, num_classes, n_channel, d_model, n_modes=8):
        super().__init__()
        self.n_channel = n_channel
        self.d_model = d_model
        self.n_modes = n_modes
        assert d_model % n_modes == 0
        self.d_mode = d_model // n_modes

        self.class_emb = nn.Embedding(num_classes, d_model)

        # 每个模式有独立的空间权重模式 (哪些通道被激活)
        self.spatial_modes = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_channel * n_modes),
        )
        # 每个模式有独立的特征方向 (如何调制)
        self.feat_modes = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),  # n_modes * d_mode = d_model
        )

    def forward(self, labels):
        """labels: (B,) → (B, n_channel, d_model)"""
        emb = self.class_emb(labels)  # (B, d_model)
        B = emb.shape[0]

        # n_modes 个空间模式
        spatial = self.spatial_modes(emb)                          # (B, C*M)
        spatial = spatial.view(B, self.n_modes, self.n_channel)    # (B, M, C)

        # n_modes 个特征方向
        feats = self.feat_modes(emb)                               # (B, M*d_mode)
        feats = feats.view(B, self.n_modes, self.d_mode)           # (B, M, d_mode)

        # 空间权重 × 特征方向 的外积, 多模式拼接 → (B, C, d_model)
        out = spatial.unsqueeze(-1) * feats.unsqueeze(2)           # (B, M, C, d_mode)
        out = out.permute(0, 2, 1, 3).reshape(B, self.n_channel, self.d_model)
        return out


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Norm with channel-specific emotion conditioning.

    timestep → 全局 scale/shift (所有通道相同, 控制去噪步进)
    label   → 通道特异 scale/shift (不同脑区不同调制, 控制情绪模式)

    两者顺序施加: 先 timestep 调制, 再 label 调制.
    """
    def __init__(self, n_embd, electrode_coords=None):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)      # timestep → global scale/shift
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

        # Channel-specific label modulation (NEW)
        self.label_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, n_embd * 2),                # label → per-channel scale/shift
        )

    def forward(self, x, timestep, label_emb=None):
        # ---- Timestep: global modulation (same for all channels) ----
        t_emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # (B, 1, 2d)
        t_scale, t_shift = torch.chunk(t_emb, 2, dim=-1)
        x = self.layernorm(x) * (1 + t_scale) + t_shift

        # ---- Label: channel-specific modulation ----
        if label_emb is not None:
            # label_emb: (B, n_channel, d_model) from SpatialEmotionConditioner
            #         or (B, d_model) for legacy backward compat
            if label_emb.dim() == 2:
                label_emb = label_emb.unsqueeze(1)  # → (B, 1, d_model)
            l_emb = self.label_film(label_emb)      # (B, C, 2d) or (B, 1, 2d)
            l_scale, l_shift = torch.chunk(l_emb, 2, dim=-1)
            x = x * (1 + l_scale) + l_shift

        return x