import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, RMSNorm, GELU2, series_decomp
import os

## hunote: our backbone network is most same as diffusion-TS. Diffusion-TS backbone has really good potential!!
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

    将信号分解为 delta/theta/alpha/beta/gamma 五个频带，
    每个频带单独建模后融合。

    原理:
      1. 将嵌入空间投影回时间域 (n_embd → n_feat)
      2. 对 n_feat 维 (200个时间点) 做 rFFT
      3. 用固定的频带 mask 提取各频带成分
      4. 每个频带经过独立的可学习网络
      5. 融合所有频带输出
    """
    # EEG 标准五频带 (Hz)
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

        # 从嵌入空间投影到时间域
        self.to_time = nn.Linear(n_embd, n_feat)

        # 每个频带的独立处理网络
        self.band_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_feat, n_feat),
                nn.GELU(),
                nn.Linear(n_feat, n_feat),
            )
            for _ in range(self.n_bands)
        ])

        # 可学习的频带权重 (注意力融合)
        self.band_attention = nn.Sequential(
            nn.Linear(n_feat * self.n_bands, self.n_bands),
            nn.Softmax(dim=-1),
        )

        # 预计算频带 mask (注册为 buffer，不参与训练)
        n_freq = n_feat // 2 + 1
        freqs = torch.fft.rfftfreq(n_feat, d=1.0 / sfreq)  # 频率轴 (Hz)
        masks = []
        for band_name, (low, high) in self.BANDS.items():
            mask = ((freqs >= low) & (freqs < high)).float()
            masks.append(mask)
        # (n_bands, n_freq)
        self.register_buffer('band_masks', torch.stack(masks, dim=0))

    def forward(self, x):
        """
        x: (batch, n_channel, n_embd) — 来自 Decoder attention 的嵌入表示
        return: (batch, n_channel, n_feat) — 频带分解后的时间域信号
        """
        # 投影到时间域
        x_time = self.to_time(x)  # (b, c, n_feat)

        # rFFT 沿时间维度
        x_freq = torch.fft.rfft(x_time, dim=2)  # (b, c, n_freq)

        # 对每个频带做 mask → iFFT → 独立网络处理
        band_outputs = []
        for i in range(self.n_bands):
            mask = self.band_masks[i]  # (n_freq,)
            x_band_freq = x_freq * mask.unsqueeze(0).unsqueeze(0)  # (b, c, n_freq)
            x_band = torch.fft.irfft(x_band_freq, n=self.n_feat, dim=2)  # (b, c, n_feat)
            x_band = self.band_nets[i](x_band)  # (b, c, n_feat)
            band_outputs.append(x_band)

        # 频带注意力融合
        # 拼接所有频带 → 计算每个频带的权重
        concat = torch.cat(band_outputs, dim=2)  # (b, c, n_bands * n_feat)
        weights = self.band_attention(concat)     # (b, c, n_bands)

        # 加权求和
        stacked = torch.stack(band_outputs, dim=-1)  # (b, c, n_feat, n_bands)
        weights = weights.unsqueeze(2)                # (b, c, 1, n_bands)
        out = (stacked * weights).sum(dim=-1)         # (b, c, n_feat)

        return out
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
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
    """
    Model seasonality of time series using the Fourier series.
    """
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


### precompute_freqs_cis/reshape_for_broadcast/apply_rotary_emb are rope code adapted from LLama code
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 max_len = None
                 
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
            max_len * 4,  ## if we use register, the overall length can be longer.
            50000,  ## 
        )

     
        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))


    def forward(self, x, mask=None):
        # x = torch.cat([self.register.repeat(x.shape[0],1,1), x], 1)

        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)


        k = self.k_norm(k) + 0.1*k
        q = self.q_norm(q) + 0.1*q

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        if int(os.environ.get('hucfg_attention_rope_use', '-1')) == 1: 
            freqs_cis = self.freqs_cis.cuda()[0 : T]
            q, k = apply_rotary_emb(q.permute(0,2,1,3), k.permute(0,2,1,3), freqs_cis=freqs_cis)
            q, k = q.permute(0,2,1,3), k.permute(0,2,1,3)



        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        # att = torch.sigmoid(att)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        # y = y[:,self.regi_num:,:]

        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 max_len = None
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        self.q_norm = RMSNorm(n_embd)
        self.k_norm = RMSNorm(n_embd)

        self.freqs_cis = precompute_freqs_cis(
            n_embd // n_head,
            max_len * 4,
            50000,  ## hucfg913
        )
     

        self.regi_num = 128
        self.register = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))
        self.register_2 = nn.Parameter(torch.randn([1, self.regi_num, n_embd]))


    def forward(self, x, encoder_output, mask=None):
        
        # x = torch.cat([self.register.repeat(x.shape[0],1,1), x], 1)

        # encoder_output = torch.cat([self.register_2.repeat(x.shape[0],1,1), encoder_output], 1)

        
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output)
        q = self.query(x)



        k = self.k_norm(k) + 0.1*k  ## residual qk norm
        q = self.q_norm(q) + 0.1*q

        k = k.view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        freqs_cis = self.freqs_cis.cuda()[0 : T]
        q, k = apply_rotary_emb(q.permute(0,2,1,3), k.permute(0,2,1,3), freqs_cis=freqs_cis)
        q, k = q.permute(0,2,1,3), k.permute(0,2,1,3)

        

        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        # att = torch.sigmoid(att)  ## sigmoid attention infact 

        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)


        y = self.resid_drop(self.proj(y))
        # y = y[:,self.regi_num:,:]

        return y, att
        

class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 max_len = None
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len = max_len
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
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len = None
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                max_len = max_len
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
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
                 max_len = None
                 ):
        super().__init__()
        
        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.ln2 = AdaLayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop, 
                resid_pdrop=resid_pdrop,
                max_len = max_len
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                max_len = max_len
                )
        
        self.ln1_1 = AdaLayerNorm(n_embd)
        # self.ln1_1 = nn.LayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        # EEG 频带分解 (替代原来的 TrendBlock + FourierLayer)
        self.band_block = EEGBandBlock(n_channel, n_embd, n_feat, sfreq=200)

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

        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a

        # EEG 频带分解 (替代 trend + season)
        band_out = self.band_block(x)  # (b, c, n_feat)

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
        max_len = None
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
                max_len = max_len
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
        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate, max_len = self.max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd, max_len = self.max_len)

    def forward(self, input, t, padding_masks=None, return_res=False, label_emb=None):
        emb = self.emb(input)

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