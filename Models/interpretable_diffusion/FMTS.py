import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.interpretable_diffusion.transformer import Transformer
import os



class FM_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            num_classes=0,       # ← 新增: 0=无条件, >0=条件生成的类别数
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        # 条件生成: 将离散类别标签映射为 d_model 维向量
        if num_classes > 0:
            _d_model = d_model if d_model is not None else (n_heads * seq_length)
            self.label_embedding = nn.Embedding(num_classes, _d_model)
        else:
            self.label_embedding = None

        self.alpha = 3  ## t shifting
        self.time_scalar = 1000

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))
    
    def output(self, x, t, padding_masks=None, label_emb=None):
        output = self.model(x, t, padding_masks=None, label_emb=label_emb)
        return output


    @torch.no_grad()
    def sample(self, shape, labels=None):
        """
        生成样本。
        labels: (batch_size,) LongTensor, 每个样本的目标类别。
                None 时为无条件生成。
        """
        self.eval()

        zt = torch.randn(shape).cuda()

        # 条件 embedding
        label_emb = None
        if labels is not None and self.label_embedding is not None:
            label_emb = self.label_embedding(labels.cuda())

        ## t shifting from stable diffusion 3
        timesteps = torch.linspace(0, 1, self.num_timesteps+1)
        t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            v = self.output(
                zt.clone(),
                torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).cuda().squeeze(),
                padding_masks=None,
                label_emb=label_emb,
            )
            zt = zt.clone() + step * v 

        return zt 


    def generate_mts(self, batch_size=16, labels=None):
        """
        生成多变量时序。
        labels: (batch_size,) LongTensor 或 None。
                传入 int 时，生成该类别的 batch_size 个样本。
        """
        feature_size, seq_length = self.feature_size, self.seq_length
        if isinstance(labels, int):
            labels = torch.full((batch_size,), labels, dtype=torch.long)
        return self.sample((batch_size, seq_length, feature_size), labels=labels)


    def _train_loss(self, x_start, labels=None):
        """
        Flow Matching 训练损失。
        labels: (batch,) LongTensor 或 None。
        """
        z0 = torch.randn_like(x_start) 
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t =  t * z1 + (1.-t) * z0
        target = z1 - z0

        # 条件 embedding
        label_emb = None
        if labels is not None and self.label_embedding is not None:
            label_emb = self.label_embedding(labels)

        model_out = self.output(z_t, t.squeeze()*self.time_scalar, None, label_emb=label_emb)
        train_loss = F.mse_loss(model_out, target, reduction='none')

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss.mean()
        return train_loss.mean()

    def forward(self, x, labels=None):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        return self._train_loss(x_start=x, labels=labels)


    def fast_sample_infill(self, shape, target, partial_mask=None, labels=None):
        z0 = torch.randn(shape).cuda()
        z1 = zt = z0

        label_emb = None
        if labels is not None and self.label_embedding is not None:
            label_emb = self.label_embedding(labels.cuda())

        for t in range(self.num_timesteps):
            t = t/self.num_timesteps
            t = t**(float(os.environ['hucfg_Kscale']))

            z0 = torch.randn(shape).cuda()
            target_t = target*t + z0*(1-t)
            zt = z1*t + z0*(1-t)
            zt[partial_mask] = target_t[partial_mask]
            v = self.output(zt, torch.tensor([t*self.time_scalar]).cuda(), None, label_emb=label_emb) 

            z1 = zt.clone() + (1 - t) * v
            z1 = torch.clamp(z1, min=-1, max=1)

        return z1