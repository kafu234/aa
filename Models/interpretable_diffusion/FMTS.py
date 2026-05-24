import math
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import SpatialEmotionConditioner
import os


# ============================================================
#  Guidance Classifier (可微分, 用于训练时引导生成)
# ============================================================

class GuidanceClassifier(nn.Module):
    """
    可微分的 DE 分类器, 用于 Classifier Guidance.

    与 eval 里的 DEClassifier 区别:
      - DE 提取 **没有** @torch.no_grad(), 梯度能回传到生成器
      - 结构更轻量 (2层, d=64), 减少额外计算开销
      - 预训练后冻结, 只做 "裁判" 不参与更新

    训练时的梯度流:
      cls_loss → logits → classifier(x_hat) → x_hat → model_out → 生成器参数
                          ↑ classifier 参数冻结, 不更新
    """
    BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]

    def __init__(self, n_channels=62, n_timepoints=200, sfreq=200,
                 d_model=64, n_heads=4, n_layers=2, num_classes=3, dropout=0.1):
        super().__init__()

        # Band masks for DE extraction (differentiable)
        freqs = torch.fft.rfftfreq(n_timepoints, d=1.0 / sfreq)
        masks = []
        for low, high in self.BANDS:
            masks.append(((freqs >= low) & (freqs < high)).float())
        self.register_buffer('band_masks', torch.stack(masks))
        self.n_timepoints = n_timepoints

        # Classifier
        self.embed = nn.Sequential(
            nn.Linear(len(self.BANDS), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def extract_de(self, x):
        """
        可微分 DE 提取. 无 @torch.no_grad(), 梯度可通过 FFT 回传.
        x: (B, C, T) → (B, C, 5)
        """
        T = x.shape[-1]
        x_fft = torch.fft.rfft(x, dim=-1)
        de_bands = []
        for i in range(len(self.BANDS)):
            mask = self.band_masks[i]
            x_band = torch.fft.irfft(x_fft * mask, n=T, dim=-1)
            var = x_band.var(dim=-1, keepdim=True).clamp(min=1e-10)
            de_bands.append(0.5 * torch.log(var))
        return torch.cat(de_bands, dim=-1)

    def forward(self, x):
        """x: (B, 62, 200) → (B, num_classes) logits"""
        de = self.extract_de(x)    # (B, 62, 5) — differentiable!
        h = self.embed(de)         # (B, 62, d_model)
        h = self.encoder(h)        # (B, 62, d_model)
        h = h.mean(dim=1)          # (B, d_model)
        return self.head(h)        # (B, num_classes)


# ============================================================
#  FM_TS with Classifier Guidance
# ============================================================

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
            num_classes=0,
            classifier_weight=0.02,
            cfg_dropout=0.15,        # ← NEW: CFG 训练时随机丢弃标签的概率
            spectral_weight=0.1,     # ← NEW: 频谱一致性损失权重
            guidance_scale=2.0,      # ← NEW: CFG 推理时的引导强度
            **kwargs
    ):
        super(FM_TS, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.classifier_weight = classifier_weight
        self.cfg_dropout = cfg_dropout
        self.spectral_weight = spectral_weight
        self.guidance_scale = guidance_scale

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        # 条件生成: 通道感知的情绪嵌入
        if num_classes > 0:
            _d_model = d_model if d_model is not None else (n_heads * seq_length)
            self.label_embedding = SpatialEmotionConditioner(
                num_classes=num_classes,
                n_channel=seq_length,
                d_model=_d_model,
                n_modes=8,
            )
        else:
            self.label_embedding = None

        # Guidance classifier (initially None, call pretrain/load to set)
        self.guidance_classifier = None

        self.alpha = 3
        self.time_scalar = 1000
        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))

    # ---- Classifier Guidance 相关方法 ----

    def pretrain_classifier(self, train_data, train_labels,
                            epochs=50, batch_size=256, lr=1e-3, device=None):
        """
        用真实数据预训练引导分类器, 然后冻结.

        Args:
            train_data:   (N, seq_length, feature_size) numpy array
            train_labels: (N,) numpy array
            epochs: 预训练轮数
            batch_size: batch size
            lr: 学习率
            device: 训练设备
        """
        if device is None:
            device = next(self.parameters()).device

        print(f"\n{'='*50}")
        print(f"  预训练 Guidance Classifier")
        print(f"  数据: {train_data.shape[0]} 样本, {len(np.unique(train_labels))} 类")
        print(f"{'='*50}")

        classifier = GuidanceClassifier(
            n_channels=self.seq_length,
            n_timepoints=self.feature_size,
            num_classes=self.num_classes,
        ).to(device)

        optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        X = torch.from_numpy(train_data).float()
        y = torch.from_numpy(train_labels).long()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        classifier.train()
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = classifier(xb)
                loss = F.cross_entropy(logits, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                total += xb.size(0)
            scheduler.step()

            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

            if epoch % 10 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:>3d}/{epochs}: loss={total_loss/total:.4f}, acc={acc:.4f}")

        # Load best and freeze
        classifier.load_state_dict(best_state)
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False

        self.guidance_classifier = classifier
        print(f"  Classifier 预训练完成, best acc={best_acc:.4f}, 已冻结")

        # 根据准确率自动调整引导强度
        if best_acc < 0.5:
            print(f"  ⚠️ 准确率过低 (<50%), 自动关闭 classifier guidance")
            self.classifier_weight = 0.0
        elif best_acc < 0.75:
            recommended = min(self.classifier_weight, 0.02)
            print(f"  ⚡ 准确率一般 ({best_acc:.0%}), 建议 classifier_weight ≤ {recommended}")
            self.classifier_weight = recommended
        else:
            print(f"  ✅ 准确率良好 ({best_acc:.0%}), classifier_weight={self.classifier_weight}")

        print(f"{'='*50}\n")
        return best_acc

    def save_classifier(self, path):
        """保存预训练好的引导分类器."""
        if self.guidance_classifier is not None:
            torch.save(self.guidance_classifier.state_dict(), path)
            print(f"Guidance classifier saved to {path}")

    def load_classifier(self, path, device=None):
        """加载预训练好的引导分类器."""
        if device is None:
            device = next(self.parameters()).device
        self.guidance_classifier = GuidanceClassifier(
            n_channels=self.seq_length,
            n_timepoints=self.feature_size,
            num_classes=self.num_classes,
        ).to(device)
        self.guidance_classifier.load_state_dict(torch.load(path, map_location=device))
        self.guidance_classifier.eval()
        for p in self.guidance_classifier.parameters():
            p.requires_grad = False
        print(f"Guidance classifier loaded from {path}")

    # ---- 原有方法 ----

    def output(self, x, t, padding_masks=None, label_emb=None):
        output = self.model(x, t, padding_masks=None, label_emb=label_emb)
        return output

    @torch.no_grad()
    def sample(self, shape, labels=None):
        """
        采样, 支持 Classifier-Free Guidance.

        CFG 推理: v = v_uncond + guidance_scale * (v_cond - v_uncond)
        guidance_scale=1.0 等同于普通条件生成, >1.0 增强条件信号.
        """
        self.eval()
        zt = torch.randn(shape).cuda()

        label_emb = None
        use_cfg = False
        if labels is not None and self.label_embedding is not None:
            label_emb = self.label_embedding(labels.cuda())
            # 只在 guidance_scale != 1.0 且训练时用了 cfg_dropout 时启用 CFG
            if self.guidance_scale != 1.0 and self.cfg_dropout > 0:
                use_cfg = True

        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr * self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).cuda().squeeze()

            # 条件预测
            v_cond = self.output(zt.clone(), t_input, padding_masks=None, label_emb=label_emb)

            if use_cfg:
                # 无条件预测 (label_emb=None → 零向量)
                zero_emb = torch.zeros_like(label_emb)
                v_uncond = self.output(zt.clone(), t_input, padding_masks=None, label_emb=zero_emb)
                # CFG 插值: 增强条件方向
                v = v_uncond + self.guidance_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            zt = zt.clone() + step * v

        return zt

    def generate_mts(self, batch_size=16, labels=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        if isinstance(labels, int):
            labels = torch.full((batch_size,), labels, dtype=torch.long)
        return self.sample((batch_size, seq_length, feature_size), labels=labels)

    def _train_loss(self, x_start, labels=None):
        """
        Flow Matching 训练损失 + 频谱一致性损失 + Classifier-Free Guidance 训练.

        总损失 = FM_loss + spectral_weight * spectral_loss

        CFG 训练: 以 cfg_dropout 概率随机丢弃 label_emb,
                  让模型同时学会条件生成和无条件生成.
        """
        z0 = torch.randn_like(x_start)
        z1 = x_start

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0

        # ---- 条件 embedding (CFG: 随机 drop) ----
        label_emb = None
        if labels is not None and self.label_embedding is not None:
            label_emb = self.label_embedding(labels)
            # Classifier-Free Guidance 训练: 随机丢弃条件
            if self.training and self.cfg_dropout > 0:
                drop_mask = torch.rand(z0.shape[0], device=z0.device) < self.cfg_dropout
                if drop_mask.any():
                    # 被选中的样本不传条件 → 模型学无条件生成
                    label_emb = label_emb.clone()
                    label_emb[drop_mask] = 0.0

        model_out = self.output(z_t, t.squeeze() * self.time_scalar, None, label_emb=label_emb)

        # ---- Flow Matching loss ----
        fm_loss = F.mse_loss(model_out, target, reduction='none')
        fm_loss = reduce(fm_loss, 'b ... -> b (...)', 'mean').mean()

        total_loss = fm_loss

        # ---- 频谱一致性损失 ----
        if self.spectral_weight > 0:
            # 重建干净样本
            x_hat = z_t + (1.0 - t) * model_out        # (B, C, T)
            # 真实 vs 重建的频谱幅度
            spec_real = torch.fft.rfft(x_start, dim=-1).abs()
            spec_hat  = torch.fft.rfft(x_hat, dim=-1).abs()
            spectral_loss = F.mse_loss(spec_hat, spec_real)
            # t 越大重建越可靠, 频谱损失权重越高
            t_weight = t.squeeze().mean().item()
            total_loss = total_loss + self.spectral_weight * t_weight * spectral_loss

        # ---- Classifier Guidance loss (可选, 与 CFG 共存) ----
        if self.guidance_classifier is not None and labels is not None and self.classifier_weight > 0:
            t_flat = t.squeeze()
            high_t_mask = (t_flat > 0.5)
            if high_t_mask.any():
                if not self.spectral_weight > 0:
                    x_hat = z_t + (1.0 - t) * model_out
                x_hat_clamped = x_hat.clamp(-1, 1)
                logits = self.guidance_classifier(x_hat_clamped)
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values
                confident_mask = (confidence > 0.5) & high_t_mask
                if confident_mask.any():
                    cls_loss = F.cross_entropy(logits[confident_mask], labels[confident_mask])
                    t_w = t_flat[confident_mask].mean().item()
                    total_loss = total_loss + self.classifier_weight * t_w * cls_loss

        return total_loss

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