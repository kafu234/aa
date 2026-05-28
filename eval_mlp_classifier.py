"""
eval_mlp_classifier.py — EEGNet / DE-Transformer 分类器评估生成数据质量
=========================================================

设计原则:
  1. 默认使用 EEGNet 直接输入 raw EEG: (B, 62, T) → 3 类
  2. 保留原 DE-Transformer 作为可选 baseline: --model de_transformer
  3. 支持只用真实数据评估: --real_only 或 --no_synthetic
  4. 不加 Mixup / Label Smoothing 等额外增强 → 保证对比公平

数据划分:
  默认 --split_mode session (SEED 标准协议):
    每个被试 session 1+2 → 训练, session 3 → 测试

用法:
    # 自动对比 (推荐)
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/result/generated_SEED_RAW_200.npz \
        --compare

    # 只用真实数据跑 EEGNet baseline
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --split_mode subject --test_subject 1 \
        --window 800 --model eegnet --real_only
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  SEED 62-channel electrode 3D coordinates
#  Source: MNE-Python standard_1020 montage
#  CB1/CB2: interpolated from midpoint of (O1,PO7) and (O2,PO8)
#  Coordinate: x(left-/right+), y(posterior-/anterior+), z(inferior-/superior+)
#  Unit: meters
# ============================================================

def _get_seed_62_coords():
    """
    SEED 62-channel 3D electrode coordinates (meters).
    Source: MNE-Python standard_1020 montage (mne.channels.make_standard_montage).
    CB1/CB2 (cerebellar): interpolated from O1/O2 + PO7/PO8.
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
        [-0.042127, -0.120449,  0.000815],  # CB1 (interpolated)
        [-0.029413, -0.112449,  0.008839],  # O1
        [ 0.000108, -0.114892,  0.014657],  # OZ
        [ 0.029843, -0.112156,  0.008800],  # O2
        [ 0.042755, -0.120156,  0.000765],  # CB2 (interpolated)
    ], dtype=torch.float32)
    # fmt: on
    return coords  # (62, 3)


# ============================================================
#  DE Feature Extraction (GPU, no grad)
# ============================================================

class DEFeatureExtractor(nn.Module):
    """
    Differential Entropy 特征提取.

    对每个通道、每个频带:
      FFT → 频带 mask → IFFT → 方差 → DE = 0.5 * log(2πe * σ²)

    支持任意窗口长度 (200, 400, 800 等), mask 在 forward 时动态计算.

    Input:  (B, 62, T) raw EEG
    Output: (B, 62, 5)  五频带 DE
    """
    BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 50)]

    def __init__(self, sfreq=200):
        super().__init__()
        self.sfreq = sfreq
        self.log_2pie = 0.5 * np.log(2 * np.pi * np.e)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 62, T) → (B, 62, 5), T 可以是任意长度"""
        T = x.shape[-1]
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq).to(x.device)
        x_fft = torch.fft.rfft(x, dim=-1)

        de_bands = []
        for low, high in self.BANDS:
            mask = ((freqs >= low) & (freqs < high)).float()
            x_band = torch.fft.irfft(x_fft * mask, n=T, dim=-1)
            var = x_band.var(dim=-1, keepdim=True).clamp(min=1e-10)
            de_bands.append(0.5 * torch.log(var) + self.log_2pie)
        return torch.cat(de_bands, dim=-1)  # (B, 62, 5)


# ============================================================
#  DE Classifier: 轻量 Transformer (带空间位置编码)
# ============================================================

class DEClassifier(nn.Module):
    """
    DE 特征分类器.

    (B, 62, 5) → band embedding → + spatial PE → Transformer → pool → 3 类

    不含任何数据增强, 作为评估生成数据质量的公平基线.
    """
    def __init__(self, n_channels=62, n_bands=5, d_model=128,
                 n_heads=4, n_layers=3, dropout=0.2, num_classes=3):
        super().__init__()

        # DE 特征提取 (frozen)
        self.de_extractor = DEFeatureExtractor()

        # Band embedding: 5 → d_model
        self.band_embed = nn.Sequential(
            nn.Linear(n_bands, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Spatial positional encoding (电极 3D 坐标)
        coords = _get_seed_62_coords()
        self.register_buffer('coords', coords)
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Classification head (mean pool → linear)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """x: (B, 62, 200) raw EEG → (B, 3) logits"""
        de = self.de_extractor(x)  # (B, 62, 5)

        # Embed + spatial PE
        h = self.band_embed(de)                             # (B, 62, d_model)
        h = h + self.spatial_proj(self.coords).unsqueeze(0) # + spatial PE

        # Transformer
        h = self.transformer(h)  # (B, 62, d_model)

        # Mean pool over channels → classify
        h = h.mean(dim=1)  # (B, d_model)
        return self.head(h)



# ============================================================
#  EEGNet Classifier: 直接输入 raw EEG (B, 62, T)
# ============================================================

class Conv2dWithMaxNorm(nn.Conv2d):
    """
    Conv2d with max-norm constraint, used by EEGNet's spatial/depthwise layer.

    Standard EEGNet constrains the spatial-filter weights rather than clipping
    the layer output. This implementation keeps groups/bias correctly, unlike
    some simplified third-party wrappers that accidentally ignore them.
    """
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


class EEGNetClassifier(nn.Module):
    """
    EEGNet for SEED raw EEG classification.

    Input:  x shape (B, 62, T)
    Output: logits shape (B, num_classes)

    depthwise_mode:
      - "standard": original EEGNet-style depthwise spatial convolution
      - "libeer":   LibEER-compatible stronger variant; spatial conv uses
                    groups=1, which mixes temporal filters and usually has
                    higher capacity. Useful for reproducing LibEER-like results.
    """
    def __init__(
        self,
        n_channels=62,
        n_times=800,
        num_classes=3,
        F1=8,
        D=2,
        dropout=0.5,
        kernel_length=None,
        depthwise_mode="standard",
        max_norm=1.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.depthwise_mode = depthwise_mode

        if kernel_length is None:
            # Match LibEER: kernel_size=(1, datapoints // 2).
            # For 4s SEED raw at 200 Hz, n_times=800 => kernel_length=400.
            kernel_length = max(16, n_times // 2)

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        if depthwise_mode == "standard":
            groups = F1
        elif depthwise_mode == "libeer":
            groups = 1
        else:
            raise ValueError("depthwise_mode must be 'standard' or 'libeer'")

        self.depth_conv = Conv2dWithMaxNorm(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=groups,
            bias=False,
            max_norm=max_norm,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding="same",
            groups=F1 * D,
            bias=False,
        )
        self.sep_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # More robust than hard-coding self.F2 * (n_times // 32).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_dim = self._forward_features(dummy).shape[1]
        self.fc = nn.Linear(flat_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depth_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        # Expected x: (B, 62, T). If a channel dimension is already present,
        # accept (B, 1, 62, T) as well.
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"EEGNet expects input (B, C, T) or (B, 1, C, T), got {tuple(x.shape)}")
        x = self._forward_features(x)
        return self.fc(x)

# ============================================================
#  数据加载 (支持 session 划分)
# ============================================================

def load_data_by_session(data_root, window=200, seed=42, split_mode="session",
                         subject=None, train_trials=None, test_trials=None,
                         test_subject=None):
    """
    返回 (train_data, train_labels, test_data, test_labels).

    split_mode="session": session 1+2 训练, session 3 测试 (SEED 标准协议).
    split_mode="trial":   每个 session 前 9 个 trial 训练, 后 6 个 trial 测试.
    split_mode="subject": 14 个被试训练, 1 个被试测试 (LOSO).
    split_mode="random":  随机 60/40 划分.
    subject: None=所有被试, int=指定被试编号 (仅 session/trial 模式).
    test_subject: int=测试被试编号 (仅 subject 模式).
    """
    from Utils.Data_utils.seed_dataset import SEEDDataset

    # subject 模式不能同时指定 subject 过滤
    subjects = [subject] if (subject is not None and split_mode != "subject") else None

    common = dict(
        name="SEED_RAW", data_root=data_root, data_type="raw",
        window=window, proportion=1.0, seed=seed,
        conditional=True, sfreq=200,
        bandpass_low=0.5, bandpass_high=50.0,
        notch_freq=50.0, notch_width=2.0, baseline_correction=True,
        split_mode=split_mode,
        subjects=subjects,
    )
    if train_trials is not None:
        common["train_trials"] = train_trials
    if test_trials is not None:
        common["test_trials"] = test_trials
    if test_subject is not None:
        common["test_subject"] = test_subject

    ds_train = SEEDDataset(**common, period="train")
    ds_test  = SEEDDataset(**common, period="test")

    subj_str = f"被试 {subject}" if subject is not None else "所有被试"
    print(f"[{subj_str}] Train: {ds_train.samples.shape[0]} samples, "
          f"labels: {dict(zip(*np.unique(ds_train.labels, return_counts=True)))}")
    print(f"[{subj_str}] Test:  {ds_test.samples.shape[0]} samples, "
          f"labels: {dict(zip(*np.unique(ds_test.labels, return_counts=True)))}")

    return ds_train.samples, ds_train.labels, ds_test.samples, ds_test.labels


def load_synthetic_data(synthetic_path):
    """加载生成数据."""
    bundle = np.load(synthetic_path)
    data, labels = bundle["data"], bundle["labels"]
    CLIP_STD = 5.0
    data = np.clip(data / CLIP_STD, -1.0, 1.0)
    print(f"[Synthetic] {data.shape[0]} samples, shape {data.shape}, "
          f"labels {dict(zip(*np.unique(labels, return_counts=True)))}")
    return data.astype(np.float32), labels.astype(np.int64)


# ============================================================
#  训练 & 评估 (干净, 无任何增强技巧)
# ============================================================

def train_and_evaluate(
    train_data, train_labels, test_data, test_labels,
    epochs=150, batch_size=512, lr=3e-4,
    weight_decay=1e-4, device="cpu", verbose=True, run_name="",
    model_type="eegnet", d_model=128, n_heads=4, n_layers=3, dropout=0.3,
    eegnet_F1=8, eegnet_D=2, eegnet_kernel_length=None,
    eegnet_depthwise_mode="standard", eegnet_max_norm=1.0,
):
    X_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_labels).long()
    X_test  = torch.from_numpy(test_data).float()
    y_test  = torch.from_numpy(test_labels).long()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, drop_last=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # 类别权重 (处理不平衡)
    class_counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 3.0
    class_weights = torch.from_numpy(class_weights).to(device)

    if model_type == "eegnet":
        n_channels, n_times = train_data.shape[1], train_data.shape[2]
        model = EEGNetClassifier(
            n_channels=n_channels, n_times=n_times, num_classes=3,
            F1=eegnet_F1, D=eegnet_D, dropout=dropout,
            kernel_length=eegnet_kernel_length,
            depthwise_mode=eegnet_depthwise_mode,
            max_norm=eegnet_max_norm,
        ).to(device)
    elif model_type == "de_transformer":
        model = DEClassifier(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_acc = 0.0
    best_state = None
    best_epoch = 0
    patience = 30
    no_improve = 0

    pbar = tqdm(range(1, epochs + 1), desc=f"Training {run_name}", disable=not verbose)
    for epoch in pbar:
        # ---- Train ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)

        scheduler.step()

        # ---- Eval ----
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(xb)
                else:
                    logits = model(xb)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_true.append(yb.numpy())

        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        test_acc = accuracy_score(all_true, all_preds)

        pbar.set_postfix(
            loss=f"{total_loss/total:.4f}",
            tr=f"{correct/total:.3f}",
            te=f"{test_acc:.3f}",
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best={best_epoch})")
            break

    # ---- Final eval with best model ----
    model.load_state_dict(best_state)
    model.to(device).eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
            else:
                logits = model(xb)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    acc = accuracy_score(all_true, all_preds)
    f1_mac = f1_score(all_true, all_preds, average="macro")
    f1_wt = f1_score(all_true, all_preds, average="weighted")
    cm = confusion_matrix(all_true, all_preds, labels=[0, 1, 2])

    per_class_acc = {}
    label_names = {0: "negative", 1: "neutral", 2: "positive"}
    for c in range(3):
        mask = all_true == c
        per_class_acc[label_names[c]] = (all_preds[mask] == c).mean() if mask.sum() > 0 else 0.0

    report = classification_report(
        all_true, all_preds,
        target_names=["negative", "neutral", "positive"], digits=4,
    )

    return {
        "accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wt,
        "per_class_acc": per_class_acc, "confusion_matrix": cm,
        "report": report, "best_epoch": best_epoch,
        "train_samples": len(train_labels), "test_samples": len(test_labels),
    }


# ============================================================
#  打印结果
# ============================================================

def print_results(results, title="Results"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  训练样本: {results['train_samples']}")
    print(f"  测试样本: {results['test_samples']}")
    print(f"  最佳 Epoch: {results['best_epoch']}")
    print(f"  ---")
    print(f"  Accuracy:    {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  F1 (macro):  {results['f1_macro']:.4f}")
    print(f"  F1 (weight): {results['f1_weighted']:.4f}")
    print(f"  ---")
    print(f"  各类别准确率:")
    for cls, acc in results["per_class_acc"].items():
        print(f"    {cls:>10s}: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\n  分类报告:")
    for line in results["report"].split("\n"):
        print(f"    {line}")
    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    cm = results["confusion_matrix"]
    print(f"    {'':>10s}  {'negative':>8s}  {'neutral':>8s}  {'positive':>8s}")
    for i, name in enumerate(["negative", "neutral", "positive"]):
        row = "  ".join(f"{cm[i,j]:>8d}" for j in range(3))
        print(f"    {name:>10s}  {row}")
    print(f"{'='*60}\n")


def print_comparison(res_no_syn, res_with_syn):
    print(f"\n{'#'*60}")
    print(f"  对比: 生成数据对分类性能的影响")
    print(f"{'#'*60}")

    print(f"\n  {'指标':<16s} {'无生成数据':>12s} {'有生成数据':>12s} {'差值':>10s} {'变化':>8s}")
    print(f"  {'-'*58}")
    for name, key in [("Accuracy","accuracy"),("F1 (macro)","f1_macro"),("F1 (weighted)","f1_weighted")]:
        v1, v2 = res_no_syn[key], res_with_syn[key]
        diff = v2 - v1
        sign = "↑" if diff > 0.001 else ("↓" if diff < -0.001 else "→")
        print(f"  {name:<16s} {v1:>11.4f}  {v2:>11.4f}  {diff:>+9.4f}  {sign:>6s}")

    print(f"\n  各类别准确率:")
    print(f"  {'类别':<12s} {'无生成数据':>12s} {'有生成数据':>12s} {'差值':>10s}")
    print(f"  {'-'*48}")
    for cls in ["negative", "neutral", "positive"]:
        v1, v2 = res_no_syn["per_class_acc"][cls], res_with_syn["per_class_acc"][cls]
        print(f"  {cls:<12s} {v1:>11.4f}  {v2:>11.4f}  {v2-v1:>+9.4f}")

    print(f"\n  训练集: {res_no_syn['train_samples']} → {res_with_syn['train_samples']}"
          f" (+{res_with_syn['train_samples'] - res_no_syn['train_samples']} 生成样本)")
    print(f"  测试集: {res_no_syn['test_samples']} (相同)")

    acc_diff = res_with_syn["accuracy"] - res_no_syn["accuracy"]
    f1_diff = res_with_syn["f1_macro"] - res_no_syn["f1_macro"]
    print(f"\n  结论: ", end="")
    if acc_diff > 0.01 and f1_diff > 0.01:
        print("生成数据显著提升了分类性能 ✅")
    elif acc_diff > 0.0 or f1_diff > 0.0:
        print("生成数据略微提升了分类性能")
    elif abs(acc_diff) < 0.01 and abs(f1_diff) < 0.01:
        print("生成数据对分类性能无明显影响")
    else:
        print("生成数据降低了分类性能 ⚠️")
    print(f"{'#'*60}\n")


# ============================================================
#  主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="EEGNet / DE-Transformer 分类器评估生成数据质量"
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--no_synthetic", action="store_true",
                        help="不加载生成数据，只用真实数据训练/评估")
    parser.add_argument("--real_only", action="store_true",
                        help="只用真实数据训练/评估；等价于 --no_synthetic，并会关闭 --compare")
    parser.add_argument("--compare", action="store_true",
                        help="先跑真实数据 baseline，再跑真实+生成数据，并打印对比")
    parser.add_argument("--split_mode", type=str, default="session",
                        choices=["random", "session", "trial", "subject"],
                        help="session=跨session, trial=跨trial, subject=跨被试(LOSO), random=随机")
    parser.add_argument("--subject", type=int, default=None,
                        help="指定被试编号 (1-15), 不指定则使用所有被试")
    parser.add_argument("--test_subject", type=int, default=None,
                        help="跨被试模式: 测试被试编号 (如 15), 其余为训练")
    parser.add_argument("--train_trials", type=str, default=None,
                        help="训练用的 trial 编号, 如 '0,1,2,3,4,5,6,7,8' (默认前9个)")
    parser.add_argument("--test_trials", type=str, default=None,
                        help="测试用的 trial 编号, 如 '9,10,11,12,13,14' (默认后6个)")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # 模型选择
    parser.add_argument("--model", type=str, default="eegnet",
                        choices=["eegnet", "de_transformer"],
                        help="默认 eegnet；de_transformer 为原来的 DE+Transformer 评估器")

    # 原 DE-Transformer 超参数，仅 --model de_transformer 时使用
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)

    # EEGNet 超参数，仅 --model eegnet 时使用
    parser.add_argument("--eegnet_F1", type=int, default=8)
    parser.add_argument("--eegnet_D", type=int, default=2)
    parser.add_argument("--eegnet_kernel_length", type=int, default=None,
                        help="EEGNet temporal kernel length；默认 window//2，和 LibEER 一致")
    parser.add_argument("--eegnet_depthwise_mode", type=str, default="standard",
                        choices=["standard", "libeer"],
                        help="standard=标准 EEGNet depthwise spatial conv；libeer=更接近 LibEER 的高容量空间卷积")
    parser.add_argument("--eegnet_max_norm", type=float, default=1.0)

    parser.add_argument("--syn_ratio", type=float, default=1.0,
                        help="使用多少比例的生成数据 (可 >1.0, 会重复采样)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="分类器 dropout (默认 0.3)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3)

    args = parser.parse_args()
    if args.real_only:
        args.no_synthetic = True
        args.compare = False

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    subj_str = f"被试 {args.subject}" if args.subject else "所有被试"
    print(f"Device: {device}, Split: {args.split_mode}, {subj_str}, Model: {args.model}")
    if args.model == "eegnet":
        print(f"EEGNet: F1={args.eegnet_F1}, D={args.eegnet_D}, "
              f"kernel_length={args.eegnet_kernel_length or args.window // 2}, "
              f"depthwise_mode={args.eegnet_depthwise_mode}")

    # ---- 加载数据 (按 session/trial 划分) ----
    print(f"\n加载 SEED 数据...")
    train_trials = [int(x) for x in args.train_trials.split(",")] if args.train_trials else None
    test_trials = [int(x) for x in args.test_trials.split(",")] if args.test_trials else None
    train_orig, train_orig_labels, test_data, test_labels = load_data_by_session(
        args.data_root, args.window, args.seed, args.split_mode, args.subject,
        train_trials, test_trials, args.test_subject,
    )

    # ---- 加载生成数据 ----
    syn_data, syn_labels = None, None
    if args.synthetic_path and not args.no_synthetic:
        print("\n加载生成数据...")
        syn_data, syn_labels = load_synthetic_data(args.synthetic_path)
        if args.syn_ratio != 1.0:
            n_target = int(len(syn_labels) * args.syn_ratio)
            if n_target > len(syn_labels):
                # syn_ratio > 1.0: 重复采样放大
                repeats = int(np.ceil(n_target / len(syn_labels)))
                syn_data = np.tile(syn_data, (repeats, 1, 1))[:n_target]
                syn_labels = np.tile(syn_labels, repeats)[:n_target]
            else:
                # syn_ratio < 1.0: 随机采样缩小
                idx = np.random.choice(len(syn_labels), n_target, replace=False)
                syn_data, syn_labels = syn_data[idx], syn_labels[idx]
            print(f"使用 syn_ratio={args.syn_ratio}: {len(syn_labels)} 生成样本")

    # ---- 运行模式 ----
    if args.compare:
        assert syn_data is not None, "--compare 需要 --synthetic_path"
        modes = ["no_synthetic", "with_synthetic"]
    elif args.no_synthetic or syn_data is None:
        modes = ["no_synthetic"]
    else:
        modes = ["with_synthetic"]

    all_results = {}

    for mode in modes:
        print(f"\n{'*'*60}\n  模式: {mode}\n{'*'*60}")
        run_accs, run_f1s, run_results = [], [], []

        for run_i in range(args.n_runs):
            run_seed = args.seed + run_i

            if mode == "with_synthetic":
                train_data = np.concatenate([train_orig, syn_data], axis=0)
                train_labels = np.concatenate([train_orig_labels, syn_labels], axis=0)
                rn = f"With Syn ({run_i+1}/{args.n_runs})"
            else:
                train_data, train_labels = train_orig, train_orig_labels
                rn = f"No Syn ({run_i+1}/{args.n_runs})"

            print(f"\n  Run {run_i+1}: train={len(train_labels)}, test={len(test_labels)}")

            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            results = train_and_evaluate(
                train_data, train_labels, test_data, test_labels,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                device=device, verbose=True, run_name=rn,
                model_type=args.model,
                d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=args.dropout,
                eegnet_F1=args.eegnet_F1, eegnet_D=args.eegnet_D,
                eegnet_kernel_length=args.eegnet_kernel_length,
                eegnet_depthwise_mode=args.eegnet_depthwise_mode,
                eegnet_max_norm=args.eegnet_max_norm,
            )
            run_accs.append(results["accuracy"])
            run_f1s.append(results["f1_macro"])
            run_results.append(results)
            print_results(results, title=rn)

        if args.n_runs > 1:
            print(f"\n  === {mode} 汇总 ({args.n_runs} runs) ===")
            print(f"  Accuracy: {np.mean(run_accs):.4f} ± {np.std(run_accs):.4f}")
            print(f"  F1 macro: {np.mean(run_f1s):.4f} ± {np.std(run_f1s):.4f}")

        avg_result = run_results[-1].copy()
        avg_result["accuracy"] = np.mean(run_accs)
        avg_result["f1_macro"] = np.mean(run_f1s)
        avg_result["f1_weighted"] = np.mean([r["f1_weighted"] for r in run_results])
        avg_result["per_class_acc"] = {
            cls: np.mean([r["per_class_acc"][cls] for r in run_results])
            for cls in ["negative", "neutral", "positive"]
        }
        all_results[mode] = avg_result

    if args.compare and len(all_results) == 2:
        print_comparison(all_results["no_synthetic"], all_results["with_synthetic"])

    print("Done!")


if __name__ == "__main__":
    main()