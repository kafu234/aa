"""
eval_mlp_classifier.py — DE特征 + 分类器评估生成数据质量
=========================================================

思路:
  1. GPU 上提取 5 频带 DE 特征: (B, 62, 200) → (B, 62, 5)
  2. 用一个轻量 Transformer 分类器 (带空间位置编码)
  3. 不加任何数据增强技巧 (无 Mixup / Label Smoothing)
     → 保证对比的公平性: 性能差异只来自生成数据本身

用法:
    # baseline (只用原始数据)
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG --no_synthetic

    # 原始 + 生成数据
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/result/generated_SEED_RAW_200.npz

    # 自动对比 (推荐)
    python eval_mlp_classifier.py \
        --data_root /root/autodl-tmp/Preprocessed_EEG \
        --synthetic_path /root/autodl-tmp/result/generated_SEED_RAW_200.npz \
        --compare
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
#  SEED 62-channel electrode 3D coordinates (10-20 system)
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
    coords_2d = {
        'FP1': (-0.15, 0.92), 'FPZ': (0.00, 0.95), 'FP2': (0.15, 0.92),
        'AF3': (-0.25, 0.82), 'AF4': (0.25, 0.82),
        'F7': (-0.70, 0.60), 'F5': (-0.52, 0.60), 'F3': (-0.35, 0.60),
        'F1': (-0.15, 0.60), 'FZ': (0.00, 0.60), 'F2': (0.15, 0.60),
        'F4': (0.35, 0.60), 'F6': (0.52, 0.60), 'F8': (0.70, 0.60),
        'FT7': (-0.80, 0.35), 'FC5': (-0.55, 0.35), 'FC3': (-0.35, 0.35),
        'FC1': (-0.15, 0.35), 'FCZ': (0.00, 0.35), 'FC2': (0.15, 0.35),
        'FC4': (0.35, 0.35), 'FC6': (0.55, 0.35), 'FT8': (0.80, 0.35),
        'T7': (-0.90, 0.00), 'C5': (-0.58, 0.00), 'C3': (-0.35, 0.00),
        'C1': (-0.15, 0.00), 'CZ': (0.00, 0.00), 'C2': (0.15, 0.00),
        'C4': (0.35, 0.00), 'C6': (0.58, 0.00), 'T8': (0.90, 0.00),
        'TP7': (-0.80, -0.35), 'CP5': (-0.55, -0.35), 'CP3': (-0.35, -0.35),
        'CP1': (-0.15, -0.35), 'CPZ': (0.00, -0.35), 'CP2': (0.15, -0.35),
        'CP4': (0.35, -0.35), 'CP6': (0.55, -0.35), 'TP8': (0.80, -0.35),
        'P7': (-0.70, -0.60), 'P5': (-0.52, -0.60), 'P3': (-0.35, -0.60),
        'P1': (-0.15, -0.60), 'PZ': (0.00, -0.60), 'P2': (0.15, -0.60),
        'P4': (0.35, -0.60), 'P6': (0.52, -0.60), 'P8': (0.70, -0.60),
        'PO7': (-0.55, -0.78), 'PO5': (-0.38, -0.78), 'PO3': (-0.22, -0.78),
        'POZ': (0.00, -0.78), 'PO4': (0.22, -0.78), 'PO6': (0.38, -0.78),
        'PO8': (0.55, -0.78),
        'CB1': (-0.35, -0.92), 'O1': (-0.15, -0.92), 'OZ': (0.00, -0.95),
        'O2': (0.15, -0.92), 'CB2': (0.35, -0.92),
    }
    xy = torch.tensor([coords_2d[ch] for ch in SEED_62_CHANNELS], dtype=torch.float32)
    r2 = (xy ** 2).sum(dim=1, keepdim=True).clamp(max=0.99)
    z = torch.sqrt(1.0 - r2)
    return torch.cat([xy, z], dim=1)  # (62, 3)


# ============================================================
#  DE Feature Extraction (GPU, no grad)
# ============================================================

class DEFeatureExtractor(nn.Module):
    """
    Differential Entropy 特征提取.

    对每个通道、每个频带:
      FFT → 频带 mask → IFFT → 方差 → DE = 0.5 * log(2πe * σ²)

    Input:  (B, 62, 200) raw EEG
    Output: (B, 62, 5)   五频带 DE
    """
    BANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50),
    }

    def __init__(self, n_timepoints=200, sfreq=200):
        super().__init__()
        freqs = torch.fft.rfftfreq(n_timepoints, d=1.0 / sfreq)
        masks = []
        for _, (low, high) in self.BANDS.items():
            masks.append(((freqs >= low) & (freqs < high)).float())
        self.register_buffer('band_masks', torch.stack(masks))  # (5, n_freq)
        self.n_timepoints = n_timepoints
        self.log_2pie = 0.5 * np.log(2 * np.pi * np.e)

    @torch.no_grad()
    def forward(self, x):
        """x: (B, 62, 200) → (B, 62, 5)"""
        T = x.shape[-1]
        x_fft = torch.fft.rfft(x, dim=-1)
        de_bands = []
        for i in range(len(self.BANDS)):
            mask = self.band_masks[i]
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
#  数据加载
# ============================================================

def load_original_data(data_root, window=200, seed=42):
    from Utils.Data_utils.seed_dataset import SEEDDataset
    ds = SEEDDataset(
        name="SEED_RAW", data_root=data_root, data_type="raw",
        window=window, proportion=1.0, seed=seed, period="train",
        conditional=True, sfreq=200, bandpass_low=0.5, bandpass_high=50.0,
        notch_freq=50.0, notch_width=2.0, baseline_correction=True,
    )
    return ds.samples, ds.labels


def load_synthetic_data(synthetic_path):
    bundle = np.load(synthetic_path)
    data, labels = bundle["data"], bundle["labels"]
    CLIP_STD = 5.0
    data = np.clip(data / CLIP_STD, -1.0, 1.0)
    print(f"[Synthetic] {data.shape[0]} samples, shape {data.shape}, labels {np.unique(labels)}")
    return data.astype(np.float32), labels.astype(np.int64)


def split_data(data, labels, train_ratio=0.6, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(labels))
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        c_idx = indices[labels == c]
        np.random.shuffle(c_idx)
        n_train = int(len(c_idx) * train_ratio)
        train_idx.extend(c_idx[:n_train])
        test_idx.extend(c_idx[n_train:])
    train_idx, test_idx = np.array(train_idx), np.array(test_idx)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]


# ============================================================
#  训练 & 评估 (干净, 无任何增强技巧)
# ============================================================

def train_and_evaluate(
    train_data, train_labels, test_data, test_labels,
    epochs=150, batch_size=512, lr=3e-4,
    weight_decay=1e-4, device="cpu", verbose=True, run_name="",
    d_model=128, n_heads=4, n_layers=3,
):
    X_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_labels).long()
    X_test  = torch.from_numpy(test_data).float()
    y_test  = torch.from_numpy(test_labels).long()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=2, pin_memory=True,
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

    model = DEClassifier(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
    ).to(device)

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

    # ---- Final eval ----
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
#  打印
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
    parser = argparse.ArgumentParser(description="DE特征 + 分类器评估生成数据质量")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_path", type=str, default=None)
    parser.add_argument("--no_synthetic", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--syn_ratio", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 加载原始数据 ----
    print("\n加载原始 SEED 数据...")
    orig_data, orig_labels = load_original_data(args.data_root, args.window, args.seed)
    print(f"原始数据: {orig_data.shape}, 标签: {dict(zip(*np.unique(orig_labels, return_counts=True)))}")

    # ---- 加载生成数据 ----
    syn_data, syn_labels = None, None
    if args.synthetic_path and not args.no_synthetic:
        print("\n加载生成数据...")
        syn_data, syn_labels = load_synthetic_data(args.synthetic_path)
        if args.syn_ratio < 1.0:
            n_use = int(len(syn_labels) * args.syn_ratio)
            idx = np.random.choice(len(syn_labels), n_use, replace=False)
            syn_data, syn_labels = syn_data[idx], syn_labels[idx]

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
            train_orig, train_orig_labels, test_data, test_labels = split_data(
                orig_data, orig_labels, args.train_ratio, run_seed,
            )

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
                d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
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